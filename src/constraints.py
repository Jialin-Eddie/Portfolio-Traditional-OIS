"""
constraints.py — Post-hoc constraint verification for Portfolio-Traditional-OIS.

Verifies weight bounds, factor exposure neutrality, and relative drawdown limits
after optimization, and generates a compliance report.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BETA_COLS, MAX_FACTOR_DEV, MAX_REL_DD_MONTHLY, MAX_WEIGHT_MULT, OUTPUTS


# ---------------------------------------------------------------------------
# Weight bounds verification
# ---------------------------------------------------------------------------


def verify_weight_bounds(
    opt_weights: pd.DataFrame,
    bench_weights: pd.DataFrame,
    max_mult: float = MAX_WEIGHT_MULT,
) -> pd.DataFrame:
    """Verify that optimized weights stay within [0, max_mult * bench_weight].

    Parameters
    ----------
    opt_weights:
        DataFrame with MultiIndex (permno, month_end) and column ``opt_weight``.
    bench_weights:
        DataFrame with MultiIndex (permno, month_end) and column ``bench_weight``.
    max_mult:
        Maximum allowed multiple of benchmark weight (default: MAX_WEIGHT_MULT).

    Returns
    -------
    pd.DataFrame
        Columns: month_end, n_total, n_violations, max_violation, pct_compliant.
    """
    merged = opt_weights[["opt_weight"]].join(
        bench_weights[["bench_weight"]], how="inner"
    )
    merged["upper_bound"] = max_mult * merged["bench_weight"]
    # Tolerance for floating point and weight renormalization effects
    tol = 1e-4
    merged["in_bounds"] = (merged["opt_weight"] >= -tol) & (
        merged["opt_weight"] <= merged["upper_bound"] + tol
    )
    merged["violation_amt"] = (
        (merged["opt_weight"] - merged["upper_bound"])
        .clip(lower=0)
        .where(merged["opt_weight"] < 0, other=(merged["opt_weight"] - merged["upper_bound"]).clip(lower=0))
    )
    # Recalculate: violation = max(0, opt_weight - upper_bound) + max(0, -opt_weight)
    merged["violation_amt"] = np.maximum(0, merged["opt_weight"] - merged["upper_bound"]) + np.maximum(
        0, -merged["opt_weight"]
    )

    def _agg(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        n_viol = int((~grp["in_bounds"]).sum())
        max_viol = float(grp["violation_amt"].max())
        pct = float(grp["in_bounds"].mean())
        return pd.Series(
            {"n_total": n, "n_violations": n_viol, "max_violation": max_viol, "pct_compliant": pct}
        )

    result = merged.groupby(level="month_end").apply(_agg).reset_index()
    result["n_total"] = result["n_total"].astype(int)
    result["n_violations"] = result["n_violations"].astype(int)

    total_violations = int(result["n_violations"].sum())
    overall_pct = float(result["pct_compliant"].mean())
    print(
        f"[constraints] Weight bounds: {total_violations} total violations across "
        f"{len(result)} months, avg compliance={overall_pct:.2%}"
    )
    return result


# ---------------------------------------------------------------------------
# Factor exposure verification
# ---------------------------------------------------------------------------


def verify_factor_exposure(
    opt_weights: pd.DataFrame,
    bench_weights: pd.DataFrame,
    monthly: pd.DataFrame,
    max_dev: float = MAX_FACTOR_DEV,
) -> pd.DataFrame:
    """Verify that QS portfolio factor exposures stay within max_dev of benchmark.

    For each month and each factor k:
        QS_beta_k = sum_i(w_i * beta_i_k)
        BNCH_beta_k = sum_i(b_i * beta_i_k)
        Check |QS_beta_k - BNCH_beta_k| <= max_dev

    Parameters
    ----------
    opt_weights:
        DataFrame with MultiIndex (permno, month_end) and column ``opt_weight``.
    bench_weights:
        DataFrame with MultiIndex (permno, month_end) and column ``bench_weight``.
    monthly:
        DataFrame with MultiIndex (permno, month_end) including BETA_COLS columns.
    max_dev:
        Maximum allowed absolute deviation in factor exposure.

    Returns
    -------
    pd.DataFrame
        Columns: month_end, factor, qs_exposure, bench_exposure, deviation, compliant.
    """
    # Only keep beta columns that are present in monthly
    available_betas = [c for c in BETA_COLS if c in monthly.columns]
    if not available_betas:
        print("[constraints] WARNING: No BETA_COLS found in monthly data — skipping factor check")
        return pd.DataFrame(
            columns=["month_end", "factor", "qs_exposure", "bench_exposure", "deviation", "compliant"]
        )

    factor_data = monthly[available_betas]

    qs_merged = opt_weights[["opt_weight"]].join(factor_data, how="inner")
    bench_merged = bench_weights[["bench_weight"]].join(factor_data, how="inner")

    records = []
    months = qs_merged.index.get_level_values("month_end").unique()

    for month in months:
        try:
            qs_month = qs_merged.xs(month, level="month_end")
            bench_month = bench_merged.xs(month, level="month_end")
        except KeyError:
            continue

        for factor in available_betas:
            qs_exp = float((qs_month["opt_weight"] * qs_month[factor]).sum())
            bench_exp = float((bench_month["bench_weight"] * bench_month[factor]).sum())
            dev = abs(qs_exp - bench_exp)
            records.append(
                {
                    "month_end": month,
                    "factor": factor,
                    "qs_exposure": qs_exp,
                    "bench_exposure": bench_exp,
                    "deviation": dev,
                    "compliant": dev <= max_dev,
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    n_violations = int((~result["compliant"]).sum())
    overall_pct = float(result["compliant"].mean())
    print(
        f"[constraints] Factor exposure: {n_violations} violations across "
        f"{len(months)} months x {len(available_betas)} factors, "
        f"avg compliance={overall_pct:.2%}"
    )
    return result


# ---------------------------------------------------------------------------
# Relative drawdown verification
# ---------------------------------------------------------------------------


def verify_relative_drawdown(
    qs_returns: pd.Series,
    bench_returns: pd.Series,
    max_dd: float = MAX_REL_DD_MONTHLY,
) -> pd.DataFrame:
    """Verify that monthly active return stays above -max_dd.

    Parameters
    ----------
    qs_returns:
        Monthly QS portfolio returns.
    bench_returns:
        Monthly benchmark returns.
    max_dd:
        Maximum allowed monthly active drawdown (default: MAX_REL_DD_MONTHLY).

    Returns
    -------
    pd.DataFrame
        Columns: month_end, active_return, compliant.
    """
    active = qs_returns.sub(bench_returns).dropna()
    df = pd.DataFrame({"active_return": active})
    df["compliant"] = df["active_return"] >= -max_dd
    df = df.reset_index()
    df.rename(columns={"index": "month_end"}, inplace=True)
    if "month_end" not in df.columns:
        # handle if active.index.name is month_end
        df = df.rename(columns={active.index.name: "month_end"})

    n_violations = int((~df["compliant"]).sum())
    pct = float(df["compliant"].mean())
    print(
        f"[constraints] Relative drawdown: {n_violations} months with active_ret < -{max_dd}, "
        f"compliance={pct:.2%}"
    )
    return df


# ---------------------------------------------------------------------------
# Compliance report
# ---------------------------------------------------------------------------


def generate_compliance_report(
    weight_check: pd.DataFrame,
    factor_check: pd.DataFrame,
    dd_check: pd.DataFrame,
) -> str:
    """Generate a text summary of all constraint checks and save to outputs/.

    Parameters
    ----------
    weight_check:
        Output of verify_weight_bounds().
    factor_check:
        Output of verify_factor_exposure().
    dd_check:
        Output of verify_relative_drawdown().

    Returns
    -------
    str
        Full report text.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  PORTFOLIO CONSTRAINT COMPLIANCE REPORT")
    lines.append("=" * 60)

    # --- Weight bounds ---
    lines.append("\n[1] Weight Bounds")
    lines.append("-" * 40)
    if weight_check.empty:
        lines.append("  No data.")
    else:
        total_stocks = int(weight_check["n_total"].sum())
        total_viol = int(weight_check["n_violations"].sum())
        avg_compliance = float(weight_check["pct_compliant"].mean())
        lines.append(f"  Total stock-months checked : {total_stocks}")
        lines.append(f"  Total violations           : {total_viol}")
        lines.append(f"  Average compliance         : {avg_compliance:.2%}")
        lines.append(f"  Max single violation       : {weight_check['max_violation'].max():.6f}")
        worst_month = weight_check.loc[weight_check["n_violations"].idxmax(), "month_end"]
        lines.append(f"  Worst month (most violations): {worst_month}")

    # --- Factor exposure ---
    lines.append("\n[2] Factor Exposure")
    lines.append("-" * 40)
    if factor_check.empty:
        lines.append("  No data (BETA_COLS not found in monthly).")
    else:
        for factor, grp in factor_check.groupby("factor"):
            n_viol = int((~grp["compliant"]).sum())
            pct = float(grp["compliant"].mean())
            max_dev = float(grp["deviation"].max())
            lines.append(
                f"  {factor:20s}: violations={n_viol:3d}, compliance={pct:.2%}, max_dev={max_dev:.4f}"
            )
        overall_pct = float(factor_check["compliant"].mean())
        lines.append(f"  Overall factor compliance  : {overall_pct:.2%}")

    # --- Relative drawdown ---
    lines.append("\n[3] Relative Drawdown (Monthly Active Return)")
    lines.append("-" * 40)
    if dd_check.empty:
        lines.append("  No data.")
    else:
        n_months = len(dd_check)
        n_viol = int((~dd_check["compliant"]).sum())
        pct = float(dd_check["compliant"].mean())
        min_active = float(dd_check["active_return"].min())
        lines.append(f"  Months checked             : {n_months}")
        lines.append(f"  Violations (active < limit): {n_viol}")
        lines.append(f"  Compliance                 : {pct:.2%}")
        lines.append(f"  Worst monthly active return: {min_active:.4f}")

    # --- Overall ---
    lines.append("\n[OVERALL COMPLIANCE SUMMARY]")
    lines.append("-" * 40)

    checks = []
    if not weight_check.empty:
        checks.append(("Weight Bounds", float(weight_check["pct_compliant"].mean())))
    if not factor_check.empty:
        checks.append(("Factor Exposure", float(factor_check["compliant"].mean())))
    if not dd_check.empty:
        checks.append(("Relative Drawdown", float(dd_check["compliant"].mean())))

    if checks:
        for name, pct in checks:
            status = "PASS" if pct >= 0.95 else "WARN" if pct >= 0.80 else "FAIL"
            lines.append(f"  {name:25s}: {pct:.2%}  [{status}]")
        overall = float(np.mean([p for _, p in checks]))
        lines.append(f"\n  Overall portfolio compliance: {overall:.2%}")
    else:
        lines.append("  No constraint checks available.")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    print(report)

    # Save to file
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS / "compliance_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[constraints] Compliance report saved to {report_path}")

    return report
