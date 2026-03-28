"""
robustness.py
Computes statistical significance tests, IC time series, EMA sensitivity,
and turnover/TC estimates. Outputs CSVs and PNG for the report.
"""

import os
import sys
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"

sys.path.insert(0, str(BASE / "src"))
from config import SPLIT_DATE, EMA_ALPHA

SPLIT_TS = pd.Timestamp(SPLIT_DATE)

DARK_BLUE = "#1B3A6B"
MED_BLUE = "#2E6DA4"
ACCENT = "#E8534A"


def load_data():
    sp = pd.read_parquet(OUTPUTS / "signal_predictions.parquet")
    mp = pd.read_parquet(OUTPUTS / "monthly_processed.parquet")
    br = pd.read_parquet(OUTPUTS / "backtest_results.parquet")
    pw = pd.read_parquet(OUTPUTS / "portfolio_weights.parquet")
    return sp, mp, br, pw


def compute_monthly_ic(signal_preds, monthly_proc, period="OOS"):
    if period == "OOS":
        mask_fn = lambda idx: idx.get_level_values("month_end") > SPLIT_TS
    elif period == "IS":
        mask_fn = lambda idx: idx.get_level_values("month_end") <= SPLIT_TS
    else:
        mask_fn = lambda idx: np.ones(len(idx), dtype=bool)

    sp_f = signal_preds[mask_fn(signal_preds.index)].copy()
    mp_f = monthly_proc[mask_fn(monthly_proc.index)].copy()

    months = sorted(sp_f.index.get_level_values("month_end").unique())
    records = []
    for m in months:
        try:
            sp_m = sp_f.xs(m, level="month_end")
            mp_m = mp_f.xs(m, level="month_end")
        except KeyError:
            continue
        joined = sp_m[["y_pred"]].join(mp_m[["fwd_ret"]], how="inner").dropna()
        if len(joined) < 5:
            continue
        corr, _ = spearmanr(joined["fwd_ret"], joined["y_pred"])
        records.append({"month_end": m, "IC": corr})

    df = pd.DataFrame(records).set_index("month_end")
    return df


def compute_ic_significance(ic_series):
    arr = ic_series["IC"].dropna().values
    n = len(arr)
    mean_ic = arr.mean()
    std_ic = arr.std(ddof=1)
    se_ic = std_ic / sqrt(n) if n > 1 else np.nan
    t_stat = mean_ic / se_ic if se_ic > 0 else np.nan
    p_value = 2 * t_dist.sf(abs(t_stat), df=n - 1) if n > 1 else np.nan
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    return dict(mean_ic=mean_ic, std_ic=std_ic, icir=icir,
                t_stat=t_stat, p_value=p_value, n_months=n)


def compute_sharpe_significance(ret_series):
    arr = ret_series.dropna().values
    n = len(arr)
    mu = arr.mean()
    sig = arr.std(ddof=1)
    sharpe_monthly = mu / sig if sig > 0 else np.nan
    t_stat = sharpe_monthly * sqrt(n) if not np.isnan(sharpe_monthly) else np.nan
    p_value = 2 * t_dist.sf(abs(t_stat), df=n - 1) if n > 1 else np.nan
    ann_ret = (1 + mu) ** 12 - 1
    ann_vol = sig * sqrt(12)
    sharpe_ann = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return dict(sharpe_ann=sharpe_ann, t_stat=t_stat, p_value=p_value, n_months=n)


def compute_ir_significance(active_series):
    arr = active_series.dropna().values
    n = len(arr)
    mu = arr.mean()
    sig = arr.std(ddof=1)
    ir_monthly = mu / sig if sig > 0 else np.nan
    t_stat = ir_monthly * sqrt(n) if not np.isnan(ir_monthly) else np.nan
    p_value = 2 * t_dist.sf(abs(t_stat), df=n - 1) if n > 1 else np.nan
    ann_active = mu * 12
    te = sig * sqrt(12)
    ir_ann = ann_active / te if te > 0 else np.nan
    return dict(ir_ann=ir_ann, t_stat=t_stat, p_value=p_value, n_months=n)


def compute_ema_sensitivity(raw_preds, monthly_proc, alphas):
    records = []
    for alpha in alphas:
        smoothed = raw_preds.copy()
        smoothed["y_pred"] = smoothed.groupby(level="permno")["y_pred"].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )
        ic_df = compute_monthly_ic(smoothed, monthly_proc, period="OOS")
        stats = compute_ic_significance(ic_df)
        records.append({
            "Alpha": alpha,
            "OOS Mean IC": round(stats["mean_ic"], 4),
            "OOS Std IC": round(stats["std_ic"], 4),
            "OOS ICIR": round(stats["icir"], 4),
            "IC t-stat": round(stats["t_stat"], 3),
            "IC p-value": round(stats["p_value"], 3),
        })
    return pd.DataFrame(records)


def sig_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def build_stat_tests_df(ic_oos, sharpe_oos, ir_oos, ic_full, sharpe_full, ir_full):
    rows = [
        ["IC", "OOS", f"{ic_oos['mean_ic']:.4f}", f"{ic_oos['t_stat']:.3f}",
         f"{ic_oos['p_value']:.3f}", sig_stars(ic_oos['p_value'])],
        ["IC", "Full", f"{ic_full['mean_ic']:.4f}", f"{ic_full['t_stat']:.3f}",
         f"{ic_full['p_value']:.3f}", sig_stars(ic_full['p_value'])],
        ["Sharpe", "OOS", f"{sharpe_oos['sharpe_ann']:.2f}", f"{sharpe_oos['t_stat']:.3f}",
         f"{sharpe_oos['p_value']:.3f}", sig_stars(sharpe_oos['p_value'])],
        ["Sharpe", "Full", f"{sharpe_full['sharpe_ann']:.2f}", f"{sharpe_full['t_stat']:.3f}",
         f"{sharpe_full['p_value']:.3f}", sig_stars(sharpe_full['p_value'])],
        ["IR", "OOS", f"{ir_oos['ir_ann']:.2f}", f"{ir_oos['t_stat']:.3f}",
         f"{ir_oos['p_value']:.3f}", sig_stars(ir_oos['p_value'])],
        ["IR", "Full", f"{ir_full['ir_ann']:.2f}", f"{ir_full['t_stat']:.3f}",
         f"{ir_full['p_value']:.3f}", sig_stars(ir_full['p_value'])],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Period", "Value", "t-stat", "p-value", "Sig."])


def plot_ic_timeseries(ic_oos, save_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = [MED_BLUE if v >= 0 else ACCENT for v in ic_oos["IC"]]
    ax.bar(ic_oos.index, ic_oos["IC"], color=colors, width=25)
    ax.axhline(0, color="black", linewidth=0.8)
    mean_ic = ic_oos["IC"].mean()
    ax.axhline(mean_ic, color=DARK_BLUE, linewidth=1.2, linestyle="--",
               label=f"Mean IC = {mean_ic:.4f}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Spearman IC")
    ax.set_title("OOS Monthly Spearman IC Time Series (2021-2024)")
    ax.legend(frameon=False)
    ax.set_xlim(ic_oos.index[0], ic_oos.index[-1])
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[robustness] Loading data...")
    sp, mp, br, pw = load_data()

    oos_mask = br.index > SPLIT_TS

    print("[robustness] Computing IC time series...")
    ic_oos = compute_monthly_ic(sp, mp, period="OOS")
    ic_full = compute_monthly_ic(sp, mp, period="Full")

    ic_stats_oos = compute_ic_significance(ic_oos)
    ic_stats_full = compute_ic_significance(ic_full)

    print("[robustness] Computing Sharpe/IR significance...")
    sharpe_oos = compute_sharpe_significance(br.loc[oos_mask, "qs_ret"])
    sharpe_full = compute_sharpe_significance(br["qs_ret"])
    ir_oos = compute_ir_significance(br.loc[oos_mask, "active_ret"])
    ir_full = compute_ir_significance(br["active_ret"])

    print("[robustness] Computing EMA sensitivity...")
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    ema_df = compute_ema_sensitivity(sp, mp, alphas)

    print("[robustness] Computing turnover & TC...")
    pw_reset = pw.reset_index()
    w_wide = pw_reset.pivot(index="month_end", columns="permno", values="opt_weight").fillna(0.0).sort_index()
    turnover = 0.5 * w_wide.diff().abs().sum(axis=1).dropna()
    oos_turnover = turnover[turnover.index > SPLIT_TS]
    tc_summary = pd.DataFrame([{
        "mean_monthly_turnover": round(turnover.mean(), 4),
        "annual_turnover": round(turnover.mean() * 12, 4),
        "tc_drag_annual_10bps": round(turnover.mean() * 12 * 0.001, 4),
        "oos_mean_monthly_turnover": round(oos_turnover.mean(), 4),
        "oos_tc_drag_annual_10bps": round(oos_turnover.mean() * 12 * 0.001, 4),
    }])

    stat_tests_df = build_stat_tests_df(
        ic_stats_oos, sharpe_oos, ir_oos,
        ic_stats_full, sharpe_full, ir_full
    )

    ic_oos.to_csv(OUTPUTS / "ic_timeseries.csv")
    stat_tests_df.to_csv(OUTPUTS / "statistical_tests.csv", index=False)
    ema_df.to_csv(OUTPUTS / "ema_sensitivity.csv", index=False)
    tc_summary.to_csv(OUTPUTS / "tc_summary.csv", index=False)
    plot_ic_timeseries(ic_oos, str(OUTPUTS / "ic_timeseries_oos.png"))

    print("\n[robustness] Statistical Significance Tests:")
    print(stat_tests_df.to_string(index=False))
    print(f"\n[robustness] EMA Sensitivity:")
    print(ema_df.to_string(index=False))
    print(f"\n[robustness] Turnover: monthly={turnover.mean():.1%}, "
          f"OOS monthly={oos_turnover.mean():.1%}, "
          f"OOS TC drag @10bps={oos_turnover.mean() * 12 * 0.001:.2%}")
    print("[robustness] All outputs saved to outputs/")


if __name__ == "__main__":
    main()
