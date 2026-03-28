"""Performance analysis and reporting for the constrained portfolio strategy vs benchmark."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats
from pathlib import Path
from typing import Optional

from src.config import (
    OUTPUTS, NW_LAGS, ANNUALIZE_MONTHLY,
    IS_START, IS_END, OOS_START, OOS_END, SPLIT_DATE,
)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_sharpe_nw(
    returns: pd.Series,
    freq: int = ANNUALIZE_MONTHLY,
    nlags: int = NW_LAGS,
) -> float:
    """Compute Newey-West HAC adjusted Sharpe ratio for monthly returns.

    CRITICAL: cov_hac returns Var(mean estimator), NOT Var(series).
    We scale back by N to obtain Var(series) before taking sqrt.

    Parameters
    ----------
    returns : pd.Series
        Monthly return series (NaN values are dropped).
    freq : int
        Annualisation factor (12 for monthly returns).
    nlags : int
        Number of Newey-West lags.

    Returns
    -------
    float
        Annualised Newey-West adjusted Sharpe ratio.
    """
    y = returns.dropna().to_numpy(dtype=np.float64, na_value=np.nan)
    if len(y) < 2:
        return np.nan
    model = sm.OLS(y, np.ones(len(y))).fit()
    nw_var_mean = cov_hac(model, nlags=nlags)[0, 0]   # Var(mean estimator)
    nw_var_series = nw_var_mean * len(y)               # scale back to Var(series)
    nw_std = np.sqrt(nw_var_series)
    sharpe = (y.mean() / nw_std) * np.sqrt(freq)
    return float(sharpe)


def compute_information_ratio(
    active_returns: pd.Series,
    freq: int = ANNUALIZE_MONTHLY,
) -> float:
    """Compute Information Ratio (geometric annualisation).

    IR = ann_excess_return / tracking_error

    Parameters
    ----------
    active_returns : pd.Series
        Active (QS minus benchmark) return series.
    freq : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised Information Ratio.
    """
    ar = active_returns.dropna().to_numpy(dtype=np.float64, na_value=np.nan)
    if len(ar) < 2:
        return np.nan
    ann_excess = (1 + ar.mean()) ** freq - 1
    te = ar.std(ddof=1) * np.sqrt(freq)
    return float(ann_excess / te) if te > 0 else np.nan


def compute_metrics(
    returns: pd.Series,
    bench_returns: Optional[pd.Series] = None,
    freq: int = ANNUALIZE_MONTHLY,
) -> dict:
    """Compute comprehensive performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Portfolio monthly return series.
    bench_returns : pd.Series, optional
        Benchmark monthly return series aligned to same index.
    freq : int
        Annualisation factor (12 for monthly data).

    Returns
    -------
    dict
        Dictionary of performance metrics.  If bench_returns is provided,
        also includes relative metrics (excess_return, tracking_error,
        information_ratio, max_relative_dd).
    """
    r = returns.dropna()
    arr = r.to_numpy(dtype=np.float64, na_value=np.nan)

    # Annualised return (geometric / compounded) and volatility
    ann_return = float((1 + arr.mean()) ** freq - 1)
    ann_vol = float(arr.std(ddof=1) * np.sqrt(freq))

    # Sharpe ratios
    sharpe_nw = compute_sharpe_nw(r, freq=freq)
    sharpe_simple = float(ann_return / ann_vol) if ann_vol > 0 else np.nan

    # Max drawdown from cumulative returns (compounded)
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    # Calmar ratio
    calmar = float(ann_return / abs(max_drawdown)) if max_drawdown != 0 else np.nan

    # Sortino ratio
    downside = arr[arr < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else np.nan
    sortino = float(ann_return / (downside_std * np.sqrt(freq))) if downside_std and downside_std > 0 else np.nan

    # Hit rate and distributional stats
    hit_rate = float((arr > 0).mean())
    skewness = float(stats.skew(arr))
    kurtosis = float(stats.kurtosis(arr))  # excess kurtosis

    metrics = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe_nw": sharpe_nw,
        "sharpe_simple": sharpe_simple,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "sortino": sortino,
        "hit_rate": hit_rate,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

    # Relative metrics (require benchmark)
    if bench_returns is not None:
        aligned_qs, aligned_bm = returns.align(bench_returns, join="inner")
        active = aligned_qs - aligned_bm
        active_arr = active.dropna().to_numpy(dtype=np.float64, na_value=np.nan)

        excess_return = float((1 + active_arr.mean()) ** freq - 1)
        tracking_error = float(active_arr.std(ddof=1) * np.sqrt(freq))
        information_ratio = compute_information_ratio(active, freq=freq)
        max_relative_dd = float(active_arr.min())  # worst single-month underperformance

        metrics.update({
            "excess_return": excess_return,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "max_relative_dd": max_relative_dd,
        })

    return metrics


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    qs_cum: pd.Series,
    bench_cum: pd.Series,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot cumulative returns for QS portfolio vs benchmark.

    Parameters
    ----------
    qs_cum : pd.Series
        Cumulative return series for the QS portfolio.
    bench_cum : pd.Series
        Cumulative return series for the benchmark.
    title : str
        Chart title.
    save_path : str, optional
        File path to save the figure. If None, figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(qs_cum.index, qs_cum.values, label="QS Portfolio", linewidth=1.8, color="#1f77b4")
    ax.plot(bench_cum.index, bench_cum.values, label="Benchmark", linewidth=1.8, color="#ff7f0e", linestyle="--")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlim(qs_cum.index[0], qs_cum.index[-1])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_active_returns_bar(
    active_returns: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Monthly bar chart of active returns, color-coded green/red.

    Parameters
    ----------
    active_returns : pd.Series
        Monthly active return series (QS - benchmark).
    save_path : str, optional
        File path to save the figure.
    """
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in active_returns.values]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(active_returns.index, active_returns.values, color=colors, width=20)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Active Returns (QS - Benchmark)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Return")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 12,
    freq: int = ANNUALIZE_MONTHLY,
    save_path: Optional[str] = None,
) -> None:
    """Plot rolling Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Monthly return series.
    window : int
        Rolling window length in months.
    freq : int
        Annualisation factor.
    save_path : str, optional
        File path to save the figure.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std(ddof=1)
    rolling_sharpe = (roll_mean / roll_std) * np.sqrt(freq)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.6, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_factor_exposure_deviation(
    factor_check_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Time series of factor exposure deviations with ±0.1 bound shading.

    Parameters
    ----------
    factor_check_df : pd.DataFrame
        DataFrame with dates as index and factor names as columns,
        values representing deviations from target exposures.
    save_path : str, optional
        File path to save the figure.
    """
    factors = factor_check_df.columns.tolist()
    n = len(factors)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, factor in zip(axes, factors):
        ax.plot(factor_check_df.index, factor_check_df[factor].values, linewidth=1.4, label=factor)
        ax.axhspan(-0.1, 0.1, alpha=0.12, color="#2ca02c", label="±0.1 bound")
        ax.axhline(0.1, color="#2ca02c", linewidth=0.8, linestyle="--")
        ax.axhline(-0.1, color="#2ca02c", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(f"{factor}\nDeviation")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_title("Factor Exposure Deviations", fontsize=13)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_weight_distribution(
    opt_weights: pd.Series,
    bench_weights: pd.Series,
    month_end: str,
    save_path: Optional[str] = None,
) -> None:
    """Histogram of active weights (w_QS - w_BNCH) for a sample month.

    Parameters
    ----------
    opt_weights : pd.Series
        Optimised portfolio weights indexed by stock identifier.
    bench_weights : pd.Series
        Benchmark weights indexed by stock identifier.
    month_end : str
        Label for the sample month (used in title).
    save_path : str, optional
        File path to save the figure.
    """
    aligned_opt, aligned_bm = opt_weights.align(bench_weights, fill_value=0.0)
    active_weights = aligned_opt - aligned_bm

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(active_weights.values, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Active Weight Distribution — {month_end}", fontsize=13)
    ax.set_xlabel("Active Weight (w_QS - w_BNCH)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_drawdown(
    qs_cum: pd.Series,
    bench_cum: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Underwater chart showing drawdown of QS and relative drawdown vs benchmark.

    Parameters
    ----------
    qs_cum : pd.Series
        Cumulative return series for the QS portfolio.
    bench_cum : pd.Series
        Cumulative return series for the benchmark.
    save_path : str, optional
        File path to save the figure.
    """
    qs_dd = (qs_cum - qs_cum.cummax()) / qs_cum.cummax()
    bm_dd = (bench_cum - bench_cum.cummax()) / bench_cum.cummax()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].fill_between(qs_dd.index, qs_dd.values, 0, color="#1f77b4", alpha=0.5, label="QS Drawdown")
    axes[0].plot(qs_dd.index, qs_dd.values, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Drawdown")
    axes[0].set_title("Portfolio Drawdown", fontsize=13)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].fill_between(bm_dd.index, bm_dd.values, 0, color="#ff7f0e", alpha=0.5, label="Benchmark Drawdown")
    axes[1].plot(bm_dd.index, bm_dd.values, color="#ff7f0e", linewidth=1.0)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Benchmark Drawdown", fontsize=13)
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    qs_returns: pd.Series,
    bench_returns: pd.Series,
    active_returns: pd.Series,
    opt_weights: pd.DataFrame,
    bench_weights: pd.DataFrame,
    factor_check: pd.DataFrame,
    compliance_text: str,
    period_label: str = "Full",
) -> str:
    """Generate comprehensive performance report with metrics, plots, and CSV.

    Parameters
    ----------
    qs_returns : pd.Series
        Monthly QS portfolio returns.
    bench_returns : pd.Series
        Monthly benchmark returns.
    active_returns : pd.Series
        Monthly active returns (QS - benchmark).
    opt_weights : pd.DataFrame
        Optimised weights over time (index=dates, columns=stocks).
    bench_weights : pd.DataFrame
        Benchmark weights over time (index=dates, columns=stocks).
    factor_check : pd.DataFrame
        Factor exposure deviations (index=dates, columns=factors).
    compliance_text : str
        Pre-formatted compliance summary string to embed in the report.
    period_label : str
        Label for the evaluation period (e.g. "IS", "OOS", "Full").

    Returns
    -------
    str
        Full text of the report.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # ---- compute metrics -----------------------------------------------
    qs_metrics = compute_metrics(qs_returns, bench_returns=bench_returns)
    bm_metrics = compute_metrics(bench_returns)

    # ---- cumulative returns for plots ----------------------------------
    qs_cum = (1 + qs_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    # ---- save plots ----------------------------------------------------
    plot_cumulative_returns(
        qs_cum, bench_cum,
        title=f"Cumulative Returns — {period_label}",
        save_path=str(OUTPUTS / f"cumulative_returns_{period_label}.png"),
    )
    plot_active_returns_bar(
        active_returns,
        save_path=str(OUTPUTS / f"active_returns_{period_label}.png"),
    )
    plot_rolling_sharpe(
        qs_returns,
        save_path=str(OUTPUTS / f"rolling_sharpe_{period_label}.png"),
    )
    if factor_check is not None and not factor_check.empty:
        plot_factor_exposure_deviation(
            factor_check,
            save_path=str(OUTPUTS / f"factor_exposure_{period_label}.png"),
        )
    if opt_weights is not None and not opt_weights.empty:
        sample_date = opt_weights.index[-1]
        sample_label = str(sample_date)[:10]
        plot_weight_distribution(
            opt_weights.loc[sample_date],
            bench_weights.loc[sample_date] if sample_date in bench_weights.index else pd.Series(dtype=float),
            month_end=sample_label,
            save_path=str(OUTPUTS / f"weight_distribution_{period_label}.png"),
        )
    plot_drawdown(
        qs_cum, bench_cum,
        save_path=str(OUTPUTS / f"drawdown_{period_label}.png"),
    )

    # ---- metrics summary CSV -------------------------------------------
    metrics_df = pd.DataFrame({
        "QS Portfolio": qs_metrics,
        "Benchmark": {k: bm_metrics.get(k, np.nan) for k in qs_metrics},
    })
    csv_path = OUTPUTS / f"metrics_summary_{period_label}.csv"
    metrics_df.to_csv(csv_path)

    # ---- build text report ---------------------------------------------
    sep = "=" * 68
    thin = "-" * 68

    def fmt(val: float, pct: bool = False, decimals: int = 4) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if pct:
            return f"{val * 100:.2f}%"
        return f"{val:.{decimals}f}"

    report_lines = [
        sep,
        f"  PORTFOLIO PERFORMANCE REPORT — {period_label}",
        sep,
        "",
        "  RETURN & RISK",
        thin,
        f"  {'Metric':<30} {'QS Portfolio':>15} {'Benchmark':>15}",
        thin,
        f"  {'Annualised Return':<30} {fmt(qs_metrics['ann_return'], pct=True):>15} {fmt(bm_metrics['ann_return'], pct=True):>15}",
        f"  {'Annualised Volatility':<30} {fmt(qs_metrics['ann_vol'], pct=True):>15} {fmt(bm_metrics['ann_vol'], pct=True):>15}",
        f"  {'Sharpe (Newey-West)':<30} {fmt(qs_metrics['sharpe_nw']):>15} {fmt(bm_metrics['sharpe_nw']):>15}",
        f"  {'Sharpe (Simple)':<30} {fmt(qs_metrics['sharpe_simple']):>15} {fmt(bm_metrics['sharpe_simple']):>15}",
        f"  {'Max Drawdown':<30} {fmt(qs_metrics['max_drawdown'], pct=True):>15} {fmt(bm_metrics['max_drawdown'], pct=True):>15}",
        f"  {'Calmar Ratio':<30} {fmt(qs_metrics['calmar']):>15} {fmt(bm_metrics['calmar']):>15}",
        f"  {'Sortino Ratio':<30} {fmt(qs_metrics['sortino']):>15} {fmt(bm_metrics['sortino']):>15}",
        f"  {'Hit Rate':<30} {fmt(qs_metrics['hit_rate'], pct=True):>15} {fmt(bm_metrics['hit_rate'], pct=True):>15}",
        f"  {'Skewness':<30} {fmt(qs_metrics['skewness']):>15} {fmt(bm_metrics['skewness']):>15}",
        f"  {'Excess Kurtosis':<30} {fmt(qs_metrics['kurtosis']):>15} {fmt(bm_metrics['kurtosis']):>15}",
        "",
        "  RELATIVE PERFORMANCE (vs Benchmark)",
        thin,
        f"  {'Excess Return (Ann.)':<30} {fmt(qs_metrics.get('excess_return'), pct=True):>15}",
        f"  {'Tracking Error (Ann.)':<30} {fmt(qs_metrics.get('tracking_error'), pct=True):>15}",
        f"  {'Information Ratio':<30} {fmt(qs_metrics.get('information_ratio')):>15}",
        f"  {'Max Monthly Underperf.':<30} {fmt(qs_metrics.get('max_relative_dd'), pct=True):>15}",
        "",
        "  COMPLIANCE SUMMARY",
        thin,
        compliance_text,
        "",
        "  OUTPUT FILES",
        thin,
        f"  Plots saved to: {OUTPUTS}",
        f"  Metrics CSV:    {csv_path}",
        sep,
    ]

    report_text = "\n".join(report_lines)

    # ---- print to console ----------------------------------------------
    print(report_text)

    # ---- save text report ----------------------------------------------
    report_path = OUTPUTS / f"performance_report_{period_label}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    return report_text


# ---------------------------------------------------------------------------
# IS / OOS split analysis
# ---------------------------------------------------------------------------

def compute_period_metrics(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    period_label: str,
    start: str,
    end: str,
) -> dict:
    """Compute metrics dict for a specific time slice.

    Parameters
    ----------
    port_rets : pd.Series
        Full monthly portfolio returns, indexed by month_end.
    bench_rets : pd.Series
        Full monthly benchmark returns, indexed by month_end.
    period_label : str
        Label for this period (e.g. "IS", "OOS", "Full").
    start : str
        Start date string (inclusive), e.g. "2015-01-01".
    end : str
        End date string (inclusive), e.g. "2020-12-31".

    Returns
    -------
    dict
        Metrics dict with an added "period" and "n_months" key.
    """
    p = port_rets.loc[start:end].dropna()
    b = bench_rets.loc[start:end].dropna()
    aligned_p, aligned_b = p.align(b, join="inner")

    m = compute_metrics(aligned_p, bench_returns=aligned_b)
    m["period"] = period_label
    m["n_months"] = len(aligned_p)
    return m


def compute_all_period_metrics(
    port_rets: pd.Series,
    bench_rets: pd.Series,
) -> pd.DataFrame:
    """Compute IS, OOS, and Full-period metrics tables.

    Parameters
    ----------
    port_rets : pd.Series
        Monthly portfolio returns indexed by month_end.
    bench_rets : pd.Series
        Monthly benchmark returns indexed by month_end.

    Returns
    -------
    pd.DataFrame
        Three-row metrics table with index ["IS", "OOS", "Full"].
    """
    print("[performance] Computing IS / OOS / Full period metrics...")

    rows = []
    for label, start, end in [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("Full", str(port_rets.dropna().index.min().date()), str(port_rets.dropna().index.max().date())),
    ]:
        m = compute_period_metrics(port_rets, bench_rets, label, start, end)
        rows.append(m)

    df = pd.DataFrame(rows).set_index("period")
    return df


# ---------------------------------------------------------------------------
# Standalone pipeline orchestrator
# ---------------------------------------------------------------------------

def load_returns_from_outputs() -> tuple[pd.Series, pd.Series]:
    """Load portfolio and benchmark returns from outputs/ parquet files.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (port_rets, bench_rets) both indexed by month_end.

    Raises
    ------
    FileNotFoundError
        If required parquet files are not found in outputs/.
    """
    port_path = OUTPUTS / "portfolio_returns.parquet"
    bench_path = OUTPUTS / "benchmark_returns.parquet"

    if not port_path.exists():
        raise FileNotFoundError(
            f"Portfolio returns not found at {port_path}. "
            "Run the backtest pipeline first."
        )
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Benchmark returns not found at {bench_path}. "
            "Run benchmark.build_benchmark() first."
        )

    port_df = pd.read_parquet(port_path)
    bench_df = pd.read_parquet(bench_path)

    port_rets = port_df["port_ret"] if "port_ret" in port_df.columns else port_df.iloc[:, 0]
    bench_rets = bench_df["bench_ret"] if "bench_ret" in bench_df.columns else bench_df.iloc[:, 0]

    port_rets.name = "port_ret"
    bench_rets.name = "bench_ret"
    port_rets.index.name = "month_end"
    bench_rets.index.name = "month_end"

    print(f"[performance] Loaded {len(port_rets)} portfolio months, "
          f"{len(bench_rets)} benchmark months.")
    return port_rets, bench_rets


def run_performance_analysis(
    port_rets: Optional[pd.Series] = None,
    bench_rets: Optional[pd.Series] = None,
    opt_weights: Optional[pd.DataFrame] = None,
    bench_weights_df: Optional[pd.DataFrame] = None,
    factor_check: Optional[pd.DataFrame] = None,
    compliance_text: str = "No compliance data provided.",
) -> dict:
    """Full performance analysis pipeline.

    Can be called with pre-computed series (from the backtest pipeline) or
    with no arguments to load saved parquet files from outputs/.

    Parameters
    ----------
    port_rets : pd.Series, optional
        Monthly portfolio returns. If None, loads from outputs/portfolio_returns.parquet.
    bench_rets : pd.Series, optional
        Monthly benchmark returns. If None, loads from outputs/benchmark_returns.parquet.
    opt_weights : pd.DataFrame, optional
        Wide-form optimised weights (index=month_end, columns=permno). Used for
        weight distribution plot. If None, that plot is skipped.
    bench_weights_df : pd.DataFrame, optional
        Wide-form benchmark weights (index=month_end, columns=permno).
    factor_check : pd.DataFrame, optional
        Factor exposure deviations (index=month_end, columns=factor names).
    compliance_text : str
        Pre-formatted compliance summary for embedding in text report.

    Returns
    -------
    dict with keys:
        "metrics_is"   : dict — IS metrics
        "metrics_oos"  : dict — OOS metrics
        "metrics_full" : dict — Full period metrics
        "metrics_df"   : pd.DataFrame — IS/OOS/Full table
        "report_is"    : str — IS text report
        "report_oos"   : str — OOS text report
        "report_full"  : str — Full text report
    """
    print("[performance] Starting full performance analysis pipeline...")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Load returns if not provided
    if port_rets is None or bench_rets is None:
        port_rets, bench_rets = load_returns_from_outputs()

    # Compute IS/OOS/Full metrics table and save
    metrics_df = compute_all_period_metrics(port_rets, bench_rets)
    metrics_csv = OUTPUTS / "metrics_all_periods.csv"
    metrics_df.to_csv(metrics_csv)
    print(f"[performance] Saved metrics table -> {metrics_csv}")

    print("\n" + "=" * 68)
    print("PERFORMANCE SUMMARY — ALL PERIODS")
    print("=" * 68)
    print(metrics_df.T.to_string())
    print("=" * 68 + "\n")

    # Run generate_report for each period
    results: dict = {"metrics_df": metrics_df}

    for label, start, end in [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("Full", str(port_rets.dropna().index.min().date()),
                 str(port_rets.dropna().index.max().date())),
    ]:
        p = port_rets.loc[start:end].dropna()
        b = bench_rets.loc[start:end].dropna()
        p, b = p.align(b, join="inner")
        active = p - b

        # Slice opt_weights and bench_weights to period
        ow = None
        bw = None
        if opt_weights is not None:
            ow = opt_weights.loc[start:end]
        if bench_weights_df is not None:
            bw = bench_weights_df.loc[start:end]

        fc = None
        if factor_check is not None:
            fc = factor_check.loc[start:end]

        try:
            report_text = generate_report(
                qs_returns=p,
                bench_returns=b,
                active_returns=active,
                opt_weights=ow if ow is not None else pd.DataFrame(),
                bench_weights=bw if bw is not None else pd.DataFrame(),
                factor_check=fc if fc is not None else pd.DataFrame(),
                compliance_text=compliance_text,
                period_label=label,
            )
            results[f"report_{label.lower()}"] = report_text
            results[f"metrics_{label.lower()}"] = compute_metrics(p, bench_returns=b)
        except Exception as exc:
            print(f"[performance] WARNING: report for {label} failed: {exc}")

    print("[performance] Analysis complete.")
    return results
