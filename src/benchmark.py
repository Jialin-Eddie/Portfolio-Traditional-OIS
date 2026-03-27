"""
benchmark.py — Market-cap-weighted benchmark construction for Portfolio-Traditional-OIS.

Builds a market-cap-weighted (MCW) benchmark from the monthly S&P 500 universe,
computing weights, benchmark returns, and factor exposures.
"""

import pandas as pd
import numpy as np

from src.config import MKTCAP_COL, BETA_COLS, OUTPUTS


def compute_benchmark_weights(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-cap-weighted benchmark weights at each month_end.

    At each month_end, w_bench_i = mktcap_i / sum(mktcap) for all stocks
    with non-NaN mktcap.

    Parameters
    ----------
    monthly : pd.DataFrame
        MultiIndex (permno, month_end) DataFrame with at minimum a `mktcap` column.

    Returns
    -------
    pd.DataFrame
        MultiIndex (permno, month_end) DataFrame with column `bench_weight`.
        Weights sum to ~1.0 per month (tolerance 1e-6).
    """
    print("[benchmark] Computing market-cap weights...")

    mktcap = monthly[[MKTCAP_COL]].copy()
    mktcap = mktcap.dropna(subset=[MKTCAP_COL])

    # Sum of mktcap per month_end
    month_totals = mktcap.groupby(level="month_end")[MKTCAP_COL].sum()

    # Align month totals back to the stock-level index
    mktcap["bench_weight"] = mktcap[MKTCAP_COL] / mktcap.index.get_level_values("month_end").map(month_totals)

    weights_df = mktcap[["bench_weight"]].copy()

    # Validate: weights sum to ~1.0 per month
    weight_sums = weights_df.groupby(level="month_end")["bench_weight"].sum()
    max_deviation = (weight_sums - 1.0).abs().max()
    assert max_deviation < 1e-6, (
        f"Benchmark weights do not sum to 1.0 per month. Max deviation: {max_deviation:.2e}"
    )

    n_months = weight_sums.shape[0]
    n_stocks = weights_df.shape[0]
    print(f"[benchmark] Weights computed: {n_months} months, {n_stocks} stock-month observations.")

    return weights_df


def compute_benchmark_returns(
    monthly: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute monthly benchmark returns.

    R_BNCH(t) = sum_i( w_bench_i(t) * fwd_ret_i(t) )

    That is, at each month_end t, the benchmark return equals the
    weighted sum of each stock's fwd_ret (next-month return) using
    that same month's benchmark weights.

    Parameters
    ----------
    monthly : pd.DataFrame
        MultiIndex (permno, month_end) with column `fwd_ret`.
    weights : pd.DataFrame
        MultiIndex (permno, month_end) with column `bench_weight`,
        as returned by compute_benchmark_weights().

    Returns
    -------
    pd.Series
        Series indexed by month_end with benchmark monthly returns.
        Named `bench_ret`.
    """
    print("[benchmark] Computing benchmark returns...")

    combined = weights.join(monthly[["fwd_ret"]], how="inner")
    combined = combined.dropna(subset=["fwd_ret", "bench_weight"])

    bench_rets = (
        combined.groupby(level="month_end")
        .apply(lambda g: (g["bench_weight"] * g["fwd_ret"]).sum())
    )
    bench_rets.name = "bench_ret"

    n_months = bench_rets.shape[0]
    print(
        f"[benchmark] Benchmark returns computed: {n_months} months, "
        f"mean={bench_rets.mean():.4f}, std={bench_rets.std():.4f}."
    )

    return bench_rets


def compute_benchmark_factor_exposures(
    monthly: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute market-cap-weighted benchmark factor exposures (betas) per month.

    beta_BNCH_k(t) = sum_i( w_bench_i(t) * beta_i_k(t) )
    for k in {mktrf, smb, hml, mom}.

    Parameters
    ----------
    monthly : pd.DataFrame
        MultiIndex (permno, month_end) with columns in BETA_COLS.
    weights : pd.DataFrame
        MultiIndex (permno, month_end) with column `bench_weight`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by month_end with columns:
        bench_beta_mktrf, bench_beta_smb, bench_beta_hml, bench_beta_mom.
    """
    print("[benchmark] Computing benchmark factor exposures...")

    combined = weights.join(monthly[BETA_COLS], how="inner")
    combined = combined.dropna(subset=["bench_weight"])

    result_parts = {}
    for beta_col in BETA_COLS:
        factor_name = beta_col.replace("beta_", "bench_beta_")
        valid = combined.dropna(subset=[beta_col])
        result_parts[factor_name] = (
            valid.groupby(level="month_end")
            .apply(lambda g, col=beta_col: (g["bench_weight"] * g[col]).sum())
        )

    factor_exposures = pd.DataFrame(result_parts)
    factor_exposures.index.name = "month_end"

    n_months = factor_exposures.shape[0]
    print(f"[benchmark] Factor exposures computed: {n_months} months x {len(BETA_COLS)} factors.")

    return factor_exposures


def build_benchmark(
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Orchestrate full benchmark construction and save outputs.

    Steps:
    1. Compute market-cap weights.
    2. Compute benchmark returns.
    3. Compute benchmark factor exposures.
    4. Save all three to outputs/.

    Parameters
    ----------
    monthly : pd.DataFrame
        MultiIndex (permno, month_end) universe DataFrame with columns:
        mktcap, fwd_ret, beta_mktrf, beta_smb, beta_hml, beta_mom, etc.

    Returns
    -------
    tuple of:
        - weights_df    : pd.DataFrame  — (permno, month_end) x bench_weight
        - returns_series: pd.Series     — month_end indexed bench_ret
        - factor_exposures_df: pd.DataFrame — month_end x bench_beta_*
    """
    print("[benchmark] Building benchmark...")

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    weights_df = compute_benchmark_weights(monthly)
    returns_series = compute_benchmark_returns(monthly, weights_df)
    factor_exposures_df = compute_benchmark_factor_exposures(monthly, weights_df)

    # Save outputs
    weights_path = OUTPUTS / "benchmark_weights.parquet"
    returns_path = OUTPUTS / "benchmark_returns.parquet"
    exposures_path = OUTPUTS / "benchmark_factor_exposures.parquet"

    weights_df.to_parquet(weights_path)
    print(f"[benchmark] Saved weights -> {weights_path}")

    returns_series.to_frame().to_parquet(returns_path)
    print(f"[benchmark] Saved returns -> {returns_path}")

    factor_exposures_df.to_parquet(exposures_path)
    print(f"[benchmark] Saved factor exposures -> {exposures_path}")

    print("[benchmark] Done.")
    return weights_df, returns_series, factor_exposures_df
