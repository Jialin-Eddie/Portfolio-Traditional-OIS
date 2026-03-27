"""
backtest.py — Monthly rebalancing engine for Portfolio-Traditional-OIS.

Computes portfolio returns, active returns, cumulative returns, drawdowns,
and orchestrates the full backtest pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ANNUALIZE_MONTHLY, OUTPUTS


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------


def compute_portfolio_returns(
    weights: pd.DataFrame,
    monthly: pd.DataFrame,
    weight_col: str = "opt_weight",
) -> pd.Series:
    """Compute monthly portfolio returns from weights and forward returns.

    For each month_end t: R_portfolio(t) = sum(w_i(t) * fwd_ret_i(t)).

    Parameters
    ----------
    weights:
        DataFrame with MultiIndex (permno, month_end) and a weight column.
    monthly:
        DataFrame with MultiIndex (permno, month_end) including column ``fwd_ret``.
    weight_col:
        Name of the weight column in *weights*.

    Returns
    -------
    pd.Series
        Monthly portfolio returns indexed by month_end.
    """
    # Align on (permno, month_end) index
    w = weights[[weight_col]].rename(columns={weight_col: "w"})
    ret = monthly[["fwd_ret"]]

    merged = w.join(ret, how="inner")
    merged["weighted_ret"] = merged["w"] * merged["fwd_ret"]

    port_returns = merged.groupby(level="month_end")["weighted_ret"].sum()
    port_returns.name = "portfolio_ret"

    print(f"[backtest] compute_portfolio_returns: {len(port_returns)} months computed")
    return port_returns


def compute_active_returns(
    qs_returns: pd.Series,
    bench_returns: pd.Series,
) -> pd.Series:
    """Compute active (excess) returns relative to benchmark.

    R_active(t) = R_QS(t) - R_BNCH(t)

    Parameters
    ----------
    qs_returns:
        Monthly quantitative strategy returns.
    bench_returns:
        Monthly benchmark returns.

    Returns
    -------
    pd.Series
        Active returns indexed by month_end.
    """
    active = qs_returns.sub(bench_returns).dropna()
    active.name = "active_ret"
    print(
        f"[backtest] compute_active_returns: mean active={active.mean():.4f}, "
        f"std={active.std():.4f}"
    )
    return active


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative compounded returns.

    cum_ret(t) = prod(1 + R(s) for s <= t) - 1

    Parameters
    ----------
    returns:
        Monthly returns series.

    Returns
    -------
    pd.Series
        Cumulative returns indexed by month_end.
    """
    cum = (1 + returns).cumprod() - 1
    cum.name = "cum_ret"
    print(
        f"[backtest] compute_cumulative_returns: terminal cum_ret={cum.iloc[-1]:.4f}"
    )
    return cum


def compute_drawdown(cum_returns: pd.Series) -> pd.Series:
    """Compute drawdown from peak for a cumulative return series.

    DD(t) = (1 + cum_ret(t)) / max_{s<=t}(1 + cum_ret(s)) - 1

    Parameters
    ----------
    cum_returns:
        Cumulative returns series (as produced by compute_cumulative_returns).

    Returns
    -------
    pd.Series
        Drawdown series (values <= 0) indexed by month_end.
    """
    wealth = 1 + cum_returns
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    dd.name = "drawdown"
    print(f"[backtest] compute_drawdown: max drawdown={dd.min():.4f}")
    return dd


def compute_relative_drawdown(
    qs_returns: pd.Series,
    bench_returns: pd.Series,
) -> pd.Series:
    """Compute relative drawdown of QS vs benchmark.

    Relative wealth ratio W_rel(t) = cumprod(1+R_QS) / cumprod(1+R_BNCH).
    Relative DD(t) = W_rel(t) / max_{s<=t}(W_rel(s)) - 1.

    Parameters
    ----------
    qs_returns:
        Monthly QS returns.
    bench_returns:
        Monthly benchmark returns.

    Returns
    -------
    pd.Series
        Relative drawdown series indexed by month_end.
    """
    qs_wealth = (1 + qs_returns).cumprod()
    bench_wealth = (1 + bench_returns).cumprod()
    rel_wealth = qs_wealth / bench_wealth
    running_max = rel_wealth.cummax()
    rel_dd = rel_wealth / running_max - 1
    rel_dd.name = "relative_drawdown"
    print(
        f"[backtest] compute_relative_drawdown: max relative drawdown={rel_dd.min():.4f}"
    )
    return rel_dd


# ---------------------------------------------------------------------------
# Summary statistics helpers
# ---------------------------------------------------------------------------


def _annualized_return(returns: pd.Series) -> float:
    """Compound annualized return from monthly returns."""
    n = len(returns)
    if n == 0:
        return float("nan")
    return float((1 + returns).prod() ** (ANNUALIZE_MONTHLY / n) - 1)


def _annualized_vol(returns: pd.Series) -> float:
    """Annualized volatility from monthly returns."""
    return float(returns.std() * np.sqrt(ANNUALIZE_MONTHLY))


def _sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio (assumes returns are already excess returns)."""
    vol = _annualized_vol(returns)
    if vol == 0:
        return float("nan")
    return float(_annualized_return(returns) / vol)


def _compute_summary(
    qs_ret: pd.Series,
    bench_ret: pd.Series,
    active_ret: pd.Series,
    dd: pd.Series,
    rel_dd: pd.Series,
) -> dict:
    """Assemble a dict of summary statistics."""
    return {
        "qs_ann_ret": _annualized_return(qs_ret),
        "bench_ann_ret": _annualized_return(bench_ret),
        "qs_ann_vol": _annualized_vol(qs_ret),
        "bench_ann_vol": _annualized_vol(bench_ret),
        "qs_sharpe": _sharpe(qs_ret),
        "bench_sharpe": _sharpe(bench_ret),
        "active_ann_ret": _annualized_return(active_ret),
        "active_ann_vol": _annualized_vol(active_ret),
        "info_ratio": _sharpe(active_ret),
        "max_drawdown": float(dd.min()),
        "max_relative_drawdown": float(rel_dd.min()),
        "n_months": len(qs_ret),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_backtest(
    opt_weights: pd.DataFrame,
    bench_weights: pd.DataFrame,
    monthly: pd.DataFrame,
) -> dict:
    """Run the full backtest pipeline.

    Parameters
    ----------
    opt_weights:
        DataFrame with MultiIndex (permno, month_end) and column ``opt_weight``.
    bench_weights:
        DataFrame with MultiIndex (permno, month_end) and column ``bench_weight``.
    monthly:
        DataFrame with MultiIndex (permno, month_end) including ``fwd_ret``.

    Returns
    -------
    dict
        Keys:
        - ``qs_returns``       : monthly QS portfolio returns
        - ``bench_returns``    : monthly benchmark returns
        - ``active_returns``   : monthly active returns
        - ``qs_cumulative``    : cumulative QS returns
        - ``bench_cumulative`` : cumulative benchmark returns
        - ``drawdown``         : QS drawdown series
        - ``relative_drawdown``: relative drawdown series
        - ``summary``          : dict of annualized stats
    """
    print("[backtest] Starting backtest pipeline ...")

    qs_returns = compute_portfolio_returns(opt_weights, monthly, weight_col="opt_weight")
    bench_returns = compute_portfolio_returns(
        bench_weights, monthly, weight_col="bench_weight"
    )

    # Align to common months
    common_idx = qs_returns.index.intersection(bench_returns.index)
    qs_returns = qs_returns.loc[common_idx]
    bench_returns = bench_returns.loc[common_idx]

    active_returns = compute_active_returns(qs_returns, bench_returns)
    qs_cum = compute_cumulative_returns(qs_returns)
    bench_cum = compute_cumulative_returns(bench_returns)
    dd = compute_drawdown(qs_cum)
    rel_dd = compute_relative_drawdown(qs_returns, bench_returns)

    summary = _compute_summary(qs_returns, bench_returns, active_returns, dd, rel_dd)

    # Print summary
    print("\n[backtest] ---- Summary Statistics ----")
    for k, v in summary.items():
        print(f"  {k:35s}: {v:.4f}" if isinstance(v, float) else f"  {k:35s}: {v}")

    # Save outputs
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        {
            "qs_ret": qs_returns,
            "bench_ret": bench_returns,
            "active_ret": active_returns,
            "qs_cum_ret": qs_cum,
            "bench_cum_ret": bench_cum,
            "drawdown": dd,
            "relative_drawdown": rel_dd,
        }
    )
    out_path = OUTPUTS / "backtest_results.parquet"
    results_df.to_parquet(out_path)
    print(f"[backtest] Results saved to {out_path}")

    summary_path = OUTPUTS / "backtest_summary.csv"
    pd.Series(summary).to_csv(summary_path, header=["value"])
    print(f"[backtest] Summary saved to {summary_path}")

    return {
        "qs_returns": qs_returns,
        "bench_returns": bench_returns,
        "active_returns": active_returns,
        "qs_cumulative": qs_cum,
        "bench_cumulative": bench_cum,
        "drawdown": dd,
        "relative_drawdown": rel_dd,
        "summary": summary,
    }
