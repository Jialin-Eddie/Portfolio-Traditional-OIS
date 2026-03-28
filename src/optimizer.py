"""
optimizer.py — Constrained portfolio optimization using cvxpy.

For each month-end rebalance date, solves for optimal portfolio weights
given XGBoost signal predictions subject to factor exposure, weight
cap, and tracking error constraints.
"""

import warnings
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import (
    BETA_COLS,
    COV_WINDOW_MONTHS,
    CVAR_ALPHA,
    CVAR_LIMIT_MONTHLY,
    DD_SHRINK_FACTOR,
    DD_WARNING_THRESHOLD,
    MAX_FACTOR_DEV,
    MAX_FACTOR_DEV_OPTIM,
    MAX_WEIGHT_MULT,
    OUTPUTS,
    TE_MAX_MONTHLY,
)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def estimate_covariance(
    monthly_returns: pd.DataFrame,
    month_end: pd.Timestamp,
    window: int = COV_WINDOW_MONTHS,
) -> np.ndarray:
    """
    Estimate a Ledoit-Wolf shrinkage covariance matrix for the stock universe.

    Parameters
    ----------
    monthly_returns : pd.DataFrame
        Wide-format monthly returns. Index = month-end dates, columns = permno.
    month_end : pd.Timestamp
        The rebalance date. Window ends at this date (inclusive).
    window : int
        Rolling window length in months (default 36).

    Returns
    -------
    np.ndarray
        (N, N) covariance matrix, positive-semidefinite with small ridge added.
    """
    # Select rows up to and including month_end
    mask = monthly_returns.index <= month_end
    hist = monthly_returns.loc[mask].tail(window)

    if len(hist) == 0:
        N = monthly_returns.shape[1]
        return np.eye(N) * 1e-4

    if len(hist) < window:
        print(
            f"  [Cov] Only {len(hist)} months available (< {window}); "
            "using available history."
        )

    # Fill NaN with 0 — missing = no information
    X = hist.fillna(0.0).values  # shape (T, N)

    lw = LedoitWolf()
    lw.fit(X)
    cov = lw.covariance_  # (N, N)

    # Small ridge to ensure PSD
    N = cov.shape[0]
    cov += 1e-6 * np.eye(N)

    return cov


# ---------------------------------------------------------------------------
# Single-period portfolio optimizer
# ---------------------------------------------------------------------------

def solve_portfolio(
    mu: np.ndarray,
    w_bench: np.ndarray,
    betas: np.ndarray,
    beta_bench: np.ndarray,
    cov: np.ndarray,
    te_max: float = TE_MAX_MONTHLY,
    max_factor_dev: float = MAX_FACTOR_DEV_OPTIM,
    target_sum: float = 1.0,
    historical_returns: Optional[np.ndarray] = None,
    cvar_alpha: float = CVAR_ALPHA,
    cvar_limit: float = CVAR_LIMIT_MONTHLY,
) -> np.ndarray:
    """
    Solve the mean-variance optimization problem with factor and TE constraints.

    Problem
    -------
    maximize    mu^T w
    subject to  sum(w) = 1
                w >= 0
                w <= MAX_WEIGHT_MULT * w_bench
                |betas[:, k]^T w - beta_bench[k]| <= max_factor_dev  for each k
                (w - w_bench)^T Sigma (w - w_bench) <= te_max^2

    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Expected returns from XGBoost signal.
    w_bench : np.ndarray, shape (N,)
        Benchmark weights for the active universe.
    betas : np.ndarray, shape (N, 4)
        Factor betas for each stock (FF4 factors).
    beta_bench : np.ndarray, shape (4,)
        Benchmark factor exposures.
    cov : np.ndarray, shape (N, N)
        Covariance matrix (Ledoit-Wolf, with ridge).
    te_max : float
        Maximum monthly tracking error (annot: monthly std).
    max_factor_dev : float
        Maximum allowed absolute deviation from benchmark factor exposure.

    Returns
    -------
    np.ndarray, shape (N,)
        Optimal portfolio weights. Falls back to benchmark on persistent failure.
    """
    N = len(mu)

    def _build_and_solve(te: float, cv_limit: float) -> tuple[Optional[np.ndarray], str]:
        w = cp.Variable(N)
        objective = cp.Maximize(mu @ w)

        constraints = [
            cp.sum(w) == target_sum,
            w >= 0,
            w <= MAX_WEIGHT_MULT * w_bench,
        ]

        # Factor exposure constraints — one inequality pair per factor
        for k in range(betas.shape[1]):
            constraints.append(betas[:, k] @ w - beta_bench[k] <= max_factor_dev)
            constraints.append(beta_bench[k] - betas[:, k] @ w <= max_factor_dev)

        # Tracking error as SOCP via quad_form
        # (w - w_bench)^T Sigma (w - w_bench) <= te^2
        active = w - w_bench
        # psd_wrap signals to cvxpy that cov is PSD, avoids unnecessary decomposition
        cov_psd = cp.psd_wrap(cov)
        constraints.append(cp.quad_form(active, cov_psd) <= te**2)

        # CVaR constraint on active returns
        if historical_returns is not None and len(historical_returns) >= 12:
            S = historical_returns.shape[0]
            zeta = cp.Variable()
            u = cp.Variable(S, nonneg=True)
            active_w = w - w_bench
            for s in range(S):
                constraints.append(u[s] >= -(historical_returns[s] @ active_w) - zeta)
            constraints.append(
                zeta + cp.sum(u) / ((1 - cvar_alpha) * S) <= cv_limit
            )

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except cp.SolverError:
                return None, "SolverError"

        status = prob.status
        if status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return w.value, status
        return None, status if status else "unknown"

    # Relaxation ladder: (cvar_limit, te_max) pairs
    relaxation_ladder = [
        (cvar_limit, te_max),
        (0.025, 0.02),
        (0.03, 0.025),
        (0.04, 0.03),
    ]

    for i, (cv_lim, te_lim) in enumerate(relaxation_ladder):
        w_opt, status = _build_and_solve(te_lim, cv_lim)
        if w_opt is not None:
            if i > 0:
                warnings.warn(
                    f"  [Solver] Solved with relaxed cvar_limit={cv_lim:.3f}, "
                    f"te_max={te_lim:.4f} (ladder step {i}).",
                    stacklevel=2,
                )
            return w_opt
        if i == 0:
            warnings.warn(
                f"  [Solver] Infeasible at cvar_limit={cv_lim:.3f}, "
                f"te_max={te_lim:.4f} (status={status}). Relaxing constraints.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"  [Solver] Still infeasible at cvar_limit={cv_lim:.3f}, "
                f"te_max={te_lim:.4f} (status={status}).",
                stacklevel=2,
            )

    # Final fallback: return benchmark weights
    warnings.warn(
        "  [Solver] All relaxations failed. Falling back to benchmark weights.",
        stacklevel=2,
    )
    return w_bench.copy()


# ---------------------------------------------------------------------------
# Post-hoc drawdown guard
# ---------------------------------------------------------------------------

def _apply_drawdown_guard(
    w_opt: np.ndarray,
    w_bench: np.ndarray,
    realized_active_returns: list,
    warning_threshold: float = DD_WARNING_THRESHOLD,
    shrink_factor: float = DD_SHRINK_FACTOR,
) -> tuple:
    """Shrink active weights if recent active return is dangerously negative.

    Returns (adjusted_weights, was_shrunk).
    """
    if len(realized_active_returns) < 1:
        return w_opt, False
    last_active = realized_active_returns[-1]
    if last_active < warning_threshold:
        active = w_opt - w_bench
        return w_bench + shrink_factor * active, True
    return w_opt, False


# ---------------------------------------------------------------------------
# Full optimization loop
# ---------------------------------------------------------------------------

def optimize_all_months(
    monthly: pd.DataFrame,
    predictions: pd.DataFrame,
    bench_weights: pd.DataFrame,
    bench_betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the portfolio optimizer for every rebalance month in predictions.

    Parameters
    ----------
    monthly : pd.DataFrame
        Wide-format monthly excess returns.
        Index = month-end dates, columns = permno (int or str).
    predictions : pd.DataFrame
        ML signal predictions. Must have columns ['permno', 'month_end', 'pred'].
        Or MultiIndex (permno, month_end) with a 'pred' column.
    bench_weights : pd.DataFrame
        Benchmark weights. Must have columns ['permno', 'month_end', 'weight'].
    bench_betas : pd.DataFrame
        Benchmark factor betas. Must have columns
        ['month_end'] + BETA_COLS (e.g., beta_mktrf, beta_smb, beta_hml, beta_mom).

    Returns
    -------
    pd.DataFrame
        MultiIndex (permno, month_end) with column 'opt_weight'.
    """
    # Normalize to long form with explicit columns
    preds = _ensure_long(predictions, value_col="pred")
    bweights = _ensure_long(bench_weights, value_col="weight")

    rebalance_dates = sorted(preds["month_end"].unique())
    print(f"[Optimizer] Running {len(rebalance_dates)} rebalance months.")

    records = []

    # Minimum months of return history needed before optimizing
    MIN_HISTORY = 12

    # Drawdown guard state
    realized_active: list = []
    prev_qs_weights: Optional[dict] = None
    prev_bench_weights: Optional[dict] = None
    prev_month: Optional[pd.Timestamp] = None

    for month_end in rebalance_dates:
        # --- Compute previous month's realized active return (backward-looking) ---
        if prev_qs_weights is not None and prev_month is not None and prev_month in monthly.index:
            fwd_ret_row = monthly.loc[prev_month].fillna(0.0)
            qs_ret = sum(
                prev_qs_weights.get(int(p), prev_qs_weights.get(p, 0)) * fwd_ret_row.get(p, 0)
                for p in fwd_ret_row.index
            )
            bench_ret = sum(
                prev_bench_weights.get(int(p), prev_bench_weights.get(p, 0)) * fwd_ret_row.get(p, 0)
                for p in fwd_ret_row.index
            )
            realized_active.append(qs_ret - bench_ret)
        # --- Slice this month's data ---
        pred_m = preds[preds["month_end"] == month_end].set_index("permno")["pred"]
        bw_m = bweights[bweights["month_end"] == month_end].set_index("permno")["weight"]

        # Benchmark factor exposures for this month
        bb_row = bench_betas[bench_betas["month_end"] == month_end]
        if bb_row.empty:
            print(f"  [{month_end.date()}] No benchmark betas found — skipping.")
            continue
        beta_bench_vec = bb_row[BETA_COLS].values.ravel().astype(float)  # (4,)

        # Check if enough history for covariance
        hist_mask = monthly.index <= month_end
        n_hist = hist_mask.sum()
        if n_hist < MIN_HISTORY:
            print(f"  [{month_end.date()}] Only {n_hist} months history (< {MIN_HISTORY}) — using benchmark weights.")
            for permno in bw_m.index:
                records.append({"permno": permno, "month_end": month_end, "opt_weight": bw_m.get(permno, 0.0)})
            continue

        # --- Align universe (inner join) ---
        universe = pred_m.index.intersection(bw_m.index)
        if len(universe) == 0:
            print(f"  [{month_end.date()}] Empty universe after alignment — skipping.")
            continue

        stock_betas = _get_stock_betas(preds, bweights, bench_betas, month_end, universe)

        pred_m = pred_m.loc[universe]
        bw_m = bw_m.loc[universe]

        # --- Drop stocks with NaN in predictions, betas, or weights ---
        valid_mask = (
            pred_m.notna()
            & bw_m.notna()
            & stock_betas.notna().all(axis=1)
        )
        valid_stocks = valid_mask[valid_mask].index
        if len(valid_stocks) < 10:
            print(f"  [{month_end.date()}] Only {len(valid_stocks)} valid stocks — using benchmark weights.")
            for permno in bw_m.index:
                records.append({"permno": permno, "month_end": month_end, "opt_weight": bw_m.get(permno, 0.0)})
            continue

        pred_m = pred_m.loc[valid_stocks]
        bw_m = bw_m.loc[valid_stocks]
        stock_betas = stock_betas.loc[valid_stocks]

        # Compute how much weight budget goes to excluded stocks
        all_month_bw = bweights[bweights["month_end"] == month_end].set_index("permno")["weight"]
        excluded_stocks = all_month_bw.index.difference(valid_stocks)
        excluded_weight = all_month_bw.loc[excluded_stocks].sum() if len(excluded_stocks) > 0 else 0.0
        target_sum = 1.0 - excluded_weight

        bw_sum = bw_m.sum()
        if bw_sum <= 0 or target_sum <= 0:
            print(f"  [{month_end.date()}] Zero benchmark weights — skipping.")
            continue

        # --- Covariance + historical return scenarios for CVaR ---
        ret_cols = monthly.columns.intersection(valid_stocks)
        ret_subset = monthly[ret_cols].reindex(columns=valid_stocks, fill_value=np.nan)
        cov = estimate_covariance(ret_subset, month_end)

        hist_mask = monthly.index <= month_end
        hist_returns = ret_subset.loc[hist_mask].tail(COV_WINDOW_MONTHS).fillna(0.0).values

        # --- Arrays (ensure no NaN) ---
        mu = np.nan_to_num(pred_m.values.astype(float), nan=0.0)
        w_bench = np.nan_to_num(bw_m.values.astype(float), nan=0.0)
        betas_arr = np.nan_to_num(stock_betas.values.astype(float), nan=0.0)
        cov = np.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)

        # Check for NaN/Inf in benchmark betas
        beta_bench_vec = np.nan_to_num(beta_bench_vec, nan=0.0)

        # --- Solve ---
        w_opt = solve_portfolio(
            mu, w_bench, betas_arr, beta_bench_vec, cov,
            target_sum=target_sum,
            historical_returns=hist_returns,
        )

        # --- Post-hoc drawdown guard ---
        w_opt, was_shrunk = _apply_drawdown_guard(w_opt, w_bench, realized_active)
        if was_shrunk:
            print(f"  [{month_end.date()}] DRAWDOWN GUARD: shrunk active weights by {DD_SHRINK_FACTOR}")

        # --- Diagnostics ---
        active = w_opt - w_bench
        te_realised = float(np.sqrt(active @ cov @ active))
        factor_dev = np.abs(betas_arr.T @ w_opt - beta_bench_vec).max()
        print(
            f"  [{month_end.date()}] N={len(valid_stocks):4d} | "
            f"TE={te_realised:.4f} | MaxFactorDev={factor_dev:.4f} | "
            f"SumW={w_opt.sum():.4f}"
        )

        # Record optimized weights for valid stocks
        for permno, wt in zip(valid_stocks, w_opt):
            records.append({"permno": permno, "month_end": month_end, "opt_weight": wt})

        # Assign benchmark weights to excluded stocks (keeps factor exposure neutral)
        all_month_bw = bweights[bweights["month_end"] == month_end].set_index("permno")["weight"]
        excluded = all_month_bw.index.difference(valid_stocks)
        for permno in excluded:
            records.append({"permno": permno, "month_end": month_end, "opt_weight": all_month_bw[permno]})

        # --- Store weights for next iteration's realized active return calc ---
        prev_qs_weights = dict(zip(valid_stocks, w_opt))
        for permno in excluded:
            prev_qs_weights[permno] = all_month_bw[permno]
        prev_bench_weights = dict(zip(bw_m.index, bw_m.values))
        for permno in excluded:
            prev_bench_weights[permno] = all_month_bw[permno]
        prev_month = month_end

    if not records:
        print("[Optimizer] No records produced — returning empty DataFrame.")
        return pd.DataFrame(columns=["permno", "month_end", "opt_weight"]).set_index(
            ["permno", "month_end"]
        )

    result = pd.DataFrame(records).set_index(["permno", "month_end"])
    print(f"[Optimizer] Done. Total (permno, month_end) records: {len(result):,}")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_long(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Normalise a DataFrame to have ['permno', 'month_end', value_col, ...] columns.

    Preserves ALL columns (including beta columns) — only converts the index.
    Handles both:
    - MultiIndex (permno, month_end) DataFrames
    - Flat DataFrames that already have those columns
    """
    if isinstance(df.index, pd.MultiIndex):
        out = df.reset_index()
        if "level_0" in out.columns:
            out = out.rename(columns={"level_0": "permno", "level_1": "month_end"})
        return out
    required = {"permno", "month_end", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    return df.copy()


def _get_stock_betas(
    preds: pd.DataFrame,
    bweights: pd.DataFrame,
    bench_betas: pd.DataFrame,
    month_end: pd.Timestamp,
    universe: pd.Index,
) -> pd.DataFrame:
    """
    Retrieve per-stock factor betas aligned to `universe`.

    Checks preds first, then bweights, then falls back to zeros with a warning.

    Returns
    -------
    pd.DataFrame
        shape (N, 4) indexed by permno, columns = BETA_COLS
    """
    for source, name in [(preds, "predictions"), (bweights, "bench_weights")]:
        cols_present = [c for c in BETA_COLS if c in source.columns]
        if len(cols_present) == len(BETA_COLS):
            month_slice = source[source["month_end"] == month_end].set_index("permno")
            betas_df = month_slice.loc[month_slice.index.intersection(universe), BETA_COLS]
            betas_df = betas_df.reindex(universe, fill_value=0.0)
            return betas_df

    warnings.warn(
        f"  [Betas] No per-stock betas found in predictions or bench_weights "
        f"for {month_end.date()}. Using zeros.",
        stacklevel=3,
    )
    return pd.DataFrame(0.0, index=universe, columns=BETA_COLS)
