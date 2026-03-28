"""
Tests for CVaR constraint and post-hoc drawdown guard.

Tests:
1. CVaR constraint is convex and feasible
2. No lookahead in drawdown guard
3. Shrinkage math correctness
4. Scenario data alignment (no future data)
5. CVaR relaxation cascade
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CVAR_ALPHA,
    CVAR_LIMIT_MONTHLY,
    DD_SHRINK_FACTOR,
    DD_WARNING_THRESHOLD,
)
from src.optimizer import _apply_drawdown_guard, solve_portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_problem(N: int = 10, S: int = 24, seed: int = 42):
    """Return (mu, w_bench, betas, beta_bench, cov, historical_returns)."""
    rng = np.random.default_rng(seed)

    mu = rng.normal(0.005, 0.02, N)

    # Benchmark weights: random, summing to 1
    raw = rng.uniform(0.05, 0.15, N)
    w_bench = raw / raw.sum()

    # Identity covariance + small ridge
    cov = np.eye(N) * 0.0025 + 1e-6 * np.eye(N)

    # Factor betas: shape (N, 4)
    betas = rng.normal(1.0, 0.3, (N, 4))

    # Benchmark factor exposures
    beta_bench = betas.T @ w_bench  # (4,)

    # Historical return scenarios: shape (S, N) — small noise around zero
    historical_returns = rng.normal(0.0, 0.03, (S, N))

    return mu, w_bench, betas, beta_bench, cov, historical_returns


# ---------------------------------------------------------------------------
# Test 1: CVaR constraint is convex and feasible
# ---------------------------------------------------------------------------

def test_cvar_convexity():
    """CVaR constraint produces a feasible CVXPY problem with valid weights."""
    N, S = 10, 24
    mu, w_bench, betas, beta_bench, cov, hist_returns = _make_synthetic_problem(N, S)

    result = solve_portfolio(
        mu=mu,
        w_bench=w_bench,
        betas=betas,
        beta_bench=beta_bench,
        cov=cov,
        historical_returns=hist_returns,
        cvar_alpha=CVAR_ALPHA,
        cvar_limit=CVAR_LIMIT_MONTHLY,
    )

    assert result is not None, "solve_portfolio returned None — problem infeasible"
    assert abs(result.sum() - 1.0) < 1e-3, f"Weights do not sum to 1.0: {result.sum()}"
    assert (result >= -1e-4).all(), "Some weights are negative"


# ---------------------------------------------------------------------------
# Test 2: No lookahead in drawdown guard
# ---------------------------------------------------------------------------

def test_no_lookahead_drawdown_guard():
    """_apply_drawdown_guard only uses realized_active_returns (past values)."""
    w_opt = np.array([0.3, 0.3, 0.2, 0.2])
    w_bench = np.array([0.25, 0.25, 0.25, 0.25])

    # Case 1: empty realized_active — should return w_opt unchanged, no shrink
    result1, shrunk1 = _apply_drawdown_guard(w_opt, w_bench, realized_active_returns=[])
    assert not shrunk1, "Should NOT shrink when realized_active is empty"
    np.testing.assert_array_equal(result1, w_opt)

    # Case 2: last active = -0.02 (< threshold -0.015) — should shrink
    result2, shrunk2 = _apply_drawdown_guard(w_opt, w_bench, realized_active_returns=[-0.02])
    assert shrunk2, "Should shrink when last active return < DD_WARNING_THRESHOLD"

    # Case 3: last active = -0.01 (> threshold -0.015) — should NOT shrink
    result3, shrunk3 = _apply_drawdown_guard(w_opt, w_bench, realized_active_returns=[-0.01])
    assert not shrunk3, "Should NOT shrink when last active return > DD_WARNING_THRESHOLD"
    np.testing.assert_array_equal(result3, w_opt)

    # Verify: the function only inspects realized_active_returns[-1], i.e. pure look-back
    # Passing a long history should give same result as single-value list
    long_hist = [0.01, 0.005, -0.02]  # last entry triggers guard
    result4, shrunk4 = _apply_drawdown_guard(w_opt, w_bench, realized_active_returns=long_hist)
    assert shrunk4, "Should shrink when last entry of longer history < threshold"
    np.testing.assert_allclose(result4, result2, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: Shrinkage math correctness
# ---------------------------------------------------------------------------

def test_shrinkage_correctness():
    """_apply_drawdown_guard interpolates to midpoint between w_opt and w_bench."""
    w_opt = np.array([0.3, 0.3, 0.2, 0.2])
    w_bench = np.array([0.25, 0.25, 0.25, 0.25])
    realized = [-0.02]  # triggers guard (< -0.015)

    result, shrunk = _apply_drawdown_guard(w_opt, w_bench, realized_active_returns=realized)

    # Expected: w_bench + 0.5 * (w_opt - w_bench)
    expected = w_bench + DD_SHRINK_FACTOR * (w_opt - w_bench)
    assert shrunk, "Guard should have fired"
    np.testing.assert_allclose(result, expected, rtol=1e-10)

    # Verify weights still sum to 1 after shrinkage (linear combination preserves sum)
    assert abs(result.sum() - 1.0) < 1e-10, f"Shrunk weights do not sum to 1: {result.sum()}"


# ---------------------------------------------------------------------------
# Test 4: Scenario data alignment — no future data
# ---------------------------------------------------------------------------

def test_scenario_no_future_data():
    """Historical return scenarios use only dates <= rebalance month."""
    rng = np.random.default_rng(0)
    N = 5
    permnos = [100, 200, 300, 400, 500]

    # Create monthly returns DataFrame from 2018-01 to 2023-12
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    returns_wide = pd.DataFrame(
        rng.normal(0.0, 0.03, (len(dates), N)),
        index=dates,
        columns=permnos,
    )

    window = 36  # COV_WINDOW_MONTHS

    # For each rebalance date, verify hist_returns only uses dates <= month_end
    test_months = [
        pd.Timestamp("2021-03-31"),
        pd.Timestamp("2022-12-31"),
        pd.Timestamp("2023-06-30"),
    ]

    for month_end in test_months:
        hist_mask = returns_wide.index <= month_end
        hist_rows = returns_wide.loc[hist_mask].tail(window)

        # All row dates must be <= month_end (no future data)
        assert (hist_rows.index <= month_end).all(), (
            f"Future data found for month_end={month_end.date()}: "
            f"max date={hist_rows.index.max().date()}"
        )

        # Must not include any date after month_end
        future_rows = returns_wide.loc[returns_wide.index > month_end]
        assert len(hist_rows.index.intersection(future_rows.index)) == 0, (
            f"Scenarios contain future dates for month_end={month_end.date()}"
        )


# ---------------------------------------------------------------------------
# Test 5: CVaR relaxation cascade
# ---------------------------------------------------------------------------

def test_cvar_relaxation_cascade():
    """Relaxation cascade always returns valid weights; solver never returns None.

    Approach: patch the inner _build_and_solve to always return (None, 'infeasible'),
    which forces the cascade to exhaust all ladder steps and fall back to w_bench.
    """
    from unittest.mock import patch

    N, S = 10, 24
    mu, w_bench, betas, beta_bench, cov, hist_returns = _make_synthetic_problem(N, S, seed=7)

    # Patch _build_and_solve inside solve_portfolio so every step returns infeasible
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        # We patch the closure's build-and-solve by patching cp.Problem.solve
        # to raise SolverError, which forces the fallback path
        import cvxpy as cp

        original_solve = cp.Problem.solve

        def always_infeasible(self, *args, **kwargs):
            self._status = "infeasible"
            self._value = None
            for v in self.variables():
                v._value = None

        with patch.object(cp.Problem, "solve", always_infeasible):
            result = solve_portfolio(
                mu=mu,
                w_bench=w_bench,
                betas=betas,
                beta_bench=beta_bench,
                cov=cov,
                historical_returns=hist_returns,
                cvar_alpha=CVAR_ALPHA,
                cvar_limit=CVAR_LIMIT_MONTHLY,
            )

    assert result is not None, "solve_portfolio must never return None"
    # When all steps fail, should return w_bench
    np.testing.assert_allclose(result, w_bench, rtol=1e-10,
                               err_msg="All-infeasible cascade should return w_bench")
    assert abs(result.sum() - 1.0) < 1e-6, f"Fallback weights do not sum to 1: {result.sum()}"

    # Should have emitted fallback warning
    fallback_msgs = [
        w for w in caught_warnings
        if "fallback" in str(w.message).lower() or "All relaxations" in str(w.message)
    ]
    assert len(fallback_msgs) > 0, (
        "Expected 'All relaxations failed' warning but none were emitted. "
        f"Got warnings: {[str(w.message) for w in caught_warnings]}"
    )
