"""Cross-sectional preprocessing pipeline: winsorize -> Box-Cox -> z-score.

Strict train-only fitting: winsorization bounds and Box-Cox lambda are
computed on training data only. OOS dates fall back to their own
cross-section quantiles for winsorization, and use the stored lambda.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config import ALL_FEATURES, OUTPUTS, WINSOR_HIGH, WINSOR_LOW


def _shift_positive(series: pd.Series) -> tuple[pd.Series, float]:
    """Shift a series so all values are strictly positive (required for Box-Cox)."""
    s = series.dropna()
    if len(s) == 0:
        return series, 0.0
    shift = 0.0
    if s.min() <= 0:
        shift = abs(s.min()) + 1e-6
    return series + shift, shift


def _fit_boxcox_lambda(train_df: pd.DataFrame, feat: str) -> float:
    """Fit Box-Cox lambda on pooled training data for one feature.

    Pools all non-NaN values across all training months, shifts to positive,
    and fits lambda via MLE. Returns lambda (float).
    """
    vals = train_df[feat].dropna().values.astype(np.float64)
    if len(vals) < 20:
        return 1.0
    shifted = vals - vals.min() + 1e-6
    try:
        _, lmbda = sp_stats.boxcox(shifted)
        lmbda = np.clip(lmbda, -5.0, 5.0)
    except Exception:
        lmbda = 1.0
    return float(lmbda)


def fit_preprocess_pipeline(
    train_df: pd.DataFrame,
    features: list[str],
) -> dict:
    """Fit winsorization bounds and Box-Cox lambda on training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        MultiIndex (permno, month_end) DataFrame containing feature columns.
    features : list[str]
        Feature columns to fit.

    Returns
    -------
    dict with keys:
        "features" : list[str]
        "winsor"   : {feat: {"p01": pd.Series, "p99": pd.Series}}
        "boxcox"   : {feat: {"lambda": float}}
    """
    print(f"[preprocessing] Fitting on {train_df.shape[0]} rows, "
          f"{len(features)} features.")

    month_ends = train_df.index.get_level_values("month_end")
    params: dict = {"features": list(features), "winsor": {}, "boxcox": {}}

    for feat in features:
        series = train_df[feat].copy()
        grouped = series.groupby(month_ends)
        p01 = grouped.quantile(WINSOR_LOW)
        p99 = grouped.quantile(WINSOR_HIGH)
        p01.index.name = "month_end"
        p99.index.name = "month_end"
        params["winsor"][feat] = {"p01": p01, "p99": p99}

        lmbda = _fit_boxcox_lambda(train_df, feat)
        params["boxcox"][feat] = {"lambda": lmbda}

        print(f"  [{feat}] winsor bounds: {len(p01)} months, "
              f"Box-Cox lambda={lmbda:.4f}")

    print("[preprocessing] Fit done.")
    return params


def apply_preprocess_pipeline(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Apply winsorize -> Box-Cox -> z-score pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (permno, month_end) DataFrame.
    params : dict
        Output of fit_preprocess_pipeline.

    Returns
    -------
    pd.DataFrame
        Processed copy. Features are z-scored (mean~0, std~1) per month.
        NaN filled with 0 (= cross-sectional mean).
    """
    print(f"[preprocessing] Processing {df.shape[0]} rows.")
    out = df.copy()
    features: list[str] = params["features"]
    month_ends = out.index.get_level_values("month_end")
    known_months: set = set(
        next(iter(params["winsor"].values()))["p01"].index
    )

    # ================================================================== #
    # Step 1 — Winsorize (per month cross-section)                        #
    # ================================================================== #
    print("[preprocessing] Step 1: Winsorizing...")
    for feat in features:
        stored_p01: pd.Series = params["winsor"][feat]["p01"]
        stored_p99: pd.Series = params["winsor"][feat]["p99"]

        def _winsor_group(grp: pd.Series) -> pd.Series:
            date = grp.index.get_level_values("month_end")[0]
            if date in known_months:
                lo = stored_p01.loc[date]
                hi = stored_p99.loc[date]
            else:
                lo = grp.quantile(WINSOR_LOW)
                hi = grp.quantile(WINSOR_HIGH)
            return grp.clip(lower=lo, upper=hi)

        out[feat] = (
            out[feat]
            .groupby(month_ends, group_keys=False)
            .apply(_winsor_group)
        )

    # ================================================================== #
    # Step 2 — Box-Cox transform (per month, using stored lambda)         #
    # ================================================================== #
    print("[preprocessing] Step 2: Box-Cox transforming...")
    for feat in features:
        lmbda = params["boxcox"][feat]["lambda"]

        def _boxcox_group(grp: pd.Series) -> pd.Series:
            vals = grp.copy()
            non_null = vals.dropna()
            if len(non_null) < 2:
                return vals
            shift = 0.0
            if non_null.min() <= 0:
                shift = abs(non_null.min()) + 1e-6
            shifted = vals + shift
            shifted = shifted.clip(lower=1e-10)
            if abs(lmbda) < 1e-10:
                transformed = np.log(shifted)
            else:
                transformed = (shifted ** lmbda - 1.0) / lmbda
            return transformed

        out[feat] = (
            out[feat]
            .groupby(month_ends, group_keys=False)
            .apply(_boxcox_group)
        )

    # ================================================================== #
    # Step 3 — Z-score (per month cross-section)                          #
    # ================================================================== #
    print("[preprocessing] Step 3: Z-scoring...")
    for feat in features:
        def _zscore_group(grp: pd.Series) -> pd.Series:
            mu = grp.mean()
            sigma = grp.std()
            if sigma is None or sigma == 0 or np.isnan(sigma):
                return grp - mu if not np.isnan(mu) else grp * 0.0
            return (grp - mu) / sigma

        out[feat] = (
            out[feat]
            .groupby(month_ends, group_keys=False)
            .apply(_zscore_group)
        )

    # ================================================================== #
    # Step 4 — Fill NaN with 0 (cross-sectional mean after z-score)       #
    # ================================================================== #
    print("[preprocessing] Step 4: Filling NaNs with 0...")
    out[features] = out[features].fillna(0.0)

    # ================================================================== #
    # Assertions                                                           #
    # ================================================================== #
    for feat in features:
        col = out[feat]
        nan_count = col.isna().sum()
        assert nan_count == 0, (
            f"[preprocessing] {feat} still has {nan_count} NaNs after fillna."
        )
        inf_count = np.isinf(col).sum()
        assert inf_count == 0, (
            f"[preprocessing] {feat} has {inf_count} Inf values."
        )

    print("[preprocessing] All assertions passed.")
    return out


def save_pipeline(params: dict, path: Path) -> None:
    """Serialize pipeline params to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(params, path)
    print(f"[preprocessing] Saved to {path}")


def load_pipeline(path: Path) -> dict:
    """Load pipeline params from disk."""
    path = Path(path)
    params = joblib.load(path)
    print(f"[preprocessing] Loaded from {path} "
          f"(features: {params['features']})")
    return params
