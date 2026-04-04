"""Fast data preparation — vectorized resampling to monthly frequency.

Replaces the slow row-by-row loop in data_loader.resample_monthly() with
groupby-last + groupby-apply for compounding returns.

Saves outputs/universe_monthly.parquet for downstream use.
"""
import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    FINAL_PANEL, ALL_FEATURES, BETA_COLS, FF4_FACTORS,
    RET_COL, MKTCAP_COL, UNIVERSE_COVERAGE_THRESHOLD,
    UNIVERSE_MAX_STOCKS, UNIVERSE_MIN_STOCKS,
    RESID_SIGNAL_WIN, RESID_SIGNAL_SKIP, OUTPUTS,
)

SNAPSHOT_COLS = (
    [MKTCAP_COL] + BETA_COLS +
    ["SKEW", "AIV", "GLB", "Mom12m", "IdioVol3F", "BM", "resid_signal"]
)


def main():
    t0 = time.time()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("[fast-prep] Loading final_panel.parquet ...")
    df = pd.read_parquet(FINAL_PANEL)
    for col in df.columns:
        if df[col].dtype == pd.Float64Dtype():
            df[col] = df[col].astype("float64")
    print(f"  Loaded {df.shape[0]:,} rows, {df.index.get_level_values('permno').nunique()} stocks")

    print("[fast-prep] Computing residual signal ...")
    factors = df[FF4_FACTORS].to_numpy(dtype=np.float64, na_value=np.nan)
    betas = df[BETA_COLS].to_numpy(dtype=np.float64, na_value=np.nan)
    ret = df[RET_COL].to_numpy(dtype=np.float64)
    predicted_ret = np.nansum(betas * factors, axis=1)
    resid = ret - predicted_ret
    df["_resid"] = resid

    win = RESID_SIGNAL_WIN
    skip = RESID_SIGNAL_SKIP

    def _signal_per_stock(g):
        r = g["_resid"]
        roll_std = r.rolling(win, min_periods=win // 2).std()
        roll_mean_skip = r.shift(skip).rolling(win - skip, min_periods=(win - skip) // 2).mean()
        return roll_mean_skip / roll_std.replace(0, np.nan)

    df["resid_signal"] = df.groupby(level="permno", group_keys=False).apply(_signal_per_stock)
    df.drop(columns=["_resid"], inplace=True)
    print(f"  resid_signal NaN rate: {df['resid_signal'].isna().mean():.1%}")

    print("[fast-prep] Filtering universe by coverage ...")
    core_features = BETA_COLS + ["SKEW", "AIV"]
    features_to_check = [f for f in core_features if f in df.columns]
    dates = df.index.get_level_values("date")
    df["_year"] = dates.year

    coverage = df.groupby([df.index.get_level_values("permno"), "_year"])[features_to_check].apply(
        lambda x: x.notna().mean().mean()
    )
    coverage.name = "coverage"
    coverage = coverage.reset_index()
    coverage.columns = ["permno", "year", "coverage"]
    good = coverage[coverage["coverage"] >= UNIVERSE_COVERAGE_THRESHOLD]
    good_pairs = set(zip(good["permno"], good["year"]))

    permnos = df.index.get_level_values("permno")
    mask = pd.Series(
        [((p, y) in good_pairs) for p, y in zip(permnos, df["_year"])],
        index=df.index
    )
    df = df[mask.values].drop(columns=["_year"])
    n_stocks = df.index.get_level_values("permno").nunique()
    print(f"  Universe: {n_stocks} stocks after coverage filter")

    print("[fast-prep] Resampling to monthly (VECTORIZED) ...")
    t_resamp = time.time()

    dates = df.index.get_level_values("date")
    permnos = df.index.get_level_values("permno")
    df["_month_end"] = dates.to_period("M").to_timestamp("M")
    df["_permno"] = permnos

    monthly_ret = (
        df.groupby(["_permno", "_month_end"])[RET_COL]
        .apply(lambda x: np.expm1(np.log1p(x.dropna()).sum()))
    )
    monthly_ret.index.names = ["permno", "month_end"]
    monthly_ret.name = "monthly_ret"

    snap_cols = [c for c in SNAPSHOT_COLS if c in df.columns]
    snapshots = df.groupby(["_permno", "_month_end"])[snap_cols].last()
    snapshots.index.names = ["permno", "month_end"]

    monthly = monthly_ret.to_frame().join(snapshots, how="inner")

    fwd = monthly["monthly_ret"].groupby(level="permno").shift(-1)
    monthly["fwd_ret"] = fwd

    print(f"  Resampled in {time.time() - t_resamp:.1f}s")
    print(f"  Monthly panel: {monthly.shape[0]:,} rows, "
          f"{monthly.index.get_level_values('permno').nunique()} stocks, "
          f"{monthly.index.get_level_values('month_end').nunique()} months")

    print(f"[fast-prep] Filtering to top {UNIVERSE_MAX_STOCKS} by market cap ...")
    before = monthly.shape[0]

    def _top_n(group):
        valid = group.dropna(subset=[MKTCAP_COL])
        return valid.nlargest(min(UNIVERSE_MAX_STOCKS, len(valid)), MKTCAP_COL)

    monthly = monthly.groupby(level="month_end", group_keys=False).apply(_top_n)

    counts = monthly.groupby(level="month_end").size()
    min_count = counts.min()
    max_count = counts.max()
    assert min_count >= UNIVERSE_MIN_STOCKS, (
        f"Month {counts.idxmin().date()} has only {min_count} stocks "
        f"(minimum required: {UNIVERSE_MIN_STOCKS})"
    )
    after = monthly.shape[0]
    print(f"  {before:,} → {after:,} rows (dropped {before - after:,})")
    print(f"  Stocks per month: min={min_count}, max={max_count}")

    # Also include FF4 factor returns for Experiment 4
    ff4_in = [c for c in FF4_FACTORS if c in df.columns]
    if ff4_in:
        ff4_monthly = df.groupby(["_permno", "_month_end"])[ff4_in].last()
        ff4_monthly.index.names = ["permno", "month_end"]
        for c in ff4_in:
            if c not in monthly.columns:
                monthly = monthly.join(ff4_monthly[[c]], how="left")

    out_path = OUTPUTS / "universe_monthly.parquet"
    monthly.to_parquet(out_path, engine="pyarrow")
    print(f"[fast-prep] Saved {out_path}")
    print(f"[fast-prep] Total time: {time.time() - t0:.1f}s")

    del df
    gc.collect()


if __name__ == "__main__":
    main()
