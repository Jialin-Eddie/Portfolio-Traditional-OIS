"""Load final_panel from 01HW and prepare monthly universe."""
import gc
import numpy as np
import pandas as pd

from src.config import (
    FINAL_PANEL, ALL_FEATURES, BETA_COLS, FF4_FACTORS,
    RET_COL, MKTCAP_COL, UNIVERSE_COVERAGE_THRESHOLD,
    UNIVERSE_MAX_STOCKS, UNIVERSE_MIN_STOCKS,
    RESID_SIGNAL_WIN, RESID_SIGNAL_SKIP, OUTPUTS,
)


def load_panel() -> pd.DataFrame:
    """Load final_panel.parquet and convert nullable Float64 to float64."""
    print("[data_loader] Loading final_panel.parquet ...")
    df = pd.read_parquet(FINAL_PANEL)
    for col in df.columns:
        if df[col].dtype == pd.Float64Dtype():
            df[col] = df[col].astype("float64")
    print(f"  Loaded {df.shape[0]:,} rows, {df.index.get_level_values('permno').nunique()} stocks")
    return df


def compute_resid_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual momentum signal (Blitz, Huij & Martens 2011).

    resid(i,t) = excess_ret(i,t) - sum_k beta_k(i,t) * factor_k(t)
    signal(i,t) = mean(resid[t-win..t-skip-1]) / std(resid[t-win..t-1])
    """
    print("[data_loader] Computing residual signal ...")
    factors = df[FF4_FACTORS].to_numpy(dtype=np.float64, na_value=np.nan)
    betas = df[BETA_COLS].to_numpy(dtype=np.float64, na_value=np.nan)
    ret = df[RET_COL].to_numpy(dtype=np.float64)

    predicted_ret = np.nansum(betas * factors, axis=1)
    resid = ret - predicted_ret

    df = df.copy()
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
    return df


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep stocks with sufficient feature coverage per year.

    Only checks CORE features (betas + option-implied) that have consistent
    data coverage across 2015-2024. OpenAssetPricing features (Mom12m, BM)
    and GLB have data gaps in 2023-2024 and are NOT used for filtering.
    """
    print("[data_loader] Filtering universe by coverage ...")
    core_features = BETA_COLS + ["SKEW", "AIV"]
    features_to_check = [f for f in core_features if f in df.columns]
    df = df.copy()
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
    print(f"  Universe: {n_stocks} stocks after coverage filter (threshold={UNIVERSE_COVERAGE_THRESHOLD})")
    return df


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily panel to monthly (last trading day per month).

    Returns monthly-frequency data with:
    - Monthly return: compounded from daily excess returns
    - Month-end mktcap, features, betas (snapshot)
    - Forward 1-month return for labels
    """
    print("[data_loader] Resampling to monthly frequency ...")
    dates = df.index.get_level_values("date")
    df = df.copy()
    df["_month_end"] = dates.to_period("M").to_timestamp("M")

    snapshot_cols = (
        [MKTCAP_COL] + BETA_COLS + ["SKEW", "AIV", "GLB", "Mom12m", "IdioVol3F", "BM", "resid_signal"]
    )

    monthly_ret = (
        df.groupby([df.index.get_level_values("permno"), "_month_end"])[RET_COL]
        .apply(lambda x: np.expm1(np.log1p(x.dropna()).sum()))
    )
    monthly_ret.name = "monthly_ret"

    last_day_idx = df.groupby(
        [df.index.get_level_values("permno"), "_month_end"]
    ).apply(lambda g: g.index.get_level_values("date").max())

    snapshots = []
    for (permno, month_end), last_date in last_day_idx.items():
        try:
            row = df.loc[(permno, last_date), snapshot_cols]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            snap = row.to_dict()
            snap["permno"] = permno
            snap["month_end"] = month_end
            snapshots.append(snap)
        except KeyError:
            continue

    snap_df = pd.DataFrame(snapshots).set_index(["permno", "month_end"])

    monthly = monthly_ret.reset_index()
    monthly.columns = ["permno", "month_end", "monthly_ret"]
    monthly = monthly.set_index(["permno", "month_end"])
    monthly = monthly.join(snap_df, how="inner")

    fwd = monthly["monthly_ret"].groupby(level="permno").shift(-1)
    monthly["fwd_ret"] = fwd

    print(f"  Monthly panel: {monthly.shape[0]:,} rows, "
          f"{monthly.index.get_level_values('permno').nunique()} stocks, "
          f"{monthly.index.get_level_values('month_end').nunique()} months")
    return monthly


def filter_top_n(monthly: pd.DataFrame) -> pd.DataFrame:
    """Keep top UNIVERSE_MAX_STOCKS stocks by market cap per month.

    Stocks with NaN mktcap are excluded from ranking.
    Asserts each month retains at least UNIVERSE_MIN_STOCKS stocks.
    """
    print(f"[data_loader] Filtering to top {UNIVERSE_MAX_STOCKS} by market cap per month ...")
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
    n_stocks = monthly.index.get_level_values("permno").nunique()
    print(f"  Before: {before:,} rows → After: {after:,} rows (dropped {before - after:,})")
    print(f"  Stocks per month: min={min_count}, max={max_count}, unique={n_stocks}")
    return monthly


def load_and_prepare() -> pd.DataFrame:
    """Full pipeline: load -> resid_signal -> filter -> monthly -> top_n."""
    df = load_panel()
    df = compute_resid_signal(df)
    df = filter_universe(df)
    monthly = resample_monthly(df)
    monthly = filter_top_n(monthly)
    del df
    gc.collect()

    out_path = OUTPUTS / "universe_monthly.parquet"
    monthly.to_parquet(out_path, engine="pyarrow")
    print(f"[data_loader] Saved {out_path}")
    return monthly


if __name__ == "__main__":
    load_and_prepare()
