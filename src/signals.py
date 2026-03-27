"""XGBoost signal generation with walk-forward monthly re-estimation.

IS period:  single model trained on 2015-2020
OOS period: expanding-window walk-forward, one model per month
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from src.config import (
    XGB_PARAMS,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_VAL_FRACTION,
    PURGE_MONTHS,
    EMBARGO_MONTHS,
    SPLIT_DATE,
    ALL_FEATURES,
    OUTPUTS,
    RANDOM_SEED,
)


def _ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman IC between two series aligned on index."""
    valid = y_true.notna() & y_pred.notna()
    if valid.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(y_true[valid], y_pred[valid])
    return float(corr)


def train_is_model(
    train_df: pd.DataFrame,
    features: list[str],
) -> tuple[XGBRegressor, pd.DataFrame]:
    """Train XGBoost on IS data and return (model, is_predictions).

    Parameters
    ----------
    train_df:
        Monthly panel with MultiIndex (permno, month_end).
        Must contain `features` columns and `fwd_ret` target.
    features:
        List of feature column names.

    Returns
    -------
    model:
        Fitted XGBRegressor.
    is_preds:
        DataFrame with MultiIndex (permno, month_end) and column `y_pred`.
    """
    df = train_df.copy()

    df = df.dropna(subset=["fwd_ret"] + features)

    months = df.index.get_level_values("month_end").unique().sort_values()
    n_val = max(1, int(len(months) * XGB_VAL_FRACTION))
    val_months = months[-n_val:]
    train_months = months[:-n_val]

    # PURGE: remove last PURGE_MONTHS from val (their fwd_ret leaks into first OOS month)
    n_drop = PURGE_MONTHS + EMBARGO_MONTHS
    if n_drop > 0 and len(val_months) > n_drop:
        val_months = val_months[:-n_drop]
    purged = n_drop

    val_mask = df.index.get_level_values("month_end").isin(val_months)
    X_tr, y_tr = df.loc[~val_mask, features].values, df.loc[~val_mask, "fwd_ret"].values
    X_val, y_val = df.loc[val_mask, features].values, df.loc[val_mask, "fwd_ret"].values

    print(f"[IS] Training on {len(X_tr):,} rows, val on {len(X_val):,} rows "
          f"({len(train_months)} train months, {len(val_months)} val months, "
          f"purge={PURGE_MONTHS}, embargo={EMBARGO_MONTHS}), "
          f"{len(features)} features, early_stopping={XGB_EARLY_STOPPING_ROUNDS}")

    model = XGBRegressor(**XGB_PARAMS, verbosity=0,
                          early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    best_iter = model.best_iteration
    print(f"[IS] Best iteration: {best_iter} / {XGB_PARAMS['n_estimators']}")

    preds = model.predict(df[features].values)
    is_preds = pd.DataFrame({"y_pred": preds}, index=df.index)

    ic_val = _ic(df["fwd_ret"], is_preds["y_pred"])
    print(f"[IS] In-sample IC (Spearman): {ic_val:.4f}")

    return model, is_preds


def walk_forward_oos(
    monthly: pd.DataFrame,
    features: list[str],
    split_date: str,
) -> pd.DataFrame:
    """Walk-forward OOS signal generation with expanding training window.

    For each OOS month t:
      - Train on all months with month_end < t (features + realized fwd_ret known)
      - Predict fwd_ret for all stocks at month t (using features at t)
      - LOOK-AHEAD ASSERTION: max(train month_end) < predict month_end

    Parameters
    ----------
    monthly:
        Full panel with MultiIndex (permno, month_end).
    features:
        List of feature column names.
    split_date:
        Last IS date string (e.g. '2020-12-31'). OOS starts after this.

    Returns
    -------
    oos_preds:
        DataFrame with MultiIndex (permno, month_end) and column `y_pred`.
    """
    split_ts = pd.Timestamp(split_date)

    month_ends = monthly.index.get_level_values("month_end").unique().sort_values()
    oos_months = month_ends[month_ends > split_ts]

    print(f"\n[OOS] Walk-forward over {len(oos_months)} OOS months "
          f"({oos_months[0].date()} → {oos_months[-1].date()})")

    all_preds: list[pd.DataFrame] = []
    ic_series: list[float] = []

    for predict_month in oos_months:
        # Training data: all months strictly before predict_month
        train_mask = monthly.index.get_level_values("month_end") < predict_month
        train_df = monthly.loc[train_mask].dropna(subset=["fwd_ret"] + features)

        if len(train_df) == 0:
            print(f"  {predict_month.date()} — skip (no training data)")
            continue

        train_months = train_df.index.get_level_values("month_end")

        # CRITICAL look-ahead guard
        assert train_months.max() < predict_month, (
            f"LOOK-AHEAD VIOLATION: training data includes month "
            f"{train_months.max()} >= prediction month {predict_month}"
        )

        tr_months = train_df.index.get_level_values("month_end").unique().sort_values()
        n_val = max(1, int(len(tr_months) * XGB_VAL_FRACTION))
        val_mo = tr_months[-n_val:]

        # PURGE: remove last PURGE_MONTHS from val
        # (their fwd_ret uses price data from predict_month)
        # EMBARGO: additional gap for serial correlation
        n_drop = PURGE_MONTHS + EMBARGO_MONTHS
        if n_drop > 0 and len(val_mo) > n_drop:
            val_mo = val_mo[:-n_drop]

        val_mask_wf = train_df.index.get_level_values("month_end").isin(val_mo)

        X_tr_wf = train_df.loc[~val_mask_wf, features].values
        y_tr_wf = train_df.loc[~val_mask_wf, "fwd_ret"].values
        X_val_wf = train_df.loc[val_mask_wf, features].values
        y_val_wf = train_df.loc[val_mask_wf, "fwd_ret"].values

        if len(X_val_wf) == 0:
            model = XGBRegressor(**XGB_PARAMS, verbosity=0)
            model.fit(X_tr_wf, y_tr_wf, verbose=False)
        else:
            model = XGBRegressor(**XGB_PARAMS, verbosity=0,
                                  early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
            model.fit(X_tr_wf, y_tr_wf, eval_set=[(X_val_wf, y_val_wf)], verbose=False)

        # Prediction targets: all stocks at predict_month with complete features
        pred_mask = monthly.index.get_level_values("month_end") == predict_month
        pred_df = monthly.loc[pred_mask].dropna(subset=features)

        if len(pred_df) == 0:
            print(f"  {predict_month.date()} — skip (no prediction rows)")
            continue

        X_pred = pred_df[features].values
        preds = model.predict(X_pred)

        month_preds = pd.DataFrame(
            {"y_pred": preds},
            index=pred_df.index,
        )
        all_preds.append(month_preds)

        # Monthly IC (against realized fwd_ret if available)
        realized = pred_df["fwd_ret"]
        month_ic = _ic(realized, month_preds["y_pred"])
        ic_series.append(month_ic)

        ic_str = f"{month_ic:.4f}" if not np.isnan(month_ic) else "n/a"
        print(f"  {predict_month.date()} | n_train={len(train_df):,} "
              f"n_pred={len(pred_df):,} | IC={ic_str}")

    if not all_preds:
        raise RuntimeError("No OOS predictions generated — check split_date and data range.")

    oos_preds = pd.concat(all_preds)

    valid_ics = [v for v in ic_series if not np.isnan(v)]
    if valid_ics:
        mean_ic = np.mean(valid_ics)
        std_ic = np.std(valid_ics, ddof=1)
        icir = mean_ic / std_ic if std_ic > 0 else float("nan")
        print(f"\n[OOS] Mean IC={mean_ic:.4f}  ICIR={icir:.4f}  "
              f"(over {len(valid_ics)} months with realized returns)")

    return oos_preds


def generate_signals(monthly: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate IS training + OOS walk-forward and save predictions.

    Parameters
    ----------
    monthly:
        Full panel with MultiIndex (permno, month_end), all features
        rank-normalized to [-1, 1], `fwd_ret` column present.

    Returns
    -------
    all_preds:
        Combined IS + OOS predictions with MultiIndex (permno, month_end)
        and column `y_pred`.
    """
    features = ALL_FEATURES  # may be extended by caller if resid_signal is present
    # Include resid_signal if present in the data
    if "resid_signal" in monthly.columns and "resid_signal" not in features:
        features = features + ["resid_signal"]
        print(f"[signals] resid_signal detected — using {len(features)} features total")

    split_ts = pd.Timestamp(SPLIT_DATE)

    # IS split
    is_mask = monthly.index.get_level_values("month_end") <= split_ts
    is_df = monthly.loc[is_mask]

    print(f"[signals] IS rows: {is_mask.sum():,}  |  "
          f"OOS rows: {(~is_mask).sum():,}")

    # IS model
    model, is_preds = train_is_model(is_df, features)

    # OOS walk-forward
    oos_preds = walk_forward_oos(monthly, features, SPLIT_DATE)

    # Merge IS + OOS
    all_preds = pd.concat([is_preds, oos_preds]).sort_index()

    # Persist
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "signal_predictions.parquet"
    all_preds.to_parquet(out_path)
    print(f"\n[signals] Saved {len(all_preds):,} predictions → {out_path}")

    return all_preds
