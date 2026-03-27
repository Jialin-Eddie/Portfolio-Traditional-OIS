"""Feature ablation analysis: compare signal quality across feature subsets.

Three variants:
  1. Traditional only: FF4 betas + Mom12m + IdioVol3F + BM + resid_signal
  2. Option-implied only: SKEW + AIV + GLB
  3. All features (baseline)

For each, trains XGBoost on IS, evaluates IS IC/ICIR, and runs walk-forward OOS.
Does NOT re-run the optimizer — only compares signal quality (IC/ICIR).
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from src.config import (
    XGB_PARAMS, SPLIT_DATE, BETA_COLS, OUTPUTS,
)


def _ic(y_true, y_pred):
    valid = y_true.notna() & y_pred.notna()
    if valid.sum() < 5:
        return float("nan")
    return float(spearmanr(y_true[valid], y_pred[valid])[0])


FEATURE_SETS = {
    "Traditional": BETA_COLS + ["Mom12m", "IdioVol3F", "BM", "resid_signal"],
    "Option-Implied": ["SKEW", "AIV", "GLB"],
    "Betas-Only": BETA_COLS,
    "All": BETA_COLS + ["SKEW", "AIV", "GLB", "Mom12m", "IdioVol3F", "BM", "resid_signal"],
}


def run_ablation(monthly_proc: pd.DataFrame) -> pd.DataFrame:
    """Run feature ablation across all feature sets.

    Returns DataFrame with rows = feature sets, columns = metrics.
    """
    split_ts = pd.Timestamp(SPLIT_DATE)
    is_mask = monthly_proc.index.get_level_values("month_end") <= split_ts

    results = []

    for name, features in FEATURE_SETS.items():
        available = [f for f in features if f in monthly_proc.columns]
        if not available:
            print(f"[{name}] No features available — skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Feature Set: {name} ({len(available)} features)")
        print(f"  Features: {available}")
        print(f"{'='*50}")

        is_df = monthly_proc.loc[is_mask].dropna(subset=["fwd_ret"] + available)
        X_is = is_df[available].values
        y_is = is_df["fwd_ret"].values

        model = XGBRegressor(**XGB_PARAMS, verbosity=0)
        model.fit(X_is, y_is)

        is_preds = model.predict(X_is)
        is_ic = _ic(pd.Series(y_is), pd.Series(is_preds))
        print(f"  IS IC: {is_ic:.4f}")

        oos_months = monthly_proc.index.get_level_values("month_end").unique()
        oos_months = oos_months[oos_months > split_ts].sort_values()

        monthly_ics = []
        for predict_month in oos_months:
            train_mask = monthly_proc.index.get_level_values("month_end") < predict_month
            train = monthly_proc.loc[train_mask].dropna(subset=["fwd_ret"] + available)
            pred_mask = monthly_proc.index.get_level_values("month_end") == predict_month
            pred_df = monthly_proc.loc[pred_mask].dropna(subset=available)

            if len(train) == 0 or len(pred_df) == 0:
                continue

            m = XGBRegressor(**XGB_PARAMS, verbosity=0)
            m.fit(train[available].values, train["fwd_ret"].values)
            preds = m.predict(pred_df[available].values)

            month_ic = _ic(pred_df["fwd_ret"], pd.Series(preds, index=pred_df.index))
            if not np.isnan(month_ic):
                monthly_ics.append(month_ic)

        oos_ic = np.mean(monthly_ics) if monthly_ics else float("nan")
        oos_icir = oos_ic / np.std(monthly_ics, ddof=1) if len(monthly_ics) > 1 and np.std(monthly_ics) > 0 else float("nan")

        print(f"  OOS IC:   {oos_ic:.4f}")
        print(f"  OOS ICIR: {oos_icir:.4f}")
        print(f"  OOS months: {len(monthly_ics)}")

        fi = dict(zip(available, model.feature_importances_))
        top3 = sorted(fi.items(), key=lambda x: -x[1])[:3]
        fi_str = ", ".join(f"{k}={v:.3f}" for k, v in top3)
        print(f"  Top-3 IS importance: {fi_str}")

        results.append({
            "Feature Set": name,
            "N Features": len(available),
            "IS IC": is_ic,
            "OOS IC": oos_ic,
            "OOS ICIR": oos_icir,
            "OOS Months": len(monthly_ics),
            "Top Feature": top3[0][0] if top3 else "",
        })

    df = pd.DataFrame(results).set_index("Feature Set")
    print(f"\n{'='*60}")
    print("  FEATURE ABLATION SUMMARY")
    print(f"{'='*60}")
    print(df.to_string())

    out_path = OUTPUTS / "feature_ablation.csv"
    df.to_csv(out_path)
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    proc_path = OUTPUTS / "monthly_processed.parquet"
    monthly_proc = pd.read_parquet(proc_path)
    run_ablation(monthly_proc)
