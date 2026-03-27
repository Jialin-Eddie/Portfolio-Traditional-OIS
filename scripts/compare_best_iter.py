"""Compare best_iteration (number of trees) with and without purge."""
import sys; sys.path.insert(0, "C:/Users/zhaoj/Desktop_backup/InternProject/QT/Portfolio-Tradtional-Ois")
import pandas as pd; import numpy as np
from xgboost import XGBRegressor
from src.config import XGB_PARAMS, XGB_EARLY_STOPPING_ROUNDS, XGB_VAL_FRACTION, SPLIT_DATE, ALL_FEATURES, OUTPUTS

monthly = pd.read_parquet(OUTPUTS / "monthly_processed.parquet")
features = [f for f in ALL_FEATURES + ["resid_signal"] if f in monthly.columns]
split_ts = pd.Timestamp(SPLIT_DATE)
month_ends = monthly.index.get_level_values("month_end").unique().sort_values()
oos_months = month_ends[month_ends > split_ts]

results = []
for predict_month in oos_months:
    train_mask = monthly.index.get_level_values("month_end") < predict_month
    train_df = monthly.loc[train_mask].dropna(subset=["fwd_ret"] + features)
    if len(train_df) == 0:
        continue
    tr_months = train_df.index.get_level_values("month_end").unique().sort_values()
    n_val = max(1, int(len(tr_months) * XGB_VAL_FRACTION))
    val_mo_raw = tr_months[-n_val:]
    val_mo_purged = val_mo_raw[:-1] if len(val_mo_raw) > 1 else val_mo_raw
    for label, val_mo in [("no_purge", val_mo_raw), ("purge_1", val_mo_purged)]:
        val_mask = train_df.index.get_level_values("month_end").isin(val_mo)
        X_tr = train_df.loc[~val_mask, features].values
        y_tr = train_df.loc[~val_mask, "fwd_ret"].values
        X_val = train_df.loc[val_mask, features].values
        y_val = train_df.loc[val_mask, "fwd_ret"].values
        if len(X_val) == 0:
            best_iter = -1
        else:
            m = XGBRegressor(**XGB_PARAMS, verbosity=0, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            best_iter = m.best_iteration
        results.append({"predict_month": predict_month.date(), "mode": label, "best_iteration": best_iter, "n_trees": best_iter + 1})

df = pd.DataFrame(results)
pivot = df.pivot(index="predict_month", columns="mode", values="n_trees")
pivot["diff"] = pivot["purge_1"] - pivot["no_purge"]
print("=" * 70)
print("  best_iteration comparison: no_purge vs purge_1")
print("=" * 70)
print(pivot.to_string())
print()
print("--- Summary ---")
print(f"no_purge  mean trees: {pivot['no_purge'].mean():.2f}, median: {pivot['no_purge'].median():.0f}")
print(f"purge_1   mean trees: {pivot['purge_1'].mean():.2f}, median: {pivot['purge_1'].median():.0f}")
print(f"months where purge changed n_trees: {(pivot['diff'] != 0).sum()} / {len(pivot)}")
print(f"months where purge used MORE trees: {(pivot['diff'] > 0).sum()}")
print(f"months where purge used FEWER trees: {(pivot['diff'] < 0).sum()}")
print(f"months where same: {(pivot['diff'] == 0).sum()}")
