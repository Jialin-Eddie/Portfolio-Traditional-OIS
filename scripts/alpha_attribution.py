"""Alpha Source Attribution Investigation.

Answers: where does the strategy's Sharpe come from — factors, model, or optimizer?

Experiments:
  1. Single-Factor IC Deep Dive — individual factor predictive power
  2. Model Comparison — XGBoost vs OLS vs naive vs random
  3. Portfolio Construction Comparison — optimizer vs simpler approaches
  4. Factor Tilt Attribution — Brinson-style decomposition via FF4 regression

Usage:
  python scripts/alpha_attribution.py
"""

import sys
import time
import warnings
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    ALL_FEATURES, BETA_COLS, SPLIT_DATE, ANNUALIZE_MONTHLY,
    XGB_PARAMS, XGB_EARLY_STOPPING_ROUNDS, XGB_VAL_FRACTION,
    PURGE_MONTHS, EMBARGO_MONTHS, EMA_ALPHA, OUTPUTS,
    MKTCAP_COL, FF4_FACTORS, NW_LAGS, RANDOM_SEED,
)
from src.data_loader import load_and_prepare
from src.preprocessing import fit_preprocess_pipeline, apply_preprocess_pipeline
from src.benchmark import build_benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
np.random.seed(RANDOM_SEED)

OUT_DIR = OUTPUTS / "alpha_attribution"
SPLIT_TS = pd.Timestamp(SPLIT_DATE)
FEATURES = ALL_FEATURES + ["resid_signal"]


def _ic(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return float("nan")
    return float(spearmanr(y_true[valid], y_pred[valid])[0])


def _ann_ret(returns):
    n = len(returns)
    if n == 0:
        return float("nan")
    return float((1 + returns).prod() ** (ANNUALIZE_MONTHLY / n) - 1)


def _ann_vol(returns):
    return float(returns.std() * np.sqrt(ANNUALIZE_MONTHLY))


def _sharpe(returns):
    vol = _ann_vol(returns)
    if vol == 0:
        return float("nan")
    return _ann_ret(returns) / vol


def _max_dd(returns):
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def _ir(active_returns):
    ar = active_returns.dropna()
    if len(ar) < 2:
        return float("nan")
    te = ar.std() * np.sqrt(ANNUALIZE_MONTHLY)
    if te == 0:
        return float("nan")
    excess = _ann_ret(ar)
    return excess / te


def load_data():
    print("=" * 60)
    print("  Loading & Preparing Data")
    print("=" * 60)

    universe_path = OUTPUTS / "universe_monthly.parquet"
    if universe_path.exists():
        print(f"[data] Loading cached panel from {universe_path}")
        monthly = pd.read_parquet(universe_path)
    else:
        monthly = load_and_prepare()

    bench_weights, bench_returns_series, bench_betas = build_benchmark(monthly)

    split_ts = pd.Timestamp(SPLIT_DATE)
    is_mask = monthly.index.get_level_values("month_end") <= split_ts
    train_df = monthly.loc[is_mask]

    features_present = [f for f in FEATURES if f in monthly.columns]
    params = fit_preprocess_pipeline(train_df, features_present)
    monthly_proc = apply_preprocess_pipeline(monthly, params)

    return monthly, monthly_proc, bench_weights, bench_returns_series, features_present


def experiment_1_single_factor_ic(monthly_proc, features):
    print("\n" + "=" * 60)
    print("  Experiment 1: Single-Factor IC Deep Dive")
    print("=" * 60)

    oos_mask = monthly_proc.index.get_level_values("month_end") > SPLIT_TS
    oos_df = monthly_proc.loc[oos_mask].copy()
    oos_months = oos_df.index.get_level_values("month_end").unique().sort_values()

    results = []
    for feat in features:
        monthly_ics = []
        for m in oos_months:
            m_df = oos_df.xs(m, level="month_end", drop_level=False)
            valid = m_df[[feat, "fwd_ret"]].dropna()
            if len(valid) < 10:
                continue
            ic = _ic(valid["fwd_ret"].values, valid[feat].values)
            monthly_ics.append(ic)

        if not monthly_ics:
            continue

        ics = np.array(monthly_ics)
        ics_clean = ics[np.isfinite(ics)]
        n = len(ics_clean)
        mean_ic = ics_clean.mean()
        std_ic = ics_clean.std(ddof=1) if n > 1 else float("nan")
        icir = mean_ic / std_ic if std_ic > 0 else float("nan")
        t_stat = mean_ic / (std_ic / sqrt(n)) if std_ic > 0 and n > 1 else float("nan")
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else float("nan")

        ic_series = pd.Series(ics_clean)
        ac1 = ic_series.autocorr(lag=1) if n > 2 else float("nan")

        results.append({
            "Factor": feat,
            "OOS Mean IC": mean_ic,
            "Std IC": std_ic,
            "ICIR": icir,
            "t-stat": t_stat,
            "p-value": p_val,
            "Significant?": "*" if (np.isfinite(p_val) and p_val < 0.05) else "",
            "N_months": n,
            "IC_AC1": ac1,
        })

    df = pd.DataFrame(results).sort_values("t-stat", ascending=False, key=abs)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return df


def _walk_forward_model(monthly_proc, features, model_type="xgb"):
    oos_months = monthly_proc.index.get_level_values("month_end").unique()
    oos_months = oos_months[oos_months > SPLIT_TS].sort_values()
    monthly_ics = []

    for predict_month in oos_months:
        train_mask = monthly_proc.index.get_level_values("month_end") < predict_month
        train = monthly_proc.loc[train_mask].dropna(subset=["fwd_ret"] + features)
        pred_mask = monthly_proc.index.get_level_values("month_end") == predict_month
        pred_df = monthly_proc.loc[pred_mask].dropna(subset=features)

        if len(train) < 100 or len(pred_df) < 10:
            continue

        X_tr = train[features].values
        y_tr = train["fwd_ret"].values
        X_pred = pred_df[features].values

        if model_type == "xgb":
            tr_months = train.index.get_level_values("month_end").unique().sort_values()
            n_val = max(1, int(len(tr_months) * XGB_VAL_FRACTION))
            val_mo = tr_months[-n_val:]
            n_drop = PURGE_MONTHS + EMBARGO_MONTHS
            if n_drop > 0 and len(val_mo) > n_drop:
                val_mo = val_mo[:-n_drop]
            val_mask_wf = train.index.get_level_values("month_end").isin(val_mo)
            X_tr_wf = train.loc[~val_mask_wf, features].values
            y_tr_wf = train.loc[~val_mask_wf, "fwd_ret"].values
            X_val_wf = train.loc[val_mask_wf, features].values
            y_val_wf = train.loc[val_mask_wf, "fwd_ret"].values

            model = XGBRegressor(**XGB_PARAMS, verbosity=0,
                                  early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
            if len(X_val_wf) > 0:
                model.fit(X_tr_wf, y_tr_wf, eval_set=[(X_val_wf, y_val_wf)], verbose=False)
            else:
                model = XGBRegressor(**XGB_PARAMS, verbosity=0)
                model.fit(X_tr_wf, y_tr_wf, verbose=False)
            preds = model.predict(X_pred)

        elif model_type == "ols":
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_pred)

        elif model_type == "equal_z":
            z_scores = pred_df[features].values
            preds = np.nanmean(z_scores, axis=1)

        elif model_type == "random":
            preds = np.random.randn(len(X_pred))

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        realized = pred_df["fwd_ret"].values
        ic = _ic(realized, preds)
        monthly_ics.append(ic)

    return monthly_ics


def experiment_2_model_comparison(monthly_proc, features):
    print("\n" + "=" * 60)
    print("  Experiment 2: Model Comparison (Signal Quality)")
    print("=" * 60)

    models = {
        "XGBoost": "xgb",
        "OLS": "ols",
        "Equal-Z": "equal_z",
        "Random": "random",
    }

    results = []
    all_ics = {}
    for name, mtype in models.items():
        print(f"  Running {name} walk-forward ...")
        t0 = time.time()
        ics = _walk_forward_model(monthly_proc, features, model_type=mtype)
        elapsed = time.time() - t0
        ics_arr = np.array([x for x in ics if np.isfinite(x)])
        n = len(ics_arr)
        mean_ic = ics_arr.mean() if n > 0 else float("nan")
        std_ic = ics_arr.std(ddof=1) if n > 1 else float("nan")
        icir = mean_ic / std_ic if std_ic > 0 else float("nan")
        hit_rate = (ics_arr > 0).mean() if n > 0 else float("nan")

        all_ics[name] = ics_arr
        results.append({
            "Model": name,
            "OOS Mean IC": mean_ic,
            "ICIR": icir,
            "Hit Rate": hit_rate,
            "N_months": n,
            "Time(s)": elapsed,
        })
        print(f"    IC={mean_ic:.4f}  ICIR={icir:.4f}  Hit={hit_rate:.1%}  ({elapsed:.0f}s)")

    random_ics = all_ics.get("Random", np.array([]))
    for r in results:
        name = r["Model"]
        if name == "Random" or len(random_ics) == 0:
            r["vs_Random_p"] = float("nan")
            continue
        model_ics = all_ics[name]
        min_n = min(len(model_ics), len(random_ics))
        diff = model_ics[:min_n] - random_ics[:min_n]
        if len(diff) > 1 and diff.std() > 0:
            t_val = diff.mean() / (diff.std(ddof=1) / sqrt(len(diff)))
            p_val = 1 - t_dist.cdf(t_val, df=len(diff) - 1)
            r["vs_Random_p"] = p_val
        else:
            r["vs_Random_p"] = float("nan")

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return df


def experiment_3_portfolio_construction(monthly, monthly_proc, bench_weights, features):
    print("\n" + "=" * 60)
    print("  Experiment 3: Portfolio Construction Comparison")
    print("=" * 60)

    oos_months_all = monthly.index.get_level_values("month_end").unique()
    oos_months = oos_months_all[oos_months_all > SPLIT_TS].sort_values()

    print("  Generating XGBoost OOS signals via walk-forward ...")
    from src.signals import walk_forward_oos, smooth_signals_ema
    oos_preds = walk_forward_oos(monthly_proc, features, SPLIT_DATE)
    oos_preds = smooth_signals_ema(oos_preds, alpha=EMA_ALPHA)

    pred_months = oos_preds.index.get_level_values("month_end").unique().sort_values()

    methods = {}

    signal_wt_returns = []
    topq_ew_returns = []
    optimizer_returns = []
    bench_returns_list = []

    for m in pred_months:
        m_preds = oos_preds.xs(m, level="month_end")["y_pred"]
        bw_m = bench_weights.xs(m, level="month_end")["bench_weight"] if m in bench_weights.index.get_level_values("month_end") else None
        if bw_m is None or len(bw_m) == 0:
            continue

        universe = m_preds.index.intersection(bw_m.index)
        if len(universe) < 10:
            continue

        sig = m_preds.loc[universe]
        bw = bw_m.loc[universe]

        fwd = monthly.xs(m, level="month_end")["fwd_ret"].reindex(universe).fillna(0)

        sig_pos = np.maximum(sig.values, 0) * bw.values
        sig_sum = sig_pos.sum()
        if sig_sum > 0:
            w_signal = sig_pos / sig_sum
        else:
            w_signal = bw.values / bw.values.sum()
        ret_signal = float(np.dot(w_signal, fwd.values))
        signal_wt_returns.append((m, ret_signal))

        n_top = max(1, len(universe) // 5)
        top_idx = sig.nlargest(n_top).index
        w_topq = np.zeros(len(universe))
        for i, p in enumerate(universe):
            if p in top_idx:
                w_topq[i] = 1.0 / n_top
        ret_topq = float(np.dot(w_topq, fwd.values))
        topq_ew_returns.append((m, ret_topq))

        w_bench_norm = bw.values / bw.values.sum()
        ret_bench = float(np.dot(w_bench_norm, fwd.values))
        bench_returns_list.append((m, ret_bench))

    signal_wt_ret = pd.Series(dict(signal_wt_returns), name="Signal-Wt")
    topq_ew_ret = pd.Series(dict(topq_ew_returns), name="Top-Q EW")
    bench_ret = pd.Series(dict(bench_returns_list), name="Benchmark")

    from src.optimizer import optimize_all_months, estimate_covariance
    from src.benchmark import compute_benchmark_factor_exposures

    fwd_ret_wide = monthly["fwd_ret"].unstack(level="permno")
    fwd_ret_wide.index.name = "month_end"

    beta_cols_in = [c for c in BETA_COLS if c in monthly.columns]

    bench_betas_raw = compute_benchmark_factor_exposures(monthly, bench_weights)
    bench_betas_for_opt = bench_betas_raw.rename(
        columns={f"bench_beta_{f.replace('beta_', '')}": f for f in beta_cols_in}
    ).reset_index()

    predictions_with_betas = oos_preds.join(monthly[beta_cols_in], how="left")

    opt_weights = optimize_all_months(
        monthly=fwd_ret_wide,
        predictions=predictions_with_betas.rename(columns={"y_pred": "pred"}),
        bench_weights=bench_weights.rename(columns={"bench_weight": "weight"}),
        bench_betas=bench_betas_for_opt,
    )

    from src.backtest import compute_portfolio_returns
    opt_ret = compute_portfolio_returns(opt_weights, monthly, weight_col="opt_weight")
    opt_ret = opt_ret[opt_ret.index > SPLIT_TS]

    common_months = signal_wt_ret.index.intersection(bench_ret.index).intersection(opt_ret.index)
    signal_wt_ret = signal_wt_ret.loc[common_months]
    topq_ew_ret = topq_ew_ret.loc[common_months]
    bench_ret = bench_ret.loc[common_months]
    opt_ret = opt_ret.loc[common_months]

    rows = []
    for name, rets in [("Optimizer", opt_ret), ("Signal-Wt", signal_wt_ret),
                        ("Top-Q EW", topq_ew_ret), ("Benchmark", bench_ret)]:
        active = rets - bench_ret if name != "Benchmark" else pd.Series(0, index=rets.index)
        rows.append({
            "Method": name,
            "OOS Return": _ann_ret(rets),
            "Sharpe": _sharpe(rets),
            "IR": _ir(active) if name != "Benchmark" else float("nan"),
            "MaxDD": _max_dd(rets),
            "Vol": _ann_vol(rets),
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return df


def experiment_4_factor_tilt_attribution(monthly, monthly_proc, bench_weights):
    print("\n" + "=" * 60)
    print("  Experiment 4: Factor Tilt Attribution (FF4 Regression)")
    print("=" * 60)

    br_path = OUTPUTS / "backtest_results.parquet"
    if br_path.exists():
        br = pd.read_parquet(br_path)
        active_ret = br["active_ret"].dropna()
    else:
        print("  [!] No backtest_results.parquet found, computing from scratch ...")
        from src.signals import walk_forward_oos, smooth_signals_ema
        from src.optimizer import optimize_all_months
        from src.benchmark import compute_benchmark_factor_exposures
        from src.backtest import run_backtest

        features_present = [f for f in FEATURES if f in monthly_proc.columns]
        oos_preds = walk_forward_oos(monthly_proc, features_present, SPLIT_DATE)
        oos_preds = smooth_signals_ema(oos_preds, alpha=EMA_ALPHA)

        fwd_ret_wide = monthly["fwd_ret"].unstack(level="permno")
        fwd_ret_wide.index.name = "month_end"
        beta_cols_in = [c for c in BETA_COLS if c in monthly.columns]
        predictions_with_betas = oos_preds.join(monthly[beta_cols_in], how="left")
        bench_betas_raw = compute_benchmark_factor_exposures(monthly, bench_weights)
        bench_betas_for_opt = bench_betas_raw.rename(
            columns={f"bench_beta_{f.replace('beta_', '')}": f for f in beta_cols_in}
        ).reset_index()

        opt_weights = optimize_all_months(
            monthly=fwd_ret_wide,
            predictions=predictions_with_betas.rename(columns={"y_pred": "pred"}),
            bench_weights=bench_weights.rename(columns={"bench_weight": "weight"}),
            bench_betas=bench_betas_for_opt,
        )

        bt = run_backtest(opt_weights, bench_weights, monthly)
        active_ret = bt["active_returns"]

    oos_active = active_ret[active_ret.index > SPLIT_TS]

    ff4_returns = monthly[FF4_FACTORS].groupby(level="month_end").first()
    ff4_oos = ff4_returns.loc[ff4_returns.index > SPLIT_TS]

    common = oos_active.index.intersection(ff4_oos.index)
    y = oos_active.loc[common].values.astype(np.float64)
    X = ff4_oos.loc[common].values.astype(np.float64)
    X_c = sm.add_constant(X)

    model = sm.OLS(y, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    print(f"\n  FF4 Regression on OOS Active Returns")
    print(f"  {'='*50}")
    print(f"  R-squared:  {model.rsquared:.4f}")
    print(f"  Adj R-sq:   {model.rsquared_adj:.4f}")
    print(f"  N months:   {len(common)}")
    print()

    names = ["Alpha (intercept)"] + list(FF4_FACTORS)
    print(f"  {'Variable':25s} {'Coeff':>10s} {'t-stat':>10s} {'p-value':>10s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        coef = model.params[i]
        t = model.tvalues[i]
        p = model.pvalues[i]
        sig = " *" if p < 0.05 else "  " if p < 0.10 else ""
        print(f"  {name:25s} {coef:10.4f} {t:10.2f} {p:10.4f}{sig}")

    alpha_monthly = model.params[0]
    alpha_ann = alpha_monthly * 12
    print(f"\n  Annualized alpha = {alpha_ann:.2%}")
    print(f"  Interpretation: {model.rsquared:.0%} of active return variance explained by FF4 factor tilts")
    if model.rsquared > 0.5:
        print("  -> Most alpha comes from FACTOR TILTS, not stock selection")
    elif model.rsquared > 0.2:
        print("  -> Mixed: both factor tilts and stock selection contribute")
    else:
        print("  -> Low R-sq: alpha is mostly from STOCK SELECTION (or noise)")

    result = {
        "R_squared": model.rsquared,
        "Alpha_monthly": alpha_monthly,
        "Alpha_annual": alpha_ann,
        "Alpha_t": model.tvalues[0],
        "Alpha_p": model.pvalues[0],
    }
    for i, f in enumerate(FF4_FACTORS):
        result[f"beta_{f}"] = model.params[i + 1]
        result[f"t_{f}"] = model.tvalues[i + 1]

    return pd.DataFrame([result])


def main():
    t_total = time.time()

    monthly, monthly_proc, bench_weights, bench_returns_series, features = load_data()
    features_present = [f for f in features if f in monthly_proc.columns]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df1 = experiment_1_single_factor_ic(monthly_proc, features_present)
    df1.to_csv(OUT_DIR / "exp1_single_factor_ic.csv", index=False)

    df2 = experiment_2_model_comparison(monthly_proc, features_present)
    df2.to_csv(OUT_DIR / "exp2_model_comparison.csv", index=False)

    df3 = experiment_3_portfolio_construction(monthly, monthly_proc, bench_weights, features_present)
    df3.to_csv(OUT_DIR / "exp3_portfolio_construction.csv", index=False)

    df4 = experiment_4_factor_tilt_attribution(monthly, monthly_proc, bench_weights)
    df4.to_csv(OUT_DIR / "exp4_factor_tilt_attribution.csv", index=False)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {time.time() - t_total:.0f}s")
    print(f"  Results saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
