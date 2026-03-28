"""6-way signal transform experiment: 3 transforms x 2 optimizer configs.

Transforms: EMA (alpha=0.3), CS Z-score, CS Rank
Optimizers: B (CVaR + TE + drawdown guard), C (TE only)

Usage: python scripts/experiment_signal_transforms.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

from src.config import (
    BETA_COLS, COV_WINDOW_MONTHS, MAX_FACTOR_DEV_OPTIM,
    MAX_WEIGHT_MULT, TE_MAX_MONTHLY, SPLIT_DATE, OUTPUTS,
)

SPLIT_TS = pd.Timestamp(SPLIT_DATE)
CVAR_ALPHA = 0.95
CVAR_LIMIT = 0.02
DD_THRESHOLD = -0.015
DD_SHRINK = 0.5


def apply_ema(preds, alpha=0.3):
    out = preds.copy()
    out["y_pred"] = out.groupby(level="permno")["y_pred"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )
    return out


def apply_cs_zscore(preds):
    out = preds.copy()
    out["y_pred"] = out.groupby(level="month_end")["y_pred"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    return out


def apply_cs_rank(preds):
    out = preds.copy()
    out["y_pred"] = out.groupby(level="month_end")["y_pred"].transform(
        lambda x: 2 * (x.rank() - 1) / max(len(x) - 1, 1) - 1
    )
    return out


def estimate_cov(monthly_ret, month_end, window=COV_WINDOW_MONTHS):
    mask = monthly_ret.index <= month_end
    hist = monthly_ret.loc[mask].tail(window).fillna(0.0).values
    if len(hist) == 0:
        return np.eye(monthly_ret.shape[1]) * 1e-4
    lw = LedoitWolf()
    lw.fit(hist)
    cov = lw.covariance_ + 1e-6 * np.eye(hist.shape[1])
    return cov


def solve_te_only(mu, w_bench, betas, beta_bench, cov, target_sum=1.0):
    N = len(mu)
    for te in [TE_MAX_MONTHLY, 0.02, 0.025, 0.03]:
        w = cp.Variable(N)
        constraints = [cp.sum(w) == target_sum, w >= 0, w <= MAX_WEIGHT_MULT * w_bench]
        for k in range(betas.shape[1]):
            constraints.append(betas[:, k] @ w - beta_bench[k] <= MAX_FACTOR_DEV_OPTIM)
            constraints.append(beta_bench[k] - betas[:, k] @ w <= MAX_FACTOR_DEV_OPTIM)
        active = w - w_bench
        constraints.append(cp.quad_form(active, cp.psd_wrap(cov)) <= te**2)
        prob = cp.Problem(cp.Maximize(mu @ w), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return w.value
    return w_bench.copy()


def solve_cvar(mu, w_bench, betas, beta_bench, cov, hist_returns, target_sum=1.0):
    N = len(mu)
    ladder = [(CVAR_LIMIT, TE_MAX_MONTHLY), (0.025, 0.02), (0.03, 0.025), (0.04, 0.03)]
    for cv_lim, te in ladder:
        w = cp.Variable(N)
        constraints = [cp.sum(w) == target_sum, w >= 0, w <= MAX_WEIGHT_MULT * w_bench]
        for k in range(betas.shape[1]):
            constraints.append(betas[:, k] @ w - beta_bench[k] <= MAX_FACTOR_DEV_OPTIM)
            constraints.append(beta_bench[k] - betas[:, k] @ w <= MAX_FACTOR_DEV_OPTIM)
        active_w = w - w_bench
        constraints.append(cp.quad_form(active_w, cp.psd_wrap(cov)) <= te**2)
        if hist_returns is not None and len(hist_returns) >= 12:
            S = hist_returns.shape[0]
            zeta = cp.Variable()
            u = cp.Variable(S, nonneg=True)
            for s in range(S):
                constraints.append(u[s] >= -(hist_returns[s] @ active_w) - zeta)
            constraints.append(zeta + cp.sum(u) / ((1 - CVAR_ALPHA) * S) <= cv_lim)
        prob = cp.Problem(cp.Maximize(mu @ w), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return w.value
    return w_bench.copy()


def run_experiment(predictions, monthly_panel, bench_weights, bench_betas,
                   monthly_ret_wide, solver_fn, label):
    print(f"\n  Running {label} ...")
    preds = predictions.reset_index()
    if "level_0" in preds.columns:
        preds = preds.rename(columns={"level_0": "permno", "level_1": "month_end"})
    bweights = bench_weights.reset_index()
    if "level_0" in bweights.columns:
        bweights = bweights.rename(columns={"level_0": "permno", "level_1": "month_end"})

    rebalance_dates = sorted(preds["month_end"].unique())
    records = []
    realized_active = []
    prev_qs_w = None
    prev_bench_w = None
    prev_month = None

    for month_end in rebalance_dates:
        if prev_qs_w is not None and prev_month is not None and prev_month in monthly_ret_wide.index:
            fwd = monthly_ret_wide.loc[prev_month].fillna(0.0)
            qr = sum(prev_qs_w.get(p, 0) * fwd.get(p, 0) for p in fwd.index)
            br = sum(prev_bench_w.get(p, 0) * fwd.get(p, 0) for p in fwd.index)
            realized_active.append(qr - br)

        pred_m = preds[preds["month_end"] == month_end].set_index("permno")["y_pred" if "y_pred" in preds.columns else "pred"]
        bw_m = bweights[bweights["month_end"] == month_end].set_index("permno")["bench_weight" if "bench_weight" in bweights.columns else "weight"]
        bb_row = bench_betas[bench_betas["month_end"] == month_end]
        if bb_row.empty:
            continue
        beta_bench_vec = bb_row[BETA_COLS].values.ravel().astype(float)

        hist_mask = monthly_ret_wide.index <= month_end
        if hist_mask.sum() < 12:
            for p in bw_m.index:
                records.append({"permno": p, "month_end": month_end, "opt_weight": bw_m.get(p, 0.0)})
            continue

        universe = pred_m.index.intersection(bw_m.index)
        if len(universe) < 10:
            continue

        stock_betas_src = preds[preds["month_end"] == month_end].set_index("permno")
        beta_cols_present = [c for c in BETA_COLS if c in stock_betas_src.columns]
        if len(beta_cols_present) == len(BETA_COLS):
            stock_betas = stock_betas_src.loc[stock_betas_src.index.intersection(universe), BETA_COLS].reindex(universe, fill_value=0.0)
        else:
            stock_betas = pd.DataFrame(0.0, index=universe, columns=BETA_COLS)

        pred_m = pred_m.reindex(universe, fill_value=0.0)
        bw_m = bw_m.reindex(universe, fill_value=0.0)
        valid_mask = pred_m.notna() & bw_m.notna() & stock_betas.notna().all(axis=1)
        valid = valid_mask[valid_mask].index
        if len(valid) < 10:
            continue

        pred_m = pred_m.loc[valid]
        bw_m = bw_m.loc[valid]
        stock_betas = stock_betas.loc[valid]

        all_bw = bweights[bweights["month_end"] == month_end].set_index("permno")
        bw_col = "bench_weight" if "bench_weight" in all_bw.columns else "weight"
        all_bw_s = all_bw[bw_col]
        excluded = all_bw_s.index.difference(valid)
        excluded_wt = all_bw_s.loc[excluded].sum() if len(excluded) > 0 else 0.0
        target_sum = 1.0 - excluded_wt

        if bw_m.sum() <= 0 or target_sum <= 0:
            continue

        ret_cols = monthly_ret_wide.columns.intersection(valid)
        ret_sub = monthly_ret_wide[ret_cols].reindex(columns=valid, fill_value=np.nan)
        cov = estimate_cov(ret_sub, month_end)
        hist_ret = ret_sub.loc[hist_mask].tail(COV_WINDOW_MONTHS).fillna(0.0).values

        mu = np.nan_to_num(pred_m.values.astype(float), nan=0.0)
        w_b = np.nan_to_num(bw_m.values.astype(float), nan=0.0)
        betas_arr = np.nan_to_num(stock_betas.values.astype(float), nan=0.0)
        cov = np.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
        beta_bench_vec = np.nan_to_num(beta_bench_vec, nan=0.0)

        if solver_fn == "cvar":
            w_opt = solve_cvar(mu, w_b, betas_arr, beta_bench_vec, cov, hist_ret, target_sum)
        else:
            w_opt = solve_te_only(mu, w_b, betas_arr, beta_bench_vec, cov, target_sum)

        if solver_fn == "cvar" and len(realized_active) > 0 and realized_active[-1] < DD_THRESHOLD:
            w_opt = w_b + DD_SHRINK * (w_opt - w_b)

        for p, wt in zip(valid, w_opt):
            records.append({"permno": p, "month_end": month_end, "opt_weight": wt})
        for p in excluded:
            records.append({"permno": p, "month_end": month_end, "opt_weight": all_bw_s.get(p, 0.0)})

        prev_qs_w = dict(zip(valid, w_opt))
        for p in excluded:
            prev_qs_w[p] = all_bw_s.get(p, 0.0)
        prev_bench_w = dict(zip(bw_m.index, bw_m.values))
        for p in excluded:
            prev_bench_w[p] = all_bw_s.get(p, 0.0)
        prev_month = month_end

    if not records:
        return None
    result = pd.DataFrame(records).set_index(["permno", "month_end"])
    return result


def backtest_weights(opt_weights, bench_weights, monthly_panel):
    from src.backtest import compute_portfolio_returns, compute_active_returns

    qs_ret = compute_portfolio_returns(opt_weights, monthly_panel, "opt_weight")
    bench_ret = compute_portfolio_returns(bench_weights, monthly_panel, "bench_weight")
    common = qs_ret.index.intersection(bench_ret.index)
    qs_ret = qs_ret.loc[common]
    bench_ret = bench_ret.loc[common]
    active = qs_ret - bench_ret

    oos_mask = qs_ret.index > SPLIT_TS
    qs_oos = qs_ret[oos_mask]
    bench_oos = bench_ret[oos_mask]
    active_oos = active[oos_mask]

    if len(qs_oos) == 0:
        return None

    ann_ret_qs = (1 + qs_oos.mean()) ** 12 - 1
    ann_ret_bn = (1 + bench_oos.mean()) ** 12 - 1
    ann_vol_qs = qs_oos.std() * np.sqrt(12)
    ann_vol_bn = bench_oos.std() * np.sqrt(12)
    sharpe_qs = ann_ret_qs / ann_vol_qs if ann_vol_qs > 0 else 0
    sharpe_bn = ann_ret_bn / ann_vol_bn if ann_vol_bn > 0 else 0
    maxdd_qs = ((1 + qs_oos).cumprod() / (1 + qs_oos).cumprod().cummax() - 1).min()
    maxdd_bn = ((1 + bench_oos).cumprod() / (1 + bench_oos).cumprod().cummax() - 1).min()
    excess = (1 + active_oos.mean()) ** 12 - 1
    te = active_oos.std() * np.sqrt(12)
    ir = excess / te if te > 0 else 0
    worst_active = active_oos.min()
    violations = (active_oos < -0.02).sum()

    return {
        "ann_ret_qs": ann_ret_qs, "ann_ret_bn": ann_ret_bn,
        "ann_vol_qs": ann_vol_qs, "ann_vol_bn": ann_vol_bn,
        "sharpe_qs": sharpe_qs, "sharpe_bn": sharpe_bn,
        "maxdd_qs": maxdd_qs, "maxdd_bn": maxdd_bn,
        "ir": ir, "worst_active": worst_active, "violations": violations,
    }


def main():
    print("=" * 70)
    print("  6-Way Signal Transform Experiment")
    print("=" * 70)

    monthly_panel = pd.read_parquet(OUTPUTS / "universe_monthly.parquet")
    predictions = pd.read_parquet(OUTPUTS / "signal_predictions.parquet")
    bench_weights = pd.read_parquet(OUTPUTS / "benchmark_weights.parquet")
    bench_betas_raw = pd.read_parquet(OUTPUTS / "benchmark_factor_exposures.parquet").reset_index()
    bench_betas = bench_betas_raw.rename(columns={f"bench_beta_{f.replace('beta_', '')}": f for f in BETA_COLS})

    fwd_ret_wide = monthly_panel["fwd_ret"].unstack(level="permno")
    fwd_ret_wide.index.name = "month_end"

    beta_cols_in = [c for c in BETA_COLS if c in monthly_panel.columns]
    preds_with_betas = predictions.join(monthly_panel[beta_cols_in], how="left")
    preds_with_betas = preds_with_betas.rename(columns={"y_pred": "y_pred"})

    transforms = {
        "Raw": lambda p: p,
        "EMA": apply_ema,
        "ZScore": apply_cs_zscore,
        "Rank": apply_cs_rank,
    }
    solvers = {"C_TE": "te", "B_CVaR": "cvar"}

    results = {}

    for t_name, t_fn in transforms.items():
        if t_name == "Raw":
            continue
        transformed = t_fn(preds_with_betas.copy())
        for s_name, s_type in solvers.items():
            label = f"{s_name}_{t_name}"
            opt_w = run_experiment(
                transformed.rename(columns={"y_pred": "pred"}) if "y_pred" in transformed.columns else transformed,
                monthly_panel, bench_weights, bench_betas, fwd_ret_wide, s_type, label
            )
            if opt_w is not None:
                metrics = backtest_weights(opt_w, bench_weights, monthly_panel)
                if metrics:
                    results[label] = metrics
                    print(f"    IR={metrics['ir']:.4f}, Sharpe={metrics['sharpe_qs']:.4f}, Violations={metrics['violations']}")

    print("\n" + "=" * 100)
    print("  OOS RESULTS (2021-2024)")
    print("=" * 100)
    header = f"{'Variant':<18} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'IR':>8} {'WorstAct':>10} {'Viol':>5}"
    print(header)
    print("-" * 100)

    if results:
        bn = list(results.values())[0]
        print(f"{'BNCH':<18} {bn['ann_ret_bn']*100:>7.2f}% {bn['ann_vol_bn']*100:>7.2f}% {bn['sharpe_bn']:>8.4f} {bn['maxdd_bn']*100:>7.2f}% {'—':>8} {'—':>10} {'—':>5}")

    for name, m in sorted(results.items()):
        print(f"{name:<18} {m['ann_ret_qs']*100:>7.2f}% {m['ann_vol_qs']*100:>7.2f}% {m['sharpe_qs']:>8.4f} {m['maxdd_qs']*100:>7.2f}% {m['ir']:>8.4f} {m['worst_active']*100:>9.2f}% {m['violations']:>5d}")

    print("=" * 100)


if __name__ == "__main__":
    main()
