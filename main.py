"""main.py — Full pipeline orchestrator for Portfolio-Traditional-OIS.

Pipeline stages:
  1. Load & prepare monthly panel          (data_loader)
  2. Build MCW benchmark (raw betas)       (benchmark)
  3. Fit & apply preprocessing pipeline    (preprocessing)
  4. Generate XGBoost signals              (signals)
  5. Optimize portfolio (raw betas for constraints)  (optimizer)
  6. Backtest + constraint verification    (backtest, constraints)
  7. Performance report                    (performance)

Usage:
  python main.py
  python main.py --skip-data-load   (reload from cached universe_monthly.parquet)
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    BETA_COLS,
    OUTPUTS,
    SPLIT_DATE,
)
from src.data_loader import load_and_prepare
from src.preprocessing import (
    apply_preprocess_pipeline,
    fit_preprocess_pipeline,
    save_pipeline,
)
from src.benchmark import build_benchmark
from src.signals import generate_signals, smooth_signals_ema
from src.optimizer import optimize_all_months
from src.backtest import run_backtest
from src.constraints import (
    verify_weight_bounds,
    verify_factor_exposure,
    verify_relative_drawdown,
    generate_compliance_report,
)
from src.performance import generate_report


FEATURES: list[str] = ALL_FEATURES + ["resid_signal"]
PIPELINE_PATH: Path = OUTPUTS / "preprocess_params.joblib"


def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def main(skip_data_load: bool = False) -> None:
    t0 = time.time()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Stage 1: Load & prepare monthly panel
    # ==================================================================
    _section("Stage 1: Data Loading & Preparation")
    universe_path = OUTPUTS / "universe_monthly.parquet"

    if skip_data_load and universe_path.exists():
        print(f"[main] Loading cached panel from {universe_path}")
        monthly = pd.read_parquet(universe_path)
        print(f"  Loaded {monthly.shape[0]:,} rows")
    else:
        monthly = load_and_prepare()

    print(f"  Stage 1 done ({_elapsed(t0)})")

    # ==================================================================
    # Stage 2: Build benchmark (using RAW betas, not rank-normalized)
    # ==================================================================
    _section("Stage 2: Benchmark Construction")
    t2 = time.time()

    bench_weights, bench_returns, bench_betas = build_benchmark(monthly)
    print(f"  Stage 2 done ({_elapsed(t2)})")

    # ==================================================================
    # Stage 3: Fit & apply preprocessing pipeline (features only)
    # ==================================================================
    _section("Stage 3: Preprocessing Pipeline")
    t3 = time.time()

    split_ts = pd.Timestamp(SPLIT_DATE)
    is_mask = monthly.index.get_level_values("month_end") <= split_ts
    train_df = monthly.loc[is_mask]

    features_present = [f for f in FEATURES if f in monthly.columns]
    missing = [f for f in FEATURES if f not in monthly.columns]
    if missing:
        warnings.warn(f"[main] Features missing from panel: {missing}")

    params = fit_preprocess_pipeline(train_df, features_present)
    save_pipeline(params, PIPELINE_PATH)

    monthly_proc = apply_preprocess_pipeline(monthly, params)
    monthly_proc.to_parquet(OUTPUTS / "monthly_processed.parquet")
    print(f"  Stage 3 done ({_elapsed(t3)})")

    # ==================================================================
    # Stage 4: Generate XGBoost signals (using processed features)
    # ==================================================================
    _section("Stage 4: Signal Generation (IS + OOS Walk-Forward)")
    t4 = time.time()

    predictions = generate_signals(monthly_proc)
    predictions = smooth_signals_ema(predictions)
    print(f"  Stage 4 done ({_elapsed(t4)})")

    # ==================================================================
    # Stage 5: Portfolio optimization
    # Key: use RAW betas from `monthly` for factor constraints,
    #       NOT rank-normalized betas from `monthly_proc`
    # ==================================================================
    _section("Stage 5: Portfolio Optimization")
    t5 = time.time()

    fwd_ret_wide = monthly["fwd_ret"].unstack(level="permno")
    fwd_ret_wide.index.name = "month_end"

    beta_cols_in = [c for c in BETA_COLS if c in monthly.columns]
    predictions_with_betas = predictions.join(
        monthly[beta_cols_in], how="left"
    )

    bench_betas_for_opt = bench_betas.rename(
        columns={f"bench_beta_{f.replace('beta_', '')}": f for f in beta_cols_in}
    ).reset_index()

    opt_weights = optimize_all_months(
        monthly=fwd_ret_wide,
        predictions=predictions_with_betas.rename(columns={"y_pred": "pred"}),
        bench_weights=bench_weights.rename(columns={"bench_weight": "weight"}),
        bench_betas=bench_betas_for_opt,
    )

    opt_weights.to_parquet(OUTPUTS / "portfolio_weights.parquet")
    print(f"  Stage 5 done ({_elapsed(t5)})")

    # ==================================================================
    # Stage 6: Backtest + constraint verification
    # ==================================================================
    _section("Stage 6: Backtest & Constraint Verification")
    t6 = time.time()

    bt_results = run_backtest(opt_weights, bench_weights, monthly)

    weight_check = verify_weight_bounds(opt_weights, bench_weights)
    factor_check = verify_factor_exposure(opt_weights, bench_weights, monthly)
    dd_check = verify_relative_drawdown(
        bt_results["qs_returns"], bt_results["bench_returns"]
    )
    compliance_report = generate_compliance_report(weight_check, factor_check, dd_check)
    print(f"  Stage 6 done ({_elapsed(t6)})")

    # ==================================================================
    # Stage 7: Performance report
    # ==================================================================
    _section("Stage 7: Performance Report")
    t7 = time.time()

    factor_dev_pivot = None
    if not factor_check.empty:
        fc = factor_check.copy()
        fc["signed_dev"] = fc["qs_exposure"] - fc["bench_exposure"]
        factor_dev_pivot = fc.pivot_table(
            index="month_end", columns="factor", values="signed_dev"
        )

    qs_ret = bt_results["qs_returns"]
    bm_ret = bt_results["bench_returns"]
    act_ret = bt_results["active_returns"]

    # Full period report
    generate_report(
        qs_returns=qs_ret, bench_returns=bm_ret, active_returns=act_ret,
        opt_weights=opt_weights, bench_weights=bench_weights,
        factor_check=factor_dev_pivot, compliance_text=compliance_report,
        period_label="Full",
    )

    # IS period report (2015-2020)
    is_mask = qs_ret.index <= split_ts
    if is_mask.any():
        is_fc = factor_dev_pivot.loc[factor_dev_pivot.index <= split_ts] if factor_dev_pivot is not None else None
        generate_report(
            qs_returns=qs_ret[is_mask], bench_returns=bm_ret[is_mask],
            active_returns=act_ret[act_ret.index <= split_ts],
            opt_weights=opt_weights, bench_weights=bench_weights,
            factor_check=is_fc, compliance_text="(see Full report)",
            period_label="IS_2015_2020",
        )

    # OOS period report (2021-2024)
    oos_mask = qs_ret.index > split_ts
    if oos_mask.any():
        oos_fc = factor_dev_pivot.loc[factor_dev_pivot.index > split_ts] if factor_dev_pivot is not None else None
        generate_report(
            qs_returns=qs_ret[oos_mask], bench_returns=bm_ret[oos_mask],
            active_returns=act_ret[act_ret.index > split_ts],
            opt_weights=opt_weights, bench_weights=bench_weights,
            factor_check=oos_fc, compliance_text="(see Full report)",
            period_label="OOS_2021_2024",
        )

    print(f"  Stage 7 done ({_elapsed(t7)})")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete. Total elapsed: {_elapsed(t0)}")
    print(f"  Outputs in: {OUTPUTS}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio-Traditional-OIS pipeline")
    parser.add_argument(
        "--skip-data-load",
        action="store_true",
        help="Load cached universe_monthly.parquet instead of reprocessing raw panel",
    )
    args = parser.parse_args()
    main(skip_data_load=args.skip_data_load)
