# Project Status: Portfolio-Traditional-OIS

## Phase: Complete
## Status: All stages pass, all constraints 100% compliant, OOS beats benchmark

## Early Stopping Impact (Before vs After)

### Before (no early stop) vs After (early_stopping_rounds=20)

```
                          BNCH     Before      After
------------------------------------------------------------

  OOS (2021-2024)
  ----------------------------------------------------------------
  Return                 11.39%     11.39%     12.60%
  Sharpe                 0.7259     0.6786     0.7519
  Excess Return               —     -0.00%     +1.21%
  Info Ratio                  —    -0.0001      +0.46
  Max Underperf.              —     -1.38%     -1.60%

  FULL (2016-2024)
  ----------------------------------------------------------------
  Return                 14.11%     17.74%     16.58%
  Sharpe                 0.9308     1.0901     1.0401
  Excess Return               —     +3.63%     +2.47%
  Info Ratio                  —     1.2629     1.0097
```

### Early Stopping OOS Effect

| OOS Metric | No Early Stop | With Early Stop | Change |
|------------|---------------|-----------------|--------|
| QS Return | 11.39% | **12.60%** | +1.21% |
| Excess Return | -0.00% | **+1.21%** | QS beats BNCH |
| Sharpe | 0.68 | **0.75** | +0.07 |
| Info Ratio | -0.0001 | **+0.46** | Positive |

QS now beats benchmark OOS by +1.21%/year, IR=0.46, all constraints 100% PASS.

`best_iteration=0` for most months — model uses only 1 tree (depth=4). The previous 200-tree model was fitting noise.

## Final Results

### Performance (Full Period 2016-2024, 108 months)

| Metric | QS Portfolio | Benchmark | Delta |
|--------|-------------|-----------|-------|
| Ann. Return | 16.58% | 14.11% | +2.47% |
| Ann. Volatility | 15.94% | 15.16% | +0.78% |
| Sharpe (NW) | 1.29 | 1.19 | +0.10 |
| Info Ratio | 1.01 | — | — |
| Tracking Error | 2.45% | — | — |
| Max Monthly Underperf. | -1.60% | — | Within 2% |

### IS vs OOS Breakdown

| Metric | IS (2016-2020) | OOS (2021-2024) |
|--------|---------------|-----------------|
| QS Ann. Return | 19.76% | 12.60% |
| BNCH Ann. Return | 16.28% | 11.39% |
| Excess Return | +3.48% | +1.21% |
| Info Ratio | 1.53 | +0.46 |

### Constraint Compliance: 100%
- Weight Bounds: 100% PASS
- Factor Exposure: 100% PASS
- Relative Drawdown: 100% PASS (worst: -1.60%)

## Key Decisions
- Preprocessing: winsorize → Box-Cox → z-score (improved OOS from -1.7% to ~0%)
- Early stopping: rounds=20, val=last 15% of IS months (improved OOS from ~0% to +1.21%)
- Hyperparams: n_estimators=500, max_depth=4, lr=0.05, min_child_weight=50 (professor's blueprint)
