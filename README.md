# Portfolio-Traditional-OIS

Constrained long-only portfolio optimization combining traditional and option-implied signals to outperform a market-cap-weighted S&P 500 benchmark.

## Key Results

### OOS Performance (2021-2024)

| Metric | QS Portfolio | Benchmark | Delta |
|--------|-------------|-----------|-------|
| Ann. Return | 14.07% | 12.01% | +2.06% |
| Ann. Volatility | 16.98% | 15.68% | +1.30% |
| Sharpe (Simple) | 0.83 | 0.77 | +0.06 |
| Max Drawdown | -22.90% | -23.25% | +0.34% |
| Information Ratio | 0.61 | -- | -- |
| Tracking Error | 3.02% | -- | -- |
| Hit Rate | 64.58% | 62.50% | +2.08% |

### Full Period (2016-2024)

| Metric | QS Portfolio | Benchmark | Delta |
|--------|-------------|-----------|-------|
| Ann. Return | 19.46% | 15.04% | +4.42% |
| Sharpe (Simple) | 1.19 | 1.00 | +0.20 |
| Information Ratio | 1.28 | -- | -- |

### Constraint Compliance: 100% (0 violations across 108 months)
- **Weight Bounds**: [0, 2x benchmark] per stock
- **Factor Exposure**: <= 0.10 deviation per FF4 factor
- **Relative Drawdown**: <= 2% monthly active return

## Data

Source: `final_panel.parquet` from HW1 pipeline (1.58M rows, 26 columns, 500 S&P 500 stocks/month, 2015-2024).

**Download**: https://drive.google.com/file/d/1kczsMeHAfiBnMxei667iLGbsRnDGLCwo/view?usp=sharing

Place at `../01HW_QT/Data/Output/final_panel.parquet`.

### Features (11)
- **FF4 Betas** (4): beta_mktrf, beta_smb, beta_hml, beta_mom
- **Option-Implied** (3): SKEW, AIV, GLB
- **Traditional** (3): Mom12m, IdioVol3F, BM
- **Residual Momentum** (1): resid_signal

## Architecture

```
main.py                 # Pipeline orchestrator (7 stages)
src/
  config.py             # Constants, paths, hyperparameters
  data_loader.py        # Load panel, universe filter, monthly resample
  benchmark.py          # Market-cap-weighted benchmark
  preprocessing.py      # Winsorize -> Box-Cox -> Z-score
  signals.py            # XGBoost IS training + OOS walk-forward + EMA smoothing
  optimizer.py          # cvxpy constrained optimization (CVaR + post-hoc guard)
  backtest.py           # Monthly rebalancing engine
  constraints.py        # Post-hoc constraint verification
  performance.py        # Metrics (NW Sharpe, IR) + plots
  feature_ablation.py   # Feature set comparison
  robustness.py         # Statistical significance tests + EMA sensitivity
  generate_report.py    # 15-page academic PDF report
```

## Methodology

1. **Benchmark**: Market-cap-weighted portfolio of top 500 S&P 500 stocks
2. **Signal**: XGBoost (n_est=500 max, lr=0.05, depth=4) with early stopping (patience=20) and walk-forward monthly re-estimation in OOS
3. **EMA Smoothing**: alpha=0.3 applied to raw signal before optimizer (half-life ~2 months)
4. **Preprocessing**: Winsorize(p1/p99) -> Box-Cox(IS-fitted lambda) -> Z-score
5. **Optimizer**: `maximize mu'w` s.t. sum(w)=1, 0<=w<=2*w_bench, |delta_beta|<=0.10, CVaR<=2%, TE<=1.5%
6. **Covariance**: Ledoit-Wolf shrinkage, 36-month rolling window
7. **Post-hoc Guard**: Shrink active weights 50% toward benchmark if prior month < -1.5%

## Usage

```bash
pip install -r requirements.txt
python main.py                    # Full pipeline
python src/robustness.py          # Statistical tests + EMA sensitivity
python src/generate_report.py     # Generate 15-page report.pdf
```

## Report (15 pages)

The PDF report includes: executive summary, strategy pipeline, performance comparison (OOS + Full), cumulative returns, active returns, factor exposure, rolling Sharpe/drawdown/weight distribution, constraint compliance, feature ablation, preprocessing comparison, methodology, discussion, and robustness/statistical significance analysis.
