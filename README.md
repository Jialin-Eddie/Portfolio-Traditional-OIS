# Portfolio-Traditional-OIS

Constrained long-only portfolio optimization combining traditional and option-implied signals to improve upon a market-cap-weighted benchmark of S&P 500 stocks.

## Key Results

### OOS Performance (2021–2024)

| | BNCH | QS Portfolio |
|--|------|-------------|
| Ann. Return | 11.39% | 11.39% |
| Volatility | 15.69% | 16.79% |
| Sharpe | 0.73 | 0.68 |
| Max Drawdown | -23.17% | -23.72% |
| Information Ratio | — | -0.0001 |

### Constraint Compliance: 100%
- **Weight Bounds**: [0, 2× benchmark] per stock — 100% PASS
- **Factor Exposure**: ≤ 0.1 deviation per FF4 factor — 100% PASS
- **Relative Drawdown**: ≤ 2% per month vs benchmark — 100% PASS

## Architecture

```
main.py                 # Pipeline orchestrator (7 stages)
src/
  config.py             # Constants, paths, hyperparameters
  data_loader.py        # Load panel, universe filter, monthly resample
  benchmark.py          # Market-cap-weighted benchmark
  preprocessing.py      # Winsorize → Box-Cox → Z-score
  signals.py            # XGBoost IS training + OOS walk-forward
  optimizer.py          # cvxpy constrained optimization
  backtest.py           # Monthly rebalancing engine
  constraints.py        # Post-hoc constraint verification
  performance.py        # Metrics (NW Sharpe, IR) + plots
  feature_ablation.py   # Feature set comparison
```

## Data

Source: `final_panel.parquet` from HW1 pipeline (1.58M rows, 26 columns, 736 S&P 500 stocks, 2015–2024).

### Features (11)
- **FF4 Betas** (4): beta_mktrf, beta_smb, beta_hml, beta_mom
- **Option-Implied** (3): SKEW, AIV, GLB
- **Traditional** (3): Mom12m, IdioVol3F, BM
- **Residual Momentum** (1): resid_signal (Blitz et al. 2011)

## Usage

```bash
pip install -r requirements.txt
python main.py                    # Full pipeline (~40 min first run)
python main.py --skip-data-load   # Reuse cached data (~2 min)
```

## Methodology

1. **Benchmark**: Value-weighted (MCW) portfolio of ~700 S&P 500 stocks
2. **Signal**: XGBoost (n_est=200, lr=0.05, depth=4) with monthly walk-forward re-estimation in OOS
3. **Preprocessing**: Per-month cross-section Winsorize(p1/p99) → Box-Cox(λ on IS) → Z-score
4. **Optimizer**: `maximize μᵀw` s.t. Σw=1, 0≤w≤2w_bench, |β_QS-β_BNCH|≤0.1, TE≤1.5%
5. **Covariance**: Ledoit-Wolf shrinkage, 36-month rolling window
