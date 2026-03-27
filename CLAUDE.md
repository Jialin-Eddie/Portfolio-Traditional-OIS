# Portfolio-Traditional-OIS

## What
Constrained long-only portfolio optimization combining traditional + option-implied signals.
MCW benchmark of S&P 500 stocks. XGBoost ML signal. cvxpy optimizer with 3 constraints.

## Data
- Source: `../01HW_QT/Data/Output/final_panel.parquet` (1.58M rows, 26 cols)
- Frequency: Monthly rebalancing
- IS: 2015-2020, OOS: 2021-2024 (walk-forward monthly re-estimation)

## Constraints
- Factor exposure deviation from BNCH: max 0.1 per FF4 factor
- Weight bounds: [0, 2x benchmark weight] per stock
- Monthly relative drawdown: max 2% (enforced via tracking error proxy)

## Key Rules
- `statsmodels` needs `.to_numpy(dtype=np.float64, na_value=np.nan)` — not nullable Float64
- NW Sharpe: `cov_hac[0,0] * N` before sqrt
- Walk-forward assertion: `max(train_date) < min(predict_date)`

@.claude/rules/windows-cli-resolution.md
