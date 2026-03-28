"""Central configuration for Portfolio-Traditional-OIS project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Data: https://drive.google.com/file/d/1kczsMeHAfiBnMxei667iLGbsRnDGLCwo/view?usp=sharing
# Download final_panel.parquet and place at ../01HW_QT/Data/Output/final_panel.parquet
HW1_ROOT = ROOT.parent / "01HW_QT"
FINAL_PANEL = HW1_ROOT / "Data" / "Output" / "final_panel.parquet"

SPLIT_DATE = "2020-12-31"
IS_START = "2015-01-01"
IS_END = "2020-12-31"
OOS_START = "2021-01-01"
OOS_END = "2024-12-31"

RET_COL = "excess_ret"
MKTCAP_COL = "mktcap"

FF4_FACTORS = ["mktrf", "smb", "hml", "mom"]
BETA_COLS = ["beta_mktrf", "beta_smb", "beta_hml", "beta_mom"]

OPTION_FEATURES = ["SKEW", "AIV", "GLB"]
TRADITIONAL_FEATURES = ["Mom12m", "IdioVol3F", "BM"]
ALL_FEATURES = BETA_COLS + OPTION_FEATURES + TRADITIONAL_FEATURES

RESID_SIGNAL_WIN = 60
RESID_SIGNAL_SKIP = 5

MAX_FACTOR_DEV = 0.1
MAX_FACTOR_DEV_OPTIM = 0.05  # tighter bound in optimizer to leave headroom for excluded stocks
MAX_WEIGHT_MULT = 2.0
MAX_REL_DD_MONTHLY = 0.02
TE_MAX_MONTHLY = 0.015

WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99
UNIVERSE_COVERAGE_THRESHOLD = 0.80
UNIVERSE_MAX_STOCKS = 500
UNIVERSE_MIN_STOCKS = 100

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    random_state=42,
    n_jobs=-1,
)
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_VAL_FRACTION = 0.15
PURGE_MONTHS = 1   # remove val rows whose fwd_ret overlaps predict_month
EMBARGO_MONTHS = 0  # additional gap between val end and predict_month

CVAR_ALPHA = 0.95              # CVaR confidence level (worst 5% of scenarios)
CVAR_LIMIT_MONTHLY = 0.02      # max expected active loss in worst 5% scenarios
DD_WARNING_THRESHOLD = -0.015   # post-hoc: shrink if previous month active < this
DD_SHRINK_FACTOR = 0.5          # post-hoc: shrink active weights by this factor

EMA_ALPHA = 0.3                 # signal smoothing: 30% new + 70% history, half-life ~2 months

COV_WINDOW_MONTHS = 36
NW_LAGS = 4
ANNUALIZE_MONTHLY = 12

RANDOM_SEED = 42
