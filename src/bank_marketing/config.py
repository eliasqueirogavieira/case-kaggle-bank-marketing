"""Project-wide constants: paths, random seed and the feature contract."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
N_JOBS = 4  # LightGBM threads; CV and Optuna always run with n_jobs=1 (see README)

TARGET = "y"
PERIOD = "period"  # "YYYY-MM", reconstructed from the row order (see data.add_period)

# Features available before the campaign starts (feature set F0).
CATEGORICAL = ["job", "marital", "education", "contact", "poutcome"]
BINARY = ["default", "housing", "loan"]
NUMERIC = ["age", "balance", "pdays", "previous"]
ENGINEERED = ["contacted_before"]
FEATURES = CATEGORICAL + BINARY + NUMERIC + ENGINEERED
RAW_INPUTS = CATEGORICAL + BINARY + NUMERIC  # columns a scoring file must have (the flag is derived from pdays)
REASON_EXCLUDE = ["age"]  # never shown to call-center agents as a reason (conduct risk)

# Never used by the production score:
#   duration  -> only known after the call (target leakage)
#   campaign  -> total attempts until the outcome (depends on the outcome)
#   day/month -> date of the *last* contact, confounded with the period
EXCLUDED = ["duration", "campaign", "day", "month"]
