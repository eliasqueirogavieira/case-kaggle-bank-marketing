"""Fit the final model on the training split, save it and record its hold-out metrics.

    uv run python -m bank_marketing.train              # hyperparameters from models/best_params.json
    uv run python -m bank_marketing.train --tune 50    # re-run the Optuna search first (slow)
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from bank_marketing.config import DATA_RAW, EXCLUDED, FEATURES, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SEED, TARGET
from bank_marketing.data import load_dataset, split_random
from bank_marketing.evaluation import model_metrics, save_metrics
from bank_marketing.models import make_lgbm, tune_lgbm


def display_path(path: Path) -> str:
    """Path relative to the project when inside it, so logs and notebook outputs carry no machine paths."""
    path = Path(path)
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the month-offset LightGBM and record hold-out metrics.")
    parser.add_argument("--params", type=Path, default=MODELS_DIR / "best_params.json")
    parser.add_argument("--tune", type=int, default=0, metavar="N_TRIALS", help="re-run Optuna with N trials first")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    train, test = split_random(load_dataset())
    if args.tune:
        params, _ = tune_lgbm(train, train[TARGET], n_trials=args.tune)
        args.params.parent.mkdir(parents=True, exist_ok=True)
        args.params.write_text(json.dumps(params, indent=2) + "\n")
    params = json.loads(args.params.read_text())

    model = make_lgbm(**params).fit(train, train[TARGET])
    metrics = model_metrics(model, test)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.models_dir / "model.joblib")
    metadata = {
        "model": "LightGBM with month offset (bank_marketing.models.PeriodOffsetClassifier)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "excluded": EXCLUDED,
        "params": params,
        "seed": SEED,
        "n_train": len(train),
        "n_test": len(test),
        "train_base_rate": float(train[TARGET].mean()),
        "data_sha256": hashlib.sha256(DATA_RAW.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version()}
        | {pkg: version(pkg) for pkg in ("scikit-learn", "lightgbm", "pandas", "numpy")},
        "holdout_metrics": metrics,
    }
    (args.models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_metrics("final_model", metrics, args.reports_dir / "metrics.json")

    print(f"model saved to {display_path(args.models_dir / 'model.joblib')}")
    for name in ("group_auc", "lift_at_20_within_month", "roc_auc", "ks", "pr_auc"):
        print(f"  {name:<26} {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
