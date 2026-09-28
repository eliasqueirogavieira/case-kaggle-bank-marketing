"""Models: month-offset LightGBM, logistic regression with month fixed effects, and simple baselines."""

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from bank_marketing.config import N_JOBS, PERIOD, SEED
from bank_marketing.evaluation import group_auc
from bank_marketing.features import build_preprocessor

LGBM_BASE = {
    "objective": "binary",
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def smoothed_period_rates(periods, y, smoothing: float):
    """Conversion rate per period, shrunk toward the global rate by `smoothing` pseudo-observations."""
    y = np.asarray(y, dtype=float)
    global_rate = float(y.mean())
    stats = pd.DataFrame({"period": np.asarray(periods), "y": y}).groupby("period")["y"].agg(["sum", "count"])
    rates = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    return rates.to_dict(), global_rate


def calibrate_to_rate(score, rate: float) -> np.ndarray:
    """Probabilities expit(score + c) with the single offset c that makes the batch average equal `rate`.

    One offset for everyone keeps the ranking; solving for it (instead of adding logit(rate)) accounts for a
    batch whose clients are, on average, better or worse than the training mix.
    """
    score = np.asarray(score, dtype=float)
    offset = brentq(lambda c: expit(score + c).mean() - rate, -50.0, 50.0, xtol=1e-12)
    return expit(score + offset)


class PeriodOffsetClassifier(ClassifierMixin, BaseEstimator):
    """Gradient boosting that only learns differences *within* a period.

    Every training row starts from the logit of its period's conversion rate (LightGBM `init_score`), so the
    trees explain how a client deviates from the month it was called in. `decision_function` returns that
    deviation, which is what ranks clients inside one campaign; `predict_proba` adds the period rate back,
    or, for a new campaign, calibrates the batch to its expected conversion rate (`base_rate`).
    """

    def __init__(self, preprocessor, estimator, period_col=PERIOD, smoothing=50.0):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.period_col = period_col
        self.smoothing = smoothing

    def fit(self, X, y):
        y = np.asarray(y)
        self.period_rates_, self.global_rate_ = smoothed_period_rates(X[self.period_col], y, self.smoothing)
        offset = logit(X[self.period_col].map(self.period_rates_).to_numpy(dtype=float))
        self.preprocessor_ = clone(self.preprocessor)
        Xt = self.preprocessor_.fit_transform(X, y)
        self.estimator_ = clone(self.estimator).fit(Xt, y, init_score=offset)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.estimator_.predict(self.preprocessor_.transform(X), raw_score=True)

    def contributions(self, X) -> pd.DataFrame:
        """Exact TreeSHAP values of the within-period score, one column per model feature plus `bias`."""
        Xt = self.preprocessor_.transform(X)
        values = self.estimator_.predict(Xt, pred_contrib=True)
        return pd.DataFrame(values, columns=[*Xt.columns, "bias"], index=X.index)

    def predict_proba(self, X, base_rate=None):
        if base_rate is not None:
            p = calibrate_to_rate(self.decision_function(X), base_rate)
            return np.column_stack([1 - p, p])
        if self.period_col in X.columns:
            rates = X[self.period_col].map(self.period_rates_).fillna(self.global_rate_)
            offset = logit(rates.to_numpy(dtype=float))
        else:
            offset = logit(self.global_rate_)
        p = expit(self.decision_function(X) + offset)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PeriodRateBaseline(ClassifierMixin, BaseEstimator):
    """"Month only" baseline: the training conversion rate of the period (global rate for unseen periods)."""

    def __init__(self, period_col=PERIOD, smoothing=50.0):
        self.period_col = period_col
        self.smoothing = smoothing

    def fit(self, X, y):
        self.period_rates_, self.global_rate_ = smoothed_period_rates(X[self.period_col], y, self.smoothing)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X[self.period_col].map(self.period_rates_).fillna(self.global_rate_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        return logit(self.predict_proba(X)[:, 1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BusinessRule(ClassifierMixin, BaseEstimator):
    """Transparent points a campaign manager could use without a model: the bar a model has to clear."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        points = (
            3 * (X["poutcome"] == "success")
            + (X["housing"] == "no")
            + (X["loan"] == "no")
            + (X["balance"] >= 0)
            + (X["contact"] == "cellular")
        )
        return points.to_numpy(dtype=float)


def make_logistic(C: float = 1.0) -> Pipeline:
    """Logistic regression with month fixed effects: coefficients are effects *within* a month."""
    return Pipeline(
        [("prep", build_preprocessor("linear", period_fe=True)), ("model", LogisticRegression(C=C, max_iter=5000))]
    )


def make_lgbm(**params) -> PeriodOffsetClassifier:
    return PeriodOffsetClassifier(build_preprocessor("tree"), LGBMClassifier(**{**LGBM_BASE, **params}))


def make_lgbm_pooled(extra_numeric=(), extra_categorical=(), **params) -> Pipeline:
    """LightGBM without period control, optionally with excluded columns: used only for the ablation."""
    pre = build_preprocessor("tree", extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    return Pipeline([("prep", pre), ("model", LGBMClassifier(**{**LGBM_BASE, **params}))])


def group_auc_scorer(estimator, X, y) -> float:
    """Scorer for sklearn CV: AUC between clients called in the same period."""
    return group_auc(y, estimator.decision_function(X), X[PERIOD])


def lgbm_search_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune_lgbm(X, y, n_trials: int = 50, n_splits: int = 5, seed: int = SEED):
    """Optuna TPE search maximizing the mean cross-validated group AUC of the month-offset LightGBM."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        model = make_lgbm(**lgbm_search_space(trial))
        return cross_val_score(model, X, y, cv=cv, scoring=group_auc_scorer, n_jobs=1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "subsample_freq": 1}, study
