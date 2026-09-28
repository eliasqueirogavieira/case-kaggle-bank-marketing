import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from scipy.special import logit

from bank_marketing.data import load_dataset
from bank_marketing.features import build_preprocessor
from bank_marketing.models import (
    BusinessRule,
    calibrate_to_rate,
    PeriodOffsetClassifier,
    PeriodRateBaseline,
    make_lgbm,
    make_logistic,
    tune_lgbm,
)


def _clients(n, rng, period, balance):
    """Valid F0 rows; `balance` and `period` are set by the caller."""
    return pd.DataFrame(
        {
            "job": rng.choice(["admin.", "technician", "retired"], n),
            "marital": rng.choice(["married", "single"], n),
            "education": rng.choice(["secondary", "tertiary"], n),
            "contact": rng.choice(["cellular", "telephone"], n),
            "poutcome": rng.choice(["nonexistent", "failure"], n),
            "default": "no",
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n),
            "age": rng.integers(20, 70, n),
            "balance": balance,
            "pdays": -1,
            "previous": 0,
            "contacted_before": 0,
            "period": period,
        }
    )


@pytest.fixture(scope="module")
def simpson():
    """Balance is high in the good month and low in the bad one, but never matters within a month."""
    rng = np.random.default_rng(0)
    bad = _clients(3000, rng, "2008-05", rng.integers(0, 100, 3000))
    good = _clients(3000, rng, "2010-05", rng.integers(5000, 5100, 3000))
    X = pd.concat([bad, good], ignore_index=True)
    y = np.concatenate([rng.random(3000) < 0.05, rng.random(3000) < 0.5]).astype(int)
    return X, y


def _small_lgbm(**kwargs):
    return LGBMClassifier(n_estimators=100, random_state=0, verbose=-1, deterministic=True,
                          force_row_wise=True, n_jobs=1, **kwargs)


def test_period_rates_are_smoothed_toward_the_global_rate():
    rng = np.random.default_rng(1)
    X = pd.concat([_clients(10, rng, "A", 0), _clients(10, rng, "B", 0)], ignore_index=True)
    y = np.array([1] + [0] * 9 + [1] * 5 + [0] * 5)
    model = PeriodOffsetClassifier(build_preprocessor("tree"), _small_lgbm(min_child_samples=1), smoothing=10)
    model.fit(X, y)
    assert model.period_rates_["A"] == pytest.approx((1 + 10 * 0.3) / 20)
    assert model.period_rates_["B"] == pytest.approx((5 + 10 * 0.3) / 20)


def test_offset_model_does_not_learn_a_pure_period_effect(simpson):
    X, y = simpson
    # A flexible learner fits within-month noise, and in a 5% month that noise alone drags the mean log-odds
    # down (leaves without positives); a strongly regularized one isolates the period mechanism under test.
    regularized = {"num_leaves": 4, "min_child_samples": 200}
    offset = PeriodOffsetClassifier(build_preprocessor("tree"), _small_lgbm(**regularized)).fit(X, y)
    pooled = _small_lgbm(**regularized).fit(build_preprocessor("tree").fit_transform(X), y)

    is_good = (X["period"] == "2010-05").to_numpy()
    raw_offset = offset.decision_function(X)
    raw_pooled = pooled.predict(build_preprocessor("tree").fit_transform(X), raw_score=True)
    assert abs(raw_offset[is_good].mean() - raw_offset[~is_good].mean()) < 0.3
    assert raw_pooled[is_good].mean() - raw_pooled[~is_good].mean() > 2.0  # logit(.5) - logit(.05) = 2.94


def test_decision_function_ignores_the_period_column(simpson):
    X, y = simpson
    model = PeriodOffsetClassifier(build_preprocessor("tree"), _small_lgbm()).fit(X, y)
    moved = X.assign(period="2099-01")
    np.testing.assert_array_equal(model.decision_function(X), model.decision_function(moved))


def test_predict_proba_adds_the_training_rate_of_known_periods(simpson):
    X, y = simpson
    model = PeriodOffsetClassifier(build_preprocessor("tree"), _small_lgbm()).fit(X, y)
    rows = X.head(5)
    p = model.predict_proba(rows)[:, 1]
    np.testing.assert_allclose(logit(p) - model.decision_function(rows), logit(model.period_rates_["2008-05"]))


def test_predict_proba_matches_the_expected_rate_of_a_new_campaign(simpson):
    X, y = simpson
    model = PeriodOffsetClassifier(build_preprocessor("tree"), _small_lgbm()).fit(X, y)
    rows = X.drop(columns="period").sample(400, random_state=0)
    proba = model.predict_proba(rows, base_rate=0.3)
    assert proba[:, 1].mean() == pytest.approx(0.3, abs=1e-9)  # the batch converts at the expected rate
    shift = logit(proba[:, 1]) - model.decision_function(rows)
    np.testing.assert_allclose(shift, shift[0])  # one offset for the whole batch: ranking unchanged
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_calibrate_to_rate_keeps_the_order_and_hits_the_rate():
    score = np.array([-2.0, 0.0, 0.5, 3.0])
    p = calibrate_to_rate(score, 0.25)
    assert p.mean() == pytest.approx(0.25, abs=1e-9)
    assert list(np.argsort(p)) == list(np.argsort(score))
    np.testing.assert_allclose(calibrate_to_rate(np.zeros(3), 0.4), [0.4, 0.4, 0.4])


def test_period_rate_baseline_predicts_the_smoothed_rate_and_falls_back_to_global():
    X = pd.DataFrame({"period": ["A"] * 10 + ["B"] * 10})
    y = np.array([1] + [0] * 9 + [1] * 5 + [0] * 5)
    base = PeriodRateBaseline(smoothing=10).fit(X, y)
    p = base.predict_proba(pd.DataFrame({"period": ["A", "B", "Z"]}))[:, 1]
    np.testing.assert_allclose(p, [0.2, 0.4, 0.3])


def test_business_rule_scores_hand_checked_rows():
    X = pd.DataFrame(
        {
            "poutcome": ["success", "failure", "nonexistent"],
            "housing": ["no", "yes", "no"],
            "loan": ["no", "yes", "yes"],
            "balance": [100, -50, 0],
            "contact": ["cellular", "telephone", "unknown"],
        }
    )
    np.testing.assert_array_equal(BusinessRule().fit(X, None).decision_function(X), [7, 0, 2])


def test_fitted_models_survive_a_joblib_round_trip(tmp_path):
    df = load_dataset().sample(3000, random_state=0)
    for model in (make_lgbm(n_estimators=20), make_logistic()):
        model.fit(df, df["y"])
        joblib.dump(model, tmp_path / "m.joblib")
        restored = joblib.load(tmp_path / "m.joblib")
        np.testing.assert_allclose(restored.decision_function(df), model.decision_function(df))


def test_tune_lgbm_runs_the_requested_trials_and_returns_lgbm_params():
    df = load_dataset().sample(3000, random_state=0)
    best, study = tune_lgbm(df, df["y"], n_trials=2, n_splits=3)
    assert len(study.trials) == 2
    assert {"learning_rate", "num_leaves", "n_estimators"} <= best.keys()
    assert 0.5 < study.best_value < 1.0
