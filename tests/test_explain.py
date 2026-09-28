import numpy as np
import pandas as pd
import pytest

from bank_marketing.config import FEATURES
from bank_marketing.data import load_dataset
from bank_marketing.explain import feature_contributions, reason_codes
from bank_marketing.models import make_lgbm


@pytest.fixture(scope="module")
def fitted():
    df = load_dataset().sample(4000, random_state=0)
    return make_lgbm(n_estimators=40).fit(df, df["y"]), df.head(200)


def test_contributions_add_up_to_the_within_month_score(fitted):
    model, X = fitted
    contribs = feature_contributions(model, X)
    np.testing.assert_allclose(contribs.sum(axis=1), model.decision_function(X), atol=1e-9)


def test_contributions_are_grouped_by_original_variable(fitted):
    model, X = fitted
    assert set(feature_contributions(model, X).columns) == set(FEATURES) | {"bias"}


def test_reason_codes_list_the_variables_that_push_the_score_up():
    contribs = pd.DataFrame(
        {"poutcome": [0.9, -0.1], "housing": [0.2, 0.3], "age": [-0.5, 0.05], "bias": [-2.0, -2.0]}
    )
    X = pd.DataFrame({"poutcome": ["success", "failure"], "housing": ["no", "no"], "age": [30, 70]})
    codes = reason_codes(contribs, X, top=2)
    assert codes.loc[0].tolist() == ["poutcome=success", "housing=no"]
    assert codes.loc[1].tolist() == ["housing=no", "age=70"]


def test_reason_codes_leave_blank_when_nothing_pushes_up():
    contribs = pd.DataFrame({"poutcome": [-0.4], "housing": [0.1], "bias": [-2.0]})
    X = pd.DataFrame({"poutcome": ["failure"], "housing": ["no"]})
    assert reason_codes(contribs, X, top=2).loc[0].tolist() == ["housing=no", ""]
