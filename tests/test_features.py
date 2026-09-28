import math

import numpy as np
import pandas as pd
import pytest

from bank_marketing.config import EXCLUDED
from bank_marketing.data import load_dataset
from bank_marketing.features import (
    age_band,
    build_preprocessor,
    feature_origin,
    log1p_positive,
    signed_log1p,
)

E1 = math.e - 1  # log1p(E1) == 1


@pytest.fixture(scope="module")
def sample():
    return load_dataset().sample(3000, random_state=0)


def test_signed_log1p_keeps_the_sign_of_negative_balances():
    np.testing.assert_allclose(signed_log1p(np.array([-E1, 0.0, E1])), [-1.0, 0.0, 1.0])


def test_log1p_positive_sends_never_contacted_pdays_to_zero():
    np.testing.assert_allclose(log1p_positive(np.array([-1.0, 0.0, E1])), [0.0, 0.0, 1.0])


def test_age_band_boundaries():
    ages = pd.DataFrame({"age": [18, 25, 26, 35, 36, 55, 56, 65, 66, 95]})
    assert age_band(ages)["age"].tolist() == [
        "18-25", "18-25", "26-35", "26-35", "36-45", "46-55", "56-65", "56-65", "66+", "66+",
    ]


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_excluded_columns_never_reach_the_model(sample, kind):
    pre = build_preprocessor(kind).fit(sample)
    origins = set(feature_origin(pre).values())
    assert origins.isdisjoint(EXCLUDED)

    poisoned = sample.assign(duration=10**6, campaign=99, day=31, month="dec")
    pd.testing.assert_frame_equal(pre.transform(sample), pre.transform(poisoned))


def test_ablation_columns_are_opt_in(sample):
    pre = build_preprocessor("tree", extra_numeric=["duration", "campaign", "day"], extra_categorical=["month"])
    origins = set(feature_origin(pre.fit(sample)).values())
    assert {"duration", "campaign", "day", "month"} <= origins


def test_tree_keeps_raw_numbers_while_linear_standardizes(sample):
    tree = build_preprocessor("tree").fit(sample).transform(sample)
    linear = build_preprocessor("linear").fit(sample).transform(sample)
    np.testing.assert_array_equal(tree["balance"].to_numpy(), sample["balance"].to_numpy())
    assert abs(linear["balance"].mean()) < 1e-9
    assert abs(linear["balance"].std(ddof=0) - 1) < 1e-9


def test_unseen_category_becomes_all_zeros(sample):
    pre = build_preprocessor("tree").fit(sample[sample["job"] != "student"])
    out = pre.transform(sample[sample["job"] == "student"].head(3))
    job_cols = [c for c in out.columns if c.startswith("job_")]
    assert out[job_cols].to_numpy().sum() == 0


def test_period_fixed_effects_only_when_requested(sample):
    with_fe = feature_origin(build_preprocessor("linear", period_fe=True).fit(sample))
    without = feature_origin(build_preprocessor("linear").fit(sample))
    assert "period" in with_fe.values()
    assert "period" not in without.values()


def test_feature_origin_maps_columns_back_to_the_original_variable(sample):
    origin = feature_origin(build_preprocessor("linear").fit(sample))
    assert origin["job_blue-collar"] == "job"
    assert origin["job_admin."] == "job"
    assert origin["education_unknown"] == "education"
    assert origin["housing_yes"] == "housing"
    assert origin["age_18-25"] == "age"
    assert origin["balance"] == "balance"
    assert origin["contacted_before"] == "contacted_before"
