import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from bank_marketing.evaluation import (
    capture_at_k,
    decile_table,
    gains_curve,
    gini,
    group_auc,
    ks_statistic,
    lift_at_k,
    paired_bootstrap,
    psi,
)

# Two months. Within A the positive beats one of two negatives; within B both positives beat the negative.
Y = np.array([1, 0, 0, 1, 1, 0])
S = np.array([0.9, 0.5, 0.95, 0.2, 0.3, 0.1])
G = np.array(["A", "A", "A", "B", "B", "B"])


def test_group_auc_only_compares_pairs_from_the_same_group():
    assert group_auc(Y, S, G) == pytest.approx(3 / 4)
    assert roc_auc_score(Y, S) == pytest.approx(4 / 9)  # pooled AUC mixes the months


def test_group_auc_counts_ties_as_half():
    assert group_auc(np.array([1, 0]), np.array([0.5, 0.5]), np.array(["A", "A"])) == pytest.approx(0.5)


def test_group_auc_skips_groups_with_a_single_class():
    y = np.append(Y, [0, 0])
    s = np.append(S, [0.99, 0.01])
    g = np.append(G, ["C", "C"])
    assert group_auc(y, s, g) == pytest.approx(3 / 4)


TOP_Y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
TOP_S = np.arange(10, 0, -1).astype(float)


def test_lift_at_k_is_precision_over_base_rate():
    assert lift_at_k(TOP_Y, TOP_S, k=0.2) == pytest.approx(5.0)
    assert capture_at_k(TOP_Y, TOP_S, k=0.2) == pytest.approx(1.0)


def test_lift_at_k_within_groups_takes_the_top_of_each_group():
    y = np.array([1, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    s = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=float)
    g = np.array(["A"] * 5 + ["B"] * 5)
    # top-1 of A is a positive (expected 0.2 at random); top-1 of B is a negative (expected 0.4)
    assert lift_at_k(y, s, k=0.2, groups=g) == pytest.approx(1 / 0.6)
    assert capture_at_k(y, s, k=0.2, groups=g) == pytest.approx(1 / 3)


def test_ks_and_gini_on_a_hand_checked_example():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert ks_statistic(y, s) == pytest.approx(0.5)
    assert gini(y, s) == pytest.approx(0.5)  # AUC 0.75


def test_ks_is_one_for_a_perfect_ranking():
    assert ks_statistic(TOP_Y, TOP_S) == pytest.approx(1.0)


def test_decile_table_orders_bins_from_highest_score():
    table = decile_table(TOP_Y, TOP_S, n_bins=5)
    assert table["n"].tolist() == [2, 2, 2, 2, 2]
    assert table.loc[0, "rate"] == pytest.approx(1.0)
    assert table.loc[0, "lift"] == pytest.approx(5.0)
    assert table["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_gains_curve_reaches_full_capture_at_the_base_rate_for_a_perfect_ranking():
    frac_contacted, frac_captured = gains_curve(TOP_Y, TOP_S)
    assert frac_captured[np.searchsorted(frac_contacted, 0.2)] == pytest.approx(1.0)
    assert frac_contacted[-1] == pytest.approx(1.0) and frac_captured[-1] == pytest.approx(1.0)


def test_gains_curve_within_groups_ranks_each_group_separately():
    # Scores of group B are all below group A, but within each group the positive ranks first.
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    g = np.array(["A", "A", "B", "B"])
    frac_contacted, frac_captured = gains_curve(y, s, groups=g)
    assert frac_captured[np.searchsorted(frac_contacted, 0.5)] == pytest.approx(1.0)


def test_paired_bootstrap_of_identical_scores_is_exactly_zero():
    res = paired_bootstrap(TOP_Y, TOP_S, TOP_S, metric=roc_auc_score, n_boot=50)
    assert res["diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    noise = rng.normal(0, 1, 400)
    res = paired_bootstrap(y, good, noise, metric=roc_auc_score, n_boot=200)
    assert res["ci_low"] > 0


def test_psi_of_a_known_categorical_shift():
    expected = ["a"] * 50 + ["b"] * 50
    actual = ["a"] * 25 + ["b"] * 75
    assert psi(expected, actual) == pytest.approx(0.25 * math.log(2) + 0.25 * math.log(1.5))


def test_psi_is_zero_for_identical_numeric_samples_and_grows_with_a_shift():
    base = np.arange(1000, dtype=float)
    assert psi(base, base) == pytest.approx(0.0)
    assert psi(base, base + 300) > psi(base, base + 100) > 0


def test_model_metrics_of_a_constant_model_is_chance_level():
    from sklearn.dummy import DummyClassifier

    from bank_marketing.data import load_dataset
    from bank_marketing.evaluation import model_metrics

    df = load_dataset().sample(4000, random_state=0)
    m = model_metrics(DummyClassifier(strategy="prior").fit(df, df["y"]), df)
    assert m["group_auc"] == pytest.approx(0.5) and m["roc_auc"] == pytest.approx(0.5)
    assert m["pr_auc"] == pytest.approx(df["y"].mean())
    assert m["brier"] == pytest.approx(df["y"].mean() * (1 - df["y"].mean()), rel=1e-3)


def test_model_metrics_ranks_with_scores_when_there_are_no_probabilities():
    from bank_marketing.data import load_dataset
    from bank_marketing.evaluation import model_metrics
    from bank_marketing.models import BusinessRule

    df = load_dataset().sample(4000, random_state=0)
    m = model_metrics(BusinessRule().fit(df), df)
    assert np.isnan(m["brier"])
    assert m["group_auc"] > 0.55 and m["roc_auc"] > 0.55


def test_lift_and_capture_do_not_depend_on_row_order_when_scores_tie():
    tied = np.ones(4)
    positives_first = np.array([1, 1, 0, 0])
    positives_last = np.array([0, 0, 1, 1])
    assert lift_at_k(positives_first, tied, k=0.5) == pytest.approx(1.0)
    assert lift_at_k(positives_last, tied, k=0.5) == pytest.approx(1.0)
    assert capture_at_k(positives_first, tied, k=0.5) == pytest.approx(0.5)


def test_lift_splits_only_the_tie_at_the_cut():
    # top 2 of 4: score 3 is in; the two rows tied at 2 share the last slot (one of them is positive)
    y = np.array([1, 1, 0, 0])
    s = np.array([3.0, 2.0, 2.0, 1.0])
    assert lift_at_k(y, s, k=0.5) == pytest.approx((1 + 0.5) / (2 * 0.5))


def test_psi_treats_low_cardinality_numbers_as_categories():
    expected = [0] * 97 + [1] * 3
    actual = [0] * 34 + [1] * 66
    want = (0.34 - 0.97) * math.log(0.34 / 0.97) + (0.66 - 0.03) * math.log(0.66 / 0.03)
    assert psi(expected, actual) == pytest.approx(want)


def test_paired_bootstrap_with_groups_compares_within_group_ranking():
    rng = np.random.default_rng(1)
    g = np.repeat(["A", "B"], 300)
    y = rng.integers(0, 2, 600)
    informative = y + rng.normal(0, 0.7, 600)
    by_group_only = (g == "B").astype(float)  # constant inside each group: within-group AUC 0.5
    res = paired_bootstrap(y, informative, by_group_only, group_auc, n_boot=200, groups=g)
    assert res["ci_low"] > 0 and res["ci_low"] <= res["diff"] <= res["ci_high"]
