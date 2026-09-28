import pandas as pd
import pytest

from bank_marketing.data import load_dataset
from bank_marketing.eda import attempt_hazard, attempt_lift, cap_attempts_impact, lift_by, unknown_summary


def test_lift_by_separates_period_effect_from_segment_effect():
    # Period A converts 10%, period B 50%. Segment x only exists in A, z1/z2 only in B.
    df = pd.DataFrame(
        {
            "period": ["A"] * 10 + ["B"] * 10,
            "seg": ["x"] * 10 + ["z1"] * 5 + ["z2"] * 5,
            "y": [1] + [0] * 9 + [1, 1, 1, 1, 0] + [1, 0, 0, 0, 0],
        }
    )
    table = lift_by(df, "seg")
    assert table.loc["x", "pooled_lift"] == pytest.approx(0.1 / 0.3)
    assert table.loc["x", "adjusted_lift"] == pytest.approx(1.0)  # x is just "called in the bad month"
    assert table.loc["z1", "pooled_lift"] == pytest.approx(0.8 / 0.3)
    assert table.loc["z1", "adjusted_lift"] == pytest.approx(4 / 2.5)
    assert table.loc["z2", "adjusted_lift"] == pytest.approx(1 / 2.5)
    assert table.loc["z1", "n"] == 5


CAMPAIGN = pd.DataFrame({"campaign": [1, 1, 2, 3], "y": [1, 0, 1, 0]})


def test_attempt_hazard_divides_by_clients_that_reached_the_attempt():
    table = attempt_hazard(CAMPAIGN)
    assert table["reached"].tolist() == [4, 2, 1]
    assert table["hazard"].tolist() == pytest.approx([0.25, 0.5, 0.0])
    assert table["cum_conversion_share"].tolist() == pytest.approx([0.5, 1.0, 1.0])
    assert table["cum_call_share"].tolist() == pytest.approx([4 / 7, 6 / 7, 1.0])


def test_cap_attempts_impact_on_a_hand_checked_example():
    impact = cap_attempts_impact(CAMPAIGN, cap=1)
    assert impact["calls_saved_share"] == pytest.approx(3 / 7)
    assert impact["conversions_lost_share"] == pytest.approx(0.5)
    assert impact["conversion_rate_of_saved_calls"] == pytest.approx(1 / 3)


def test_cap_at_three_attempts_on_the_real_data():
    impact = cap_attempts_impact(load_dataset(), cap=3)
    assert impact["calls_saved_share"] == pytest.approx(36916 / 124956)
    assert impact["conversions_lost_share"] == pytest.approx(709 / 5289)


def test_unknown_summary_compares_conversion_of_unknown_and_known_rows():
    df = pd.DataFrame({"job": ["unknown", "admin.", "admin.", "unknown"], "y": [1, 0, 0, 0]})
    row = unknown_summary(df, ["job"]).loc["job"]
    assert row["n_unknown"] == 2
    assert row["pct_unknown"] == pytest.approx(0.5)
    assert row["rate_unknown"] == pytest.approx(0.5)
    assert row["rate_known"] == pytest.approx(0.0)


# Period A: 2 clients, 3 calls, 2 conversions (per-call rate 2/3). Period B: 2 clients, 2 calls, 1 conversion (1/2).
TWO_PERIODS = pd.DataFrame({"period": ["A", "A", "B", "B"], "campaign": [1, 2, 1, 1], "y": [1, 1, 0, 1]})


def test_attempt_lift_compares_each_attempt_with_a_call_of_the_same_period():
    table = attempt_lift(TWO_PERIODS).set_index("attempt")
    assert table.loc[1, "expected"] == pytest.approx(2 / 3 + 2 / 3 + 1 / 2 + 1 / 2)
    assert table.loc[1, "ratio"] == pytest.approx(2 / (7 / 3))
    assert table.loc[2, "ratio"] == pytest.approx(1 / (2 / 3))


def test_cap_impact_reports_the_cut_calls_against_their_period_rate():
    impact = cap_attempts_impact(TWO_PERIODS, cap=1)
    assert impact["relative_rate_of_saved_calls"] == pytest.approx(1 / (2 / 3))
