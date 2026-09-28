import pandas as pd
import pytest

from bank_marketing.data import add_period, clean, load_raw, split_oot, split_random


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_load_raw_reads_the_semicolon_separated_file(raw):
    assert raw.shape == (45211, 17)
    assert raw.loc[0, "job"] == "management"


def test_add_period_rolls_the_year_over_when_the_month_goes_back():
    df = pd.DataFrame({"month": ["may", "may", "dec", "jan", "jan", "may"]})
    out = add_period(df)
    assert out["year"].tolist() == [2008, 2008, 2008, 2009, 2009, 2009]
    assert out["period"].tolist() == ["2008-05", "2008-05", "2008-12", "2009-01", "2009-01", "2009-05"]
    assert out["period_idx"].tolist() == [0, 0, 7, 8, 8, 12]


def test_add_period_on_the_real_file_spans_may_2008_to_nov_2010(raw):
    out = add_period(raw)
    assert out["period"].iloc[0] == "2008-05"
    assert out["period"].iloc[-1] == "2010-11"
    assert out["period"].nunique() == 30  # Sep/2008 has no calls
    assert out["period_idx"].is_monotonic_increasing


def test_clean_maps_target_to_int():
    df = _rows(y=["yes", "no"], poutcome=["success", "failure"], pdays=[10, 20])
    assert clean(df)["y"].tolist() == [1, 0]


def test_clean_recodes_poutcome_unknown_only_for_never_contacted_clients():
    df = _rows(y=["no", "no", "yes"], poutcome=["unknown", "unknown", "failure"], pdays=[-1, 90, 30])
    out = clean(df)
    assert out["poutcome"].tolist() == ["nonexistent", "unknown", "failure"]
    assert out["contacted_before"].tolist() == [0, 1, 1]


def test_clean_works_without_target_column():
    df = _rows(y=["no"], poutcome=["unknown"], pdays=[-1]).drop(columns="y")
    out = clean(df)
    assert "y" not in out.columns
    assert out["contacted_before"].tolist() == [0]


def test_split_random_is_stratified_disjoint_and_reproducible(raw):
    df = clean(add_period(raw))
    train, test = split_random(df)
    assert len(train) == 36168 and len(test) == 9043
    assert set(train.index).isdisjoint(test.index)
    assert abs(train["y"].mean() - test["y"].mean()) < 0.0005
    train2, _ = split_random(df)
    assert train.index.equals(train2.index)


def test_split_oot_cuts_by_period(raw):
    df = clean(add_period(raw))
    train, test = split_oot(df, train_end="2009-04")
    assert len(train) == 34177 and len(test) == 11034
    assert train["period"].max() == "2009-04"
    assert test["period"].min() == "2009-05"


def _rows(y, poutcome, pdays):
    n = len(y)
    return pd.DataFrame(
        {
            "age": [40] * n,
            "job": ["admin."] * n,
            "poutcome": poutcome,
            "pdays": pdays,
            "y": y,
        }
    )
