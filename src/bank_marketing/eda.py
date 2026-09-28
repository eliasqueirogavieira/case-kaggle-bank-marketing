"""Exploratory helpers whose numbers are quoted in the notebooks and in the deck."""

import numpy as np
import pandas as pd

from bank_marketing.config import PERIOD, TARGET


def unknown_summary(df: pd.DataFrame, columns, target: str = TARGET) -> pd.DataFrame:
    """How often each column holds "unknown", and how those rows convert compared with the rest."""
    rows = {}
    for col in columns:
        is_unknown = df[col] == "unknown"
        rows[col] = {
            "n_unknown": int(is_unknown.sum()),
            "pct_unknown": is_unknown.mean(),
            "rate_unknown": df.loc[is_unknown, target].mean(),
            "rate_known": df.loc[~is_unknown, target].mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def lift_by(df: pd.DataFrame, col: str, target: str = TARGET, period: str = PERIOD) -> pd.DataFrame:
    """Conversion lift per category: pooled vs adjusted for the period each client was called in.

    pooled_lift   = rate of the category / overall rate
    adjusted_lift = observed conversions / conversions expected from the period rates of the same rows
    Both are 1.0 for an irrelevant attribute; they differ when a category is concentrated in good or bad periods.
    """
    expected = df.groupby(period)[target].transform("mean")
    grouped = df.assign(_expected=expected).groupby(col, observed=True)
    table = grouped.agg(n=(target, "size"), conversions=(target, "sum"), expected=("_expected", "sum"))
    table["rate"] = table["conversions"] / table["n"]
    table["pooled_lift"] = table["rate"] / df[target].mean()
    table["adjusted_lift"] = table["conversions"] / table["expected"]
    return table.drop(columns="expected")


def attempt_hazard(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Conversion per attempt, avoiding the survivorship bias of grouping by `campaign`.

    Calls stop at a "yes", so a row with campaign == k is the last call of that client. The rate at attempt k
    is conversions at k divided by every client that received a k-th call (campaign >= k).
    """
    attempts = np.arange(1, df["campaign"].max() + 1)
    campaign = df["campaign"].to_numpy()
    y = df[target].to_numpy()
    reached = np.array([(campaign >= k).sum() for k in attempts])
    conversions = np.array([((campaign == k) & (y == 1)).sum() for k in attempts])
    table = pd.DataFrame({"attempt": attempts, "reached": reached, "conversions": conversions})
    table["hazard"] = table["conversions"] / table["reached"]
    table["cum_conversion_share"] = table["conversions"].cumsum() / y.sum()
    table["cum_call_share"] = table["reached"].cumsum() / campaign.sum()
    return table


def _per_call_rate(df: pd.DataFrame, target: str, period: str) -> pd.Series:
    """Conversion per call of the period each row belongs to."""
    per_call = df.groupby(period)[target].sum() / df.groupby(period)["campaign"].sum()
    return df[period].map(per_call)


def attempt_lift(df: pd.DataFrame, target: str = TARGET, period: str = PERIOD) -> pd.DataFrame:
    """Conversions at each attempt against what calls of the same period convert on average.

    Pooled per-attempt rates mix periods: most 4th+ calls happened in 2008, when every call converted little.
    `ratio` = conversions at attempt k / (per-call rate of each client's period, summed over clients that got
    a k-th call); 1.0 means the k-th call converts like an average call of its own period.
    """
    rate = _per_call_rate(df, target, period).to_numpy()
    campaign = df["campaign"].to_numpy()
    y = df[target].to_numpy()
    attempts = np.arange(1, campaign.max() + 1)
    table = pd.DataFrame({
        "attempt": attempts,
        "observed": [((campaign == k) & (y == 1)).sum() for k in attempts],
        "expected": [rate[campaign >= k].sum() for k in attempts],
    })
    table["ratio"] = table["observed"] / table["expected"]
    return table


def cap_attempts_impact(df: pd.DataFrame, cap: int, target: str = TARGET, period: str = PERIOD) -> dict:
    """What a maximum of `cap` calls per client would have saved and lost.

    With a period column it also reports how the cut calls converted relative to an average call of their
    own period (1.0 = no worse than usual).
    """
    campaign = df["campaign"]
    extra_calls = (campaign - cap).clip(lower=0)
    saved_calls = extra_calls.sum()
    lost = ((campaign > cap) & (df[target] == 1)).sum()
    impact = {
        "calls_saved_share": saved_calls / campaign.sum(),
        "conversions_lost_share": lost / df[target].sum(),
        "conversion_rate_of_saved_calls": lost / saved_calls,
    }
    if period in df.columns:
        impact["relative_rate_of_saved_calls"] = lost / (extra_calls * _per_call_rate(df, target, period)).sum()
    return impact
