"""Loading, time reconstruction, cleaning and splitting of the bank marketing data."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bank_marketing.config import DATA_RAW, PERIOD, SEED, TARGET

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FIRST_YEAR = 2008  # UCI: bank-full.csv is ordered by date, from May 2008 to November 2010
FIRST_MONTH = 5


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def add_period(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the calendar year from the row order: the year advances whenever the month goes back.

    Adds `year`, `month_num`, `period` ("YYYY-MM") and `period_idx` (months since May 2008).
    """
    month_num = df["month"].map({m: i for i, m in enumerate(MONTHS, start=1)}).to_numpy()
    rollovers = np.concatenate([[0], np.cumsum(np.diff(month_num) < 0)])
    year = FIRST_YEAR + rollovers
    out = df.copy()
    out["year"] = year
    out["month_num"] = month_num
    out[PERIOD] = [f"{y}-{m:02d}" for y, m in zip(year, month_num)]
    out["period_idx"] = (year - FIRST_YEAR) * 12 + month_num - FIRST_MONTH
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Target to 0/1, structural missing values made explicit and the `contacted_before` flag."""
    out = df.copy()
    if TARGET in out.columns:
        out[TARGET] = (out[TARGET] == "yes").astype(int)
    never_contacted = out["pdays"] == -1
    out["poutcome"] = out["poutcome"].where(~(never_contacted & (out["poutcome"] == "unknown")), "nonexistent")
    out["contacted_before"] = (~never_contacted).astype(int)
    return out


def load_dataset(path=DATA_RAW) -> pd.DataFrame:
    return clean(add_period(load_raw(path)))


def split_random(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)


def split_oot(df: pd.DataFrame, train_end: str, test_start: str | None = None):
    """Out-of-time split: train on periods <= train_end, test on periods >= test_start (default: right after)."""
    train = df[df[PERIOD] <= train_end]
    test = df[df[PERIOD] > train_end] if test_start is None else df[df[PERIOD] >= test_start]
    return train, test
