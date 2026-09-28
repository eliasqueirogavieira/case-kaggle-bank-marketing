"""Preprocessing pipelines: one for the linear model, one for tree models.

Transform functions live at module level (not lambdas) so fitted pipelines pickle with joblib.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from bank_marketing.config import BINARY, CATEGORICAL, ENGINEERED, NUMERIC, PERIOD

AGE_EDGES = [0, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def signed_log1p(X):
    """log1p that keeps the sign: tames the long tails of `balance`, which can be negative."""
    return np.sign(X) * np.log1p(np.abs(X))


def log1p_positive(X):
    """log1p of the positive part: `pdays = -1` (never contacted) maps to 0; the flag carries that information."""
    return np.log1p(np.clip(X, 0, None))


def age_band(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.cut(X[c], AGE_EDGES, labels=AGE_LABELS).astype(str) for c in X.columns}, index=X.index
    )


def _onehot(**kwargs) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)


def build_preprocessor(
    kind: Literal["linear", "tree"],
    period_fe: bool = False,
    extra_numeric: Sequence[str] = (),
    extra_categorical: Sequence[str] = (),
) -> ColumnTransformer:
    """Column transformer over the F0 features; anything else in the input (e.g. `duration`) is dropped.

    - linear: one-hot, age bands (readable odds ratios), log transforms for long tails, standardization;
      optional one-hot of the period = month fixed effects.
    - tree: one-hot categoricals, raw numerics (trees are invariant to monotonic transforms and scale).
      `extra_*` add excluded columns back, only for the ablation that measures what they inflate.
    """
    if (extra_numeric or extra_categorical) and kind != "tree":
        raise ValueError("extra columns are only supported for the tree preprocessor (ablation)")
    if kind == "linear":
        transformers = [
            ("cat", _onehot(), CATEGORICAL),
            ("bin", _onehot(drop="if_binary"), BINARY),
            ("age", Pipeline([("band", FunctionTransformer(age_band, feature_names_out="one-to-one")),
                              ("onehot", _onehot())]), ["age"]),
            ("balance", Pipeline([("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                                  ("scale", StandardScaler())]), ["balance"]),
            ("history", Pipeline([("log", FunctionTransformer(log1p_positive, feature_names_out="one-to-one")),
                                  ("scale", StandardScaler())]), ["pdays", "previous"]),
            ("flag", "passthrough", ENGINEERED),
        ]
        if period_fe:
            transformers.append(("period", _onehot(), [PERIOD]))
    elif kind == "tree":
        transformers = [
            ("cat", _onehot(), CATEGORICAL + BINARY + list(extra_categorical)),
            ("num", "passthrough", NUMERIC + ENGINEERED + list(extra_numeric)),
        ]
    else:
        raise ValueError(f"unknown preprocessor kind: {kind!r}")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False).set_output(
        transform="pandas"
    )


def feature_origin(pre: ColumnTransformer) -> dict[str, str]:
    """Map each output column of a fitted preprocessor to the raw variable it came from.

    One-hot columns are attributed through the encoder's `categories_`, never by splitting names
    (categories such as "blue-collar" or "admin." contain separators).
    """
    origin: dict[str, str] = {}
    for name, transformer, columns in pre.transformers_:
        if name == "remainder" or isinstance(transformer, str) and transformer == "drop":
            continue
        if isinstance(transformer, str) and transformer == "passthrough":
            origin.update({c: c for c in columns})
            continue
        out_names = list(transformer.get_feature_names_out(columns))
        encoder = transformer[-1] if isinstance(transformer, Pipeline) else transformer
        if isinstance(encoder, OneHotEncoder):
            dropped = encoder.drop_idx_
            sizes = [
                len(cats) - int(dropped is not None and dropped[i] is not None)
                for i, cats in enumerate(encoder.categories_)
            ]
            origin.update(zip(out_names, np.repeat(encoder.feature_names_in_, sizes)))
        else:
            origin.update(zip(out_names, columns))
    return origin
