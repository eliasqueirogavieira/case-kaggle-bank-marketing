"""Explanations of the month-offset model: exact TreeSHAP per original variable and reason codes."""

import numpy as np
import pandas as pd

from bank_marketing.features import feature_origin


def feature_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions to the within-month score, summed over the columns of each original variable.

    SHAP values are additive, so one-hot columns of the same variable can be summed; each row adds up to the
    model's `decision_function` (the `bias` column holds the expected score).
    """
    raw = model.contributions(X)
    origin = feature_origin(model.preprocessor_)
    grouped = raw.drop(columns="bias").T.groupby(origin).sum().T
    grouped["bias"] = raw["bias"]
    return grouped


def reason_codes(contribs: pd.DataFrame, X: pd.DataFrame, top: int = 3, exclude=()) -> pd.DataFrame:
    """The `top` variables pushing each client's score up, as "variable=value" (blank when fewer push up).

    Variables in `exclude` still count in the score but are never shown as a reason.
    """
    values = contribs.drop(columns=["bias", *exclude])
    order = np.argsort(-values.to_numpy(), axis=1, kind="stable")[:, :top]
    names = values.columns.to_numpy()[order]
    pushes = np.take_along_axis(values.to_numpy(), order, axis=1)
    codes = [
        [f"{name}={X.iloc[i][name]}" if push > 0 else "" for name, push in zip(row_names, row_pushes)]
        for i, (row_names, row_pushes) in enumerate(zip(names, pushes))
    ]
    return pd.DataFrame(codes, index=contribs.index, columns=[f"motivo_{k + 1}" for k in range(top)])
