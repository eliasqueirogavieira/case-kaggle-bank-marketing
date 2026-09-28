"""Evaluation metrics.

The production question is *ranking clients called in the same campaign*, so the headline metrics are
computed within each period (month): group AUC and lift@k. Pooled ROC-AUC, Gini, KS and PR-AUC are kept
for comparability, knowing they also reward telling periods apart.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from bank_marketing.config import PERIOD, REPORTS_DIR, SEED, TARGET


def group_auc(y, score, groups) -> float:
    """AUC over (positive, negative) pairs that belong to the same group, e.g. clients called in the same month.

    Each group adds its Mann-Whitney U (ties count 1/2); groups with a single class have no pairs.
    """
    y, score, groups = np.asarray(y), np.asarray(score, dtype=float), np.asarray(groups)
    wins = pairs = 0.0
    for g in np.unique(groups):
        in_g = groups == g
        yg = y[in_g]
        n_pos = yg.sum()
        n_neg = len(yg) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(score[in_g])
        wins += ranks[yg == 1].sum() - n_pos * (n_pos + 1) / 2
        pairs += n_pos * n_neg
    return float(wins / pairs) if pairs else float("nan")


def _group_masks(groups, n):
    if groups is None:
        yield np.ones(n, dtype=bool)
        return
    groups = np.asarray(groups)
    for g in np.unique(groups):
        yield groups == g


def _top_k_positives(y, score, k):
    """Expected positives among the top-k fraction; rows tied at the cut share the remaining slots evenly.

    Splitting the tie (instead of breaking it by row position) keeps the metric independent of row order,
    which matters for score-only models with many ties, such as the business rule.
    """
    top_n = max(1, int(round(k * len(score))))
    cut = np.sort(score)[::-1][top_n - 1]
    above, tied = score > cut, score == cut
    return y[above].sum() + (top_n - above.sum()) * y[tied].mean(), top_n


def lift_at_k(y, score, k=0.2, groups=None) -> float:
    """Conversions in the top-k fraction (of each group) divided by what the same number of random picks converts."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    captured = expected = 0.0
    for in_g in _group_masks(groups, len(y)):
        positives, top_n = _top_k_positives(y[in_g], score[in_g], k)
        captured += positives
        expected += top_n * y[in_g].mean()
    return float(captured / expected)


def capture_at_k(y, score, k=0.2, groups=None) -> float:
    """Share of all positives found in the top-k fraction (of each group)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    captured = sum(_top_k_positives(y[in_g], score[in_g], k)[0] for in_g in _group_masks(groups, len(y)))
    return float(captured / y.sum())


def ks_statistic(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def gini(y, score) -> float:
    return float(2 * roc_auc_score(y, score) - 1)


def decile_table(y, score, n_bins=10) -> pd.DataFrame:
    """Equal-sized score bins, bin 1 = highest scores: volume, conversion rate, lift and cumulative capture."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    bins = np.empty(len(y), dtype=int)
    bins[np.argsort(-score, kind="stable")] = np.arange(len(y)) * n_bins // len(y) + 1
    table = (
        pd.DataFrame({"bin": bins, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), positives=("y", "sum"))
        .reset_index()
    )
    table["rate"] = table["positives"] / table["n"]
    table["lift"] = table["rate"] / y.mean()
    table["cum_capture"] = table["positives"].cumsum() / y.sum()
    return table


def gains_curve(y, score, groups=None):
    """Cumulative share of positives captured vs share of rows contacted, starting at (0, 0).

    With groups, rows are ordered by their percentile rank inside their own group: contacting the top x%
    of every month at once, as a campaign would.
    """
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    key = score if groups is None else (
        pd.Series(score).groupby(np.asarray(groups)).rank(pct=True, method="first").to_numpy()
    )
    order = np.argsort(-key, kind="stable")
    contacted = np.arange(len(y) + 1) / len(y)
    captured = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    return contacted, captured


def paired_bootstrap(y, score_a, score_b, metric, n_boot=1000, seed=SEED, groups=None, alpha=0.05) -> dict:
    """Confidence interval of metric(a) - metric(b) on the same resampled rows.

    Resampling is stratified by class so every replicate keeps both classes.
    """
    y, a, b = np.asarray(y), np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    g = None if groups is None else np.asarray(groups)

    def diff(idx):
        if g is None:
            return metric(y[idx], a[idx]) - metric(y[idx], b[idx])
        return metric(y[idx], a[idx], g[idx]) - metric(y[idx], b[idx], g[idx])

    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    replicates = [
        diff(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])) for _ in range(n_boot)
    ]
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(diff(np.arange(len(y)))), "ci_low": float(low), "ci_high": float(high)}


def psi(expected, actual, bins=10, eps=1e-4) -> float:
    """Population Stability Index of `actual` against `expected`.

    Numeric variables with more distinct values than `bins` use expected-quantile bins; anything else
    (categories, flags, small counts) is compared value by value.
    """
    expected, actual = pd.Series(expected), pd.Series(actual)
    if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > bins:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e_dist = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a_dist = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    else:
        categories = sorted(set(expected) | set(actual))
        e_dist = expected.value_counts(normalize=True).reindex(categories, fill_value=0)
        a_dist = actual.value_counts(normalize=True).reindex(categories, fill_value=0)
    e_p = np.clip(e_dist.to_numpy(dtype=float), eps, None)
    a_p = np.clip(a_dist.to_numpy(dtype=float), eps, None)
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def model_metrics(model, df: pd.DataFrame, target: str = TARGET, period: str = PERIOD) -> dict:
    """Within-month ranking metrics (headline) plus pooled metrics, for any fitted model.

    Within-month metrics use the model's score (`decision_function`). Pooled metrics use probabilities when the
    model has them (for the month-offset model they include the month rate) and the score otherwise; Brier
    needs probabilities, so it is NaN for score-only models such as the business rule.
    """
    y = df[target].to_numpy()
    months = df[period].to_numpy()
    proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    score = model.decision_function(df) if hasattr(model, "decision_function") else proba
    pooled = proba if proba is not None else score
    return {
        "group_auc": group_auc(y, score, months),
        "lift_at_20_within_month": lift_at_k(y, score, 0.2, months),
        "capture_at_20_within_month": capture_at_k(y, score, 0.2, months),
        "roc_auc": float(roc_auc_score(y, pooled)),
        "gini": gini(y, pooled),
        "ks": ks_statistic(y, pooled),
        "pr_auc": float(average_precision_score(y, pooled)),
        "brier": float(brier_score_loss(y, proba)) if proba is not None else float("nan"),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def save_metrics(section: str, values: dict, path=None) -> None:
    """Write one section of reports/metrics.json, the single source of the numbers quoted in the deck and README."""
    path = Path(path) if path is not None else REPORTS_DIR / "metrics.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    path.write_text(text + "\n")


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")
