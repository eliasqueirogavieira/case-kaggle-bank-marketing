"""Score a batch of clients before a campaign and produce the prioritized call list.

    uv run python -m bank_marketing.predict --input clients.csv --output call_list.csv --top 0.2 --base-rate 0.1
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bank_marketing.config import MODELS_DIR, RAW_INPUTS, REASON_EXCLUDE
from bank_marketing.data import clean
from bank_marketing.explain import feature_contributions, reason_codes


def score_batch(
    model, clients: pd.DataFrame, top: float = 0.2, base_rate: float | None = None, reasons: int = 0
) -> pd.DataFrame:
    """Rank clients by their within-campaign score and flag the `top` fraction as priority.

    The score is the model's log-odds relative to the campaign average: it orders clients, it is not a
    probability. Passing the campaign's expected conversion rate as `base_rate` adds probabilities.
    Columns the model does not use (e.g. `duration`) are ignored. `reasons > 0` adds that many reason codes
    (the variables pushing each client up, from exact TreeSHAP) for the call-center agent.
    """
    missing = [c for c in RAW_INPUTS if c not in clients.columns]
    if missing:
        raise ValueError(f"missing input columns: {', '.join(missing)}")
    X = clean(clients)
    score = model.decision_function(X)
    n = len(X)
    rank = pd.Series(score).rank(ascending=False, method="first").astype(int).to_numpy()
    out = pd.DataFrame(
        {
            "row_id": clients.index,
            "score": score,
            "rank": rank,
            "decile": (rank - 1) * 10 // n + 1,
            "priority": rank <= max(1, round(top * n)),
        }
    )
    if base_rate is not None:
        out["probability"] = model.predict_proba(X, base_rate=base_rate)[:, 1]
    if reasons:
        codes = reason_codes(feature_contributions(model, X), X, top=reasons, exclude=REASON_EXCLUDE)
        out = pd.concat([out, codes.set_axis(out.index)], axis=1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score clients and write the prioritized call list.")
    parser.add_argument("--input", type=Path, required=True, help="clients in the bank-full.csv layout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "model.joblib")
    parser.add_argument("--top", type=float, default=0.2, help="fraction of the batch flagged as priority")
    parser.add_argument("--base-rate", type=float, default=None, help="expected conversion rate of the campaign")
    parser.add_argument("--reasons", type=int, default=3, help="reason codes per client (0 to skip)")
    parser.add_argument("--sep", default=";")
    args = parser.parse_args(argv)

    clients = pd.read_csv(args.input, sep=args.sep)
    out = score_batch(joblib.load(args.model), clients, top=args.top, base_rate=args.base_rate, reasons=args.reasons)
    out.sort_values("rank").to_csv(args.output, index=False)
    print(f"{len(out)} clients scored, {int(out['priority'].sum())} flagged as priority -> {args.output}")


if __name__ == "__main__":
    main()
