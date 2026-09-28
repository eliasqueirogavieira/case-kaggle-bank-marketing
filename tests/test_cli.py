import json

import joblib
import numpy as np
import pandas as pd
import pytest

from bank_marketing.data import load_dataset, load_raw
from bank_marketing.evaluation import save_metrics
from bank_marketing.models import make_lgbm
from bank_marketing.predict import main as predict_main
from bank_marketing.predict import score_batch
from bank_marketing.train import main as train_main


def test_save_metrics_merges_sections(tmp_path):
    path = tmp_path / "metrics.json"
    save_metrics("eda", {"base_rate": 0.117}, path)
    save_metrics("test", {"group_auc": 0.7}, path)
    save_metrics("eda", {"base_rate": 0.12}, path)
    assert json.loads(path.read_text()) == {"eda": {"base_rate": 0.12}, "test": {"group_auc": 0.7}}


@pytest.fixture(scope="module")
def model():
    df = load_dataset().sample(4000, random_state=0)
    return make_lgbm(n_estimators=30).fit(df, df["y"])


@pytest.fixture(scope="module")
def batch():
    return load_raw().sample(500, random_state=1).drop(columns="y")


def test_score_batch_ranks_every_client_and_flags_the_top(model, batch):
    out = score_batch(model, batch, top=0.2)
    assert sorted(out["rank"]) == list(range(1, 501))
    assert out.loc[out["rank"] == 1, "score"].item() == out["score"].max()
    assert out["priority"].sum() == 100
    assert out.loc[out["priority"], "score"].min() >= out.loc[~out["priority"], "score"].max()
    assert (out.loc[out["rank"] <= 50, "decile"] == 1).all()
    assert out["row_id"].tolist() == batch.index.tolist()


def test_score_batch_ignores_post_call_columns(model, batch):
    with_leaks = batch.assign(duration=4000, campaign=40)
    without = batch.drop(columns=["duration", "campaign", "day", "month"])
    np.testing.assert_array_equal(score_batch(model, with_leaks)["score"], score_batch(model, without)["score"])


def test_probability_needs_an_expected_base_rate(model, batch):
    assert "probability" not in score_batch(model, batch).columns
    p = score_batch(model, batch, base_rate=0.3)["probability"]
    assert p.between(0, 1).all()


def test_predict_cli_writes_the_scored_file(model, batch, tmp_path):
    joblib.dump(model, tmp_path / "model.joblib")
    batch.to_csv(tmp_path / "in.csv", sep=";", index=False)
    predict_main(["--input", str(tmp_path / "in.csv"), "--output", str(tmp_path / "out.csv"),
                  "--model", str(tmp_path / "model.joblib"), "--base-rate", "0.2"])
    out = pd.read_csv(tmp_path / "out.csv")
    assert len(out) == 500
    assert {"row_id", "score", "rank", "decile", "priority", "probability"} <= set(out.columns)


def test_train_cli_saves_model_metadata_and_test_metrics(tmp_path):
    params = tmp_path / "best_params.json"
    params.write_text(json.dumps({"n_estimators": 30, "learning_rate": 0.1}))
    train_main(["--params", str(params), "--models-dir", str(tmp_path), "--reports-dir", str(tmp_path)])

    restored = joblib.load(tmp_path / "model.joblib")
    assert restored.decision_function(load_dataset().head(3)).shape == (3,)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["excluded"] == ["duration", "campaign", "day", "month"]
    assert metadata["n_train"] == 36168 and metadata["n_test"] == 9043
    final = json.loads((tmp_path / "metrics.json").read_text())["final_model"]
    assert 0.5 < final["group_auc"] < 1
    assert {"roc_auc", "gini", "ks", "pr_auc", "brier", "lift_at_20_within_month"} <= final.keys()


def test_score_batch_can_add_reason_codes(model, batch):
    out = score_batch(model, batch, reasons=3)
    assert {"motivo_1", "motivo_2", "motivo_3"} <= set(out.columns)
    assert out["motivo_1"].str.contains("=").mean() > 0.9


def test_display_path_hides_the_absolute_project_location(tmp_path):
    from bank_marketing.config import MODELS_DIR
    from bank_marketing.train import display_path

    assert display_path(MODELS_DIR / "model.joblib") == "models/model.joblib"
    assert display_path(tmp_path / "m.joblib") == str(tmp_path / "m.joblib")


def test_deciles_are_right_for_small_batches(model, batch):
    assert score_batch(model, batch.head(1))["decile"].tolist() == [1]
    assert sorted(score_batch(model, batch.head(3))["decile"]) == [1, 4, 7]


def test_missing_input_columns_are_reported_by_name(model, batch):
    with pytest.raises(ValueError, match="pdays"):
        score_batch(model, batch.drop(columns="pdays"))


def test_reason_codes_never_show_age_to_the_agent(model, batch):
    out = score_batch(model, batch, reasons=3)
    codes = out[["motivo_1", "motivo_2", "motivo_3"]].to_numpy().ravel()
    assert not any(str(c).startswith("age=") for c in codes)
