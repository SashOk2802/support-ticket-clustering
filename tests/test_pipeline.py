import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

import experiment
from apply_pipeline import save_results
from io_utils import load_text_dataset
from preprocessing import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_semicolon_csv_preserves_blank_row_alignment(tmp_path):
    source = tmp_path / "tickets.csv"
    pd.DataFrame(
        {"id": [1, 2, 3, 4], "text": ["dns record", None, "mail delivery", "invoice payment"]}
    ).to_csv(source, sep=";", index=False)
    frame, texts, mask = load_text_dataset(source)
    assert len(frame) == 4
    assert texts == ["dns record", "mail delivery", "invoice payment"]
    assert mask.tolist() == [True, False, True, True]


def test_single_column_csv_does_not_guess_letters_as_delimiters(tmp_path):
    source = tmp_path / "texts.csv"
    pd.DataFrame({"text": ["dns record", "mail delivery", "invoice payment"]}).to_csv(
        source, index=False
    )
    _, texts, _ = load_text_dataset(source)
    assert texts == ["dns record", "mail delivery", "invoice payment"]


def test_preprocessing_uses_first_user_message_and_exports_only_required_fields():
    tickets = pd.DataFrame({"ticket_id": [1], "subject": ["Оплата"], "private_note": ["local"]})
    messages = pd.DataFrame(
        {
            "ticket_id": [1, 1, 1],
            "ticket_message_author": ["Agent", "User", "User"],
            "ticket_message": ["Ответ оператора", "  Счёт  не оплачен  ", "Позднее сообщение"],
        }
    )
    result = prepare_dataset(tickets, messages)
    assert list(result.columns) == ["ticket_id", "text"]
    assert result.iloc[0]["text"] == "Оплата\n Счёт не оплачен"


def test_cache_never_reuses_same_length_different_or_reordered_texts(tmp_path, monkeypatch):
    calls = []

    def encode(texts, model):
        calls.append(tuple(texts))
        return np.full((len(texts), 3), len(calls), dtype=float)

    monkeypatch.setattr(experiment, "get_embeddings", encode)
    first = experiment.load_or_compute_embeddings(["a", "b"], "minilm", cache_dir=tmp_path)
    reused = experiment.load_or_compute_embeddings(["a", "b"], "minilm", cache_dir=tmp_path)
    other = experiment.load_or_compute_embeddings(["c", "d"], "minilm", cache_dir=tmp_path)
    reversed_ = experiment.load_or_compute_embeddings(["b", "a"], "minilm", cache_dir=tmp_path)
    assert len(calls) == 3
    np.testing.assert_array_equal(first, reused)
    assert not np.array_equal(first, other)
    assert not np.array_equal(first, reversed_)


def test_corrupted_cache_is_recomputed(tmp_path, monkeypatch):
    path = experiment.embedding_cache_path(["text"], "minilm", tmp_path)
    path.write_bytes(b"invalid array")
    monkeypatch.setattr(experiment, "get_embeddings", lambda texts, model: np.ones((1, 2)))
    result = experiment.load_or_compute_embeddings(["text"], "minilm", cache_dir=tmp_path)
    assert result.shape == (1, 2)


@pytest.mark.parametrize(
    "texts, labels, expected",
    [
        (["сервер память", "сервер процессор"], [0, 0], {0}),
        (["dns запись", "почта письмо"], [0, 1], {0, 1}),
        (["сервер память", "почта письмо"], [-1, -1], set()),
        (["123", "!"], [0, 0], {0}),
    ],
)
def test_keywords_handle_single_cluster_unique_vocabulary_noise_and_empty_vocabulary(
    texts, labels, expected
):
    keywords = experiment.extract_keywords_ctfidf(texts, np.array(labels))
    assert set(keywords) == expected
    if labels == [0, 1]:
        assert "dns" in keywords[0] and "почта" not in keywords[0]
    if texts == ["123", "!"]:
        assert keywords[0] == []


def test_all_noise_exports_distinguish_empty_rows(tmp_path):
    frame = pd.DataFrame({"text": ["dns", None, "mail", "billing"]})
    result = {
        "pred_labels": np.array([-1, -1, -1]),
        "keywords": {},
        "llm_labels": {},
        "sample_texts": {},
    }
    summary = save_results(frame, frame["text"].notna(), result, tmp_path)
    annotated = pd.read_csv(tmp_path / "annotated_dataset.csv")
    assert annotated["cluster"].tolist() == [-1, -2, -1, -1]
    assert summary.iloc[0]["size"] == 3
    assert (tmp_path / "cluster_summary.xlsx").exists()


def test_metrics_report_noise_separately():
    result = experiment.evaluate_clustering(
        np.array([0, 0, 1, 1, -1]),
        np.array([0, 0, 1, 1, 0]),
        np.array([[0, 0], [0, 1], [10, 10], [10, 11], [5, 5]]),
    )
    assert result["noise_pct"] == 20.0
    assert result["nmi"] == result["ari"] == 1.0
    assert result["n_clusters"] == 2


def test_tsne_handles_fewer_than_fifty_samples():
    embeddings = np.random.default_rng(42).normal(size=(8, 64))
    reduced = experiment.reduce_tsne(embeddings)
    assert reduced.shape == (8, 2)
    assert np.isfinite(reduced).all()


def test_real_pca_kmeans_pipeline():
    embeddings, labels = make_blobs(n_samples=30, n_features=6, centers=3, random_state=42)
    texts = [f"server category {label}" for label in labels]
    result = experiment.run_pipeline(
        texts, embeddings, "pca", "kmeans", {"n_components": 3}, {"n_clusters": 3}, labels
    )
    assert result["metrics"]["ari"] == 1.0
    assert len(result["pred_labels"]) == len(texts)
    assert result["llm_labels"] == {}


def test_llm_cli_requires_explicit_text_upload_permission():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "apply_pipeline.py"), "--demo", "--llm-labels"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "--allow-text-upload" in proc.stderr


def test_demo_cli_runs_without_credentials_or_model_downloads(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "HF_TOKEN"}}
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "apply_pipeline.py"), "--demo", "--output-dir", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert result["n_texts"] == 32 and result["llm_enabled"] is False
    assert result["model"] == "tfidf"
    assert len(pd.read_csv(tmp_path / "annotated_dataset.csv")) == 32


def test_benchmark_reports_all_failed_trials_instead_of_fake_metrics(monkeypatch):
    monkeypatch.setattr(
        experiment, "load_or_compute_embeddings", lambda texts, model: np.ones((10, 3))
    )

    def fail(*args, **kwargs):
        raise ValueError("invalid trial")

    monkeypatch.setattr(experiment, "run_pipeline", fail)
    with pytest.raises(ValueError, match="All trials failed"):
        experiment.run_benchmark(
            ["ticket"] * 10,
            np.zeros(10),
            n_trials=1,
            models=["minilm"],
            dim_methods=["pca"],
            clust_methods=["kmeans"],
        )
