"""Cluster local support tickets; run a fictional offline demo with --demo."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from experiment import (
    BASE_RESULTS,
    OPENAI_API_KEY,
    PROJECT_ROOT,
    TRANSFORMERS,
    load_or_compute_embeddings,
    run_pipeline,
)
from io_utils import load_text_dataset


def save_results(frame, valid_mask, result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_labels = result["pred_labels"]
    keywords = result["keywords"]
    llm_labels = result["llm_labels"]

    def label_for(cid):
        if cid == -2:
            return "Not processed: empty text"
        if cid == -1:
            return "Noise"
        return llm_labels.get(cid) or " / ".join(keywords.get(cid, [])[:3]) or f"Cluster_{cid}"

    annotated = frame.copy()
    labels = np.full(len(frame), -2, dtype=int)
    labels[np.asarray(valid_mask)] = pred_labels
    annotated["cluster"] = labels
    annotated["cluster_label"] = [label_for(cid) for cid in labels]
    annotated.to_csv(output_dir / "annotated_dataset.csv", index=False, encoding="utf-8-sig")
    rows = []
    for cid in sorted(set(pred_labels)):
        rows.append(
            {
                "cluster_id": int(cid),
                "label": label_for(cid),
                "size": int((pred_labels == cid).sum()),
                "keywords": ", ".join(keywords.get(cid, [])),
                "examples": " | ".join(result["sample_texts"].get(cid, [])),
            }
        )
    summary = pd.DataFrame(rows).sort_values("size", ascending=False)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_excel(output_dir / "cluster_summary.xlsx", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="Use fictional tickets and TF-IDF/PCA/K-Means."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", type=Path, default=BASE_RESULTS / "inference")
    parser.add_argument("--model", choices=["tfidf", *TRANSFORMERS])
    parser.add_argument("--dim", choices=["umap", "pca", "tsne"])
    parser.add_argument("--cluster", choices=["hdbscan", "kmeans", "gmm"])
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=40)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--cluster-epsilon", type=float, default=0.02)
    parser.add_argument("--cluster-selection", choices=["eom", "leaf"], default="eom")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--llm-labels",
        action="store_true",
        help="Generate labels with the configured LLM provider.",
    )
    parser.add_argument(
        "--allow-text-upload",
        action="store_true",
        help="Allow ticket examples and keywords to be sent to that provider.",
    )
    args = parser.parse_args()
    if args.llm_labels and not args.allow_text_upload:
        parser.error(
            "LLM labels send ticket examples to the provider; add --allow-text-upload only for data you may share."
        )
    if args.llm_labels and not OPENAI_API_KEY:
        parser.error(
            "Set OPENAI_API_KEY in the environment or local .env before using --llm-labels."
        )
    if min(args.n_components, args.n_neighbors, args.n_clusters, args.min_samples) < 1:
        parser.error("Counts must be positive.")
    if args.min_cluster_size < 2 or not 0 <= args.min_dist <= 1 or args.cluster_epsilon < 0:
        parser.error("Invalid UMAP/HDBSCAN parameters.")

    input_path = args.input or PROJECT_ROOT / "datasets" / (
        "demo_tickets.csv" if args.demo else "cleaned_dataset.csv"
    )
    model = args.model or ("tfidf" if args.demo else "bge-m3")
    dim = args.dim or ("pca" if args.demo else "umap")
    cluster = args.cluster or ("kmeans" if args.demo else "hdbscan")
    frame, texts, valid_mask = load_text_dataset(input_path, args.text_column)
    if model == "tfidf":
        embeddings = (
            TfidfVectorizer(max_features=3000, ngram_range=(1, 2)).fit_transform(texts).toarray()
        )
    else:
        embeddings = load_or_compute_embeddings(texts, model, args.force_recompute)
    dim_params = {
        "umap": {
            "n_components": args.n_components,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
        },
        "pca": {"n_components": args.n_components},
        "tsne": {"n_components": 2},
    }[dim]
    clust_params = {
        "hdbscan": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "cluster_selection_epsilon": args.cluster_epsilon,
            "cluster_selection_method": args.cluster_selection,
        },
        "kmeans": {"n_clusters": min(args.n_clusters, len(texts) - 1)},
        "gmm": {"n_components": min(args.n_clusters, len(texts) - 1)},
    }[cluster]
    true_labels = None
    if "category" in frame and frame.loc[valid_mask, "category"].notna().all():
        true_labels = LabelEncoder().fit_transform(frame.loc[valid_mask, "category"].astype(str))
    result = run_pipeline(
        texts, embeddings, dim, cluster, dim_params, clust_params, true_labels, args.llm_labels
    )
    summary = save_results(frame, valid_mask, result, args.output_dir)
    metrics = {
        k: None if isinstance(v, float) and not np.isfinite(v) else v
        for k, v in result["metrics"].items()
    }
    run_summary = {
        "model": model,
        "dim_reduction": dim,
        "clustering": cluster,
        "n_texts": len(texts),
        "skipped_rows": int((~valid_mask).sum()),
        "dim_params": dim_params,
        "cluster_params": clust_params,
        "llm_enabled": args.llm_labels,
        "metrics": metrics,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Processed {len(texts)} texts: {model} + {dim} + {cluster}.")
    print(summary[["cluster_id", "size", "label"]].to_string(index=False))
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
