import os
import json
import time
import itertools
import argparse
import hashlib
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import CountVectorizer
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from dotenv import load_dotenv
from io_utils import configure_console_utf8, load_text_dataset, require_columns

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

CLEANED_DATASET = PROJECT_ROOT / "datasets/cleaned_dataset.csv"
TEST_DATASET = PROJECT_ROOT / "datasets/test_dataset.csv"
BASE_RESULTS = PROJECT_ROOT / "results"

TRANSFORMERS = {
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "rubert": "cointegrated/rubert-tiny2",
}

N_OPTUNA_TRIALS = 30


def load_data(path=TEST_DATASET):
    """Загружает тестовый датасеты из CSV-файлов"""

    test, _, valid_mask = load_text_dataset(path)
    require_columns(test, ["category"])
    test = test.loc[valid_mask].copy()
    if test["category"].isna().any():
        raise ValueError("Every benchmark text must have a category.")
    test["text"] = test["text"].astype(str).str.strip()
    if len(test) < 10 or test["category"].nunique() < 2:
        raise ValueError("Benchmark requires at least 10 texts and two categories.")

    return test


def get_embeddings(texts, model_name, batch_size=64):
    """Вычисляет эмбеддинги текстов с помощью SentenceTransformer"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings


def embedding_cache_path(texts, model_key, cache_dir=None):
    """Cache identity includes model, ordered input texts and encoding settings."""
    identity = json.dumps(
        {"model": TRANSFORMERS[model_key], "texts": list(texts), "normalize": True, "version": 1},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", model_key)
    return Path(cache_dir or BASE_RESULTS) / f"embeddings_{slug}_{digest}.npy"


def load_or_compute_embeddings(texts, model_key, force_recompute=False, cache_dir=None):
    """Reuse embeddings only for the same model and exact ordered texts."""
    cache_path = embedding_cache_path(texts, model_key, cache_dir)
    if not force_recompute and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if cached.ndim == 2 and cached.shape[0] == len(texts) and np.isfinite(cached).all():
                return cached
        except (ValueError, OSError):
            pass
    embeddings = get_embeddings(texts, TRANSFORMERS[model_key])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def reduce_umap(
    embeddings, n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42
):
    """Снижение размерности методом UMAP"""
    import umap

    if len(embeddings) < 4:
        raise ValueError("UMAP requires at least four texts; use PCA for smaller datasets.")
    reducer = umap.UMAP(
        n_components=min(n_components, len(embeddings) - 2),
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def reduce_pca(embeddings, n_components=50, random_state=42):
    """Снижение размерности методом PCA"""
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)
    reducer = PCA(n_components=n_components, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)

    return reduced


def reduce_tsne(embeddings, n_components=2, perplexity=30.0, random_state=42):
    """Снижение размерности методом t-SNE"""
    if embeddings.shape[1] > 50:
        components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        embeddings = PCA(n_components=components, random_state=random_state).fit_transform(
            embeddings
        )

    max_perplexity = min(embeddings.shape[0] - 1.0, max(1.0, embeddings.shape[0] / 3.0 - 1.0))
    perplexity = min(perplexity, max_perplexity)

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)

    return reduced


def cluster_hdbscan(
    reduced,
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_epsilon=0.0,
    cluster_selection_method="eom",
):
    """Кластеризация методом HDBSCAN"""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    return labels


def cluster_kmeans(reduced, n_clusters=10, random_state=42):
    """Кластеризация методом K-Means"""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(reduced)

    return labels


def cluster_gmm(reduced, n_components=10, covariance_type="full", random_state=42):
    """Кластеризация методом Gaussian Mixture Model"""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
    )
    labels = gmm.fit_predict(reduced)

    return labels


def extract_keywords_ctfidf(texts, labels, top_n=10):
    STOPWORDS = {
        "тема",
        "сообщение",
        "тему",
        "добрый",
        "день",
        "здравствуйте",
        "привет",
        "пожалуйста",
        "помогите",
        "помощь",
        "хотел",
        "хочу",
        "нужна",
        "нужно",
        "нужен",
        "это",
        "есть",
        "был",
        "была",
        "были",
        "буду",
        "будет",
        "subject",
        "message",
        "hello",
        "please",
        "help",
        "что",
        "как",
        "для",
        "при",
        "его",
        "ваш",
        "вас",
        "вам",
        "мне",
        "меня",
        "мой",
        "моя",
        "мои",
        "уже",
        "все",
        "это",
    }

    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    cluster_docs = {}
    for lbl in unique_labels:
        cluster_texts = [texts[i] for i, label in enumerate(labels) if label == lbl]
        cluster_docs[lbl] = " ".join(cluster_texts)

    if not cluster_docs:
        return {}

    doc_list = list(cluster_docs.values())
    label_list = list(cluster_docs.keys())

    vectorizer = CountVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[а-яёА-ЯЁa-zA-Z][а-яёА-ЯЁa-zA-Z]{2,}\b",
        stop_words=list(STOPWORDS),
    )
    try:
        tf_matrix = vectorizer.fit_transform(doc_list).toarray().astype(float)
    except ValueError:
        return {lbl: [] for lbl in label_list}
    vocab = np.array(vectorizer.get_feature_names_out())

    tf = tf_matrix / (tf_matrix.sum(axis=1, keepdims=True) + 1e-9)
    n_docs = len(doc_list)
    df = (tf_matrix > 0).sum(axis=0)
    idf = np.log(1 + n_docs / (df + 1))
    ctfidf = tf * idf

    keywords = {}
    for i, lbl in enumerate(label_list):
        top_idx = ctfidf[i].argsort()[::-1][:top_n]
        keywords[lbl] = [vocab[j] for j in top_idx if tf_matrix[i, j] > 0]

    return keywords


def generate_labels_gpt(keywords, sample_texts, api_key=None, model=OPENAI_MODEL):
    """Генерирует названия кластеров с помощью LLM по ключевым словам и примерам"""
    from openai import OpenAI

    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        raise ValueError("LLM labels require OPENAI_API_KEY in the environment or local .env.")
    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, timeout=30, max_retries=2)

    labels = {}
    for cluster_id, kws in keywords.items():
        samples = sample_texts.get(cluster_id, [])[:3]
        samples_str = "\n---\n".join(samples) if samples else "(нет примеров)"

        prompt = f"""Ты аналитик технической поддержки. Тебе даны ключевые слова и примеры обращений из одного кластера.
Придумай краткое (2-5 слов) русскоязычное название категории, отражающее суть этих обращений.
Ответь ТОЛЬКО названием категории, без пояснений.

Ключевые слова: {", ".join(kws)}

Примеры обращений:
{samples_str}

Название категории:"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.2,
            )
            label = response.choices[0].message.content.strip().strip('"').strip("'")
            labels[cluster_id] = label
            time.sleep(0.3)
        except Exception:
            warnings.warn("LLM label unavailable; using local keywords. Provider details omitted.")
            labels[cluster_id] = ", ".join(kws[:3])

    return labels


def evaluate_clustering(pred_labels, true_labels, reduced_embeddings):
    """Вычисляет метрики кластеризации: ARI, NMI, Silhouette, шум"""
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    mask = pred_labels != -1
    noise_pct = (~mask).mean() * 100

    metrics = {
        "noise_pct": noise_pct,
        "n_clusters": int(len(set(pred_labels)) - (1 if -1 in pred_labels else 0)),
    }

    if mask.sum() > 1 and len(set(pred_labels[mask])) > 1:
        metrics["ari"] = adjusted_rand_score(true_labels[mask], pred_labels[mask])
        metrics["nmi"] = normalized_mutual_info_score(true_labels[mask], pred_labels[mask])
        try:
            metrics["silhouette"] = silhouette_score(
                reduced_embeddings[mask], pred_labels[mask], metric="euclidean"
            )
        except Exception:
            metrics["silhouette"] = float("nan")
    else:
        metrics["ari"] = 0.0
        metrics["nmi"] = 0.0
        metrics["silhouette"] = float("nan")

    return metrics


def run_pipeline(
    texts,
    embeddings,
    dim_method,
    clust_method,
    dim_params,
    clust_params,
    true_labels=None,
    generate_llm_labels=False,
    api_key=None,
):
    """Запускает полный пайплайн: снижение размерности → кластеризация → ключевые слова → LLM-метки"""
    if dim_method == "umap":
        reduced = reduce_umap(embeddings, **dim_params)
    elif dim_method == "pca":
        reduced = reduce_pca(embeddings, **dim_params)
    elif dim_method == "tsne":
        reduced = reduce_tsne(embeddings, **dim_params)
    else:
        raise ValueError(f"Unknown dim_method: {dim_method}")

    if clust_method == "hdbscan":
        pred_labels = cluster_hdbscan(reduced, **clust_params)
    elif clust_method == "kmeans":
        pred_labels = cluster_kmeans(reduced, **clust_params)
    elif clust_method == "gmm":
        pred_labels = cluster_gmm(reduced, **clust_params)
    else:
        raise ValueError(f"Unknown clust_method: {clust_method}")

    keywords = extract_keywords_ctfidf(texts, pred_labels, top_n=10)

    sample_texts = {}
    for lbl in set(pred_labels):
        if lbl == -1:
            continue
        sample_texts[lbl] = [texts[i] for i, label in enumerate(pred_labels) if label == lbl][:5]

    llm_labels = {}
    if generate_llm_labels:
        llm_labels = generate_labels_gpt(keywords, sample_texts, api_key=api_key)

    metrics = {}
    if true_labels is not None:
        metrics = evaluate_clustering(pred_labels, true_labels, reduced)

    return {
        "pred_labels": pred_labels,
        "reduced": reduced,
        "keywords": keywords,
        "llm_labels": llm_labels,
        "sample_texts": sample_texts,
        "metrics": metrics,
    }


def make_optuna_objective(texts, embeddings, true_labels, dim_method, clust_method):
    """Создаёт целевую функцию Optuna для подбора гиперпараметров пайплайна"""

    def objective(trial):
        if dim_method == "umap":
            dim_params = {
                "n_components": trial.suggest_int("umap_n_components", 5, 30),
                "n_neighbors": trial.suggest_int("umap_n_neighbors", 5, 50),
                "min_dist": trial.suggest_float("umap_min_dist", 0.0, 0.1),
            }
        elif dim_method == "pca":
            dim_params = {
                "n_components": trial.suggest_int("pca_n_components", 10, 100),
            }
        elif dim_method == "tsne":
            max_perp = max(5.0, embeddings.shape[0] / 3.0 - 1.0)
            dim_params = {
                "n_components": trial.suggest_int("tsne_n_components", 2, 3),
                "perplexity": trial.suggest_float("tsne_perplexity", 5.0, min(50.0, max_perp)),
            }
        else:
            dim_params = {}

        if clust_method == "hdbscan":
            if dim_method == "tsne":
                mcs_max = 10
            else:
                mcs_max = 30
            clust_params = {
                "min_cluster_size": trial.suggest_int("hdbscan_min_cluster_size", 2, mcs_max),
                "min_samples": trial.suggest_int("hdbscan_min_samples", 1, 10),
                "cluster_selection_epsilon": trial.suggest_float("hdbscan_epsilon", 0.0, 1.0),
                "cluster_selection_method": trial.suggest_categorical(
                    "hdbscan_method", ["eom", "leaf"]
                ),
            }
        elif clust_method == "kmeans":
            clust_params = {
                "n_clusters": trial.suggest_int("kmeans_n_clusters", 3, min(25, len(texts) - 1)),
            }
        elif clust_method == "gmm":
            clust_params = {
                "n_components": trial.suggest_int("gmm_n_components", 3, min(25, len(texts) - 1)),
                "covariance_type": trial.suggest_categorical(
                    "gmm_covariance_type", ["full", "tied", "diag"]
                ),
            }
        else:
            clust_params = {}

        try:
            result = run_pipeline(
                texts=texts,
                embeddings=embeddings,
                dim_method=dim_method,
                clust_method=clust_method,
                dim_params=dim_params,
                clust_params=clust_params,
                true_labels=true_labels,
                generate_llm_labels=False,
            )
            m = result["metrics"]
            nmi = m.get("nmi", 0.0)
            noise_pct = m.get("noise_pct", 100.0)
            n_clusters = m.get("n_clusters", 0)

            noise_penalty = max(0, noise_pct - 40) * 0.004
            cluster_penalty = 0.15 if n_clusters < 3 else 0.0

            score = nmi - noise_penalty - cluster_penalty
            return score
        except (ValueError, RuntimeError):
            return -1.0

    return objective


def run_benchmark(
    test_texts,
    test_labels,
    n_trials=N_OPTUNA_TRIALS,
    models=None,
    dim_methods=None,
    clust_methods=None,
):
    """Запускает эксперимент для всех комбинаций (трансформер × dim × cluster) с Optuna"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dim_methods = dim_methods or ["umap", "pca", "tsne"]
    clust_methods = clust_methods or ["hdbscan", "kmeans", "gmm"]

    rows = []
    combo_idx = 0

    for model_key in models or TRANSFORMERS:
        test_emb = load_or_compute_embeddings(test_texts, model_key)

        for dim_m, clust_m in itertools.product(dim_methods, clust_methods):
            combo_idx += 1

            t_start = time.time()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                study_name=f"{model_key}_{dim_m}_{clust_m}",
            )
            objective_fn = make_optuna_objective(
                texts=test_texts,
                embeddings=test_emb,
                true_labels=test_labels,
                dim_method=dim_m,
                clust_method=clust_m,
            )
            study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
            if study.best_value == -1.0:
                raise ValueError(f"All trials failed for {model_key}/{dim_m}/{clust_m}.")

            best_params = study.best_params
            elapsed = time.time() - t_start

            dim_params, clust_params = _split_params(best_params, dim_m, clust_m)
            try:
                final = run_pipeline(
                    texts=test_texts,
                    embeddings=test_emb,
                    dim_method=dim_m,
                    clust_method=clust_m,
                    dim_params=dim_params,
                    clust_params=clust_params,
                    true_labels=test_labels,
                    generate_llm_labels=False,
                )
                m = final["metrics"]
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"Best trial failed for {model_key}/{dim_m}/{clust_m}.") from exc

            row = {
                "Transformer": model_key,
                "Dim_Reduction": dim_m,
                "Clustering": clust_m,
                "Best_Params": json.dumps(best_params),
                "ARI": round(m.get("ari", 0.0), 4),
                "NMI": round(m.get("nmi", 0.0), 4),
                "Silhouette": round(m.get("silhouette", float("nan")), 4),
                "N_Clusters": m.get("n_clusters", 0),
                "Noise_Pct": round(m.get("noise_pct", 0.0), 2),
                "Time_Execution": round(elapsed, 1),
                "Optuna_Score": round(study.best_value, 4),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _split_params(best_params, dim_m, clust_m):
    """Разделяет словарь параметров Optuna на dim_params и clust_params"""
    prefix_map = {
        "umap": ["umap_n_components", "umap_n_neighbors", "umap_min_dist"],
        "pca": ["pca_n_components"],
        "tsne": ["tsne_n_components", "tsne_perplexity"],
        "hdbscan": [
            "hdbscan_min_cluster_size",
            "hdbscan_min_samples",
            "hdbscan_epsilon",
            "hdbscan_method",
        ],
        "kmeans": ["kmeans_n_clusters"],
        "gmm": ["gmm_n_components", "gmm_covariance_type"],
    }

    rename = {
        "umap_n_components": "n_components",
        "umap_n_neighbors": "n_neighbors",
        "umap_min_dist": "min_dist",
        "pca_n_components": "n_components",
        "tsne_n_components": "n_components",
        "tsne_perplexity": "perplexity",
        "hdbscan_min_cluster_size": "min_cluster_size",
        "hdbscan_min_samples": "min_samples",
        "hdbscan_epsilon": "cluster_selection_epsilon",
        "hdbscan_method": "cluster_selection_method",
        "kmeans_n_clusters": "n_clusters",
        "gmm_n_components": "n_components",
        "gmm_covariance_type": "covariance_type",
    }

    dim_keys = set(prefix_map.get(dim_m, []))
    clust_keys = set(prefix_map.get(clust_m, []))

    dim_params = {rename[k]: v for k, v in best_params.items() if k in dim_keys}
    clust_params = {rename[k]: v for k, v in best_params.items() if k in clust_keys}

    return dim_params, clust_params


def plot_results(results_df, save_path):
    """Строит и сохраняет графики результатов эксперимента"""
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(20, 14))

    ax_bar = fig.add_subplot(2, 1, 1)
    df_sorted = results_df.sort_values("NMI", ascending=False).reset_index(drop=True)
    combo_labels = df_sorted.apply(
        lambda r: f"{r['Transformer']}\n{r['Dim_Reduction']}+{r['Clustering']}", axis=1
    )
    colors = sns.color_palette("viridis", len(df_sorted))
    bars = ax_bar.bar(
        range(len(df_sorted)), df_sorted["NMI"], color=colors, edgecolor="white", linewidth=0.5
    )
    ax_bar.set_xticks(range(len(df_sorted)))

    ax_bar.set_xticklabels(combo_labels, fontsize=11, rotation=45, ha="right")
    ax_bar.set_ylabel("NMI", fontsize=12)
    ax_bar.set_title("Ранжирование всех комбинаций по NMI", fontsize=13)
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    for bar, val in zip(bars, df_sorted["NMI"]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    model_keys = list(results_df["Transformer"].unique())
    n_transformers = len(model_keys)
    for i, model_key in enumerate(model_keys):
        ax_hm = fig.add_subplot(2, n_transformers, n_transformers + i + 1)
        sub = results_df[results_df["Transformer"] == model_key]
        pivot = sub.pivot(index="Dim_Reduction", columns="Clustering", values="NMI")
        is_last = i == n_transformers - 1
        sns.heatmap(
            pivot,
            ax=ax_hm,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            vmax=results_df["NMI"].max() + 0.05,
            linewidths=0.5,
            cbar=is_last,
        )
        ax_hm.set_title(f"{model_key}", fontsize=11, fontweight="bold")
        ax_hm.set_xlabel("Кластеризация")
        ax_hm.set_ylabel("Снижение размерности" if i == 0 else "")
        if is_last:
            cbar = ax_hm.collections[0].colorbar
            cbar.ax.set_xlabel("NMI", fontsize=10, labelpad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    configure_console_utf8()
    parser = argparse.ArgumentParser(description="Benchmark support-ticket clustering pipelines.")
    parser.add_argument("--input", type=Path, default=TEST_DATASET)
    parser.add_argument("--output-dir", type=Path, default=BASE_RESULTS / "benchmark")
    parser.add_argument("--trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--models", nargs="+", choices=list(TRANSFORMERS))
    parser.add_argument("--dim-methods", nargs="+", choices=["umap", "pca", "tsne"])
    parser.add_argument("--cluster-methods", nargs="+", choices=["hdbscan", "kmeans", "gmm"])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    test_df = load_data(args.input)
    test_texts = test_df["text"].tolist()
    test_labels = LabelEncoder().fit_transform(test_df["category"])

    results_df = run_benchmark(
        test_texts=test_texts,
        test_labels=test_labels,
        n_trials=args.trials,
        models=args.models,
        dim_methods=args.dim_methods,
        clust_methods=args.cluster_methods,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_dir / "benchmark_results.csv", index=False)
    xlsx_path = args.output_dir / "benchmark_results.xlsx"
    results_df.to_excel(xlsx_path, index=False)

    plot_path = args.output_dir / "benchmark_nmi_chart.png"
    plot_results(results_df, plot_path)

    best_row = results_df.loc[results_df["NMI"].idxmax()]
    best_params = json.loads(best_row["Best_Params"])

    summary = {
        "best_combination": {
            "transformer": best_row["Transformer"],
            "dim_reduction": best_row["Dim_Reduction"],
            "clustering": best_row["Clustering"],
        },
        "best_params": best_params,
        "metrics": {
            "nmi": float(best_row["NMI"]),
            "ari": float(best_row["ARI"]),
            "silhouette": float(best_row["Silhouette"]),
        },
    }
    with open(args.output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"Лучшая комбинация: {best_row['Transformer']} | {best_row['Dim_Reduction']} | {best_row['Clustering']}"
    )
    print(f"NMI = {best_row['NMI']:.4f}, ARI = {best_row['ARI']:.4f}")


if __name__ == "__main__":
    main()
