"""Render an aggregate-only chart from the public historical benchmark CSV."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    frame = pd.read_csv(ROOT / "docs/benchmark_results.csv")
    top = frame.sort_values("NMI", ascending=False).head(9).iloc[::-1]
    labels = top.apply(
        lambda row: (
            f"{row['Transformer']} · {row['Dim_Reduction'].upper()} + {row['Clustering'].upper()}"
        ),
        axis=1,
    )
    colors = [
        {"bge-m3": "#2878a0", "minilm": "#37a091", "rubert": "#d39b44"}[model]
        for model in top["Transformer"]
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (bars_ax, coverage_ax) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3.4, 1.3]}
    )
    fig.patch.set_facecolor("#f8fafc")
    for ax in (bars_ax, coverage_ax):
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e2e8f0")
    bars_ax.barh(range(len(top)), top["NMI"], color=colors, height=0.62)
    bars_ax.set_yticks(range(len(top)), labels)
    bars_ax.set_xlim(0, 0.65)
    bars_ax.set_xlabel("NMI on non-noise tickets")
    for i, value in enumerate(top["NMI"]):
        bars_ax.text(value + 0.009, i, f"{value:.3f}", va="center", fontsize=10)
    coverage = 100 - top["Noise_Pct"]
    coverage_ax.barh(range(len(top)), coverage, color="#94a3b8", height=0.62)
    coverage_ax.set_yticks(range(len(top)), [""] * len(top))
    coverage_ax.set_xlim(0, 115)
    coverage_ax.set_xticks([0, 50, 100])
    coverage_ax.set_xlabel("Tickets retained (%)")
    for i, value in enumerate(coverage):
        coverage_ax.text(value + 2, i, f"{value:g}%", va="center", fontsize=10)
    fig.suptitle(
        "Support ticket clustering · historical benchmark",
        x=0.05,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.05,
        0.895,
        "Top 9 of 27 pipelines · 200 labeled tickets · 30 Optuna trials per pipeline",
        color="#64748b",
        fontsize=11,
    )
    fig.text(
        0.05,
        0.035,
        "Exploratory results: tuned and evaluated on the same private sample. Raw tickets are excluded.",
        color="#64748b",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.35, right=0.97, top=0.83, bottom=0.14, wspace=0.12)
    fig.savefig(ROOT / "docs/benchmark_nmi_chart.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
