"""
Visualization utilities — confusion matrices, metrics table, F1 bar charts.

Main entry point:
    generate_report(predictions_jsonl, ground_truth_excel, output_path)

Produces a single PDF/PNG with:
    - Metrics summary table (binary + multiclass)
    - Binary confusion matrix
    - Multi-class confusion matrix
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from evaluation.metrics import (
    load_ground_truth,
    load_predictions_jsonl,
    merge,
    binary_metrics,
    multiclass_metrics,
)


# ── Main report ───────────────────────────────────────────────────────────────

def generate_report(
    predictions_jsonl: str,
    ground_truth_excel: str,
    output_path: str = "report.pdf",
    run_id: str = "",
) -> str:
    """
    Generate a single PDF/PNG containing:
      - Metrics summary table
      - Binary confusion matrix
      - Multi-class confusion matrix

    Returns the output path.
    """
    gt = load_ground_truth(ground_truth_excel)
    preds = load_predictions_jsonl(predictions_jsonl)
    merged = merge(preds, gt)

    bm = binary_metrics(merged)
    mm = multiclass_metrics(merged)

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        f"EHS Incident Detection — Evaluation Report{f'  [{run_id}]' if run_id else ''}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[0.6, 1.4, 2.8],
        hspace=0.45,
    )

    ax_table = fig.add_subplot(gs[0])
    _plot_metrics_table(ax_table, bm, mm)

    ax_bin = fig.add_subplot(gs[1])
    _plot_binary_cm(ax_bin, merged)

    ax_mc = fig.add_subplot(gs[2])
    _plot_multiclass_cm(ax_mc, merged)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved → {output_path}")
    return output_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _plot_metrics_table(ax: plt.Axes, bm: dict, mm: dict) -> None:
    ax.axis("off")
    columns = [
        "Binary\nAccuracy", "Binary\nPrecision", "Binary\nRecall", "Binary\nF1",
        "Multiclass\nAccuracy", "Macro\nF1",
    ]
    values = [[
        f"{bm['binary_accuracy']:.3f}",
        f"{bm['binary_precision']:.3f}",
        f"{bm['binary_recall']:.3f}",
        f"{bm['binary_f1']:.3f}",
        f"{mm['multiclass_accuracy']:.3f}",
        f"{mm['macro_f1']:.3f}",
    ]]
    tbl = ax.table(cellText=values, colLabels=columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)
    for j in range(len(columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
        tbl[1, j].set_facecolor("#ecf0f1")
        tbl[1, j].set_text_props(fontsize=13, fontweight="bold")
    ax.set_title("Metrics Summary", fontsize=12, fontweight="bold", pad=8)


def _plot_binary_cm(ax: plt.Axes, merged: pd.DataFrame) -> None:
    labels = ["Accident", "No Accident"]
    cm = confusion_matrix(merged["true_binary"], merged["pred_binary"], labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.0%})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])
    sns.heatmap(
        cm_pct, ax=ax, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white", vmin=0, vmax=1, cbar=False,
        annot_kws={"size": 13},
    )
    ax.set_title("Binary Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, rotation=0)


def _plot_multiclass_cm(ax: plt.Axes, merged: pd.DataFrame) -> None:
    all_labels = sorted(
        set(merged["true_category"].tolist()) | set(merged["predicted_category"].tolist())
    )
    cm = confusion_matrix(merged["true_category"], merged["predicted_category"], labels=all_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums != 0, cm.astype(float) / np.maximum(row_sums, 1), 0.0)
    annot = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.0%})" if cm[i,j] > 0 else "" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])
    sns.heatmap(
        cm_pct, ax=ax, annot=annot, fmt="", cmap="Oranges",
        xticklabels=all_labels, yticklabels=all_labels,
        linewidths=0.5, linecolor="white", vmin=0, vmax=1, cbar=True,
        annot_kws={"size": 8},
    )
    ax.set_title("Multi-Class Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)


# ── Standalone helpers ────────────────────────────────────────────────────────

def plot_per_class_f1(
    per_class_f1: dict[str, float],
    output_path: str | None = None,
    title: str = "Per-Class F1 Scores",
) -> None:
    """Horizontal bar chart of per-class F1 scores, colour-coded by threshold."""
    sorted_items = sorted(per_class_f1.items(), key=lambda x: x[1])
    classes = [k for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]
    colors = ["#e74c3c" if s < 0.5 else "#f39c12" if s < 0.75 else "#27ae60" for s in scores]
    fig, ax = plt.subplots(figsize=(10, max(4, len(classes) * 0.45)))
    bars = ax.barh(classes, scores, color=colors)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_ablation_comparison(
    results: list[dict[str, Any]],
    metrics: list[str] | None = None,
    output_path: str | None = None,
    title: str = "Ablation Comparison",
) -> None:
    """Grouped bar chart comparing metrics across multiple runs."""
    if metrics is None:
        metrics = ["binary_f1", "macro_f1", "binary_precision", "binary_recall"]
    run_ids = [r.get("run_id", f"run{i}") for i, r in enumerate(results)]
    x = np.arange(len(run_ids))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, len(run_ids) * 2), 5))
    for i, metric in enumerate(metrics):
        vals = [r.get(metric, 0) for r in results]
        ax.bar(x + i * width, vals, width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(run_ids, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
