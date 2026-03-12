"""
Confusion analysis and failure diagnostics (Direction 7).

Identifies:
- Most confused category pairs
- Per-category failure patterns
- Low-confidence errors
- False positive / false negative breakdown
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_pairs(merged: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return the most frequently confused category pairs.

    Returns a DataFrame: true_category | predicted_category | count
    """
    errors = merged[merged["true_category"] != merged["predicted_category"]].copy()
    pairs = (
        errors.groupby(["true_category", "predicted_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return pairs


def per_category_diagnostics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    For each true category, compute:
    - total count
    - correct count
    - accuracy
    - most common misclassification
    """
    rows = []
    for cat in merged["true_category"].unique():
        subset = merged[merged["true_category"] == cat]
        correct = (subset["predicted_category"] == cat).sum()
        total = len(subset)
        accuracy = correct / total if total > 0 else 0.0

        wrong = subset[subset["predicted_category"] != cat]
        if not wrong.empty:
            most_common_mistake = wrong["predicted_category"].value_counts().idxmax()
        else:
            most_common_mistake = None

        rows.append(
            {
                "true_category": cat,
                "total": total,
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "most_common_mistake": most_common_mistake,
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy")


def false_positive_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Binary false positives: predicted Accident but true is No Accident.
    """
    fps = merged[
        (merged["pred_binary"] == "Accident") & (merged["true_binary"] == "No Accident")
    ].copy()
    return fps[["video_id_x", "predicted_category", "confidence", "description"]].sort_values(
        "confidence", ascending=False
    )


def false_negative_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Binary false negatives: predicted No Accident but true is Accident.
    """
    fns = merged[
        (merged["pred_binary"] == "No Accident") & (merged["true_binary"] == "Accident")
    ].copy()
    return fns[["video_id_x", "true_category", "confidence", "description"]].sort_values(
        "confidence"
    )


def low_confidence_errors(
    merged: pd.DataFrame,
    confidence_col: str = "confidence",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Return predictions below confidence threshold that were incorrect."""
    errors = merged[merged["true_category"] != merged["predicted_category"]].copy()
    if confidence_col in errors.columns:
        errors = errors[errors[confidence_col] < threshold]
    return errors.sort_values(confidence_col)


def print_failure_report(merged: pd.DataFrame) -> None:
    """Print a concise failure analysis to stdout."""
    print("=" * 60)
    print("FAILURE ANALYSIS REPORT")
    print("=" * 60)

    fps = false_positive_analysis(merged)
    fns = false_negative_analysis(merged)
    pairs = confusion_pairs(merged)
    per_cat = per_category_diagnostics(merged)

    print(f"\nFalse Positives (No Accident → predicted Accident): {len(fps)}")
    print(fps.to_string(index=False))

    print(f"\nFalse Negatives (Accident → predicted No Accident): {len(fns)}")
    print(fns.to_string(index=False))

    print(f"\nTop Confusion Pairs:")
    print(pairs.to_string(index=False))

    print(f"\nPer-Category Accuracy:")
    print(per_cat.to_string(index=False))
