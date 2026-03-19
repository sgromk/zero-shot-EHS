"""
Evaluation metrics — binary and multi-class.

Given a predictions JSONL log and a ground-truth Excel file,
compute all standard metrics and return structured results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.categories import normalize_gt_category


# ── Ground truth loading ──────────────────────────────────────────────────────

def load_ground_truth(excel_path: str) -> pd.DataFrame:
    """
    Load the ground-truth mapping file.

    Actual columns (trailing spaces stripped):
        video_url, start_s, end_s, duration_s, incident_present,
        near_miss_present, accident_present, incident_severity,
        incident_type, description

    Notes:
    - Column names and string values have trailing spaces — stripped on load.
    - video_url is a YouTube URL (no VID number), so video_id_clean is
      assigned by 1-based row order: row 0 → "VID1", row 1 → "VID2", etc.
      This assumes the spreadsheet rows are ordered identically to the VID
      folder numbering.
    - duration_s may be an Excel formula string — ignored.
    - 1010 rows total (73 originals + augmented clips).
    """
    df = pd.read_excel(excel_path)

    # Drop entirely-empty trailing columns and rows
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Strip trailing/leading whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip string values across all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

    # Assign VID key by row order — 0-indexed, zero-padded to match folder names
    # (VID000, VID001, … VID072)
    df = df.reset_index(drop=True)
    df["video_id_clean"] = [f"VID{i:03d}" for i in df.index]

    df = df.rename(columns={"incident_type": "true_category"})
    df["true_category"] = df["true_category"].apply(normalize_gt_category)

    # Normalise accident_present → binary label (handles bool, int/float, Yes/No)
    def _to_binary(val) -> str:
        if pd.isna(val):
            return "No Accident"
        if isinstance(val, bool):
            return "Accident" if val else "No Accident"
        if isinstance(val, (int, float)):
            return "Accident" if val else "No Accident"
        if isinstance(val, str):
            return "Accident" if val.strip().lower() in {"yes", "true", "1"} else "No Accident"
        return "No Accident"

    df["true_binary"] = df["accident_present"].apply(_to_binary)

    # Near-miss binary label
    if "near_miss_present" in df.columns:
        df["true_near_miss"] = df["near_miss_present"].apply(_to_binary)

    return df


# ── Predictions loading ───────────────────────────────────────────────────────

def load_predictions_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Load predictions from a JSONL file produced by ExperimentLogger."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)

    # Extract VID key
    df["video_id_clean"] = df["video_id"].apply(
        lambda x: re.search(r"(VID\d+)", str(x)).group(1)
        if re.search(r"(VID\d+)", str(x))
        else None
    )
    df = df.dropna(subset=["video_id_clean"])

    # Binary prediction label
    df["pred_binary"] = df["predicted_category"].apply(
        lambda x: "Accident" if x != "No Accident" else "No Accident"
    )

    # Parse all_categories from JSON string if needed (JSONL reload gives strings)
    if "all_categories" in df.columns:
        def _parse_cats(v):
            if isinstance(v, list):
                return v
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
            return []
        df["all_categories"] = df["all_categories"].apply(_parse_cats)

    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions with ground truth on VID key."""
    merged = predictions.merge(ground_truth, on="video_id_clean", how="inner")
    return merged


# ── Binary metrics ────────────────────────────────────────────────────────────

def binary_metrics(merged: pd.DataFrame) -> dict[str, float]:
    """Compute binary detection metrics."""
    y_true = merged["true_binary"]
    y_pred = merged["pred_binary"]
    return {
        "binary_accuracy": accuracy_score(y_true, y_pred),
        "binary_precision": precision_score(y_true, y_pred, pos_label="Accident", zero_division=0),
        "binary_recall": recall_score(y_true, y_pred, pos_label="Accident", zero_division=0),
        "binary_f1": f1_score(y_true, y_pred, pos_label="Accident", zero_division=0),
    }


# ── Multi-class metrics ───────────────────────────────────────────────────────

def multiclass_metrics(merged: pd.DataFrame) -> dict[str, Any]:
    """Compute multi-class classification metrics."""
    y_true = merged["true_category"]
    y_pred = merged["predicted_category"]

    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {
        cls: per_class[cls]["f1-score"]
        for cls in per_class
        if cls not in ("accuracy", "macro avg", "weighted avg")
    }

    return {
        "multiclass_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_f1": per_class_f1,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


# ── Multi-label any-match metric ──────────────────────────────────────────────

def any_match_metrics(merged: pd.DataFrame) -> dict[str, float]:
    """
    Multi-label recall: for accident clips, was the GT category predicted anywhere?

    The structured prompt returns a ranked `all_categories` list. The primary
    prediction (highest confidence) is used for standard metrics. But the GT
    category may appear as a secondary prediction — this metric captures that.

    Only evaluated on clips where true_binary == "Accident".
    Returns empty dict if all_categories column is absent.
    """
    if "all_categories" not in merged.columns:
        return {}

    accident_rows = merged[merged["true_binary"] == "Accident"].copy()
    if len(accident_rows) == 0:
        return {}

    def _gt_in_any(row) -> bool:
        cats = row.get("all_categories", [])
        if not isinstance(cats, list):
            return False
        cat_names = [c.get("category", "") for c in cats if isinstance(c, dict)]
        return row["true_category"] in cat_names

    accident_rows["any_match"] = accident_rows.apply(_gt_in_any, axis=1)
    return {
        "any_match_recall": round(float(accident_rows["any_match"].mean()), 4),
    }


# ── All metrics ───────────────────────────────────────────────────────────────

def evaluate(
    predictions_jsonl: str,
    ground_truth_excel: str,
    run_id: str = "",
    total_cost_usd: float = 0.0,
    mean_latency_s: float = 0.0,
) -> dict[str, Any]:
    """
    Run full evaluation: binary + multi-class.

    Returns a metrics dict suitable for saving to metrics.json.
    """
    gt = load_ground_truth(ground_truth_excel)
    preds = load_predictions_jsonl(predictions_jsonl)
    merged = merge(preds, gt)

    n_matched = len(merged)
    if n_matched == 0:
        raise ValueError("No matching videos between predictions and ground truth.")

    bm = binary_metrics(merged)
    mm = multiclass_metrics(merged)
    am = any_match_metrics(merged)

    return {
        "run_id": run_id,
        "n_matched": n_matched,
        "total_cost_usd": total_cost_usd,
        "mean_latency_s": mean_latency_s,
        **bm,
        "multiclass_accuracy": mm["multiclass_accuracy"],
        "macro_f1": mm["macro_f1"],
        "weighted_f1": mm["weighted_f1"],
        "per_class_f1": mm["per_class_f1"],
        **am,
    }
