"""
ExperimentLogger — records per-video predictions, latency, cost, and params.

Output files in `output_dir/`:
  predictions.jsonl     — one JSON record per video
  failed_videos.csv     — error log
  metrics.json          — aggregate metrics (written on finalize)
  config_snapshot.yaml  — copy of experiment config (written by runner)
"""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.detection import DetectionResult
    from pipeline.classification import ClassificationResult


# ── Token cost estimates (USD per 1K tokens) ─────────────────────────────────
# Approximate Vertex AI / Google AI pricing — update as pricing changes.
COST_PER_1K_INPUT_TOKENS: dict[str, float] = {
    "gemini-2.5-flash":          0.000075,
    "gemini-2.5-flash-lite":     0.0000375,
    "gemini-2.5-pro":            0.00125,
    "gemini-2.0-flash":          0.000075,
    "gemini-2.0-flash-001":      0.000075,
    "gemini-2.0-flash-lite":     0.0000375,
    "gemini-2.0-flash-lite-001": 0.0000375,
    "gemini-1.5-flash":          0.000075,
    "gemini-1.5-pro":            0.00125,
}
COST_PER_1K_OUTPUT_TOKENS: dict[str, float] = {
    "gemini-2.5-flash":          0.0003,
    "gemini-2.5-flash-lite":     0.00015,
    "gemini-2.5-pro":            0.005,
    "gemini-2.0-flash":          0.0003,
    "gemini-2.0-flash-001":      0.0003,
    "gemini-2.0-flash-lite":     0.00015,
    "gemini-2.0-flash-lite-001": 0.00015,
    "gemini-1.5-flash":          0.0003,
    "gemini-1.5-pro":            0.005,
}


class ExperimentLogger:
    """
    Records predictions and metadata for a single experiment run.
    """

    def __init__(self, run_id: str, output_dir: str) -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.predictions_path = os.path.join(output_dir, "predictions.jsonl")
        self.failures_path = os.path.join(output_dir, "failed_videos.csv")
        self.metrics_path = os.path.join(output_dir, "metrics.json")

        self._records: list[dict[str, Any]] = []
        self._failures: list[dict[str, str]] = []
        self._lock = threading.Lock()  # protects all writes (for parallel video processing)

        # Open CSV for failures
        self._failures_file = open(self.failures_path, "w", newline="")
        self._failures_writer = csv.DictWriter(
            self._failures_file, fieldnames=["video_id", "error", "timestamp"]
        )
        self._failures_writer.writeheader()

    # ── Log a successful prediction ───────────────────────────────────────────

    def log(
        self,
        video_id: str,
        gcs_uri: str,
        detection: DetectionResult,
        classification: ClassificationResult | None,
        frame_fallback_used: bool,
        frame_fallback_latency: float,
        model_name: str,
        stage1_model_name: str | None = None,
        n_votes: int = 1,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        """Append one prediction record."""
        # Resolve classification fields
        if classification:
            predicted_category = classification.category
            all_categories = classification.categories
            cls_start = classification.incident_start_time
            cls_end = classification.incident_end_time
            cls_conf = classification.confidence
            description = classification.description
            root_cause = classification.root_cause_analysis
            ehs_report = classification.ehs_report
            stage2_latency = classification.latency_s
            fallback_used_cat = classification.fallback_used
        else:
            predicted_category = "No Accident"
            all_categories = [{"category": "No Accident", "confidence": detection.confidence}]
            cls_start = cls_end = None
            cls_conf = detection.confidence
            description = "No accident detected."
            root_cause = "No safety incident observed."
            ehs_report = {}
            stage2_latency = 0.0
            fallback_used_cat = False

        total_latency = (
            detection.latency_s + stage2_latency + frame_fallback_latency
        )

        # Accurate cost: n_votes Stage-1 calls + 1 Stage-2 call only if accident detected
        estimated_cost = _estimate_cost(
            stage2_model=model_name,
            stage1_model=stage1_model_name or model_name,
            n_votes=n_votes,
            stage1_detected=detection.incident_detected,
        )

        record: dict[str, Any] = {
            "run_id": self.run_id,
            "video_id": video_id,
            "gcs_uri": gcs_uri,
            "timestamp": datetime.now().isoformat(),
            # Detection
            "stage1_votes": detection.votes,
            "stage1_detected": detection.incident_detected,
            "stage1_confidence": detection.confidence,
            "stage1_latency_s": round(detection.latency_s, 3),
            # Frame fallback
            "frame_fallback_used": frame_fallback_used,
            "frame_fallback_latency_s": round(frame_fallback_latency, 3),
            # Classification
            "incident_detected": detection.incident_detected,
            "predicted_category": predicted_category,
            "all_categories": all_categories,
            "incident_start_time": cls_start,
            "incident_end_time": cls_end,
            "confidence": cls_conf,
            "category_fallback_used": fallback_used_cat,
            "description": description,
            "root_cause_analysis": root_cause,
            "ehs_report": ehs_report,
            "stage2_latency_s": round(stage2_latency, 3),
            # Totals
            "total_latency_s": round(total_latency, 3),
            "estimated_cost_usd": round(estimated_cost, 6),
            # Params
            "model": model_name,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        with self._lock:
            self._records.append(record)
            # Write immediately so partial runs are recoverable
            with open(self.predictions_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    # ── Log a failure ─────────────────────────────────────────────────────────

    def log_failure(self, video_id: str, error: str) -> None:
        """Append a failure entry."""
        row = {
            "video_id": video_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock:
            self._failures.append(row)
            self._failures_writer.writerow(row)
            self._failures_file.flush()

    # ── Finalize ──────────────────────────────────────────────────────────────

    def save_all(self) -> dict:
        """
        Write aggregate stats and return them.

        Returns the summary dict so the caller can pass cost/latency into
        evaluation.metrics.evaluate() without re-reading the file.
        """
        self._failures_file.close()

        n = len(self._records)
        if n == 0:
            return {}

        total_cost = sum(r["estimated_cost_usd"] for r in self._records)
        mean_latency = sum(r["total_latency_s"] for r in self._records) / n

        summary = {
            "run_id": self.run_id,
            "n_videos": n,
            "n_failed": len(self._failures),
            "total_cost_usd": round(total_cost, 6),
            "mean_latency_s": round(mean_latency, 3),
            "generated_at": datetime.now().isoformat(),
        }
        with open(self.metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Logged {n} predictions, {len(self._failures)} failures")
        print(f"  Estimated cost: ${total_cost:.6f} | Mean latency: {mean_latency:.1f}s")
        return summary

    # ── Export helpers ────────────────────────────────────────────────────────

    def to_dataframe(self):
        """Return all records as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._records)


# ── Cost estimation ───────────────────────────────────────────────────────────

# Approximate token counts per API call (video context is large but billed per token).
# These are rough but consistent across runs — relative comparisons are reliable.
_STAGE1_INPUT_TOKENS = 800    # video + short binary prompt
_STAGE1_OUTPUT_TOKENS = 30    # {"incident_detected": true, "confidence": 0.9}
_STAGE2_INPUT_TOKENS = 1500   # video + longer structured prompt
_STAGE2_OUTPUT_TOKENS = 500   # JSON with reasoning, incidents list, EHS report


def _estimate_cost(
    stage2_model: str,
    stage1_model: str,
    n_votes: int,
    stage1_detected: bool,
) -> float:
    """
    Estimate API cost for one video.

    Stage 1 runs n_votes times on every video (cheap model, short prompt).
    Stage 2 runs once only when Stage 1 detected an accident (skip for non-accidents).

    Parameters
    ----------
    stage2_model : str
        Model name used for Stage 2 classification.
    stage1_model : str
        Model name used for Stage 1 binary gate (may differ from stage2_model).
    n_votes : int
        Number of Stage 1 calls made (ensemble size).
    stage1_detected : bool
        Whether Stage 1 flagged an accident (determines if Stage 2 ran).
    """
    s1_in  = COST_PER_1K_INPUT_TOKENS.get(stage1_model,  0.000075)
    s1_out = COST_PER_1K_OUTPUT_TOKENS.get(stage1_model, 0.0003)
    s2_in  = COST_PER_1K_INPUT_TOKENS.get(stage2_model,  0.000075)
    s2_out = COST_PER_1K_OUTPUT_TOKENS.get(stage2_model, 0.0003)

    stage1_cost = n_votes * (
        _STAGE1_INPUT_TOKENS / 1000 * s1_in
        + _STAGE1_OUTPUT_TOKENS / 1000 * s1_out
    )
    stage2_cost = (
        _STAGE2_INPUT_TOKENS / 1000 * s2_in
        + _STAGE2_OUTPUT_TOKENS / 1000 * s2_out
    ) if stage1_detected else 0.0

    return stage1_cost + stage2_cost
