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
from datetime import datetime
from typing import Any

from pipeline.detection import DetectionResult
from pipeline.classification import ClassificationResult


# ── Token cost estimates (USD per 1K tokens) ─────────────────────────────────
# Approximate Vertex AI / Google AI pricing — update as pricing changes.
COST_PER_1K_INPUT_TOKENS: dict[str, float] = {
    "gemini-2.5-flash":   0.000075,
    "gemini-2.5-pro":     0.00125,
    "gemini-2.0-flash":   0.000075,
    "gemini-1.5-flash":   0.000075,
    "gemini-1.5-pro":     0.00125,
}
COST_PER_1K_OUTPUT_TOKENS: dict[str, float] = {
    "gemini-2.5-flash":   0.0003,
    "gemini-2.5-pro":     0.005,
    "gemini-2.0-flash":   0.0003,
    "gemini-1.5-flash":   0.0003,
    "gemini-1.5-pro":     0.005,
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
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> None:
        """Append one prediction record."""
        # Resolve classification fields
        if classification:
            predicted_category = classification.category
            cls_start = classification.incident_start_time
            cls_end = classification.incident_end_time
            cls_conf = classification.confidence
            description = classification.description
            root_cause = classification.root_cause_analysis
            stage2_latency = classification.latency_s
            fallback_used_cat = classification.fallback_used
        else:
            predicted_category = "No Accident"
            cls_start = cls_end = None
            cls_conf = detection.confidence
            description = "No accident detected."
            root_cause = "No safety incident observed."
            stage2_latency = 0.0
            fallback_used_cat = False

        total_latency = (
            detection.latency_s + stage2_latency + frame_fallback_latency
        )

        # Token cost estimation (rough — vertex response doesn't always expose tokens)
        estimated_cost = _estimate_cost(model_name, total_latency)

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
            "incident_start_time": cls_start,
            "incident_end_time": cls_end,
            "confidence": cls_conf,
            "category_fallback_used": fallback_used_cat,
            "description": description,
            "root_cause_analysis": root_cause,
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
        self._failures.append(row)
        self._failures_writer.writerow(row)
        self._failures_file.flush()

    # ── Finalize ──────────────────────────────────────────────────────────────

    def save_all(self) -> None:
        """Write aggregate stats. Call once after all videos are processed."""
        self._failures_file.close()

        n = len(self._records)
        if n == 0:
            return

        total_cost = sum(r["estimated_cost_usd"] for r in self._records)
        mean_latency = sum(r["total_latency_s"] for r in self._records) / n

        summary = {
            "run_id": self.run_id,
            "n_videos": n,
            "n_failed": len(self._failures),
            "total_cost_usd": round(total_cost, 4),
            "mean_latency_s": round(mean_latency, 3),
            "generated_at": datetime.now().isoformat(),
        }
        with open(self.metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Logged {n} predictions, {len(self._failures)} failures")
        print(f"  Estimated cost: ${total_cost:.4f} | Mean latency: {mean_latency:.1f}s")

    # ── Export helpers ────────────────────────────────────────────────────────

    def to_dataframe(self):
        """Return all records as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._records)


# ── Cost estimation ───────────────────────────────────────────────────────────

def _estimate_cost(model_name: str, total_latency_s: float) -> float:
    """
    Rough cost estimate based on token throughput approximation.

    ~500 tokens/s for Flash models is a conservative estimate for
    combined input+output tokens in a video analysis context.
    """
    # Approximate token counts based on latency (very rough heuristic)
    approx_input_tokens = 1000   # video context + prompt
    approx_output_tokens = 150   # JSON response

    input_cost = (
        approx_input_tokens / 1000
        * COST_PER_1K_INPUT_TOKENS.get(model_name, 0.0001)
    )
    output_cost = (
        approx_output_tokens / 1000
        * COST_PER_1K_OUTPUT_TOKENS.get(model_name, 0.0003)
    )
    return input_cost + output_cost
