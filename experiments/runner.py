"""
ExperimentRunner — orchestrates the full pipeline for a given config.

Usage
-----
from experiments.runner import ExperimentRunner

runner = ExperimentRunner.from_yaml("experiments/configs/attempt3.yaml")
runner.run()
"""

from __future__ import annotations

import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from vertexai.generative_models import Part

from config.categories import MERGED_CATEGORIES, DEFAULT_FALLBACK_CATEGORY


def next_attempt_number(outputs_dir: str = "outputs") -> int:
    """
    Scan outputs/ for folders named *attempt{n}* and return n+1.
    Returns 1 if no attempt folders exist yet.
    """
    pattern = re.compile(r"attempt(\d+)")
    max_n = 0
    if os.path.isdir(outputs_dir):
        for entry in os.listdir(outputs_dir):
            m = pattern.search(entry)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1
from config.settings import Config
from run_logging.experiment_logger import ExperimentLogger
from pipeline import classification, detection, frame_fallback, ingestion
from pipeline.client import get_model
from pipeline.postprocessing import safe_parse_json


class ExperimentRunner:
    """
    Runs a batch experiment over all GCS videos using a YAML config.

    The config controls:
    - model name
    - prompt variant
    - temperature, top_k, top_p
    - n_votes and vote_policy for Stage 1
    - confidence threshold
    - frame fallback settings
    - output directory
    """

    def __init__(self, exp_config: dict[str, Any], run_name: str | None = None) -> None:
        self.exp_config = exp_config
        self.run_name = run_name or exp_config.get("name", "experiment")

        # Resolve config overrides
        self.model_name: str = exp_config.get("model", "gemini-2.5-flash")
        # stage1_model: cheaper/faster model for binary gating (defaults to model)
        self.stage1_model_name: str = exp_config.get("stage1_model", self.model_name)
        self.prompt_variant: str = exp_config.get(
            "binary_prompt_variant", "animated"
        )
        self.classification_prompt_variant: str = exp_config.get(
            "classification_prompt_variant", "animated"
        )
        self.temperature: float = float(exp_config.get("temperature", 0.0))
        self.top_k: int | None = exp_config.get("top_k")
        self.top_p: float | None = exp_config.get("top_p")
        self.n_votes: int = int(exp_config.get("n_votes", 3))
        self.vote_policy: str = exp_config.get("vote_policy", "any")
        self.confidence_threshold: float = float(
            exp_config.get("confidence_threshold", 0.6)
        )
        self.use_frame_fallback: bool = bool(
            exp_config.get("use_frame_fallback", False)
        )
        self.frame_fallback_fps: float = float(
            exp_config.get("frame_fallback_fps", 2.0)
        )
        self.frame_fallback_max_frames: int = int(
            exp_config.get("frame_fallback_max_frames", 10)
        )
        self.sleep_between_videos: float = float(
            exp_config.get("sleep_between_videos", 1.0)
        )
        # max_workers > 1 enables parallel video processing within a run.
        # Each worker fires its own API calls concurrently. The Gemini SDK is
        # thread-safe. Set to 8–16 when you have high API quota.
        # sleep_between_videos is ignored when max_workers > 1.
        self.max_workers: int = int(exp_config.get("max_workers", 1))

        # ── Local mode (no GCS required) ──────────────────────────────────────
        # Set "local_videos_dir" in config to run against local .mp4 files.
        # Set "originals_only: false" to include augmented versions too.
        # Set "augmented_only: true" for Phase 2 sweep (skip originals).
        self.local_videos_dir: str | None = exp_config.get("local_videos_dir")
        self.originals_only: bool = bool(exp_config.get("originals_only", True))
        self.augmented_only: bool = bool(exp_config.get("augmented_only", False))

        # ── Spreadsheet mode ──────────────────────────────────────────────────
        # Set "spreadsheet_source" to a path to dataset_mapping.xlsx.
        # local_videos_dir must also be set — the runner looks for VID{n}_* folders.
        self.spreadsheet_source: str | None = exp_config.get("spreadsheet_source")

        # Output directory: outputs/<timestamp>_<name>/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_root = exp_config.get("outputs_dir", "outputs")
        self.output_dir = os.path.join(outputs_root, f"{timestamp}_{self.run_name}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = ExperimentLogger(
            run_id=f"{timestamp}_{self.run_name}",
            output_dir=self.output_dir,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentRunner":
        """Load config from a YAML file."""
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        run_name = Path(yaml_path).stem
        return cls(cfg, run_name=run_name)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the full experiment.

        Source is determined by config:
        - local_videos_dir is set → read .mp4 files from local disk
        - otherwise              → list blobs from GCS

        Two models are used:
        - stage1_model (binary gate)  — cheap, high-recall
        - stage2_model (classifier)   — more capable
        Both default to "model" if "stage1_model" is not set in config.
        """
        stage1_model = get_model(self.stage1_model_name)
        stage2_model = get_model(self.model_name)

        binary_prompt = detection.BINARY_PROMPTS.get(
            self.prompt_variant, detection.BINARY_PROMPT_ANIMATED
        )
        class_prompt = classification.CLASSIFICATION_PROMPTS.get(
            self.classification_prompt_variant,
            classification.CLASSIFICATION_PROMPT_ANIMATED,
        )

        # ── Build video list ──────────────────────────────────────────────────
        if self.spreadsheet_source:
            video_items = ingestion.load_clips_from_spreadsheet(self.spreadsheet_source)
            mode = "spreadsheet"
        elif self.local_videos_dir:
            video_items = ingestion.find_local_videos(
                self.local_videos_dir,
                originals_only=self.originals_only,
                augmented_only=self.augmented_only,
            )
            mode = "local"
        else:
            video_items = ingestion.list_video_blobs()
            mode = "GCS"

        print(f"[{self.run_name}] {mode} mode — {len(video_items)} videos")

        tmp_frame_dir = os.path.join(self.output_dir, "_tmp_frames")

        kwargs = dict(
            mode=mode,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            binary_prompt=binary_prompt,
            class_prompt=class_prompt,
            tmp_frame_dir=tmp_frame_dir,
        )

        try:
            if self.max_workers > 1:
                print(f"[{self.run_name}] Parallel mode — {self.max_workers} workers")
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {
                        pool.submit(self._process_video, item, **kwargs): item
                        for item in video_items
                    }
                    for future in as_completed(futures):
                        exc = future.exception()
                        if exc:
                            # _process_video already catches and logs per-video errors;
                            # this guards against unexpected escapes
                            print(f"  [parallel] unhandled error: {exc}")
            else:
                for item in video_items:
                    self._process_video(item, **kwargs)
                    time.sleep(self.sleep_between_videos)
        finally:
            agg = self.logger.save_all()
            shutil.rmtree(tmp_frame_dir, ignore_errors=True)
            self._write_full_metrics(agg)
            print(f"[{self.run_name}] Done. Results in: {self.output_dir}")

    # ── Post-run evaluation ───────────────────────────────────────────────────

    def _write_full_metrics(self, agg: dict) -> None:
        """
        Merge evaluation metrics into metrics.json after the run completes.

        Calls evaluation.metrics.evaluate() against the ground-truth spreadsheet
        so that compare_results() in ablation.py sees binary_f1, macro_f1, etc.
        Does nothing if no spreadsheet_source is configured.
        """
        if not self.spreadsheet_source:
            return
        if not os.path.exists(self.logger.predictions_path):
            return

        import json as _json
        from evaluation.metrics import evaluate

        try:
            eval_results = evaluate(
                predictions_jsonl=self.logger.predictions_path,
                ground_truth_excel=self.spreadsheet_source,
                run_id=self.run_name,
                total_cost_usd=agg.get("total_cost_usd", 0.0),
                mean_latency_s=agg.get("mean_latency_s", 0.0),
            )
            # Merge aggregate counts back in (n_videos, n_failed)
            eval_results["n_videos"] = agg.get("n_videos", 0)
            eval_results["n_failed"] = agg.get("n_failed", 0)
            with open(self.logger.metrics_path, "w") as f:
                _json.dump(eval_results, f, indent=2)
            print(
                f"  binary_f1={eval_results.get('binary_f1', float('nan')):.3f}  "
                f"macro_f1={eval_results.get('macro_f1', float('nan')):.3f}"
            )
        except Exception as exc:
            print(f"  [warn] Could not compute evaluation metrics: {exc}")

    # ── Per-video logic ───────────────────────────────────────────────────────

    def _process_video(
        self,
        item: Any,              # str (local path), Blob (GCS), or dict (spreadsheet)
        *,
        mode: str,              # "local" | "GCS" | "spreadsheet"
        stage1_model: Any,      # binary detection model (cheap, high recall)
        stage2_model: Any,      # classification model
        binary_prompt: str,
        class_prompt: str,
        tmp_frame_dir: str,
    ) -> None:
        # ── Resolve video_id, uri, Part, and local path ───────────────────────
        if mode == "spreadsheet":
            clip = item
            video_id = clip["video_id"]
            uri = clip["video_url"]

            local_path: str | None = ingestion.find_local_video_by_vid(
                self.local_videos_dir, video_id
            ) if self.local_videos_dir else None

            if local_path is None:
                print(f"    SKIP {video_id}: not found in {self.local_videos_dir}")
                self.logger.log_failure(video_id=video_id, error="no local file found")
                return

            with open(local_path, "rb") as f:
                video_part = Part.from_data(data=f.read(), mime_type="video/mp4")

        elif mode == "local":
            local_path = item
            video_id = ingestion.video_id_from_path(local_path)
            uri = local_path
            with open(local_path, "rb") as f:
                video_part = Part.from_data(data=f.read(), mime_type="video/mp4")
        else:
            local_path = None
            video_id = item.name
            uri = ingestion.gcs_uri(item.name)
            video_part = Part.from_uri(uri=uri, mime_type="video/mp4")

        print(f"  Processing: {video_id}")

        try:
            # ── Stage 1 ───────────────────────────────────────────────────────
            det_result = detection.detect(
                video_part=video_part,
                model=stage1_model,
                prompt=binary_prompt,
                n_votes=self.n_votes,
                vote_policy=self.vote_policy,
                confidence_threshold=self.confidence_threshold,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

            frame_fallback_used = False
            frame_fallback_latency = 0.0

            # ── Frame fallback ────────────────────────────────────────────────
            if not det_result.incident_detected and self.use_frame_fallback:
                # Local mode: file already on disk; GCS mode: download first
                if local_path is None:
                    tmp_vid_dir = os.path.join(self.output_dir, "_tmp_vids")
                    local_path = ingestion.download_video(item.name, tmp_vid_dir)

                safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", video_id)
                frames_dir = os.path.join(tmp_frame_dir, safe_name)

                fb_detected, _, fb_latency = frame_fallback.frame_fallback(
                    local_mp4_path=local_path,
                    frames_dir=frames_dir,
                    model=stage1_model,  # use cheap model for frame fallback too
                    fps=self.frame_fallback_fps,
                    max_frames=self.frame_fallback_max_frames,
                )
                frame_fallback_used = True
                frame_fallback_latency = fb_latency

                if fb_detected:
                    det_result.incident_detected = True

            # ── Stage 2 or No Accident ────────────────────────────────────────
            if not det_result.incident_detected:
                self.logger.log(
                    video_id=video_id,
                    gcs_uri=uri,
                    detection=det_result,
                    classification=None,
                    frame_fallback_used=frame_fallback_used,
                    frame_fallback_latency=frame_fallback_latency,
                    model_name=self.model_name,
                    stage1_model_name=self.stage1_model_name,
                    n_votes=self.n_votes,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                return

            cls_result = classification.classify(
                video_part=video_part,
                model=stage2_model,
                prompt=class_prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

            self.logger.log(
                video_id=video_id,
                gcs_uri=uri,
                detection=det_result,
                classification=cls_result,
                frame_fallback_used=frame_fallback_used,
                frame_fallback_latency=frame_fallback_latency,
                model_name=self.model_name,
                stage1_model_name=self.stage1_model_name,
                n_votes=self.n_votes,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        except Exception as exc:
            print(f"    ERROR: {exc}")
            self.logger.log_failure(video_id=video_id, error=str(exc))


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or args[0] in ("--next", "--auto"):
        # Auto-detect the next attempt number from outputs/
        n = next_attempt_number()
        config_path = f"experiments/configs/attempt{n}.yaml"
        print(f"[auto] Next attempt: {n} → {config_path}")
    else:
        config_path = args[0]

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    runner = ExperimentRunner.from_yaml(config_path)
    runner.run()
