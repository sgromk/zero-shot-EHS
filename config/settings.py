"""
Centralized configuration loaded from environment variables / .env file.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env automatically when this module is imported
load_dotenv()


@dataclass
class Config:
    # ── Google Cloud ────────────────────────────────────────────────────────────
    gcp_project_id: str = field(
        default_factory=lambda: os.environ.get("GCP_PROJECT_ID", "ehs-incident-detection-blissey")
    )
    gcp_location: str = field(
        default_factory=lambda: os.environ.get("GCP_LOCATION", "us-central1")
    )
    gcs_bucket_name: str = field(
        default_factory=lambda: os.environ.get("GCS_BUCKET_NAME", "ehs-video-analysis-2026-pavithran")
    )
    gcs_video_prefix: str = field(
        default_factory=lambda: os.environ.get("GCS_VIDEO_PREFIX", "videos/")
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    vertex_model: str = field(
        default_factory=lambda: os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    )

    # ── Pipeline behaviour ──────────────────────────────────────────────────────
    confidence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
    )
    stage1_votes: int = field(
        default_factory=lambda: int(os.environ.get("STAGE1_VOTES", "3"))
    )
    frame_fallback_fps: float = field(
        default_factory=lambda: float(os.environ.get("FRAME_FALLBACK_FPS", "2.0"))
    )
    frame_fallback_max_frames: int = field(
        default_factory=lambda: int(os.environ.get("FRAME_FALLBACK_MAX_FRAMES", "10"))
    )
    sleep_between_videos: float = field(
        default_factory=lambda: float(os.environ.get("SLEEP_BETWEEN_VIDEOS", "1.0"))
    )
    retry_attempts: int = field(
        default_factory=lambda: int(os.environ.get("RETRY_ATTEMPTS", "4"))
    )

    # ── Sampling (for Direction 4 experiments) ──────────────────────────────────
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.0"))
    )
    top_k: int | None = field(
        default_factory=lambda: (
            int(os.environ["TOP_K"]) if os.environ.get("TOP_K") else None
        )
    )
    top_p: float | None = field(
        default_factory=lambda: (
            float(os.environ["TOP_P"]) if os.environ.get("TOP_P") else None
        )
    )

    # ── Output ──────────────────────────────────────────────────────────────────
    outputs_dir: str = field(
        default_factory=lambda: os.environ.get("OUTPUTS_DIR", "outputs")
    )


# Module-level singleton — import this directly
config = Config()
