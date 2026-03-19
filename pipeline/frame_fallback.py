"""
Stage 1.5 — Frame-level fallback detection.

When the video-level Stage 1 returns No Accident, extract frames from the
local video file and classify each frame individually with Gemini.

If ANY frame is classified as an accident, the video is promoted to Accident.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vertexai.generative_models import GenerativeModel, Part

from config.settings import config
from pipeline.client import vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json


# ── Prompt ────────────────────────────────────────────────────────────────────

FRAME_PROMPT = """\
IMPORTANT: This is a single frame from an ANIMATED workplace safety clip.

Decide if the frame shows evidence of an accident involving the human character.
Count as accident if you see any of:
- person fallen / collapsed / on ground
- person being struck / hit / trapped
- visible electric shock reaction
- visible fire/steam/gas contacting or affecting the person

Return ONLY JSON:
{ "incident_detected": true/false }
"""


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    mp4_path: str,
    out_dir: str,
    fps: float | None = None,
    max_frames: int | None = None,
) -> list[str]:
    """
    Extract frames from a video using ffmpeg.

    Parameters
    ----------
    mp4_path : str
        Path to the local .mp4 file.
    out_dir : str
        Directory where frames are saved as frame_N.jpg.
    fps : float
        Frames per second to extract (default from config).
    max_frames : int
        Maximum number of frames to extract (default from config).

    Returns
    -------
    List of absolute paths to extracted JPEG frames, sorted by name.
    """
    _fps = fps if fps is not None else config.frame_fallback_fps
    _max = max_frames if max_frames is not None else config.frame_fallback_max_frames

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Clear any existing frames
    for existing in Path(out_dir).glob("frame_*.jpg"):
        existing.unlink()

    out_pattern = os.path.join(out_dir, "frame_%d.jpg")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", mp4_path,
            "-vf", f"fps={_fps}",
            "-frames:v", str(_max),
            "-q:v", "2",
            out_pattern,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    return sorted(str(p) for p in Path(out_dir).glob("frame_*.jpg"))


# ── Frame classification ──────────────────────────────────────────────────────

def classify_frame(
    frame_path: str,
    model: GenerativeModel | None = None,
    prompt: str = FRAME_PROMPT,
) -> bool:
    """
    Classify a single frame as accident (True) or not (False).
    """
    _model = model or vertex_model

    from vertexai.generative_models import Part

    with open(frame_path, "rb") as f:
        img_bytes = f.read()

    img_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")

    response = generate_with_retry(
        model=_model,
        parts=[img_part, prompt],
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )
    parsed = safe_parse_json(response.text) or {}
    return bool(parsed.get("incident_detected", False))


# ── Fallback pipeline ─────────────────────────────────────────────────────────

def frame_fallback(
    local_mp4_path: str,
    frames_dir: str,
    model: GenerativeModel | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    policy: str = "any",
) -> tuple[bool, int, float]:
    """
    Run the frame-level fallback detection pipeline.

    Parameters
    ----------
    local_mp4_path : str
        Path to the local .mp4 file (already downloaded from GCS).
    frames_dir : str
        Directory to store extracted frames.
    policy : str
        "any"      — accident if ANY frame positive  (current default)
        "majority" — accident if > half of frames positive

    Returns
    -------
    (accident_detected, num_frames_checked, latency_s)
    """
    t0 = time.perf_counter()

    frames = extract_frames(local_mp4_path, frames_dir, fps=fps, max_frames=max_frames)
    if not frames:
        return False, 0, time.perf_counter() - t0

    positive_frames = 0
    for frame_path in frames:
        is_accident = classify_frame(frame_path, model=model)
        if is_accident:
            positive_frames += 1
            if policy == "any":
                break  # early exit

    elapsed = time.perf_counter() - t0

    if policy == "any":
        detected = positive_frames > 0
    elif policy == "majority":
        detected = positive_frames > len(frames) / 2
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    return detected, len(frames), elapsed
