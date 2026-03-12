"""
Video ingestion helpers:
- List videos in GCS
- Download a video from GCS to a local temp path
- (Optional) Download + trim from YouTube using yt-dlp + moviepy
"""

import os
import subprocess
from pathlib import Path

from google.cloud.storage import Blob

from config.settings import config
from pipeline.client import gcs_bucket


# ── GCS helpers ──────────────────────────────────────────────────────────────


def list_video_blobs(prefix: str | None = None) -> list[Blob]:
    """Return all .mp4 blobs under the configured GCS prefix."""
    effective_prefix = prefix or config.gcs_video_prefix
    return [
        blob
        for blob in gcs_bucket.list_blobs(prefix=effective_prefix)
        if blob.name.endswith(".mp4")
    ]


def gcs_uri(blob_name: str) -> str:
    """Construct a gs:// URI for a given blob name."""
    return f"gs://{config.gcs_bucket_name}/{blob_name}"


def download_video(blob_name: str, local_dir: str) -> str:
    """
    Download a GCS blob to a local directory.

    Returns the local file path.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(blob_name))
    gcs_bucket.blob(blob_name).download_to_filename(local_path)
    return local_path


def upload_video(local_path: str, blob_name: str) -> str:
    """
    Upload a local file to GCS.

    Returns the gs:// URI of the uploaded blob.
    """
    blob = gcs_bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return gcs_uri(blob_name)


# ── Local dataset helpers ─────────────────────────────────────────────────────


def find_local_videos(
    root_dir: str,
    originals_only: bool = True,
) -> list[str]:
    """
    Walk the local dataset directory and return video paths.

    Expected structure:
        root_dir/
        └── VID000_<title>/
            ├── original.mp4
            ├── meta.json
            ├── type1/  (augmented .mp4 files)
            └── type2/  (augmented .mp4 files)

    Parameters
    ----------
    root_dir : str
        Path to the videos root (e.g. "data/videos").
    originals_only : bool
        If True  → return only original.mp4 files (73 videos).
        If False → return original + all augmented files (73 × 11).

    Returns
    -------
    Sorted list of absolute .mp4 paths.
    """
    import glob

    root = os.path.abspath(root_dir)
    paths: list[str] = []

    if originals_only:
        # Only files literally named "original.mp4" at depth 1
        pattern = os.path.join(root, "*", "original.mp4")
        paths = glob.glob(pattern)
    else:
        # original.mp4 + everything inside the two augmentation subfolders
        for vid_folder in sorted(os.listdir(root)):
            vid_path = os.path.join(root, vid_folder)
            if not os.path.isdir(vid_path):
                continue
            # Original
            orig = os.path.join(vid_path, "original.mp4")
            if os.path.isfile(orig):
                paths.append(orig)
            # Augmented (one level deeper)
            for sub in sorted(os.listdir(vid_path)):
                sub_path = os.path.join(vid_path, sub)
                if os.path.isdir(sub_path):
                    for f in sorted(os.listdir(sub_path)):
                        if f.lower().endswith(".mp4"):
                            paths.append(os.path.join(sub_path, f))

    return sorted(paths)


def video_id_from_path(path: str) -> str:
    """
    Extract the VID{n} identifier from a local video path.

    e.g. /data/videos/VID042_slip_on_floor/original.mp4 → "VID042"
    """
    import re
    folder = os.path.basename(os.path.dirname(path))
    match = re.search(r"(VID\d+)", folder)
    return match.group(1) if match else os.path.splitext(os.path.basename(path))[0]


def find_local_video_by_vid(root_dir: str, vid_id: str) -> str | None:
    """
    Search root_dir for a folder that starts with vid_id (e.g. "VID1") and
    return the path to original.mp4 inside it, or None if not found.
    """
    root = Path(root_dir)
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.upper().startswith(vid_id.upper()):
            candidate = folder / "original.mp4"
            if candidate.exists():
                return str(candidate)
    return None


def load_clips_from_spreadsheet(excel_path: str) -> list[dict]:
    """
    Load the dataset mapping spreadsheet and return a list of clip dicts.

    Each dict has:
        video_id     : str  — "VID1", "VID2", …
        video_url    : str  — YouTube URL
        start_s      : float — clip start in seconds
        end_s        : float — clip end in seconds
        incident_type: str  — normalised ground-truth label
        accident_present: int — 1 or 0
        near_miss_present: int — 1 or 0
        description  : str

    Rows are 0-indexed; VID number = row_index + 1.
    """
    import pandas as pd
    from config.categories import normalize_gt_category

    df = pd.read_excel(excel_path)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    df = df.reset_index(drop=True)

    clips = []
    for i, row in df.iterrows():
        clips.append({
            "video_id": f"VID{i:03d}",
            "video_url": row["video_url"],
            "start_s": parse_timestamp(row["start_s"]),
            "end_s": parse_timestamp(row["end_s"]),
            "incident_type": normalize_gt_category(row.get("incident_type", "")),
            "accident_present": int(row.get("accident_present", 0)),
            "near_miss_present": int(row.get("near_miss_present", 0)),
            "description": row.get("description", ""),
        })
    return clips


def parse_timestamp(ts) -> float:
    """
    Parse a timestamp to seconds.

    Accepts:
    - "HH:MM:SS" or "MM:SS" strings  (e.g. "00:15:00" → 900.0)
    - datetime.time objects           (e.g. time(0, 15) → 900.0)
    - int / float already in seconds  (returned as-is)
    """
    import datetime

    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime.time):
        return ts.hour * 3600 + ts.minute * 60 + ts.second
    if isinstance(ts, str):
        parts = ts.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


def download_and_trim(
    youtube_url: str,
    start_time: float,
    end_time: float,
    raw_dir: str = "ytvideos",
    trimmed_dir: str = "trimmed_videos",
) -> str:
    """
    Download a YouTube video and trim it to [start_time, end_time].

    Requires: yt-dlp, moviepy, ffmpeg.
    Returns: path to trimmed .mp4 file.
    """
    from moviepy.editor import VideoFileClip  # lazy import

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)

    raw_path = f"{raw_dir}/%(title)s.%(ext)s"

    subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f", "bv*+ba/best",
            "--extractor-args", "youtube:player_client=android",
            "-o", raw_path,
            youtube_url,
        ],
        check=True,
    )

    downloaded_files = os.listdir(raw_dir)
    video_file = max(
        [os.path.join(raw_dir, f) for f in downloaded_files],
        key=os.path.getctime,
    )

    base_name = os.path.splitext(os.path.basename(video_file))[0]

    with VideoFileClip(video_file) as video:
        trimmed = video.subclip(start_time, end_time)
        trimmed_path = f"{trimmed_dir}/{base_name}_accident.mp4"
        trimmed.write_videofile(trimmed_path, codec="libx264", audio=False)

    return trimmed_path
