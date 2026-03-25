"""
dataset.py — VideoDataset, split utilities, and ground-truth loading.

Self-contained: no imports from the project's pipeline/, evaluation/, or config/
packages so it runs identically in Colab and locally.

Augmentation policy:
  - Splits are performed on ORIGINALS ONLY (73 clips) via StratifiedKFold.
  - Augmented clips (type1/, type2/ variants) are added to TRAINING folds only.
    Val folds are always original clips, preventing data leakage.
"""
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms

# ── Canonical category list (mirrors config/categories.py) ─────────────────────
MERGED_CATEGORIES: list[str] = [
    "Arc Flash",
    "Caught In Machine",
    "Electrocution",
    "Fall",
    "Fire",
    "Gas Inhalation",
    "Lifting",
    "Slip",
    "Struck by Object",
    "Trip",
    "Vehicle Incident",
]

# ── GT normalisation map (mirrors config/categories.py) ───────────────────────
_GT_NORM: dict[str, str] = {
    "Caught in Machine":    "Caught In Machine",
    "Trip and Fall":        "Trip",
    "Slip and Fall":        "Slip",
    "Electrocution and Fall": "Electrocution",
    "No Accident":          "No Accident",
}


def normalize_gt_category(category: str) -> str:
    if not isinstance(category, str):
        return "No Accident"
    return _GT_NORM.get(category.strip(), category.strip())


# ── Ground-truth loader (mirrors evaluation/metrics.py:load_ground_truth) ──────

def load_ground_truth(excel_path: str) -> pd.DataFrame:
    """
    Load dataset_mapping.xlsx and return a tidy DataFrame with columns:
        video_id_clean, true_category, true_binary
    """
    df = pd.read_excel(excel_path)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    df = df.reset_index(drop=True)
    df["video_id_clean"] = [f"VID{i:03d}" for i in df.index]
    df = df.rename(columns={"incident_type": "true_category"})
    df["true_category"] = df["true_category"].apply(normalize_gt_category)

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
    return df

# ── ImageNet normalisation stats ───────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])


def _extract_frames(video_path: str, n_frames: int) -> np.ndarray | None:
    """Extract n_frames uniformly from a video. Returns uint8 [T, H, W, C] or None."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # Duplicate last good frame rather than drop
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return np.stack(frames)  # [T, H, W, C]


def preload_frames(paths: list[str], n_frames: int) -> dict:
    """
    Pre-extract frames for all paths into a dict {path: uint8 [T,H,W,C]}.
    Call once before training; pass the result as frame_cache to VideoDataset.
    Eliminates repeated cv2 disk seeks across epochs.
    """
    cache = {}
    for i, p in enumerate(paths):
        frames = _extract_frames(p, n_frames)
        if frames is not None:
            cache[p] = frames
        if (i + 1) % 25 == 0:
            print(f"  preloaded {i + 1}/{len(paths)}", flush=True)
    print(f"  preloaded {len(paths)}/{len(paths)} — done")
    return cache


class VideoDataset(Dataset):
    """
    Args:
        paths       : list of .mp4 paths
        labels      : list of int class indices (same length as paths)
        n_frames    : number of frames to sample uniformly from each video
        transform   : per-frame torchvision transform
        model_type  : "cnn3d" → returns [C, T, H, W]; otherwise [T, C, H, W]
        frame_cache : optional dict {path: uint8 [T,H,W,C]} from preload_frames()
    """

    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        n_frames: int = 16,
        transform=None,
        model_type: str = "lrcn",
        frame_cache: dict | None = None,
    ):
        self.paths       = paths
        self.labels      = labels
        self.n_frames    = n_frames
        self.transform   = transform or get_transform("val")
        self.model_type  = model_type
        self.frame_cache = frame_cache or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        frames_np = self.frame_cache.get(p)
        if frames_np is None:
            frames_np = _extract_frames(p, self.n_frames)
        if frames_np is None:
            frames_np = np.zeros((self.n_frames, 224, 224, 3), dtype=np.uint8)

        frames = torch.stack([self.transform(f) for f in frames_np])  # [T, C, H, W]

        if self.model_type == "cnn3d":
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, self.labels[idx]


# ── Label utilities ────────────────────────────────────────────────────────────

def extract_base_vid(path: str) -> str | None:
    """Return 'VID042' from any path containing VID042_... """
    m = re.search(r"(VID\d+)", path)
    return m.group(1) if m else None


def build_aug_labels(aug_paths: list[str], vid_to_label: dict[str, int]) -> list[int]:
    """Map augmented paths back to the label of their parent original clip."""
    labels = []
    for p in aug_paths:
        vid = extract_base_vid(p)
        if vid is None or vid not in vid_to_label:
            raise ValueError(f"Cannot find label for augmented path: {p}")
        labels.append(vid_to_label[vid])
    return labels


# ── Split builder ──────────────────────────────────────────────────────────────

def make_splits(
    orig_paths:   list[str],
    orig_labels:  list[int],
    aug_paths:    list[str],
    aug_labels:   list[int],
    n_folds:      int = 5,
    seed:         int = 42,
) -> list[tuple]:
    """
    Stratified k-fold on originals; augmented clips added to train only.

    Returns list of (train_paths, train_labels, val_paths, val_labels) per fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    orig_arr   = np.array(orig_paths)
    labels_arr = np.array(orig_labels)
    aug_arr    = np.array(aug_paths)
    aug_lbl    = np.array(aug_labels)

    folds = []
    for train_idx, val_idx in skf.split(orig_arr, labels_arr):
        train_vids = {extract_base_vid(p) for p in orig_arr[train_idx]}

        # Only augmented clips whose parent original is in this training fold
        aug_train = [(p, l) for p, l in zip(aug_arr, aug_lbl)
                     if extract_base_vid(p) in train_vids]
        aug_train_paths  = [p for p, _ in aug_train]
        aug_train_labels = [l for _, l in aug_train]

        tr_paths  = orig_arr[train_idx].tolist() + aug_train_paths
        tr_labels = labels_arr[train_idx].tolist() + aug_train_labels
        val_paths  = orig_arr[val_idx].tolist()
        val_labels = labels_arr[val_idx].tolist()
        folds.append((tr_paths, tr_labels, val_paths, val_labels))

    return folds


def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.zeros(n_classes)
    for lbl in labels:
        counts[lbl] += 1
    counts = counts.clamp(min=1)
    weights = len(labels) / (n_classes * counts)
    return weights
