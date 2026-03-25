"""
evaluate.py — Per-fold evaluation, confusion matrix, and embedding extraction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    classification_report,
)


@dataclass
class FoldResult:
    fold:            int
    binary_f1:       float
    binary_recall:   float
    binary_precision: float
    macro_f1:        float
    per_class_f1:    dict[str, float]
    preds:           np.ndarray
    labels:          np.ndarray
    probs:           np.ndarray          # [N, n_classes] softmax probabilities
    embeddings:      np.ndarray          # [N, feat_dim] penultimate features
    confusion_mat:   np.ndarray


def _extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Hook into the embedding layer to capture penultimate features."""
    embeddings = []

    def hook_fn(module, input, output):
        # Capture the input to the final linear (= penultimate representation)
        embeddings.append(input[0].detach().cpu())

    emb_layer = model.get_embedding_layer() if hasattr(model, "get_embedding_layer") else None
    handle = emb_layer.register_forward_hook(hook_fn) if emb_layer is not None else None

    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            model(batch_x)

    if handle is not None:
        handle.remove()

    return torch.cat(embeddings, dim=0).numpy() if embeddings else np.array([])


def evaluate_fold(
    model:       nn.Module,
    loader:      DataLoader,
    class_names: list[str],
    device:      torch.device | str,
    fold:        int = 0,
    binary_positive_label: str = "Accident",
) -> FoldResult:
    """
    Run inference on loader and compute all diagnostic metrics.

    For binary task: class_names = ["No Accident", "Accident"]
    For multiclass:  class_names = MERGED_CATEGORIES + ["No Accident"]
    """
    device = torch.device(device)
    model  = model.to(device).eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x  = batch_x.to(device, non_blocking=True)
            logits   = model(batch_x)
            probs    = torch.softmax(logits, dim=1).cpu()
            preds    = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(batch_y)
            all_probs.append(probs)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()

    # Embeddings (captured via hook)
    embs = _extract_embeddings(model, loader, device)

    # Per-class F1 (computed first so macro_f1 is available as fallback)
    f1_per = f1_score(labels, preds, average=None, labels=list(range(len(class_names))), zero_division=0)
    per_class_f1 = {class_names[i]: float(f1_per[i]) for i in range(len(class_names))}
    macro_f1 = float(f1_per.mean())

    # Binary metrics — only meaningful when "Accident" class exists in class_names
    if len(class_names) == 2:
        pos_idx = 1
    else:
        pos_idx = class_names.index(binary_positive_label) if binary_positive_label in class_names else -1

    if pos_idx >= 0:
        binary_labels = (labels == pos_idx).astype(int)
        binary_preds  = (preds  == pos_idx).astype(int)
        bin_f1  = f1_score(binary_labels, binary_preds, zero_division=0)
        bin_rec = recall_score(binary_labels, binary_preds, zero_division=0)
        bin_pre = precision_score(binary_labels, binary_preds, zero_division=0)
    else:
        # Multiclass task — no single positive class; fall back to macro averages
        bin_f1  = macro_f1
        bin_rec = float(recall_score(labels, preds, average="macro", zero_division=0))
        bin_pre = float(precision_score(labels, preds, average="macro", zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    return FoldResult(
        fold             = fold,
        binary_f1        = bin_f1,
        binary_recall    = bin_rec,
        binary_precision = bin_pre,
        macro_f1         = macro_f1,
        per_class_f1     = per_class_f1,
        preds            = preds,
        labels           = labels,
        probs            = probs,
        embeddings       = embs,
        confusion_mat    = cm,
    )


def summarise_cv(
    fold_results: list[FoldResult],
    class_names:  list[str],
) -> dict:
    """
    Aggregate per-fold results into mean ± std.
    Returns dict usable in a display() table.
    """
    binary_f1s = [r.binary_f1 for r in fold_results]
    macro_f1s  = [r.macro_f1  for r in fold_results]

    per_class = {c: [] for c in class_names}
    for r in fold_results:
        for c in class_names:
            per_class[c].append(r.per_class_f1.get(c, 0.0))

    summary = {
        "binary_f1_mean":  float(np.mean(binary_f1s)),
        "binary_f1_std":   float(np.std(binary_f1s)),
        "macro_f1_mean":   float(np.mean(macro_f1s)),
        "macro_f1_std":    float(np.std(macro_f1s)),
        "per_class_f1_mean": {c: float(np.mean(per_class[c])) for c in class_names},
        "per_class_f1_std":  {c: float(np.std(per_class[c]))  for c in class_names},
    }
    return summary


def best_fold(fold_results: list[FoldResult]) -> FoldResult:
    """Return the fold with the highest binary F1."""
    return max(fold_results, key=lambda r: r.binary_f1)


def print_summary(model_name: str, summary: dict):
    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")
    print(f"  Binary F1:  {summary['binary_f1_mean']:.3f} ± {summary['binary_f1_std']:.3f}")
    print(f"  Macro  F1:  {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}")
