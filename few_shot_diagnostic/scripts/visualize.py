"""
visualize.py — Publication-quality diagnostic figures.

All functions save to out_path and return the figure for optional display.
DPI=150, tight layout, consistent colour palette throughout.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive; override in notebook with %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

_PALETTE = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
            "#8bc34a", "#ff5722"]
_MODEL_COLORS = {"FrameCNN": "#3498db", "LRCN": "#2ecc71", "CNN3D": "#e74c3c"}


# ── 1. Learning curves ─────────────────────────────────────────────────────────

def plot_learning_curves(
    histories_by_model: dict[str, list[list[dict]]],
    out_path: str,
):
    """
    histories_by_model: {model_name: [fold_history, ...]}
    Each fold_history is a list of dicts with keys epoch, train_loss, val_loss, train_acc, val_acc.
    Plots mean ± std across folds per model; one row per model.
    """
    models = list(histories_by_model.keys())
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for row, name in enumerate(models):
        folds = histories_by_model[name]
        min_len = min(len(fold) for fold in folds)
        folds_  = [fold[:min_len] for fold in folds]
        epochs  = [h["epoch"] for h in folds_[0]]
        color   = _MODEL_COLORS.get(name, "#555")

        for metric, ax, title in [
            ("loss", axes[row][0], f"{name} — Loss"),
            ("acc",  axes[row][1], f"{name} — Accuracy"),
        ]:
            tr_vals = np.array([[h[f"train_{metric}"] for h in fold] for fold in folds_])
            vl_vals = np.array([[h[f"val_{metric}"]   for h in fold] for fold in folds_])

            for vals, lbl, ls in [(tr_vals, "Train", "-"), (vl_vals, "Val", "--")]:
                mean_ = vals.mean(axis=0)
                std_  = vals.std(axis=0)
                ax.plot(epochs, mean_, ls, color=color, lw=2, label=lbl)
                ax.fill_between(epochs, mean_ - std_, mean_ + std_, alpha=0.15, color=color)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss" if metric == "loss" else "Accuracy")
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

    plt.suptitle("Learning Curves — Mean ± Std across 5 Folds", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig


# ── 2. Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: str,
    normalise: bool = True,
):
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot  = cm.astype(float) / row_sums
        fmt      = ".2f"
    else:
        cm_plot, fmt = cm.astype(int), "d"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8),
                                    max(6, len(class_names) * 0.7)))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig


# ── 3. Per-class F1 heatmap (models × classes) ────────────────────────────────

def plot_per_class_f1_heatmap(
    f1_by_model: dict[str, dict[str, float]],
    class_names: list[str],
    out_path: str,
):
    """
    f1_by_model: {model_name: {class_name: mean_f1}}
    """
    model_names = list(f1_by_model.keys())
    data = np.array([[f1_by_model[m].get(c, 0.0) for c in class_names]
                     for m in model_names])

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.85), len(model_names) * 1.4 + 1))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=model_names,
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 9},
    )
    ax.set_title("Per-Class F1 Score (mean across 5 folds)", fontweight="bold", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=10, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig


# ── 4. Grad-CAM temporal strip ─────────────────────────────────────────────────

def plot_gradcam_temporal(
    model,
    frames_tensor,          # [1, T, C, H, W] or [1, C, T, H, W] for CNN3D
    target_layer,           # e.g. model.backbone.blocks[-1]
    class_idx: int,
    class_name: str,
    vid_id: str,
    out_path: str,
    model_type: str = "lrcn",
):
    """
    Generate per-frame Grad-CAM saliency and render as a horizontal strip.
    Uses pytorch-grad-cam. Falls back to blank frames if library unavailable.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("grad-cam not installed; skipping Grad-CAM plot.")
        return None

    import torch
    import numpy as np
    from torchvision import transforms as T

    _MEAN = np.array([0.485, 0.456, 0.406])
    _STD  = np.array([0.229, 0.224, 0.225])

    model.eval()
    device = next(model.parameters()).device

    if model_type == "cnn3d":
        # For CNN3D: [1, C, T, H, W]
        T_ = frames_tensor.shape[2]
        # Grad-CAM on the last conv layer of the backbone
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=frames_tensor.to(device),
                            targets=None)[0]  # [T, H, W] or [H, W]
        if grayscale_cam.ndim == 2:
            grayscale_cam = np.stack([grayscale_cam] * T_)
    else:
        # Process frame-by-frame
        T_ = frames_tensor.shape[1]
        grayscale_cam = []

        # Wrap model to accept single frame [1, C, H, W]
        class _SingleFrame(torch.nn.Module):
            def __init__(self, backbone): super().__init__(); self.backbone = backbone
            def forward(self, x): return self.backbone(x)

        single_model = _SingleFrame(model.backbone).to(device).eval()
        cam = GradCAM(model=single_model, target_layers=[target_layer])

        for t in range(T_):
            frame = frames_tensor[0, t].unsqueeze(0).to(device)
            gc    = cam(input_tensor=frame, targets=None)[0]
            grayscale_cam.append(gc)
        grayscale_cam = np.stack(grayscale_cam)  # [T, H, W]

    # Denormalize frames for display
    if model_type == "cnn3d":
        frames_np = frames_tensor[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    else:
        frames_np = frames_tensor[0].cpu().numpy()  # [T, C, H, W] → [T, H, W, C]
        frames_np = frames_np.transpose(0, 2, 3, 1)

    frames_rgb = np.clip(frames_np * _STD + _MEAN, 0, 1).astype(np.float32)

    n_show = min(T_, 8)
    indices = np.linspace(0, T_ - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    for col, idx in enumerate(indices):
        cam_img = show_cam_on_image(frames_rgb[idx], grayscale_cam[idx], use_rgb=True)
        axes[0, col].imshow(frames_rgb[idx])
        axes[0, col].set_title(f"f{idx}", fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(cam_img)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Frame", fontsize=8)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)
    fig.suptitle(f"Grad-CAM — {vid_id}  |  Predicted: {class_name}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig


# ── 5. UMAP embeddings ─────────────────────────────────────────────────────────

def plot_umap(
    embeddings:   np.ndarray,      # [N, feat_dim]
    true_labels:  np.ndarray,      # [N] int indices
    pred_labels:  np.ndarray,      # [N] int indices
    class_names:  list[str],
    out_path:     str,
    n_neighbors:  int = 10,
    min_dist:     float = 0.3,
):
    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP plot.")
        return None

    reducer   = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(embeddings)  # [N, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, labels, title in [
        (axes[0], true_labels,  "True Classes"),
        (axes[1], pred_labels,  "Predicted Classes"),
    ]:
        for i, cname in enumerate(class_names):
            mask = labels == i
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=_PALETTE[i % len(_PALETTE)], label=cname,
                s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
            )
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=7, markerscale=0.8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.2)

    plt.suptitle("UMAP — Penultimate Layer Embeddings", fontweight="bold", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig


# ── 6. Frame ablation ──────────────────────────────────────────────────────────

def plot_frame_ablation(
    acc_by_model_and_n: dict[str, dict[int, list[float]]],
    out_path: str,
    metric_name: str = "Binary F1",
):
    """
    acc_by_model_and_n: {model_name: {n_frames: [fold_scores]}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, n_frame_dict in acc_by_model_and_n.items():
        ns     = sorted(n_frame_dict.keys())
        means  = [np.mean(n_frame_dict[n]) for n in ns]
        stds   = [np.std(n_frame_dict[n])  for n in ns]
        color  = _MODEL_COLORS.get(model_name, "#555")
        ax.plot(ns, means, "o-", color=color, lw=2, ms=8, label=model_name)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=color)
        for n, m in zip(ns, means):
            ax.annotate(f"{m:.2f}", (n, m), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=color)

    ax.set_xlabel("Number of frames sampled per clip", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} vs Frames Sampled — Ablation Study", fontweight="bold", fontsize=12)
    ax.set_xticks(sorted({n for d in acc_by_model_and_n.values() for n in d}))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig


# ── 7. Results summary bar chart ───────────────────────────────────────────────

def plot_results_summary(
    summaries: dict[str, dict],
    out_path: str,
):
    """
    summaries: {model_name: {"binary_f1_mean": ..., "binary_f1_std": ...,
                              "macro_f1_mean": ...,  "macro_f1_std":  ...}}
    """
    models = list(summaries.keys())
    x      = np.arange(len(models))
    width  = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (metric, label, color) in enumerate([
        ("binary_f1", "Binary F1", "#3498db"),
        ("macro_f1",  "Macro F1",  "#2ecc71"),
    ]):
        means = [summaries[m][f"{metric}_mean"] for m in models]
        stds  = [summaries[m][f"{metric}_std"]  for m in models]
        bars  = ax.bar(x + (offset - 0.5) * width, means, width,
                       label=label, color=color, alpha=0.85,
                       yerr=stds, capsize=4, error_kw={"linewidth": 1.5})
        for bar, mean_ in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean_:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score (mean ± std, 5 folds)", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison — Binary F1 and Macro F1", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return fig
