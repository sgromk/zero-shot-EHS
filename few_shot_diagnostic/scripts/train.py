"""
train.py — Training loop with compact epoch-level output.

Output format per fold (≤ n_epochs/log_every + 2 lines):
  Epoch  5 | loss=1.234  acc=0.603 | val_loss=1.109  val_acc=0.655 ← best
  Epoch 10 | loss=0.956  acc=0.710 | val_loss=0.998  val_acc=0.690 ← best
  Early stop at epoch 22 — restoring epoch 10 (val_acc=0.690)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainResult:
    best_val_acc:   float
    best_val_f1:    float
    best_epoch:     int
    history:        list[dict]          = field(default_factory=list)
    best_state_dict: dict | None        = None


def _run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with autocast(enabled=(scaler is not None)):
                logits = model(batch_x)
                loss   = criterion(logits, batch_y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            preds = logits.argmax(dim=1)
            correct   += (preds == batch_y).sum().item()
            total_loss += loss.item() * len(batch_y)
            n += len(batch_y)

    return total_loss / n, correct / n


def train_model(
    model:          nn.Module,
    loader_train:   DataLoader,
    loader_val:     DataLoader,
    *,
    n_epochs:       int   = 50,
    log_every:      int   = 5,
    patience:       int   = 15,
    device:         str   = "cuda",
    use_amp:        bool  = True,
) -> TrainResult:
    """
    Train with AdamW + CosineAnnealingLR + optional mixed precision.
    Prints one summary line every log_every epochs (≤10 lines total).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Build optimiser with differential learning rates if model supports it
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr_backbone=1e-4, lr_head=1e-3)
    else:
        param_groups = [{"params": model.parameters(), "lr": 1e-3}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    # Loss — criterion must be passed in with class weights already set
    # (caller builds criterion with compute_class_weights)
    criterion = getattr(model, "_criterion", nn.CrossEntropyLoss())

    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    best_val_acc   = 0.0
    best_val_f1    = 0.0
    best_epoch     = 0
    best_state     = None
    no_improve     = 0
    history        = []

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, loader_train, criterion, optimizer, scaler, device, train=True)
        vl_loss, vl_acc = _run_epoch(model, loader_val,   criterion, None,      None,   device, train=False)
        scheduler.step()

        is_best = vl_acc > best_val_acc
        if is_best:
            best_val_acc = vl_acc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss":   vl_loss, "val_acc":   vl_acc,
        })

        if epoch % log_every == 0:
            overfit_tag = "  [overfit]" if vl_loss > (history[best_epoch - 1]["val_loss"] + 0.15) else ""
            best_tag    = " ← best" if is_best else ""
            print(
                f"  Epoch {epoch:3d} | "
                f"loss={tr_loss:.4f}  acc={tr_acc:.3f} | "
                f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}"
                f"{best_tag}{overfit_tag}"
            )

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} — restoring epoch {best_epoch} (val_acc={best_val_acc:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        best_val_acc    = best_val_acc,
        best_val_f1     = best_val_f1,
        best_epoch      = best_epoch,
        history         = history,
        best_state_dict = best_state,
    )


def attach_criterion(model: nn.Module, class_weights: torch.Tensor | None = None):
    """Attach a weighted CrossEntropyLoss to the model so train_model can use it."""
    model._criterion = nn.CrossEntropyLoss(weight=class_weights)
