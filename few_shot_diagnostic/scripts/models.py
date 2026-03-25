"""
models.py — FrameCNN, LRCN, CNN3D model definitions.

All models:
  - Accept a config dict with 'n_classes' and 'n_frames'
  - Return raw logits [B, n_classes]
  - Expose get_param_groups() for differential learning rates
  - Expose get_embedding_layer() for UMAP feature extraction
"""
import torch
import torch.nn as nn


# ── FrameCNN — spatial baseline (no temporal modelling) ───────────────────────

class FrameCNN(nn.Module):
    """
    EfficientNet-B2 feature extractor applied per-frame, mean-pooled over time.
    Establishes the spatial-only performance ceiling. Temporal gain is defined as
    (LRCN or CNN3D accuracy) - FrameCNN accuracy.
    """

    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1408 for B2
        self.head = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C, H, W] → [B, feat_dim]"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)          # [B*T, feat_dim]
        feats = feats.view(B, T, -1)      # [B, T, feat_dim]
        return feats.mean(dim=1)          # [B, feat_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def get_param_groups(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3):
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(),     "lr": lr_head},
        ]

    def get_embedding_layer(self):
        """Return the penultimate module for Grad-CAM / embedding hooks."""
        return self.head


# ── LRCN — Long-term Recurrent Convolutional Network ─────────────────────────

class LRCN(nn.Module):
    """
    EfficientNet-B2 frame encoder (partially frozen) + BiLSTM sequence model.
    First 3 backbone blocks frozen; later blocks fine-tuned to allow feature
    adaptation while limiting overfitting on the small dataset.
    """

    def __init__(self, n_classes: int, pretrained: bool = True,
                 lstm_hidden: int = 512, lstm_layers: int = 2, lstm_dropout: float = 0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1408

        # Freeze first 3 blocks of EfficientNet-B2
        self._freeze_early_blocks(n_blocks=3)

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(lstm_hidden * 2, n_classes)  # *2 for bidirectional

    def _freeze_early_blocks(self, n_blocks: int):
        """Freeze the first n_blocks of EfficientNet stages."""
        children = list(self.backbone.children())
        for child in children[:n_blocks]:
            for p in child.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C, H, W] → [B, lstm_hidden*2] (last hidden state)"""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.backbone(x_flat)           # [B*T, feat_dim]
        feats = feats.view(B, T, -1)            # [B, T, feat_dim]
        _, (h_n, _) = self.lstm(feats)          # h_n: [2*layers, B, hidden]
        # Concatenate forward and backward final hidden states
        h_fwd = h_n[-2]                         # forward last layer
        h_bwd = h_n[-1]                         # backward last layer
        return torch.cat([h_fwd, h_bwd], dim=1) # [B, hidden*2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.forward_features(x)))

    def get_param_groups(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3):
        frozen_ids = {id(p) for p in self.parameters() if not p.requires_grad}
        backbone_params = [p for p in self.backbone.parameters()
                           if id(p) not in frozen_ids]
        return [
            {"params": backbone_params,       "lr": lr_backbone},
            {"params": self.lstm.parameters(), "lr": lr_head},
            {"params": self.head.parameters(), "lr": lr_head},
        ]

    def get_embedding_layer(self):
        return self.head


# ── CNN3D — 3D ResNet-18 pretrained on Kinetics-400 ──────────────────────────

class CNN3D(nn.Module):
    """
    R3D-18 (3D ResNet) pretrained on Kinetics-400.
    Input convention: [B, C, T, H, W] — standard torchvision video format.
    """

    def __init__(self, n_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        base = r3d_18(weights=weights)

        # Strip the original head; keep feature extractor
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # ends at avgpool
        feat_dim = 512  # R3D-18 penultimate feature size

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] → [B, 512]"""
        feats = self.backbone(x)   # [B, 512, 1, 1, 1]
        return feats.flatten(1)    # [B, 512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.forward_features(x)))

    def get_param_groups(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3):
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(),     "lr": lr_head},
        ]

    def get_embedding_layer(self):
        return self.head


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Args:
        name: "framecnn" | "lrcn" | "cnn3d"
    """
    name = name.lower()
    if name == "framecnn":
        return FrameCNN(n_classes=n_classes, pretrained=pretrained)
    elif name == "lrcn":
        return LRCN(n_classes=n_classes, pretrained=pretrained)
    elif name == "cnn3d":
        return CNN3D(n_classes=n_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from framecnn | lrcn | cnn3d")


def count_params(model: nn.Module) -> dict:
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
