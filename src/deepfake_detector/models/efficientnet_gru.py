from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _load_efficientnet_b0(pretrained: bool):
    if pretrained:
        try:
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except AttributeError:
            return models.efficientnet_b0(pretrained=True)
    try:
        return models.efficientnet_b0(weights=None)
    except TypeError:
        return models.efficientnet_b0(pretrained=False)


class EfficientNetGRU(nn.Module):
    """Optional baseline inspired by repo-2 CNN + recurrent setup."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone = _load_efficientnet_b0(pretrained_backbone)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(
            input_size=1280,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W]
        bsz, seq_len, channels, height, width = x.shape
        x = x.view(bsz * seq_len, channels, height, width)
        feats = self.feature_extractor(x)
        feats = self.pool(feats).flatten(1)  # [B*T, 1280]
        feats = feats.view(bsz, seq_len, 1280)
        out, _ = self.gru(feats)
        out = self.dropout(out[:, -1, :])
        logits = self.classifier(out)
        return logits
