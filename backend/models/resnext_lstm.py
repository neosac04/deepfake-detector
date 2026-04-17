from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _load_resnext50(pretrained: bool):
    if pretrained:
        try:
            return models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        except AttributeError:
            return models.resnext50_32x4d(pretrained=True)
    try:
        return models.resnext50_32x4d(weights=None)
    except TypeError:
        return models.resnext50_32x4d(pretrained=False)


class ResNeXtLSTM(nn.Module):
    """Primary model inspired by repo-1 architecture."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone = _load_resnext50(pretrained_backbone)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(
            input_size=2048,
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
        feats = self.pool(feats).flatten(1)  # [B*T, 2048]
        feats = feats.view(bsz, seq_len, 2048)
        out, _ = self.lstm(feats)
        out = self.dropout(out[:, -1, :])
        logits = self.classifier(out)
        return logits
