from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.config import load_config
from backend.models.resnext_lstm import ResNeXtLSTM
from backend.models.efficientnet_gru import EfficientNetGRU
from backend.utilities.pipelines.inference_pipeline import (
    load_checkpoint,
    predict_image,
    predict_video,
)


def build_model(cfg):
    if cfg.model.name == "resnext_lstm":
        return ResNeXtLSTM(
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            pretrained_backbone=cfg.model.pretrained_backbone,
        )
    if cfg.model.name == "efficientnet_gru":
        return EfficientNetGRU(
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            pretrained_backbone=cfg.model.pretrained_backbone,
        )
    raise ValueError(f"Unknown model name: {cfg.model.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", help="Path to a video file")
    source_group.add_argument("--image", help="Path to an image file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    model = load_checkpoint(model, args.checkpoint, cfg.inference.device)

    if args.image:
        pred_idx, confidence = predict_image(
            model=model,
            image_path=args.image,
            sequence_length=cfg.data.sequence_length,
            image_size=cfg.data.image_size,
            device=cfg.inference.device,
        )
    else:
        pred_idx, confidence = predict_video(
            model=model,
            video_path=args.video,
            sequence_length=cfg.data.sequence_length,
            image_size=cfg.data.image_size,
            device=cfg.inference.device,
        )
    label = cfg.labels.get(pred_idx, str(pred_idx))
    print(f"Prediction: {label} (index={pred_idx}, confidence={confidence:.4f})")


if __name__ == "__main__":
    main()
