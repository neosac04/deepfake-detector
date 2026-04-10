from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from deepfake_detector.config import load_config
from deepfake_detector.data.dataset import VideoSequenceDataset
from deepfake_detector.models.resnext_lstm import ResNeXtLSTM
from deepfake_detector.models.efficientnet_gru import EfficientNetGRU
from deepfake_detector.pipelines.train_pipeline import run_training


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
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output", required=True, help="Checkpoint output path (.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.metadata_csv)
    required = {"video_path", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{args.metadata_csv} must contain columns: {required}")

    stratify_col = df["label"] if df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.train.val_split,
        random_state=42,
        stratify=stratify_col,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = Path(tmpdir) / "train_split.csv"
        val_csv = Path(tmpdir) / "val_split.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        train_dataset = VideoSequenceDataset(
            metadata_csv=train_csv,
            sequence_length=cfg.data.sequence_length,
            image_size=cfg.data.image_size,
            face_padding=cfg.data.face_padding,
        )
        val_dataset = VideoSequenceDataset(
            metadata_csv=val_csv,
            sequence_length=cfg.data.sequence_length,
            image_size=cfg.data.image_size,
            face_padding=cfg.data.face_padding,
        )

        model = build_model(cfg)
        run_training(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.learning_rate,
            num_workers=cfg.train.num_workers,
            device=cfg.inference.device,
            checkpoint_path=args.output,
        )


if __name__ == "__main__":
    main()
