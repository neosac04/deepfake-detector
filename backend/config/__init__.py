from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml


@dataclass
class DataConfig:
    sequence_length: int
    image_size: int
    face_padding: int


@dataclass
class ModelConfig:
    name: str
    num_classes: int
    hidden_dim: int
    dropout: float
    pretrained_backbone: bool


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    num_workers: int
    val_split: float


@dataclass
class InferenceConfig:
    device: str


@dataclass
class ProjectConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig
    labels: Dict[int, str]


def _to_dataclass(cfg: Dict[str, Any]) -> ProjectConfig:
    labels = {int(k): str(v) for k, v in cfg["labels"].items()}
    return ProjectConfig(
        data=DataConfig(**cfg["data"]),
        model=ModelConfig(**cfg["model"]),
        train=TrainConfig(**cfg["train"]),
        inference=InferenceConfig(**cfg["inference"]),
        labels=labels,
    )


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_dataclass(raw)
