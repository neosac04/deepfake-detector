from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from backend.utilities.data.video import extract_frames, detect_and_crop_faces


class VideoSequenceDataset(Dataset):
    """CSV-based dataset with columns: video_path,label."""

    def __init__(
        self,
        metadata_csv: str | Path,
        sequence_length: int,
        image_size: int,
        face_padding: int = 20,
    ) -> None:
        self.df = pd.read_csv(metadata_csv)
        required = {"video_path", "label"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(f"{metadata_csv} must contain columns: {required}")

        self.sequence_length = sequence_length
        self.image_size = image_size
        self.face_padding = face_padding
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        label = int(row["label"])

        frames = extract_frames(video_path, self.sequence_length)
        frames = detect_and_crop_faces(frames, padding=self.face_padding, fallback_full_frame=True)
        frames = frames[: self.sequence_length]

        if len(frames) < self.sequence_length:
            frames += [frames[-1]] * (self.sequence_length - len(frames))

        tensor_frames = torch.stack([self.transform(f) for f in frames], dim=0)
        return tensor_frames, torch.tensor(label, dtype=torch.long)
