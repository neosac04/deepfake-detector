from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torchvision import transforms

from deepfake_detector.data.video import extract_frames, detect_and_crop_faces


_TRANSFORM_CACHE = {}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _get_transform(image_size: int):
    if image_size not in _TRANSFORM_CACHE:
        _TRANSFORM_CACHE[image_size] = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return _TRANSFORM_CACHE[image_size]


def _load_image(image_path: str | Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def _build_input_batch(frames, sequence_length: int, image_size: int):
    frames = frames[:sequence_length]
    if not frames:
        raise ValueError("No frames available for inference.")
    if len(frames) < sequence_length:
        frames += [frames[-1]] * (sequence_length - len(frames))

    transform = _get_transform(image_size)
    return torch.stack([transform(frame) for frame in frames], dim=0).unsqueeze(0)


def load_checkpoint(model, checkpoint_path: str | Path, device: str):
    device_t = torch.device(device)
    state = torch.load(checkpoint_path, map_location=device_t)
    model.load_state_dict(state)
    model.to(device_t)
    model.eval()
    return model


@torch.no_grad()
def predict_video(
    model,
    video_path: str | Path,
    sequence_length: int,
    image_size: int,
    device: str = "cpu",
):
    frames = extract_frames(video_path, max_frames=sequence_length)
    frames = detect_and_crop_faces(frames, padding=20, fallback_full_frame=True)
    x = _build_input_batch(frames, sequence_length=sequence_length, image_size=image_size)
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence


@torch.no_grad()
def predict_image(
    model,
    image_path: str | Path,
    sequence_length: int,
    image_size: int,
    device: str = "cpu",
):
    frame = _load_image(image_path)
    frames = detect_and_crop_faces([frame], padding=20, fallback_full_frame=True)
    x = _build_input_batch(frames, sequence_length=sequence_length, image_size=image_size)
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence
