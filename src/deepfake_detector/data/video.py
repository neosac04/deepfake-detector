from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None


def extract_frames(video_path: str | Path, max_frames: int) -> List[np.ndarray]:
    """Extract up to max_frames with uniform sampling from the video."""
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    if len(frames) <= max_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


def detect_and_crop_faces(
    frames: List[np.ndarray],
    padding: int = 20,
    fallback_full_frame: bool = True,
) -> List[np.ndarray]:
    """Crop first detected face from each frame; fallback to full frame if needed."""
    cropped: List[np.ndarray] = []
    for frame in frames:
        if face_recognition is None:
            cropped.append(frame)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            if fallback_full_frame:
                cropped.append(frame)
            continue

        top, right, bottom, left = locations[0]
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        face = frame[top:bottom, left:right]
        cropped.append(face if face.size > 0 else frame)

    if not cropped:
        raise ValueError("No frames available after face extraction.")
    return cropped
