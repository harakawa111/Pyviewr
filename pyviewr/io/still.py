"""Still image saving (PNG)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_still(path: Path | str, frame: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.ndim != 2:
        raise ValueError("Expected grayscale frame (H, W).")
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write still image: {path}")
    return path
