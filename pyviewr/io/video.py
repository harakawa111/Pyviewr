"""AVI (MJPG) video writer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(
        self,
        path: Path | str,
        size: tuple[int, int],
        fps: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        # MJPG on Windows expects BGR frames; grayscale is duplicated to 3ch.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            float(fps),
            (int(width), int(height)),
            isColor=True,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.ndim != 2:
            raise ValueError("Expected grayscale frame (H, W).")
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self._writer.write(bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
