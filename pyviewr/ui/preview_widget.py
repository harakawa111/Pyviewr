"""Fit-to-widget grayscale preview."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class PreviewWidget(QWidget):
    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(title or "No signal")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(240, 240)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._label, stretch=1)

        self._last: np.ndarray | None = None

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def clear_frame(self) -> None:
        self._last = None
        self._label.setPixmap(QPixmap())
        self._label.setText("No signal")

    def show_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self._last = frame
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last is not None:
            self._render()

    def _render(self) -> None:
        frame = self._last
        if frame is None:
            return
        h, w = frame.shape[:2]
        if frame.ndim != 2:
            return
        bytes_per_line = w
        qimg = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        ).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.setText("")
