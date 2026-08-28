"""Live image enhancement for UV preview / detection / save.

Applied before sunscreen detection so thresholding sees the same image
as the preview. Still and video saves use the same enhance. Defaults are
identity (no-op) for zero cost when unused.

Typical uses under UV + skin specular glare:
- highlight_compress: soft-knee roll-off of bright specular patches
- contrast / CLAHE: lift marker vs skin separation under uneven light
- sharpness: unsharp mask for marker edges (after denoise in detection)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhanceParams:
    brightness: float = 0.0  # -100 .. 100 (pixel add)
    contrast: float = 1.0  # 0.5 .. 2.5 (pivot 128)
    sharpness: float = 0.0  # 0 .. 3 (unsharp amount)
    clahe_clip: float = 0.0  # 0 = off; else clip limit (~0.5 .. 8)
    highlight_compress: float = 0.0  # 0 .. 1 soft-knee on bright end


def is_identity(params: EnhanceParams) -> bool:
    return (
        abs(params.brightness) < 1e-6
        and abs(params.contrast - 1.0) < 1e-6
        and params.sharpness < 1e-6
        and params.clahe_clip < 1e-6
        and params.highlight_compress < 1e-6
    )


def apply(frame: np.ndarray, params: EnhanceParams) -> np.ndarray:
    """Enhance a uint8 grayscale (or pass through RGB unchanged)."""
    if frame is None or frame.size == 0 or is_identity(params):
        return frame
    if frame.ndim != 2:
        return frame

    img = frame
    if params.highlight_compress > 1e-6:
        img = _compress_highlights(img, params.highlight_compress)

    if abs(params.brightness) > 1e-6 or abs(params.contrast - 1.0) > 1e-6:
        # Pivot mid-gray: contrast*(x-128)+128+brightness
        beta = params.brightness + 128.0 * (1.0 - params.contrast)
        img = cv2.convertScaleAbs(img, alpha=params.contrast, beta=beta)

    if params.clahe_clip > 1e-6:
        clahe = cv2.createCLAHE(
            clipLimit=float(params.clahe_clip),
            tileGridSize=(8, 8),
        )
        img = clahe.apply(img)

    if params.sharpness > 1e-6:
        img = _unsharp(img, params.sharpness)

    return img


def _compress_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    """Soft-knee compression of bright values; darks stay nearly untouched."""
    amount = float(np.clip(amount, 0.0, 1.0))
    x = img.astype(np.float32) * (1.0 / 255.0)
    # Stronger roll-off near 1.0: y = x * (1 - a * x^2)
    y = x * (1.0 - amount * x * x)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _unsharp(img: np.ndarray, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    # out = img + amount * (img - blurred)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
