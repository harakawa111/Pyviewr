"""Detect sunscreen-coated regions in UV camera frames.

Sunscreen absorbs UV, so coated skin appears dark ("black marker") in a
UV-illuminated mono image. The pipeline is:

1. Downscale (preview speed) and denoise (Gaussian blur).
2. Optional flat-field normalization: divide by a heavily blurred copy of
   the frame to cancel uneven UV illumination / vignetting.
3. Threshold (Otsu / adaptive / manual) with BINARY_INV so dark = detected.
4. Morphological open + close to drop speckle noise and fill pinholes.
5. Connected-component area filter to reject tiny blobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# After flat-field division the image is re-centered around this gray level,
# so manual thresholds stay meaningful in both modes.
_FLAT_FIELD_MID = 128.0


@dataclass
class DetectionParams:
    enabled: bool = False
    view_mode: str = "overlay"  # "overlay" | "mask"
    threshold_mode: str = "manual"  # "manual" | "otsu" | "adaptive"
    manual_threshold: int = 80
    adaptive_block: int = 51
    adaptive_c: int = 10
    flat_field: bool = True
    blur_ksize: int = 5
    morph_ksize: int = 5
    min_area_pct: float = 0.05  # min blob area, % of frame area
    max_process_width: int = 960


def compute_mask(gray: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Binary mask (255 = sunscreen candidate) for a uint8 grayscale image."""
    img = gray
    if params.blur_ksize >= 3:
        k = params.blur_ksize | 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    if params.flat_field:
        # Background estimate must be much larger than the marker features.
        bg_k = max(31, (min(img.shape) // 4) | 1)
        bg = cv2.blur(img, (bg_k, bg_k)).astype(np.float32)
        norm = img.astype(np.float32) / np.maximum(bg, 1.0) * _FLAT_FIELD_MID
        img = np.clip(norm, 0, 255).astype(np.uint8)

    if params.threshold_mode == "otsu":
        _, mask = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    elif params.threshold_mode == "adaptive":
        block = max(3, params.adaptive_block | 1)
        mask = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            params.adaptive_c,
        )
    else:
        _, mask = cv2.threshold(
            img, params.manual_threshold, 255, cv2.THRESH_BINARY_INV
        )

    k = max(1, params.morph_ksize | 1)
    if k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area = params.min_area_pct / 100.0 * mask.size
    if min_area >= 1.0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros(num, dtype=np.uint8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[i] = 255
        mask = keep[labels]
    return mask


def process_frame(
    frame: np.ndarray, params: DetectionParams
) -> tuple[np.ndarray, float]:
    """Run detection on a uint8 grayscale frame.

    Returns (display RGB image, coverage ratio 0..1). The frame is
    downscaled to ``max_process_width`` for preview-rate processing.
    """
    h, w = frame.shape[:2]
    if w > params.max_process_width:
        scale = params.max_process_width / w
        small = cv2.resize(
            frame,
            (params.max_process_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame

    mask = compute_mask(small, params)
    coverage = float(np.count_nonzero(mask)) / float(mask.size)

    if params.view_mode == "mask":
        rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        sel = mask > 0
        # Translucent red fill + green contour outline.
        overlay = rgb[sel].astype(np.uint16)
        overlay = (overlay // 2) + np.array([128, 16, 16], dtype=np.uint16)
        rgb[sel] = np.clip(overlay, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)

    cv2.putText(
        rgb,
        f"coverage: {coverage * 100.0:.1f}%",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(rgb), coverage
