"""Side panels: camera feature sliders and sunscreen-detection controls."""

from __future__ import annotations

import math
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from pyviewr.camera.manager import FeatureInfo
from pyviewr.processing.enhance import EnhanceParams
from pyviewr.processing.sunscreen import DetectionParams

_SLIDER_STEPS = 1000


class _FeatureRow(QWidget):
    """Slider + value label mapped onto a float range (optionally log-scale)."""

    changed = Signal(float)

    def __init__(self, log_scale: bool, fmt, parent=None) -> None:
        super().__init__(parent)
        self._log = log_scale
        self._fmt = fmt
        self._min = 0.0
        self._max = 1.0
        self._updating = False

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _SLIDER_STEPS)
        self._label = QLabel("—")
        self._label.setMinimumWidth(72)
        self._label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

        self._slider.valueChanged.connect(self._on_slider)

    def set_range(self, minimum: float, maximum: float, value: float) -> None:
        self._min = minimum
        self._max = max(maximum, minimum + 1e-9)
        if self._log:
            self._min = max(self._min, 1e-9)
        self._updating = True
        self._slider.setValue(self._to_step(value))
        self._updating = False
        self._label.setText(self._fmt(value))

    def value(self) -> float:
        return self._from_step(self._slider.value())

    def _to_step(self, value: float) -> int:
        value = min(max(value, self._min), self._max)
        if self._log:
            t = math.log(value / self._min) / math.log(self._max / self._min)
        else:
            t = (value - self._min) / (self._max - self._min)
        return round(t * _SLIDER_STEPS)

    def _from_step(self, step: int) -> float:
        t = step / _SLIDER_STEPS
        if self._log:
            return self._min * (self._max / self._min) ** t
        return self._min + (self._max - self._min) * t

    def _on_slider(self, step: int) -> None:
        value = self._from_step(step)
        self._label.setText(self._fmt(value))
        if not self._updating:
            self.changed.emit(value)


def _fmt_exposure(us: float) -> str:
    if us >= 1000.0:
        return f"{us / 1000.0:.2f} ms"
    return f"{us:.0f} µs"


class CameraControlPanel(QGroupBox):
    """Live sliders for ExposureTime / Gain / Gamma (applied to all cameras)."""

    feature_changed = Signal(str, float)

    def __init__(self, parent=None) -> None:
        super().__init__("Camera settings", parent)
        self._rows: dict[str, _FeatureRow] = {
            "ExposureTime": _FeatureRow(log_scale=True, fmt=_fmt_exposure),
            "Gain": _FeatureRow(log_scale=False, fmt=lambda v: f"{v:.1f} dB"),
            "Gamma": _FeatureRow(log_scale=False, fmt=lambda v: f"{v:.2f}"),
        }
        form = QFormLayout(self)
        form.addRow("Exposure", self._rows["ExposureTime"])
        form.addRow("Gain", self._rows["Gain"])
        form.addRow("Gamma", self._rows["Gamma"])

        for name, row in self._rows.items():
            row.changed.connect(
                lambda value, n=name: self.feature_changed.emit(n, value)
            )
        self.setEnabled(False)

    def load_features(self, infos: dict[str, FeatureInfo | None]) -> None:
        """Populate slider ranges from camera values; enable available rows."""
        any_available = False
        for name, row in self._rows.items():
            info = infos.get(name)
            if info is None:
                row.setEnabled(False)
                continue
            maximum = info.maximum
            if name == "ExposureTime":
                # Cap at 1 s so the slider stays usable for live preview.
                maximum = min(maximum, 1_000_000.0)
            row.set_range(info.minimum, maximum, info.value)
            row.setEnabled(True)
            any_available = True
        self.setEnabled(any_available)

    def set_disconnected(self) -> None:
        self.setEnabled(False)

    def current_features(self) -> dict[str, float]:
        """Slider values for features that are currently enabled."""
        return {
            name: row.value()
            for name, row in self._rows.items()
            if row.isEnabled()
        }


class EnhanceControlPanel(QGroupBox):
    """Software enhance sliders (preview + detection input)."""

    params_changed = Signal(object)  # EnhanceParams

    def __init__(self, parent=None) -> None:
        super().__init__("Image enhance (preview + save)", parent)
        self._params = EnhanceParams()

        self._row_brightness = _FeatureRow(
            log_scale=False, fmt=lambda v: f"{v:+.0f}"
        )
        self._row_brightness.set_range(-80, 80, self._params.brightness)

        self._row_contrast = _FeatureRow(
            log_scale=False, fmt=lambda v: f"{v:.2f}"
        )
        self._row_contrast.set_range(0.50, 2.50, self._params.contrast)

        self._row_sharp = _FeatureRow(
            log_scale=False, fmt=lambda v: f"{v:.2f}"
        )
        self._row_sharp.set_range(0.0, 3.0, self._params.sharpness)

        self._row_clahe = _FeatureRow(
            log_scale=False, fmt=lambda v: "Off" if v < 0.05 else f"{v:.1f}"
        )
        self._row_clahe.set_range(0.0, 8.0, self._params.clahe_clip)

        self._row_highlights = _FeatureRow(
            log_scale=False, fmt=lambda v: f"{v * 100:.0f}%"
        )
        self._row_highlights.set_range(
            0.0, 1.0, self._params.highlight_compress
        )

        self._btn_reset = QPushButton("Reset")

        self._row_brightness.setToolTip("Overall brightness offset")
        self._row_contrast.setToolTip("Global contrast around mid-gray")
        self._row_sharp.setToolTip("Unsharp mask — strengthens marker edges")
        self._row_clahe.setToolTip(
            "Local contrast (CLAHE) — helps markers under uneven UV light; 0 = Off"
        )
        self._row_highlights.setToolTip(
            "Soft-knee compression of bright specular glare"
        )

        form = QFormLayout(self)
        form.addRow("Brightness", self._row_brightness)
        form.addRow("Contrast", self._row_contrast)
        form.addRow("Sharpness", self._row_sharp)
        form.addRow("CLAHE", self._row_clahe)
        form.addRow("Highlights", self._row_highlights)
        form.addRow(self._btn_reset)

        self._row_brightness.changed.connect(self._emit)
        self._row_contrast.changed.connect(self._emit)
        self._row_sharp.changed.connect(self._emit)
        self._row_clahe.changed.connect(self._emit)
        self._row_highlights.changed.connect(self._emit)
        self._btn_reset.clicked.connect(self._reset)

    def params(self) -> EnhanceParams:
        return self._params

    def set_params(self, params: EnhanceParams) -> None:
        """Load values into sliders without emitting params_changed."""
        self._row_brightness.set_range(-80, 80, params.brightness)
        self._row_contrast.set_range(0.50, 2.50, params.contrast)
        self._row_sharp.set_range(0.0, 3.0, params.sharpness)
        self._row_clahe.set_range(0.0, 8.0, params.clahe_clip)
        self._row_highlights.set_range(0.0, 1.0, params.highlight_compress)
        clahe = float(params.clahe_clip)
        if clahe < 0.05:
            clahe = 0.0
        self._params = EnhanceParams(
            brightness=float(params.brightness),
            contrast=float(params.contrast),
            sharpness=float(params.sharpness),
            clahe_clip=clahe,
            highlight_compress=float(params.highlight_compress),
        )

    def to_dict(self) -> dict:
        p = self._params
        return {
            "brightness": p.brightness,
            "contrast": p.contrast,
            "sharpness": p.sharpness,
            "clahe_clip": p.clahe_clip,
            "highlight_compress": p.highlight_compress,
        }

    @staticmethod
    def params_from_dict(data: object) -> EnhanceParams:
        if not isinstance(data, dict):
            return EnhanceParams()
        base = EnhanceParams()
        return EnhanceParams(
            brightness=float(data.get("brightness", base.brightness)),
            contrast=float(data.get("contrast", base.contrast)),
            sharpness=float(data.get("sharpness", base.sharpness)),
            clahe_clip=float(data.get("clahe_clip", base.clahe_clip)),
            highlight_compress=float(
                data.get("highlight_compress", base.highlight_compress)
            ),
        )

    def _reset(self) -> None:
        self.set_params(EnhanceParams())
        self.params_changed.emit(self._params)

    def _emit(self, *_args) -> None:
        clahe = float(self._row_clahe.value())
        if clahe < 0.05:
            clahe = 0.0
        self._params = EnhanceParams(
            brightness=float(self._row_brightness.value()),
            contrast=float(self._row_contrast.value()),
            sharpness=float(self._row_sharp.value()),
            clahe_clip=clahe,
            highlight_compress=float(self._row_highlights.value()),
        )
        self.params_changed.emit(self._params)


class DetectionControlPanel(QGroupBox):
    """Controls for the sunscreen (dark-region) detection overlay."""

    params_changed = Signal(object)  # DetectionParams

    def __init__(self, parent=None) -> None:
        super().__init__("Sunscreen detection (UV dark regions)", parent)
        self._params = DetectionParams()

        self._chk_enable = QCheckBox("Enable overlay")
        self._combo_view = QComboBox()
        self._combo_view.addItem("Overlay", "overlay")
        self._combo_view.addItem("Mask", "mask")

        self._combo_mode = QComboBox()
        self._combo_mode.addItem("Manual threshold", "manual")
        self._combo_mode.addItem("Otsu (auto)", "otsu")
        self._combo_mode.addItem("Adaptive", "adaptive")

        self._chk_flat = QCheckBox("Illumination correction (flat-field)")
        self._chk_flat.setChecked(self._params.flat_field)

        self._row_thresh = _FeatureRow(log_scale=False, fmt=lambda v: f"{v:.0f}")
        self._row_thresh.set_range(0, 255, self._params.manual_threshold)

        self._row_morph = _FeatureRow(log_scale=False, fmt=lambda v: f"{int(v) | 1}")
        self._row_morph.set_range(1, 31, self._params.morph_ksize)

        self._row_area = _FeatureRow(log_scale=False, fmt=lambda v: f"{v:.2f}%")
        self._row_area.set_range(0.0, 2.0, self._params.min_area_pct)

        form = QFormLayout(self)
        form.addRow(self._chk_enable)
        form.addRow("View", self._combo_view)
        form.addRow("Threshold", self._combo_mode)
        form.addRow("Level", self._row_thresh)
        form.addRow(self._chk_flat)
        form.addRow("Noise removal", self._row_morph)
        form.addRow("Min area", self._row_area)

        self._chk_enable.toggled.connect(self._emit)
        self._combo_view.currentIndexChanged.connect(self._emit)
        self._combo_mode.currentIndexChanged.connect(self._emit)
        self._chk_flat.toggled.connect(self._emit)
        self._row_thresh.changed.connect(self._emit)
        self._row_morph.changed.connect(self._emit)
        self._row_area.changed.connect(self._emit)
        self._update_row_states()

    def params(self) -> DetectionParams:
        return self._params

    def set_params(self, params: DetectionParams) -> None:
        """Load values into controls without emitting params_changed."""
        widgets = (
            self._chk_enable,
            self._combo_view,
            self._combo_mode,
            self._chk_flat,
        )
        for w in widgets:
            w.blockSignals(True)

        self._chk_enable.setChecked(params.enabled)
        view_idx = self._combo_view.findData(params.view_mode)
        if view_idx >= 0:
            self._combo_view.setCurrentIndex(view_idx)
        mode_idx = self._combo_mode.findData(params.threshold_mode)
        if mode_idx >= 0:
            self._combo_mode.setCurrentIndex(mode_idx)
        self._chk_flat.setChecked(params.flat_field)
        self._row_thresh.set_range(0, 255, params.manual_threshold)
        self._row_morph.set_range(1, 31, params.morph_ksize)
        self._row_area.set_range(0.0, 2.0, params.min_area_pct)

        for w in widgets:
            w.blockSignals(False)

        self._params = replace(
            self._params,
            enabled=params.enabled,
            view_mode=params.view_mode,
            threshold_mode=params.threshold_mode,
            manual_threshold=int(params.manual_threshold),
            flat_field=params.flat_field,
            morph_ksize=int(params.morph_ksize) | 1,
            min_area_pct=float(params.min_area_pct),
        )
        self._update_row_states()

    def to_dict(self) -> dict:
        p = self._params
        return {
            "enabled": p.enabled,
            "view_mode": p.view_mode,
            "threshold_mode": p.threshold_mode,
            "manual_threshold": p.manual_threshold,
            "flat_field": p.flat_field,
            "morph_ksize": p.morph_ksize,
            "min_area_pct": p.min_area_pct,
        }

    @staticmethod
    def params_from_dict(data: object) -> DetectionParams:
        if not isinstance(data, dict):
            return DetectionParams()
        base = DetectionParams()
        return DetectionParams(
            enabled=bool(data.get("enabled", base.enabled)),
            view_mode=str(data.get("view_mode", base.view_mode)),
            threshold_mode=str(data.get("threshold_mode", base.threshold_mode)),
            manual_threshold=int(data.get("manual_threshold", base.manual_threshold)),
            flat_field=bool(data.get("flat_field", base.flat_field)),
            morph_ksize=int(data.get("morph_ksize", base.morph_ksize)) | 1,
            min_area_pct=float(data.get("min_area_pct", base.min_area_pct)),
        )

    def _update_row_states(self) -> None:
        self._row_thresh.setEnabled(self._combo_mode.currentData() == "manual")

    def _emit(self, *_args) -> None:
        self._params = replace(
            self._params,
            enabled=self._chk_enable.isChecked(),
            view_mode=str(self._combo_view.currentData()),
            threshold_mode=str(self._combo_mode.currentData()),
            manual_threshold=int(self._row_thresh.value()),
            flat_field=self._chk_flat.isChecked(),
            morph_ksize=int(self._row_morph.value()) | 1,
            min_area_pct=float(self._row_area.value()),
        )
        self._update_row_states()
        self.params_changed.emit(self._params)
