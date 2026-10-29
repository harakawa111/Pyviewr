"""Minimal main window: connect, preview, still, video, save dir."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pyviewr import config as app_config
from pyviewr.camera.manager import TARGET_FPS, CameraManager, RecordingSummary
from pyviewr.processing import enhance, sunscreen
from pyviewr.ui.control_panels import (
    CameraControlPanel,
    DetectionControlPanel,
    EnhanceControlPanel,
)
from pyviewr.ui.preview_widget import PreviewWidget

TIMER_SECONDS = 10


class _Bridge(QObject):
    """Thread-safe bridge from grab threads to the UI thread."""

    frame = Signal(int, object)
    error = Signal(str)
    device_removed = Signal(int)
    devices_ready = Signal(object, object)  # list[DeviceInfo] | None, error str | None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pyviewr")
        self.resize(1100, 700)

        self._save_dir = app_config.load_save_dir()
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._bridge = _Bridge()
        self._bridge.frame.connect(self._on_frame)
        self._bridge.error.connect(self._on_error)
        self._bridge.device_removed.connect(self._on_device_removed)
        self._bridge.devices_ready.connect(self._on_devices_ready)
        self._refreshing = False
        self._closing = False

        self._manager = CameraManager(
            on_frame=self._emit_frame,
            on_error=self._emit_error,
            on_device_removed=self._emit_device_removed,
        )

        self._preview0 = PreviewWidget("Camera 0")
        self._preview1 = PreviewWidget("Camera 1")

        self._camera_panel = CameraControlPanel()
        self._camera_panel.feature_changed.connect(self._on_camera_feature)
        self._camera_panel.target_fps_clicked.connect(self._on_target_fps_clicked)
        self._enhance_panel = EnhanceControlPanel()
        self._enhance_panel.params_changed.connect(self._on_enhance_params)
        self._detect_panel = DetectionControlPanel()
        self._detect_panel.params_changed.connect(self._on_detection_params)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_connect = QPushButton("Connect")
        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.setToolTip(
            "Freeze the current preview so enhance / detection sliders "
            "can be tuned on a still frame"
        )
        self._btn_still = QPushButton("Still")
        self._btn_record = QPushButton("Record")
        self._btn_save_dir = QPushButton("Save folder…")
        self._btn_cancel_timer = QPushButton("Cancel timer")
        self._btn_cancel_timer.setVisible(False)

        self._timer_combo = QComboBox()
        self._timer_combo.addItem("Timer: Off", 0)
        self._timer_combo.addItem(f"Timer: {TIMER_SECONDS}s", TIMER_SECONDS)

        self._chk_lossless = QCheckBox("Lossless")
        self._chk_lossless.setToolTip(
            "On: FFV1 .mkv (bit-exact, same quality as stills, large files)\n"
            "Off: MPEG-4 .mp4 (lossy, small files)"
        )

        self._paused = False
        self._last_raw: dict[int, np.ndarray] = {}
        self._frozen_raw: dict[int, np.ndarray] = {}

        cfg = app_config.load_config()
        enhance_params = EnhanceControlPanel.params_from_dict(cfg.get("enhance"))
        detect_params = DetectionControlPanel.params_from_dict(cfg.get("detection"))
        self._enhance_panel.set_params(enhance_params)
        self._detect_panel.set_params(detect_params)
        self._enhance_params = self._enhance_panel.params()
        self._det_params = self._detect_panel.params()
        self._manager.set_enhance_params(self._enhance_params)
        saved_timer = app_config.load_timer_seconds()
        idx = self._timer_combo.findData(saved_timer)
        if idx < 0 and saved_timer > 0:
            idx = self._timer_combo.findData(TIMER_SECONDS)
        if idx >= 0:
            self._timer_combo.setCurrentIndex(idx)
        self._chk_lossless.setChecked(app_config.load_lossless_video())

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._flush_settings)

        self._lbl_countdown = QLabel("")
        self._lbl_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_countdown.setStyleSheet("font-size: 28px; font-weight: bold;")
        self._lbl_countdown.setVisible(False)

        self._lbl_recording_time = QLabel("")
        self._lbl_recording_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_recording_time.setStyleSheet(
            "color: #d32f2f; font-size: 20px; font-weight: bold;"
        )
        self._lbl_recording_time.setVisible(False)
        # Stays until the next recording so drops can be checked afterwards.
        self._lbl_last_recording = QLabel("")
        self._lbl_last_recording.setVisible(False)
        self._recording_elapsed = QElapsedTimer()
        self._recording_timer = QTimer(self)
        self._recording_timer.setInterval(1000)
        self._recording_timer.timeout.connect(self._update_recording_time)

        self._lbl_devices = QLabel("Devices: —")
        self._lbl_fps = QLabel("FPS: —")
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._update_fps_label)
        self._lbl_save = QLabel(f"Save: {self._save_dir}")
        self._lbl_save.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._pending_action: str | None = None  # "still" | "record"
        self._seconds_left = 0
        self._countdown = QTimer(self)
        self._countdown.setInterval(1000)
        self._countdown.timeout.connect(self._on_countdown_tick)

        self._btn_refresh.clicked.connect(self.refresh_devices)
        self._btn_connect.clicked.connect(self.connect_cameras)
        self._btn_disconnect.clicked.connect(self.disconnect_cameras)
        self._btn_pause.toggled.connect(self._on_pause_toggled)
        self._btn_still.clicked.connect(self.capture_still)
        self._btn_record.clicked.connect(self.toggle_record)
        self._btn_save_dir.clicked.connect(self.choose_save_dir)
        self._btn_cancel_timer.clicked.connect(self.cancel_timer)
        self._timer_combo.currentIndexChanged.connect(self._schedule_save_settings)
        self._chk_lossless.toggled.connect(self._schedule_save_settings)

        controls = QHBoxLayout()
        for w in (
            self._btn_refresh,
            self._btn_connect,
            self._btn_disconnect,
            self._btn_pause,
            self._timer_combo,
            self._btn_still,
            self._btn_record,
            self._chk_lossless,
            self._btn_cancel_timer,
            self._btn_save_dir,
        ):
            controls.addWidget(w)
        controls.addStretch(1)

        side = QVBoxLayout()
        side.addWidget(self._camera_panel)
        side.addWidget(self._enhance_panel)
        side.addWidget(self._detect_panel)
        side.addStretch(1)
        side_inner = QWidget()
        side_inner.setLayout(side)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        side_scroll.setWidget(side_inner)
        side_scroll.setFixedWidth(340)

        previews = QHBoxLayout()
        previews.addWidget(self._preview0, stretch=1)
        previews.addWidget(self._preview1, stretch=1)
        previews.addWidget(side_scroll)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self._lbl_devices)
        root.addWidget(self._lbl_fps)
        root.addWidget(self._lbl_save)
        root.addWidget(self._lbl_countdown)
        root.addWidget(self._lbl_recording_time)
        root.addWidget(self._lbl_last_recording)
        root.addLayout(previews, stretch=1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._set_connected_ui(False)
        self.refresh_devices()

    def _timer_delay_seconds(self) -> int:
        return int(self._timer_combo.currentData() or 0)

    def _emit_frame(self, index: int, frame: np.ndarray) -> None:
        self._bridge.frame.emit(index, frame)

    def _emit_error(self, message: str) -> None:
        self._bridge.error.emit(message)

    def _emit_device_removed(self, index: int) -> None:
        self._bridge.device_removed.emit(index)

    @Slot(int, object)
    def _on_frame(self, index: int, frame: object) -> None:
        arr = frame if isinstance(frame, np.ndarray) else None
        if arr is None:
            return
        # Keep a reference for Pause; copy only when freezing.
        self._last_raw[index] = arr
        if self._paused:
            # Capture the first frame seen while already paused.
            if index not in self._frozen_raw:
                self._frozen_raw[index] = arr.copy()
                self._show_processed(index, self._frozen_raw[index])
            return
        self._show_processed(index, arr)

    def _show_processed(self, index: int, raw: np.ndarray) -> None:
        arr = raw
        if arr.ndim == 2:
            try:
                arr = enhance.apply(arr, self._enhance_params)
            except Exception:
                pass
        if self._det_params.enabled and arr.ndim == 2:
            try:
                arr, _coverage = sunscreen.process_frame(arr, self._det_params)
            except Exception:
                pass
        if index == 0:
            self._preview0.show_frame(arr)
        elif index == 1:
            self._preview1.show_frame(arr)

    def _redraw_frozen(self) -> None:
        for index, raw in self._frozen_raw.items():
            self._show_processed(index, raw)

    @Slot(bool)
    def _on_pause_toggled(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._frozen_raw = {
                i: frame.copy() for i, frame in self._last_raw.items()
            }
            self._btn_pause.setText("Resume")
            if self._frozen_raw:
                self._redraw_frozen()
                self.statusBar().showMessage(
                    "Preview paused — enhance / detection still update", 0
                )
            else:
                self.statusBar().showMessage(
                    "Preview paused — waiting for a frame", 0
                )
        else:
            self._frozen_raw.clear()
            self._btn_pause.setText("Pause")
            self.statusBar().showMessage("Preview live", 3000)

    def _clear_pause(self) -> None:
        self._paused = False
        self._frozen_raw.clear()
        self._last_raw.clear()
        if self._btn_pause.isChecked():
            self._btn_pause.blockSignals(True)
            self._btn_pause.setChecked(False)
            self._btn_pause.blockSignals(False)
        self._btn_pause.setText("Pause")

    @Slot(object)
    def _on_enhance_params(self, params: object) -> None:
        if isinstance(params, enhance.EnhanceParams):
            self._enhance_params = params
            self._manager.set_enhance_params(params)
            if self._paused:
                self._redraw_frozen()
            self._schedule_save_settings()

    @Slot(object)
    def _on_detection_params(self, params: object) -> None:
        if isinstance(params, sunscreen.DetectionParams):
            self._det_params = params
            if self._paused:
                self._redraw_frozen()
            self._schedule_save_settings()

    @Slot(str, float)
    def _on_camera_feature(self, name: str, value: float) -> None:
        try:
            self._manager.set_feature(name, value)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        if name == "AcquisitionFrameRate":
            info = self._manager.get_feature_info("ExposureTime")
            if info is not None:
                self._camera_panel.update_feature("ExposureTime", info)
        self._update_fps_label()
        self._schedule_save_settings()

    @Slot()
    def _on_target_fps_clicked(self) -> None:
        """Pair the target rate with the longest exposure that sustains it."""
        try:
            reachable = self._manager.maximize_exposure_for_frame_rate()
        except Exception as exc:
            self.statusBar().showMessage(f"Exposure adjust failed: {exc}", 5000)
            return
        info = self._manager.get_feature_info("ExposureTime")
        if info is not None:
            self._camera_panel.update_feature("ExposureTime", info)
        self._update_fps_label()
        self._schedule_save_settings()
        if reachable:
            exposure_ms = info.value / 1000.0 if info is not None else 0.0
            self.statusBar().showMessage(
                f"Frame rate set to {TARGET_FPS:g} fps, exposure "
                f"{exposure_ms:.2f} ms (saved)",
                4000,
            )
        else:
            resulting = self._manager.get_resulting_fps()
            self.statusBar().showMessage(
                f"{TARGET_FPS:g} fps not reachable: camera limit "
                f"{resulting or 0:.1f} fps (ROI / bandwidth)",
                8000,
            )

    @Slot()
    def _update_fps_label(self) -> None:
        try:
            fps = self._manager.get_resulting_fps()
        except Exception:
            fps = None
        self._lbl_fps.setText("FPS: —" if fps is None else f"FPS: {fps:.1f}")

    def _load_camera_panel(self) -> None:
        infos = {
            name: self._manager.get_feature_info(name)
            for name in ("AcquisitionFrameRate", "ExposureTime", "Gain", "Gamma")
        }
        self._camera_panel.load_features(infos)

    def _apply_saved_camera_features(self) -> None:
        saved = app_config.load_camera_features()
        if "AcquisitionFrameRate" not in saved:
            saved = {"AcquisitionFrameRate": TARGET_FPS, **saved}
        for name, value in saved.items():
            try:
                self._manager.set_feature(name, value)
            except Exception:
                pass

    def _schedule_save_settings(self, *_args) -> None:
        self._save_timer.start()

    @Slot()
    def _flush_settings(self) -> None:
        camera = self._camera_panel.current_features()
        if not camera:
            camera = app_config.load_camera_features()
        app_config.update_config(
            save_dir=str(self._save_dir),
            timer_seconds=self._timer_delay_seconds(),
            lossless_video=self._chk_lossless.isChecked(),
            camera=camera,
            enhance=self._enhance_panel.to_dict(),
            detection=self._detect_panel.to_dict(),
        )

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    @Slot(int)
    def _on_device_removed(self, index: int) -> None:
        """Camera unplugged / link lost — tear down session cleanly."""
        self.cancel_timer()
        try:
            self._manager.close()
        except Exception:
            pass
        self._set_connected_ui(False)
        self._preview0.set_title("Camera 0")
        self._preview1.set_title("Camera 1")
        self.statusBar().showMessage(
            f"Camera {index} disconnected (device removed)", 8000
        )
        self.refresh_devices()
        QMessageBox.warning(
            self,
            "Pyviewr",
            f"Camera {index} was disconnected (cable unplugged or link lost).\n"
            "Re-plug the camera, then click Connect.",
        )

    def _set_connected_ui(self, connected: bool) -> None:
        self._btn_connect.setEnabled(not connected)
        self._btn_disconnect.setEnabled(connected)
        self._btn_pause.setEnabled(connected)
        counting = self._countdown.isActive()
        self._btn_still.setEnabled(connected and not counting)
        self._btn_record.setEnabled(connected and not counting)
        self._timer_combo.setEnabled(connected and not counting)
        if connected:
            self._fps_timer.start()
        else:
            self._fps_timer.stop()
            self._lbl_fps.setText("FPS: —")
            self.cancel_timer()
            self._stop_recording_time()
            self._clear_pause()
            self._btn_record.setText("Record")
            self._chk_lossless.setEnabled(True)
            self._preview0.clear_frame()
            self._preview1.clear_frame()
            self._camera_panel.set_disconnected()

    def _set_countdown_ui(self, active: bool) -> None:
        self._btn_cancel_timer.setVisible(active)
        self._lbl_countdown.setVisible(active)
        connected = self._manager.is_open
        self._btn_still.setEnabled(connected and not active)
        # Stop must stay available while recording; start waits for timer.
        if self._manager.is_recording:
            self._btn_record.setEnabled(True)
        else:
            self._btn_record.setEnabled(connected and not active)
        self._timer_combo.setEnabled(connected and not active)
        self._btn_disconnect.setEnabled(connected and not active)

    def _start_countdown(self, action: str) -> None:
        delay = self._timer_delay_seconds()
        if delay <= 0:
            self._run_action(action)
            return
        self._pending_action = action
        self._seconds_left = delay
        self._lbl_countdown.setText(str(self._seconds_left))
        self._set_countdown_ui(True)
        self.statusBar().showMessage(f"Timer: {self._seconds_left}s → {action}", 0)
        self._countdown.start()

    @Slot()
    def _on_countdown_tick(self) -> None:
        self._seconds_left -= 1
        if self._seconds_left > 0:
            self._lbl_countdown.setText(str(self._seconds_left))
            action = self._pending_action or ""
            self.statusBar().showMessage(f"Timer: {self._seconds_left}s → {action}", 0)
            return
        action = self._pending_action
        self._finish_countdown_ui()
        if action:
            self._run_action(action)

    def _finish_countdown_ui(self) -> None:
        self._countdown.stop()
        self._pending_action = None
        self._seconds_left = 0
        self._lbl_countdown.setText("")
        self._set_countdown_ui(False)

    @Slot()
    def cancel_timer(self) -> None:
        if not self._countdown.isActive() and self._pending_action is None:
            self._finish_countdown_ui()
            return
        self._finish_countdown_ui()
        self.statusBar().showMessage("Timer cancelled", 3000)

    def _run_action(self, action: str) -> None:
        if action == "still":
            self._do_capture_still()
        elif action == "record":
            self._do_start_recording()

    def _show_device_list(self, devices: list) -> None:
        if not devices:
            hint = self._manager.gige_unreachable_hint()
            self._lbl_devices.setText(
                f"Devices: none reachable — {hint}" if hint else "Devices: none found"
            )
            return
        text = "; ".join(f"[{d.index}] {d.display_name}" for d in devices)
        self._lbl_devices.setText(f"Devices: {text}")

    @Slot()
    def refresh_devices(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self._btn_refresh.setEnabled(False)
        if not self._manager.is_open:
            self._btn_connect.setEnabled(False)
        self._lbl_devices.setText("Devices: scanning…")

        def work() -> None:
            try:
                devices = self._manager.enumerate_devices()
            except Exception as exc:
                self._bridge.devices_ready.emit(None, str(exc))
                return
            self._bridge.devices_ready.emit(devices, None)

        threading.Thread(target=work, name="pyviewr-enum", daemon=True).start()

    @Slot(object, object)
    def _on_devices_ready(self, devices: object, error: object) -> None:
        self._refreshing = False
        self._btn_refresh.setEnabled(True)
        if not self._manager.is_open:
            self._btn_connect.setEnabled(True)
        if self._closing:
            return
        if error:
            self._lbl_devices.setText(f"Devices: error — {error}")
            return
        if not isinstance(devices, list):
            self._lbl_devices.setText("Devices: none found")
            return
        self._show_device_list(devices)

    @Slot()
    def connect_cameras(self) -> None:
        try:
            devices = self._manager.enumerate_devices()
            if not devices:
                hint = self._manager.gige_unreachable_hint(detailed=True)
                detail = (
                    "No cameras reachable.\n\n"
                    + (hint or "pylon found no devices. Check power, cable, and NIC link.")
                )
                self._show_device_list([])
                QMessageBox.warning(self, "Pyviewr", detail)
                return
            # Open with a stable UVL -> Camera 0 / UVR -> Camera 1 identity,
            # independent of which one happens to enumerate first over USB.
            opened = self._manager.open_identified()
            self._manager.start_grabbing()
        except Exception as exc:
            QMessageBox.critical(self, "Pyviewr", str(exc))
            self._manager.close()
            self._set_connected_ui(False)
            return

        identity = app_config.load_camera_identity()
        if len(opened) >= 1:
            label = identity.get(opened[0].serial, "?")
            self._preview0.set_title(f"Camera 0 ({label}) — {opened[0].display_name}")
        if len(opened) >= 2:
            label = identity.get(opened[1].serial, "?")
            self._preview1.set_title(f"Camera 1 ({label}) — {opened[1].display_name}")
        else:
            self._preview1.set_title("Camera 1 (UVR) — (not connected)")
            self._preview1.clear_frame()

        self._set_connected_ui(True)
        self._apply_saved_camera_features()
        self._load_camera_panel()
        self._update_fps_label()
        self.statusBar().showMessage(f"Connected: {len(opened)} camera(s)", 3000)
        self.refresh_devices()

    @Slot()
    def disconnect_cameras(self) -> None:
        self.cancel_timer()
        try:
            self._manager.close()
        except Exception as exc:
            QMessageBox.warning(self, "Pyviewr", str(exc))
        self._set_connected_ui(False)
        self._preview0.set_title("Camera 0")
        self._preview1.set_title("Camera 1")
        self.statusBar().showMessage("Disconnected", 3000)

    @Slot()
    def capture_still(self) -> None:
        if self._countdown.isActive():
            return
        self._start_countdown("still")

    def _do_capture_still(self) -> None:
        try:
            paths = self._manager.capture_still(self._save_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Pyviewr", str(exc))
            return
        names = ", ".join(p.name for p in paths)
        self.statusBar().showMessage(f"Saved still: {names}", 5000)

    @Slot()
    def toggle_record(self) -> None:
        if self._manager.is_recording:
            # Stop is always immediate (no timer).
            self.cancel_timer()
            try:
                summaries = self._manager.stop_recording()
            except Exception as exc:
                QMessageBox.critical(self, "Pyviewr", str(exc))
                return
            self._stop_recording_time()
            self._btn_record.setText("Record")
            self._chk_lossless.setEnabled(True)
            self._show_recording_summary(summaries)
            self.statusBar().showMessage("Recording stopped", 3000)
            return

        if self._countdown.isActive():
            return
        self._start_countdown("record")

    def _do_start_recording(self) -> None:
        try:
            paths = self._manager.start_recording(
                self._save_dir, lossless=self._chk_lossless.isChecked()
            )
        except Exception as exc:
            QMessageBox.critical(self, "Pyviewr", str(exc))
            return
        self._btn_record.setText("Stop")
        self._chk_lossless.setEnabled(False)
        self._recording_elapsed.start()
        self._update_recording_time()
        self._recording_timer.start()
        names = ", ".join(p.name for p in paths)
        self.statusBar().showMessage(f"Recording: {names}", 0)

    def _update_recording_time(self) -> None:
        if not self._recording_elapsed.isValid():
            return
        elapsed = self._recording_elapsed.elapsed() // 1000
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self._lbl_recording_time.setText(
            f"Recording time: {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        self._lbl_recording_time.setVisible(True)

    def _show_recording_summary(self, summaries: list[RecordingSummary]) -> None:
        if not summaries:
            self._lbl_last_recording.setVisible(False)
            return
        problem = False
        lines = []
        for s in summaries:
            if s.camera_dropped is None:
                dropped = "dropped: n/a"
            else:
                dropped = f"dropped: {s.camera_dropped}"
                problem |= s.camera_dropped > 0
            problem |= s.disk_waits > 0 or s.incomplete > 0
            underruns = "n/a" if s.buffer_underruns is None else s.buffer_underruns
            packet = "n/a" if s.packet_size is None else s.packet_size
            lines.append(
                f"Last recording cam{s.slot}: {s.fps:.1f} fps, {s.frames} frames, "
                f"{dropped}, disk waits: {s.disk_waits}, "
                f"incomplete: {s.incomplete}, no buffer: {underruns}, "
                f"packet: {packet}"
            )
        self._lbl_last_recording.setText("\n".join(lines))
        self._lbl_last_recording.setStyleSheet(
            "color: #d32f2f; font-weight: bold;" if problem else ""
        )
        self._lbl_last_recording.setVisible(True)

    def _stop_recording_time(self) -> None:
        self._recording_timer.stop()
        self._recording_elapsed.invalidate()
        self._lbl_recording_time.clear()
        self._lbl_recording_time.setVisible(False)

    @Slot()
    def choose_save_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select save folder",
            str(self._save_dir),
        )
        if not chosen:
            return
        self._save_dir = Path(chosen).expanduser()
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._lbl_save.setText(f"Save: {self._save_dir}")
        self._flush_settings()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        self.cancel_timer()
        self._flush_settings()
        try:
            self._manager.close()
        except Exception:
            pass
        super().closeEvent(event)
