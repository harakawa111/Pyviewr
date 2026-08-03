"""Minimal main window: connect, preview, still, video, save dir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pyviewr import config as app_config
from pyviewr.camera.manager import CameraManager
from pyviewr.ui.preview_widget import PreviewWidget


class _Bridge(QObject):
    """Thread-safe bridge from grab threads to the UI thread."""

    frame = Signal(int, object)
    error = Signal(str)


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

        self._manager = CameraManager(
            on_frame=self._emit_frame,
            on_error=self._emit_error,
        )

        self._preview0 = PreviewWidget("Camera 0")
        self._preview1 = PreviewWidget("Camera 1")

        self._btn_refresh = QPushButton("Refresh")
        self._btn_connect = QPushButton("Connect")
        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_still = QPushButton("Still")
        self._btn_record = QPushButton("Record")
        self._btn_save_dir = QPushButton("Save folder…")

        self._lbl_devices = QLabel("Devices: —")
        self._lbl_save = QLabel(f"Save: {self._save_dir}")
        self._lbl_save.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._btn_refresh.clicked.connect(self.refresh_devices)
        self._btn_connect.clicked.connect(self.connect_cameras)
        self._btn_disconnect.clicked.connect(self.disconnect_cameras)
        self._btn_still.clicked.connect(self.capture_still)
        self._btn_record.clicked.connect(self.toggle_record)
        self._btn_save_dir.clicked.connect(self.choose_save_dir)

        controls = QHBoxLayout()
        for w in (
            self._btn_refresh,
            self._btn_connect,
            self._btn_disconnect,
            self._btn_still,
            self._btn_record,
            self._btn_save_dir,
        ):
            controls.addWidget(w)
        controls.addStretch(1)

        previews = QHBoxLayout()
        previews.addWidget(self._preview0, stretch=1)
        previews.addWidget(self._preview1, stretch=1)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self._lbl_devices)
        root.addWidget(self._lbl_save)
        root.addLayout(previews, stretch=1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._set_connected_ui(False)
        self.refresh_devices()

    def _emit_frame(self, index: int, frame: np.ndarray) -> None:
        self._bridge.frame.emit(index, frame)

    def _emit_error(self, message: str) -> None:
        self._bridge.error.emit(message)

    @Slot(int, object)
    def _on_frame(self, index: int, frame: object) -> None:
        arr = frame if isinstance(frame, np.ndarray) else None
        if arr is None:
            return
        if index == 0:
            self._preview0.show_frame(arr)
        elif index == 1:
            self._preview1.show_frame(arr)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _set_connected_ui(self, connected: bool) -> None:
        self._btn_connect.setEnabled(not connected)
        self._btn_disconnect.setEnabled(connected)
        self._btn_still.setEnabled(connected)
        self._btn_record.setEnabled(connected)
        if not connected:
            self._btn_record.setText("Record")
            self._preview0.clear_frame()
            self._preview1.clear_frame()

    @Slot()
    def refresh_devices(self) -> None:
        try:
            devices = self._manager.enumerate_devices()
        except Exception as exc:
            self._lbl_devices.setText(f"Devices: error — {exc}")
            return
        if not devices:
            self._lbl_devices.setText("Devices: none found")
            return
        text = "; ".join(f"[{d.index}] {d.display_name}" for d in devices)
        self._lbl_devices.setText(f"Devices: {text}")

    @Slot()
    def connect_cameras(self) -> None:
        try:
            devices = self._manager.enumerate_devices()
            if not devices:
                QMessageBox.warning(self, "Pyviewr", "No cameras found.")
                return
            # Open up to first two cameras.
            opened = self._manager.open(list(range(min(2, len(devices)))))
            self._manager.start_grabbing()
        except Exception as exc:
            QMessageBox.critical(self, "Pyviewr", str(exc))
            self._manager.close()
            self._set_connected_ui(False)
            return

        if len(opened) >= 1:
            self._preview0.set_title(f"Camera 0 — {opened[0].display_name}")
        if len(opened) >= 2:
            self._preview1.set_title(f"Camera 1 — {opened[1].display_name}")
        else:
            self._preview1.set_title("Camera 1 — (not connected)")
            self._preview1.clear_frame()

        self._set_connected_ui(True)
        self.statusBar().showMessage(f"Connected: {len(opened)} camera(s)", 3000)
        self.refresh_devices()

    @Slot()
    def disconnect_cameras(self) -> None:
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
        try:
            paths = self._manager.capture_still(self._save_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Pyviewr", str(exc))
            return
        names = ", ".join(p.name for p in paths)
        self.statusBar().showMessage(f"Saved still: {names}", 5000)

    @Slot()
    def toggle_record(self) -> None:
        if not self._manager.is_recording:
            try:
                paths = self._manager.start_recording(self._save_dir)
            except Exception as exc:
                QMessageBox.critical(self, "Pyviewr", str(exc))
                return
            self._btn_record.setText("Stop")
            names = ", ".join(p.name for p in paths)
            self.statusBar().showMessage(f"Recording: {names}", 0)
        else:
            try:
                self._manager.stop_recording()
            except Exception as exc:
                QMessageBox.critical(self, "Pyviewr", str(exc))
                return
            self._btn_record.setText("Record")
            self.statusBar().showMessage("Recording stopped", 3000)

    @Slot()
    def choose_save_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select save folder",
            str(self._save_dir),
        )
        if not chosen:
            return
        self._save_dir = app_config.save_save_dir(Path(chosen))
        self._lbl_save.setText(f"Save: {self._save_dir}")

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._manager.close()
        except Exception:
            pass
        super().closeEvent(event)
