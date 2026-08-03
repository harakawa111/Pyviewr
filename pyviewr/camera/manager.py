"""Camera enumeration, grab loops, still/video capture."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from pyviewr.camera import sync as sync_mod
from pyviewr.io.still import save_still
from pyviewr.io.video import VideoRecorder

try:
    from pypylon import genicam, pylon
except ImportError:  # pragma: no cover - runtime dependency
    pylon = None  # type: ignore[assignment]
    genicam = None  # type: ignore[assignment]


FrameCallback = Callable[[int, np.ndarray], None]
ErrorCallback = Callable[[str], None]


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    model: str
    serial: str
    display_name: str


def _require_pylon() -> None:
    if pylon is None:
        raise RuntimeError(
            "pypylon is not installed, or Basler pylon Software Suite is missing. "
            "Install pylon 7.1+ then: pip install pypylon"
        )


def array_from_grab(result) -> np.ndarray:
    """Convert a pylon grab result to a contiguous uint8 grayscale image."""
    img = result.Array
    if img.ndim == 3:
        # Rare for this UV mono camera; take first channel.
        img = img[:, :, 0]
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    # Mono10/12/16 -> display/save as uint8 by simple scale.
    info = np.iinfo(img.dtype)
    if info.max <= 0:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img.astype(np.float32) * (255.0 / float(info.max))).clip(0, 255)
    return np.ascontiguousarray(scaled.astype(np.uint8))


class CameraManager:
    """Owns InstantCamera instances and background grab threads."""

    def __init__(
        self,
        on_frame: FrameCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        _require_pylon()
        self._on_frame = on_frame
        self._on_error = on_error
        self._tl_factory = pylon.TlFactory.GetInstance()
        self._cameras: list[pylon.InstantCamera] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._grabbing = False
        self._recording = False
        self._recorders: dict[int, VideoRecorder] = {}
        self._lock = threading.Lock()

    # ----- discovery / lifecycle -------------------------------------------------

    def enumerate_devices(self) -> list[DeviceInfo]:
        _require_pylon()
        devices = self._tl_factory.EnumerateDevices()
        out: list[DeviceInfo] = []
        for i, di in enumerate(devices):
            model = di.GetModelName() or "Unknown"
            serial = di.GetSerialNumber() or ""
            name = di.GetFriendlyName() or f"{model} ({serial})"
            out.append(DeviceInfo(i, model, serial, name))
        return out

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    @property
    def is_open(self) -> bool:
        return bool(self._cameras)

    @property
    def is_grabbing(self) -> bool:
        return self._grabbing

    @property
    def is_recording(self) -> bool:
        return self._recording

    def open(self, indices: list[int] | None = None) -> list[DeviceInfo]:
        """Open selected devices (default: first two, or all if fewer)."""
        self.close()
        devices = self._tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No cameras found.")

        if indices is None:
            indices = list(range(min(2, len(devices))))
        if not indices:
            raise RuntimeError("No camera indices selected.")
        for idx in indices:
            if idx < 0 or idx >= len(devices):
                raise RuntimeError(f"Invalid camera index: {idx}")

        opened: list[DeviceInfo] = []
        try:
            for local_i, idx in enumerate(indices):
                di = devices[idx]
                cam = pylon.InstantCamera(self._tl_factory.CreateDevice(di))
                cam.Open()
                # Prefer Mono8 when available for light preview/recording.
                try:
                    cam.PixelFormat.SetValue("Mono8")
                except (genicam.GenericException, AttributeError):
                    pass
                sync_mod.configure_free_run(cam)
                self._cameras.append(cam)
                opened.append(
                    DeviceInfo(
                        local_i,
                        di.GetModelName() or "Unknown",
                        di.GetSerialNumber() or "",
                        di.GetFriendlyName() or di.GetModelName() or f"cam{local_i}",
                    )
                )
        except Exception:
            self.close()
            raise
        return opened

    def close(self) -> None:
        self.stop_recording()
        self.stop_grabbing()
        for cam in self._cameras:
            try:
                if cam.IsOpen():
                    cam.Close()
            except Exception:
                pass
        self._cameras.clear()

    # ----- grabbing --------------------------------------------------------------

    def start_grabbing(self) -> None:
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        if self._grabbing:
            return
        for cam in self._cameras:
            sync_mod.configure_free_run(cam)
            if not cam.IsGrabbing():
                cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._stop.clear()
        self._grabbing = True
        self._threads = []
        for i, cam in enumerate(self._cameras):
            t = threading.Thread(
                target=self._grab_loop,
                args=(i, cam),
                name=f"pyviewr-grab-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop_grabbing(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        for cam in self._cameras:
            try:
                if cam.IsGrabbing():
                    cam.StopGrabbing()
            except Exception:
                pass
        self._grabbing = False

    def _grab_loop(self, index: int, cam: "pylon.InstantCamera") -> None:
        while not self._stop.is_set():
            try:
                if not cam.IsGrabbing():
                    time.sleep(0.01)
                    continue
                grab = cam.RetrieveResult(200, pylon.TimeoutHandling_Return)
                with grab:
                    if not grab.GrabSucceeded():
                        continue
                    frame = array_from_grab(grab)
                with self._lock:
                    rec = self._recorders.get(index)
                    if rec is not None:
                        rec.write(frame)
                if self._on_frame is not None:
                    self._on_frame(index, frame)
            except Exception as exc:  # keep loop alive
                if self._on_error is not None:
                    self._on_error(f"Camera {index}: {exc}")
                time.sleep(0.05)

    # ----- still -----------------------------------------------------------------

    def capture_still(self, save_dir: Path) -> list[Path]:
        """Capture one frame per open camera. Sync via Action Command when 2+ cams."""
        if not self._cameras:
            raise RuntimeError("No cameras open.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        paths: list[Path] = []

        was_grabbing = self._grabbing
        if was_grabbing:
            self.stop_grabbing()

        try:
            if len(self._cameras) >= 2:
                paths = self._capture_still_action(save_dir, stamp)
            else:
                paths = self._capture_still_software(save_dir, stamp)
        finally:
            if was_grabbing:
                self.start_grabbing()
        return paths

    def _capture_still_software(self, save_dir: Path, stamp: str) -> list[Path]:
        paths: list[Path] = []
        cam = self._cameras[0]
        sync_mod.configure_free_run(cam)
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            with grab:
                if not grab.GrabSucceeded():
                    raise RuntimeError("Still grab failed.")
                frame = array_from_grab(grab)
            path = save_dir / f"cam0_{stamp}.png"
            save_still(path, frame)
            paths.append(path)
        finally:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        return paths

    def _capture_still_action(self, save_dir: Path, stamp: str) -> list[Path]:
        paths: list[Path] = []
        for cam in self._cameras:
            sync_mod.configure_action_trigger(cam)
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)

        try:
            # Arm all cameras, then fire one broadcast action command.
            time.sleep(0.05)
            sync_mod.issue_action_command(self._tl_factory)

            frames: list[np.ndarray] = []
            for i, cam in enumerate(self._cameras):
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                with grab:
                    if not grab.GrabSucceeded():
                        raise RuntimeError(f"Still grab failed on camera {i}.")
                    frames.append(array_from_grab(grab))

            for i, frame in enumerate(frames):
                path = save_dir / f"cam{i}_{stamp}.png"
                save_still(path, frame)
                paths.append(path)
        finally:
            for cam in self._cameras:
                try:
                    if cam.IsGrabbing():
                        cam.StopGrabbing()
                except Exception:
                    pass
                try:
                    sync_mod.configure_free_run(cam)
                except Exception:
                    pass
        return paths

    # ----- video -----------------------------------------------------------------

    def start_recording(self, save_dir: Path, fps: float = 30.0) -> list[Path]:
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        if self._recording:
            raise RuntimeError("Already recording.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Probe size from a quick grab if not already grabbing.
        sizes: list[tuple[int, int]] = []
        if not self._grabbing:
            self.start_grabbing()
            time.sleep(0.2)

        for i, cam in enumerate(self._cameras):
            w = int(cam.Width.GetValue())
            h = int(cam.Height.GetValue())
            sizes.append((w, h))

        paths: list[Path] = []
        with self._lock:
            self._recorders.clear()
            for i, (w, h) in enumerate(sizes):
                path = save_dir / f"cam{i}_{stamp}.avi"
                self._recorders[i] = VideoRecorder(path, (w, h), fps=fps)
                paths.append(path)
            self._recording = True

        # Align start for multi-cam with a single action pulse then free-run.
        if len(self._cameras) >= 2:
            self._restart_free_run_aligned()
        return paths

    def _restart_free_run_aligned(self) -> None:
        """Stop, arm action once, then continue free-run recording."""
        was = self._grabbing
        if was:
            self.stop_grabbing()
        for cam in self._cameras:
            sync_mod.configure_action_trigger(cam)
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        try:
            time.sleep(0.05)
            sync_mod.issue_action_command(self._tl_factory)
            for idx, cam in enumerate(self._cameras):
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                with grab:
                    if grab.GrabSucceeded():
                        frame = array_from_grab(grab)
                        with self._lock:
                            rec = self._recorders.get(idx)
                            if rec is not None:
                                rec.write(frame)
        finally:
            for cam in self._cameras:
                try:
                    if cam.IsGrabbing():
                        cam.StopGrabbing()
                except Exception:
                    pass
                sync_mod.configure_free_run(cam)
            self.start_grabbing()

    def stop_recording(self) -> None:
        with self._lock:
            recorders = list(self._recorders.values())
            self._recorders.clear()
            self._recording = False
        for rec in recorders:
            rec.close()
