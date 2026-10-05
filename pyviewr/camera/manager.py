"""Camera enumeration, grab loops, still/video capture."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from pyviewr import config as app_config
from pyviewr.camera import sync as sync_mod
from pyviewr.io.still import save_still
from pyviewr.io.video import DEFAULT_VIDEO_FPS, VideoRecorder
from pyviewr.processing.enhance import EnhanceParams, apply as apply_enhance

try:
    from pypylon import genicam, pylon
except ImportError:  # pragma: no cover - runtime dependency
    pylon = None  # type: ignore[assignment]
    genicam = None  # type: ignore[assignment]


FrameCallback = Callable[[int, np.ndarray], None]
ErrorCallback = Callable[[str], None]
DeviceRemovedCallback = Callable[[int], None]

# GenICam auto-function node that must be "Off" before manual writes.
_AUTO_NODE = {"ExposureTime": "ExposureAuto", "Gain": "GainAuto"}
_PREVIEW_FPS = 10.0


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    model: str
    serial: str
    display_name: str


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    value: float
    minimum: float
    maximum: float


def _require_pylon() -> None:
    if pylon is None:
        raise RuntimeError(
            "pypylon is not installed, or Basler pylon Software Suite is missing. "
            "Install pylon 7.1+ then: pip install pypylon"
        )


def _is_physically_removed_message(message: str) -> bool:
    return "physically removed" in message.lower()


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
        on_device_removed: DeviceRemovedCallback | None = None,
    ) -> None:
        _require_pylon()
        self._on_frame = on_frame
        self._on_error = on_error
        self._on_device_removed = on_device_removed
        self._tl_factory = pylon.TlFactory.GetInstance()
        self._cameras: list[pylon.InstantCamera] = []
        self._slots: list[int] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._device_removed = threading.Event()
        self._grabbing = False
        self._recording = False
        self._recorders: dict[int, VideoRecorder] = {}
        self._lock = threading.Lock()
        self._last_preview_at: dict[int, float] = {}
        self._enhance_params = EnhanceParams()

    def set_enhance_params(self, params: EnhanceParams) -> None:
        """Use the same software enhance for Still / Record as the preview."""
        self._enhance_params = params

    def _frame_for_save(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhance for still images and post-processed video output."""
        try:
            return apply_enhance(frame, self._enhance_params)
        except Exception:
            return frame

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

    def _resolve_slot_order(
        self, devices: list[DeviceInfo]
    ) -> tuple[list[tuple[int, int]], dict[str, str]]:
        """Pair each device with its UVL(0) / UVR(1) slot.

        Matches devices to their persisted serial -> label identity first.
        Any device with an unrecognized serial fills the next open slot (UVL
        before UVR), so the *first* camera ever seen becomes UVL and the
        second becomes UVR — after that, identity sticks regardless of USB
        enumeration order or whether the other camera is connected at all.
        A slot with no matching or fallback device is simply absent, so a
        lone UVR camera keeps slot 1 rather than sliding into slot 0.
        Devices beyond the two known slots get their own trailing slots.

        Returns a list of (device_index, slot_index) pairs, ordered by slot,
        plus any newly-learned serial -> label identity to persist.
        """
        identity = app_config.load_camera_identity()
        by_label: dict[str, int] = {}
        unknown: list[int] = []
        for i, d in enumerate(devices):
            label = identity.get(d.serial)
            if label in app_config.SLOT_LABELS and label not in by_label:
                by_label[label] = i
            else:
                unknown.append(i)

        updates: dict[str, str] = {}
        ordered: list[tuple[int, int]] = []
        for slot, label in enumerate(app_config.SLOT_LABELS):
            if label in by_label:
                ordered.append((by_label[label], slot))
            elif unknown:
                idx = unknown.pop(0)
                ordered.append((idx, slot))
                if devices[idx].serial:
                    updates[devices[idx].serial] = label
        extra_slot = len(app_config.SLOT_LABELS)
        for idx in unknown:
            ordered.append((idx, extra_slot))
            extra_slot += 1
        return ordered, updates

    def open_identified(self, max_devices: int = 2) -> list[DeviceInfo]:
        """Open up to max_devices cameras in a stable UVL / UVR slot order.

        Slot 0 is always the camera identified as UVL, slot 1 always UVR,
        keyed by serial number rather than USB enumeration order — so a lone
        UVR camera shows up as "Camera 1", not "Camera 0", and which physical
        camera lands where no longer depends on connection order.
        """
        devices = self.enumerate_devices()
        if not devices:
            raise RuntimeError("No cameras found.")
        pairs, updates = self._resolve_slot_order(devices)
        for serial, label in updates.items():
            app_config.save_camera_identity(serial, label)
        pairs = pairs[:max_devices]
        opened = self.open([device_index for device_index, _ in pairs])
        self._slots = [slot for _, slot in pairs]
        return opened

    def _slot(self, index: int) -> int:
        """Translate an internal (dense) camera index to its UVL/UVR slot."""
        if 0 <= index < len(self._slots):
            return self._slots[index]
        return index

    def close(self) -> None:
        self.stop_grabbing()
        self.stop_recording()
        for cam in self._cameras:
            self._dispose_camera(cam)
        self._cameras.clear()
        self._slots.clear()
        self._device_removed.clear()

    @staticmethod
    def _dispose_camera(cam: "pylon.InstantCamera") -> None:
        """Stop / close / destroy a camera, including after physical removal."""
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass
        try:
            if cam.IsOpen():
                cam.Close()
        except Exception:
            pass
        try:
            # Required after physical removal so the TL can rediscover the device.
            cam.DestroyDevice()
        except Exception:
            pass

    # ----- camera features (exposure / gain / gamma) ------------------------------

    def get_feature_info(self, name: str) -> FeatureInfo | None:
        """Read value + range of a float feature from the first camera."""
        if not self._cameras:
            return None
        node = getattr(self._cameras[0], name, None)
        if node is None:
            return None
        try:
            return FeatureInfo(
                name=name,
                value=float(node.GetValue()),
                minimum=float(node.GetMin()),
                maximum=float(node.GetMax()),
            )
        except Exception:
            return None

    def get_resulting_fps(self) -> float | None:
        """Read the camera's achievable frame rate under current settings.

        Reflects ExposureTime / ROI / bandwidth right now (no AcquisitionFrameRate
        limit is set anywhere in this app, so this is effectively the max fps).
        """
        if not self._cameras:
            return None
        cam = self._cameras[0]
        for name in ("ResultingFrameRate", "ResultingFrameRateAbs"):
            node = getattr(cam, name, None)
            if node is None:
                continue
            try:
                return float(node.GetValue())
            except Exception:
                continue
        return None

    def set_feature(self, name: str, value: float) -> None:
        """Set a float feature on all open cameras (clamped to valid range).

        Disables the matching auto function (ExposureAuto/GainAuto) first so
        the manual value sticks. Safe to call while grabbing.
        """
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        errors: list[str] = []
        for i, cam in enumerate(self._cameras):
            try:
                auto_name = _AUTO_NODE.get(name)
                if auto_name is not None:
                    auto_node = getattr(cam, auto_name, None)
                    if auto_node is not None:
                        try:
                            if auto_node.GetValue() != "Off":
                                auto_node.SetValue("Off")
                        except Exception:
                            pass
                node = getattr(cam, name, None)
                if node is None:
                    raise RuntimeError(f"feature not available: {name}")
                clamped = min(max(value, float(node.GetMin())), float(node.GetMax()))
                node.SetValue(clamped)
            except Exception as exc:
                errors.append(f"cam{i}: {exc}")
        if errors:
            raise RuntimeError(f"Set {name} failed — " + "; ".join(errors))

    # ----- grabbing --------------------------------------------------------------

    def start_grabbing(self) -> None:
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        if self._grabbing:
            return
        self._device_removed.clear()
        self._last_preview_at.clear()
        for cam in self._cameras:
            sync_mod.configure_free_run(cam)
            if not cam.IsGrabbing():
                strategy = (
                    pylon.GrabStrategy_OneByOne
                    if self._recording
                    else pylon.GrabStrategy_LatestImageOnly
                )
                cam.StartGrabbing(strategy)
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
        self._last_preview_at.clear()
        for cam in self._cameras:
            try:
                if cam.IsGrabbing():
                    cam.StopGrabbing()
            except Exception:
                pass
        self._grabbing = False

    def _camera_device_removed(self, cam: "pylon.InstantCamera") -> bool:
        try:
            return bool(cam.IsCameraDeviceRemoved())
        except Exception:
            return False

    def _notify_device_removed(self, index: int) -> None:
        """Stop all grab loops and notify once that a camera vanished."""
        if self._device_removed.is_set():
            return
        self._device_removed.set()
        self._stop.set()
        if self._on_device_removed is not None:
            self._on_device_removed(index)
        elif self._on_error is not None:
            self._on_error(
                f"Camera {index}: device physically removed — disconnecting"
            )

    def _grab_loop(self, index: int, cam: "pylon.InstantCamera") -> None:
        while not self._stop.is_set():
            try:
                if self._camera_device_removed(cam):
                    self._notify_device_removed(self._slot(index))
                    return
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
                # Preview is intentionally throttled.  During recording the
                # acquisition threads must not flood the Qt event queue with
                # one signal per camera frame.  Emit it before the disk
                # write, which can block for a while under backpressure —
                # otherwise a slow disk would stall this camera's preview,
                # and (since the write used to happen under the lock shared
                # with the other camera's grab thread) the other camera's
                # preview too.
                now = time.monotonic()
                last_preview = self._last_preview_at.get(index, 0.0)
                if (
                    self._on_frame is not None
                    and now - last_preview >= 1.0 / _PREVIEW_FPS
                ):
                    self._last_preview_at[index] = now
                    self._on_frame(self._slot(index), frame)
                if rec is not None:
                    rec.write(frame)
            except Exception as exc:
                # Physical unplug / link loss: InstantCamera stays open but is dead.
                if self._camera_device_removed(cam) or _is_physically_removed_message(
                    str(exc)
                ):
                    self._notify_device_removed(self._slot(index))
                    return
                if self._on_error is not None:
                    self._on_error(f"Camera {self._slot(index)}: {exc}")
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
            path = save_dir / f"cam{self._slot(0)}_{stamp}.png"
            save_still(path, self._frame_for_save(frame))
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
                        raise RuntimeError(
                            f"Still grab failed on camera {self._slot(i)}."
                        )
                    frames.append(array_from_grab(grab))

            for i, frame in enumerate(frames):
                path = save_dir / f"cam{self._slot(i)}_{stamp}.png"
                save_still(path, self._frame_for_save(frame))
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

    def start_recording(
        self, save_dir: Path, fps: float = DEFAULT_VIDEO_FPS
    ) -> list[Path]:
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
                path = save_dir / f"cam{self._slot(i)}_{stamp}.mp4"
                self._recorders[i] = VideoRecorder(
                    path,
                    (w, h),
                    fps=fps,
                    frame_processor=self._frame_for_save,
                    on_complete=lambda path, index=i: self._video_postprocess_done(
                        index, path
                    ),
                    on_error=lambda message, index=i: self._video_postprocess_error(
                        index, message
                    ),
                )
                paths.append(path)
            self._recording = True

        # Align start for multi-cam with a single action pulse then free-run.
        if len(self._cameras) >= 2:
            self._restart_free_run_aligned()
        else:
            # Re-open the single-camera grab stream with OneByOne.  The
            # connected preview normally uses LatestImageOnly.
            self.stop_grabbing()
            self.start_grabbing()
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

    def _video_postprocess_done(self, index: int, path: Path) -> None:
        if self._on_error is not None:
            self._on_error(f"Camera {self._slot(index)}: video saved: {path.name}")

    def _video_postprocess_error(self, index: int, message: str) -> None:
        if self._on_error is not None:
            self._on_error(
                f"Camera {self._slot(index)}: video post-processing failed: {message}"
            )

    def stop_recording(self) -> None:
        with self._lock:
            recording = self._recording or bool(self._recorders)
        if not recording:
            return

        was_grabbing = self._grabbing
        if was_grabbing:
            # Stop retrieval first so every frame already accepted by the
            # recorder can drain to the raw spool before post-processing.
            self.stop_grabbing()

        with self._lock:
            recorders = list(self._recorders.items())
            ended_at = time.monotonic()
            self._recorders.clear()
            self._recording = False

        first_error: Exception | None = None
        for index, rec in recorders:
            try:
                rec.close(ended_at)
                if rec.backpressure_events and self._on_error is not None:
                    self._on_error(
                        f"Camera {self._slot(index)}: disk spool was busy "
                        f"({rec.backpressure_events} waits); capture may be slower "
                        "than the camera rate."
                    )
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if was_grabbing and self._cameras:
            self.start_grabbing()
        if first_error is not None:
            raise first_error
