"""Disk-backed video recording and post-processing."""

from __future__ import annotations

import fcntl
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO_FPS = 67.0
# ~256 MB per camera at 2840x2840; absorbs short disk / encoder stalls.
FRAME_QUEUE_SIZE = 32
_QUEUE_STOP = object()
_FFMPEG = shutil.which("ffmpeg")
# Two cameras encode at once; give each half the cores.
_ENCODER_THREADS = max(2, (os.cpu_count() or 4) // 2)
# The live encoder stamps frames at 1 fps.  The captured rate is only known
# when recording stops, and stream copy cannot change the Matroska frame-rate
# header, so the header is patched in place and the timestamps rescaled.
# 1 fps gives DefaultDuration = 1e9 ns, a 4-byte field any rate >= 1 fits in.
_ENCODE_FPS = 1
_F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", 1031)
_PIPE_SIZE = 1024 * 1024
_MKV_DEFAULT_DURATION = b"\x23\xe3\x83\x84"


def _ffmpeg_error(log, fallback: str) -> str:
    try:
        log.seek(0)
        text = log.read().decode("utf-8", "replace").strip()
    except (OSError, ValueError):
        text = ""
    return text.splitlines()[-1] if text else fallback


class VideoRecorder:
    def __init__(
        self,
        path: Path | str,
        size: tuple[int, int],
        fps: float = DEFAULT_VIDEO_FPS,
        frame_processor: Callable[[np.ndarray], np.ndarray] | None = None,
        lossless: bool = False,
        on_complete: Callable[[Path], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        self._size = (int(width), int(height))
        if self._size[0] <= 0 or self._size[1] <= 0:
            raise ValueError(f"Invalid video size: {self._size}")
        self._fps = float(fps)
        if self._fps <= 0:
            raise ValueError(f"Invalid video FPS: {self._fps}")
        self._frame_processor = frame_processor
        # lossless: FFV1 grayscale (bit-exact).  Otherwise MPEG-4 (lossy, small).
        self._lossless = bool(lossless)
        self._on_complete = on_complete
        self._on_error = on_error
        self._frame_bytes = self._size[0] * self._size[1]
        self._frame_count = 0
        self._started_at = time.monotonic()
        # Arrival times of the second and latest frame.  The first frame is
        # skipped: with two cameras it is the Action-triggered alignment shot,
        # followed by a free-run restart gap that is not part of the rate.
        self._second_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._frames_written = 0
        self._temporary_path = self.path.with_name(f"{self.path.name}.part.raw")
        self._temporary_output = self.path.with_name(
            f"{self.path.name}.part{self.path.suffix}"
        )
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(
            maxsize=FRAME_QUEUE_SIZE
        )
        self._closing = threading.Event()
        self._state_lock = threading.Lock()
        self._closed = False
        self._writer_error: BaseException | None = None
        self._postprocess_error: BaseException | None = None
        self._backpressure_events = 0
        self._postprocess_thread: threading.Thread | None = None
        # Lossless video is encoded to FFV1 while recording: raw frames at
        # 2840x2840 / 67 fps (~540 MB/s per camera) outrun a SATA SSD, FFV1
        # output is ~1/8 of that.  Without ffmpeg, fall back to the raw spool.
        self._encoder: subprocess.Popen | None = None
        self._encoder_log = None
        if self._lossless and _FFMPEG is not None:
            self._start_encoder()
        self._spool_thread = threading.Thread(
            target=self._spool_loop,
            name=f"pyviewr-spool-{self.path.stem}",
            daemon=True,
        )
        self._spool_thread.start()

    @property
    def frame_count(self) -> int:
        """Frames spooled so far (final once close() has returned)."""
        return self._frame_count

    def measured_fps(self, ended_at: float) -> float:
        """Capture rate from frame arrival times (fallback: creation to ended_at)."""
        if (
            self._frames_written >= 3
            and self._second_frame_at is not None
            and self._last_frame_at is not None
            and self._last_frame_at > self._second_frame_at
        ):
            return (self._frames_written - 2) / (
                self._last_frame_at - self._second_frame_at
            )
        elapsed = max(ended_at - self._started_at, 0.001)
        return self._frame_count / elapsed

    @property
    def backpressure_events(self) -> int:
        """Number of times capture had to wait for the disk spool."""
        return self._backpressure_events

    def write(self, frame: np.ndarray) -> None:
        if frame.ndim != 2:
            raise ValueError("Expected grayscale frame (H, W).")
        if frame.shape[::-1] != self._size:
            raise ValueError(
                f"Unexpected frame size: {(frame.shape[1], frame.shape[0])}; "
                f"expected {self._size}"
            )
        if frame.dtype != np.uint8:
            raise ValueError("Expected uint8 grayscale frame.")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cannot write to a closed video recorder.")

        now = time.monotonic()
        self._frames_written += 1
        if self._frames_written == 2:
            self._second_frame_at = now
        self._last_frame_at = now

        # The camera buffer must never be silently discarded because the
        # spool queue is full.  Wait for the disk writer instead.
        item = np.ascontiguousarray(frame).copy()
        while True:
            self._raise_writer_error()
            if self._closing.is_set():
                raise RuntimeError("Video recorder is closing.")
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                self._backpressure_events += 1

    def _start_encoder(self) -> None:
        width, height = self._size
        self._encoder_log = tempfile.TemporaryFile()
        self._encoder = subprocess.Popen(
            [
                _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "gray",
                "-s", f"{width}x{height}", "-framerate", str(_ENCODE_FPS),
                "-i", "pipe:0",
                "-c:v", "ffv1", "-level", "3",
                "-threads", str(_ENCODER_THREADS),
                "-slices", "24", "-slicecrc", "0",
                str(self._temporary_output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._encoder_log,
        )
        # The default 64 KiB pipe needs ~130 wakeups per 8 MB frame.
        try:
            fcntl.fcntl(self._encoder.stdin.fileno(), _F_SETPIPE_SZ, _PIPE_SIZE)
        except OSError:
            pass

    def _spool_loop(self) -> None:
        try:
            if self._encoder is not None:
                self._write_frames(self._encoder.stdin, encode=True)
            else:
                with self._temporary_path.open(
                    "wb", buffering=1024 * 1024
                ) as spool:
                    self._write_frames(spool, encode=False)
        except BaseException as exc:
            if isinstance(exc, BrokenPipeError) and self._encoder is not None:
                exc = RuntimeError(
                    _ffmpeg_error(self._encoder_log, "ffmpeg encoder exited")
                )
            self._writer_error = exc

    def _write_frames(self, out, encode: bool) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _QUEUE_STOP:
                        return
                    frame = item
                    assert isinstance(frame, np.ndarray)
                    if encode:
                        frame = self._process(frame)
                    out.write(frame.data)
                    self._frame_count += 1
                finally:
                    self._queue.task_done()
        finally:
            if encode:
                # EOF lets ffmpeg flush and finish the file.
                out.close()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        if self._frame_processor is not None:
            frame = self._frame_processor(frame)
        if frame.ndim != 2 or frame.shape[::-1] != self._size:
            raise ValueError("Frame processor returned an unexpected frame shape.")
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"Video spool failed: {self._writer_error}")

    def close(self, ended_at: float | None = None) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._closing.set()

        while True:
            if self._writer_error is not None:
                self._kill_encoder()
            self._raise_writer_error()
            try:
                self._queue.put(_QUEUE_STOP, timeout=0.2)
                break
            except queue.Full:
                continue
        self._spool_thread.join()
        if self._writer_error is not None:
            self._kill_encoder()
        self._raise_writer_error()
        self._postprocess_thread = threading.Thread(
            target=self._postprocess_loop,
            args=(ended_at,),
            name=f"pyviewr-postprocess-{self.path.stem}",
            daemon=False,
        )
        self._postprocess_thread.start()

    def _kill_encoder(self) -> None:
        if self._encoder is not None and self._encoder.poll() is None:
            self._encoder.kill()
            self._encoder.wait()

    def wait_for_completion(self) -> None:
        """Wait for post-processing, mainly for shutdown and tests."""
        if self._postprocess_thread is not None:
            self._postprocess_thread.join()
        if self._postprocess_error is not None:
            raise RuntimeError(
                f"Video post-processing failed: {self._postprocess_error}"
            )

    def _postprocess_loop(self, ended_at: float) -> None:
        try:
            self._finalize_video(ended_at)
        except BaseException as exc:
            self._postprocess_error = exc
            if self._on_error is not None:
                self._on_error(str(exc))
        else:
            if self._on_complete is not None:
                self._on_complete(self.path)

    def _finalize_video(self, ended_at: float | None = None) -> None:
        end = time.monotonic() if ended_at is None else ended_at
        actual_fps = (
            self.measured_fps(end) if self._frame_count > 0 else self._fps
        )
        # Keep metadata in a useful range for very short recordings.
        actual_fps = max(1.0, min(actual_fps, self._fps))

        if self._encoder is not None:
            self._finalize_encoded(actual_fps)
            return

        codec = "FFV1" if self._lossless else "mp4v"
        writer = cv2.VideoWriter(
            str(self._temporary_output),
            cv2.VideoWriter_fourcc(*codec),
            actual_fps,
            self._size,
            isColor=not self._lossless,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open {codec} writer: {self.path}")

        try:
            with self._temporary_path.open("rb", buffering=1024 * 1024) as spool:
                while True:
                    data = spool.read(self._frame_bytes)
                    if not data:
                        break
                    if len(data) != self._frame_bytes:
                        raise RuntimeError(
                            f"Incomplete raw frame in temporary video: "
                            f"{self._temporary_path}"
                        )
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(
                        (self._size[1], self._size[0])
                    )
                    if self._frame_processor is not None:
                        frame = self._frame_processor(frame)
                    if frame.ndim != 2 or frame.shape[::-1] != self._size:
                        raise ValueError(
                            "Frame processor returned an unexpected frame shape."
                        )
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
                    if self._lossless:
                        writer.write(frame)
                    else:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()

        self._temporary_output.replace(self.path)
        self._temporary_path.unlink(missing_ok=True)

    def _finalize_encoded(self, actual_fps: float) -> None:
        assert self._encoder is not None
        if self._encoder.wait() != 0:
            raise RuntimeError(
                _ffmpeg_error(
                    self._encoder_log,
                    f"ffmpeg exited with {self._encoder.returncode}",
                )
            )
        self._encoder_log.close()
        self._patch_default_duration(actual_fps)
        retimed = self.path.with_name(f"{self.path.name}.retime{self.path.suffix}")
        result = subprocess.run(
            [
                _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-itsscale", f"{_ENCODE_FPS / actual_fps:.9f}",
                "-i", str(self._temporary_output),
                "-c", "copy", str(retimed),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            retimed.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg retime failed: {result.stderr.strip()}")
        retimed.replace(self.path)
        self._temporary_output.unlink(missing_ok=True)

    def _patch_default_duration(self, fps: float) -> None:
        """Rewrite the track DefaultDuration (ns per frame) to match fps."""
        expected = round(1e9 / _ENCODE_FPS).to_bytes(4, "big")
        with self._temporary_output.open("r+b") as f:
            head = f.read(64 * 1024)
            at = head.find(_MKV_DEFAULT_DURATION)
            value_at = at + len(_MKV_DEFAULT_DURATION)
            if at < 0 or head[value_at:value_at + 4] != expected:
                raise RuntimeError(
                    f"Matroska DefaultDuration not found: {self._temporary_output}"
                )
            f.seek(value_at)
            f.write(round(1e9 / fps).to_bytes(4, "big"))
