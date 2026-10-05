"""Disk-backed video recording and post-processing."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO_FPS = 67.0
FRAME_QUEUE_SIZE = 8
_QUEUE_STOP = object()


class VideoRecorder:
    def __init__(
        self,
        path: Path | str,
        size: tuple[int, int],
        fps: float = DEFAULT_VIDEO_FPS,
        frame_processor: Callable[[np.ndarray], np.ndarray] | None = None,
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
        self._on_complete = on_complete
        self._on_error = on_error
        self._frame_bytes = self._size[0] * self._size[1]
        self._frame_count = 0
        self._started_at = time.monotonic()
        self._temporary_path = self.path.with_name(f"{self.path.name}.part.raw")
        self._temporary_output = self.path.with_name(f"{self.path.name}.part.mp4")
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
        self._spool_thread = threading.Thread(
            target=self._spool_loop,
            name=f"pyviewr-spool-{self.path.stem}",
            daemon=True,
        )
        self._spool_thread.start()

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

    def _spool_loop(self) -> None:
        try:
            with self._temporary_path.open("wb", buffering=1024 * 1024) as spool:
                while True:
                    item = self._queue.get()
                    try:
                        if item is _QUEUE_STOP:
                            return
                        frame = item
                        assert isinstance(frame, np.ndarray)
                        spool.write(frame.data)
                        self._frame_count += 1
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._writer_error = exc

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
            self._raise_writer_error()
            try:
                self._queue.put(_QUEUE_STOP, timeout=0.2)
                break
            except queue.Full:
                continue
        self._spool_thread.join()
        self._raise_writer_error()
        self._postprocess_thread = threading.Thread(
            target=self._postprocess_loop,
            args=(ended_at,),
            name=f"pyviewr-postprocess-{self.path.stem}",
            daemon=False,
        )
        self._postprocess_thread.start()

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
            self._finalize_mp4(ended_at)
        except BaseException as exc:
            self._postprocess_error = exc
            if self._on_error is not None:
                self._on_error(str(exc))
        else:
            if self._on_complete is not None:
                self._on_complete(self.path)

    def _finalize_mp4(self, ended_at: float | None = None) -> None:
        end = time.monotonic() if ended_at is None else ended_at
        elapsed = max(end - self._started_at, 0.001)
        actual_fps = (
            self._frame_count / elapsed if self._frame_count > 0 else self._fps
        )
        # Keep metadata in a useful range for very short recordings.
        actual_fps = max(1.0, min(actual_fps, self._fps))

        final_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self._temporary_output),
            final_fourcc,
            actual_fps,
            self._size,
            isColor=True,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open MP4 writer: {self.path}")

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
                    bgr = cv2.cvtColor(
                        np.ascontiguousarray(frame, dtype=np.uint8),
                        cv2.COLOR_GRAY2BGR,
                    )
                    writer.write(bgr)
        finally:
            writer.release()

        self._temporary_output.replace(self.path)
        self._temporary_path.unlink(missing_ok=True)
