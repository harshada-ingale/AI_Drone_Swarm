"""Video capture abstraction supporting live threaded sources and sequential video files."""

import threading
import time
from typing import Any, Union
import cv2
import numpy as np


class Camera:
    """Camera capture wrapper supporting webcams, video files, and network streams.

    Args:
        source: Device index (int or str), video file path, or stream URL.
        threaded: If True, spawns a daemon thread keeping only the freshest frame.
                  If False, performs synchronous sequential reads (ideal for files).
        loop: If True, loops back to frame 0 when the video ends.
    """

    def __init__(
        self,
        source: Union[int, str],
        threaded: bool = True,
        loop: bool = False,
    ) -> None:
        if isinstance(source, str) and source.isdigit():
            self.source: Union[int, str] = int(source)
        else:
            self.source = source

        self.threaded = threaded
        self.loop = loop

        self.cap: cv2.VideoCapture | None = None
        self.ended: bool = False

        self._seq: int = 0
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running: bool = False

    def start(self) -> "Camera":
        """Opens video capture device or file stream.

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self.ended = False
        self._running = True

        if self.threaded:
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()

        return self

    def _capture_worker(self) -> None:
        """Daemon worker loop fetching latest frames."""
        while self._running:
            if self.cap is None:
                break

            ret, frame = self.cap.read()
            if not ret:
                if self.loop and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with self._lock:
                    self.ended = True
                break

            with self._lock:
                self._frame = frame
                self._seq += 1

            time.sleep(0.001)

    def read(self) -> tuple[np.ndarray | None, int]:
        """Reads a frame and sequence number.

        Returns:
            (frame_or_None, sequence_number)
        """
        if self.threaded:
            with self._lock:
                frame_copy = (
                    self._frame.copy() if self._frame is not None else None
                )
                return (frame_copy, self._seq)

        # Synchronous sequential mode
        if self.ended or self.cap is None:
            return (None, self._seq)

        ret, frame = self.cap.read()
        if not ret:
            if self.loop and self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            if not ret:
                self.ended = True
                return (None, self._seq)

        self._seq += 1
        self._frame = frame
        return (frame, self._seq)

    def stop(self) -> None:
        """Stops capture thread and releases video resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.ended = True
