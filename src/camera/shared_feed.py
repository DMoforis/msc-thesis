"""
shared_feed.py
--------------
Single OpenCV capture thread that distributes frames to all consumers.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Problem solved:
  PoC #1 (rPPG) and PoC #2 (face landmarks) both called cv2.VideoCapture()
  independently, causing an exclusive access conflict on Windows. Only one
  process/thread can own the webcam at a time.

Solution:
  SharedCameraFeed opens ONE cv2.VideoCapture and stores the latest frame in
  a shared variable protected by a threading.Lock. Any number of consumers
  call get_frame() to get a copy of that frame — non-blocking, no queues,
  no head-of-line blocking. Slow consumers always receive the most recent
  frame without stalling the capture thread.

Usage:
  feed = SharedCameraFeed(camera_index=0, fps=30)
  feed.start()

  # In each consumer's processing loop:
  frame = feed.get_frame()   # returns latest BGR frame or None
  if frame is not None:
      # process frame ...

  feed.stop()

  # Or as a context manager:
  with SharedCameraFeed(camera_index=0) as feed:
      frame = feed.get_frame()
      ...
"""

import threading
import time

import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import CAMERA_INDEX, CAMERA_FPS


class SharedCameraFeed:
    """
    Single-capture-thread webcam feed shared across all consumer modules.

    One background thread reads from cv2.VideoCapture at the camera's native
    rate and atomically replaces self._frame on each successful read.
    Consumers call get_frame() at any time; they always receive the latest
    available BGR frame (or None if the camera hasn't delivered one yet).

    Thread safety:
      - The capture thread holds _lock only for the brief moment it replaces
        self._frame, so consumers are never blocked for long.
      - get_frame() returns a copy — callers can modify it freely.

    Attributes:
        width  (int): Frame width in pixels, set after start().
        height (int): Frame height in pixels, set after start().
    """

    def __init__(self, camera_index: int = CAMERA_INDEX, fps: int = CAMERA_FPS):
        self._camera_index = camera_index
        self._target_fps   = fps
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock          = threading.Lock()
        self._stop_event    = threading.Event()
        self._thread: threading.Thread | None = None
        self.width:  int = 0
        self.height: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open the camera and start the background capture thread.

        Blocks for up to 2 seconds waiting for the first frame so that
        callers can assume get_frame() returns a valid frame immediately
        after start() returns.

        Raises:
            RuntimeError: if the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[SharedCameraFeed] Cannot open camera at index "
                f"{self._camera_index}. Check that no other application "
                f"is using it and that the index is correct."
            )

        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="SharedCameraFeed",
            daemon=True,   # exits automatically if the main thread dies
        )
        self._thread.start()

        # Wait up to 2 s for the first valid frame before returning.
        # This prevents consumers from receiving None immediately after start().
        deadline = time.monotonic() + 2.0
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.02)

        if self._frame is None:
            self.stop()
            raise RuntimeError(
                f"[SharedCameraFeed] Camera opened but no frames arrived "
                f"within 2 seconds. Check camera connection and lighting."
            )

        print(f"[Camera] Feed started — {self.width}×{self.height} "
              f"@ {self._target_fps} FPS (index {self._camera_index})")

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
        print("[Camera] Feed stopped.")

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray | None:
        """
        Return the latest captured BGR frame, or None if none is available.

        Always returns a copy so the caller can freely draw on or modify it
        without affecting other consumers or the capture thread.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SharedCameraFeed":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: reads frames continuously from the camera."""
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                # Camera returned no frame — brief back-off to avoid busy-spin.
                # This typically happens at startup or on transient read errors.
                time.sleep(0.005)


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SharedCameraFeed smoke-test — press Q to quit.")

    with SharedCameraFeed() as feed:
        while True:
            frame = feed.get_frame()
            if frame is None:
                continue
            cv2.putText(frame,
                        f"SharedCameraFeed  {feed.width}x{feed.height}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 120), 2)
            cv2.imshow("SharedCameraFeed — Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Done.")
