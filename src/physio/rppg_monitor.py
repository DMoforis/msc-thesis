"""
rppg_monitor.py
---------------
Physiological signals via rPPG — Phase 2 (shared-feed refactor).
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Changes from Phase 1 (PoC #1):
  - No longer opens the camera directly; receives frames from SharedCameraFeed.
    This resolves the exclusive webcam access conflict with face_monitor.py.
  - HR reading frequency: every 10 s (was 30 s) per supervisor feedback.
  - HRV computed from a 5-minute rolling buffer (was 30 s — too short for LF/HF).
  - Two row types in physio_readings: 'hr_10s' and 'hrv_5min'.
  - HRV metrics (RMSSD, SDNN, LF/HF) come from open-rppg's built-in heartpy
    pipeline rather than a separate neurokit2 pass.
  - DB schema updated to match SPEC Section 2 (adds sdnn, window_type columns).
  - All config constants imported from src/utils/config.py.

Dependencies:
  pip install open-rppg opencv-python numpy
"""

import sys
import os
import time
import sqlite3
import threading
from datetime import datetime

import cv2
import numpy as np
import rppg

# ── Path bootstrap (allows running as a script from any directory) ────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.camera.shared_feed import SharedCameraFeed
from src.physio.hrv_processor import validate_hrv_metrics, _MAX_HR_DELTA
from src.utils.config import (
    DB_PATH, CAMERA_INDEX, CAMERA_FPS,
    HR_WINDOW_SECONDS, HRV_WINDOW_SECONDS,
    RPPG_MODEL,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def init_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the shared SQLite database and ensure the physio_readings
    table matches the Phase 2 schema.

    Migration strategy: use ALTER TABLE to add new columns if they don't exist,
    so existing Phase 1 data is preserved.
    """
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS physio_readings (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT    NOT NULL,
            heart_rate     REAL,
            rmssd          REAL,
            lf_hf_ratio    REAL,
            sdnn           REAL,
            signal_quality REAL,
            window_type    TEXT
        )
    """)
    conn.commit()

    # Add new Phase 2 columns to tables created by Phase 1 (safe no-ops if
    # they already exist — SQLite raises OperationalError, which we swallow).
    for column_def in ("sdnn REAL", "window_type TEXT"):
        try:
            conn.execute(f"ALTER TABLE physio_readings ADD COLUMN {column_def}")
            conn.commit()
        except sqlite3.OperationalError:
            pass   # column already exists

    print(f"[DB] physio_readings table ready → {db_path}")
    return conn


def save_reading(
    conn: sqlite3.Connection,
    heart_rate: float,
    rmssd: float | None,
    lf_hf: float | None,
    sdnn: float | None,
    signal_quality: float,
    window_type: str,
) -> None:
    """Insert one physiological reading row."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO physio_readings
            (timestamp, heart_rate, rmssd, lf_hf_ratio, sdnn, signal_quality, window_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ts, heart_rate, rmssd, lf_hf, sdnn, signal_quality, window_type))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# PHYSIO MODULE
# Thin wrapper around open-rppg. All rPPG logic lives here.
# ─────────────────────────────────────────────────────────────────────────────

class PhysioModule:
    """
    Wraps rppg.Model for clean integration into the shared-feed architecture.

    Phase 2 usage pattern:
      module = PhysioModule()
      with module.model:                          # starts inference thread
          while running:
              frame = feed.get_frame()            # BGR from SharedCameraFeed
              if frame is not None:
                  module.push_frame(frame)        # convert BGR→RGB, feed model
              # query model.hr() on schedule
    """

    def __init__(self, model_name: str = RPPG_MODEL):
        print(f"[rPPG] Loading model: {model_name} ...")
        self.model = rppg.Model(model_name)
        print("[rPPG] Model loaded.")

    def push_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Convert a BGR frame (from SharedCameraFeed) to RGB and feed it to
        the rPPG model for BVP signal accumulation.

        open-rppg's update_frame() expects RGB; OpenCV gives us BGR.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.model.update_frame(rgb)

    def get_hr(self, window_seconds: int) -> dict | None:
        """
        Query the model for HR and signal quality over the last window_seconds.

        Returns a dict with 'hr', 'SQI', and optionally 'hrv' (heartpy metrics
        populated only when SQI > 0.5 and enough signal is available).
        Returns None if the model has no signal yet.
        """
        return self.model.hr(start=-window_seconds, return_hrv=False)

    def get_hrv(self, window_seconds: int) -> dict | None:
        """
        Query the model for full HRV metrics over the last window_seconds.
        Includes RMSSD, SDNN, LF/HF from heartpy's frequency-domain analysis.
        HRV fields are populated only when SQI > 0.5.
        """
        return self.model.hr(start=-window_seconds, return_hrv=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor(feed: SharedCameraFeed) -> None:
    """
    Process frames from the shared camera feed, compute HR every 10 s and
    full HRV every 5 min, and save readings to the database.

    Parameters
    ----------
    feed : SharedCameraFeed
        Running shared camera feed. Must already be started (feed.start()
        called) before this function is invoked.
    """
    conn    = init_database()
    physio  = PhysioModule()
    stop_ev = threading.Event()

    hr_window_start  = time.monotonic()
    hrv_window_start = time.monotonic()
    last_status      = time.monotonic()
    STATUS_INTERVAL  = 5.0   # seconds between "still running" console prints

    # HR samples collected during the current HRV window — used for cross-validation
    # inside validate_hrv_metrics() (see src/physio/hrv_processor.py).
    hr_samples_for_hrv: list[float] = []

    print(f"\n[rPPG] Monitor started.")
    print(f"[rPPG] HR every {HR_WINDOW_SECONDS}s | HRV every {HRV_WINDOW_SECONDS}s.")
    print(f"[rPPG] Keep face visible and still. Press Q to quit.\n")

    try:
        with physio.model:   # initialises rPPG inference thread

            while not stop_ev.is_set():
                frame = feed.get_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # ── Feed frame to rPPG model ──────────────────────────────────
                physio.push_frame(frame)

                # ── Build annotated preview ───────────────────────────────────
                preview = frame.copy()
                box = physio.model.box
                if box is not None:
                    # box shape: ((y_start, y_end), (x_start, x_end))
                    y1, y2 = int(box[0][0]), int(box[0][1])
                    x1, x2 = int(box[1][0]), int(box[1][1])
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 100), 2)
                    cv2.putText(preview, "Face OK", (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)
                else:
                    cv2.putText(preview,
                                "No face — centre yourself in frame",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

                # Show time remaining to next HR reading
                hr_elapsed  = time.monotonic() - hr_window_start
                hr_remaining = max(0, HR_WINDOW_SECONDS - int(hr_elapsed))
                cv2.putText(preview, f"HR in {hr_remaining}s",
                            (10, preview.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("rPPG Monitor — Q to quit", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[rPPG] Q pressed — stopping.")
                    stop_ev.set()
                    break

                now = time.monotonic()

                # ── Periodic console status ───────────────────────────────────
                if now - last_status >= STATUS_INTERVAL:
                    hrv_elapsed = now - hrv_window_start
                    print(f"[rPPG] Collecting… "
                          f"HR in {max(0, HR_WINDOW_SECONDS - int(hr_elapsed))}s | "
                          f"HRV in {max(0, HRV_WINDOW_SECONDS - int(hrv_elapsed))}s")
                    last_status = now

                # ── HR reading every 10 s ─────────────────────────────────────
                if now - hr_window_start >= HR_WINDOW_SECONDS:
                    result = physio.get_hr(HR_WINDOW_SECONDS)

                    if result and result.get("hr"):
                        hr  = float(result["hr"])
                        sqi = float(result.get("SQI") or 0.0)

                        if 40.0 <= hr <= 200.0:
                            save_reading(conn,
                                         heart_rate    = round(hr, 1),
                                         rmssd         = None,
                                         lf_hf         = None,
                                         sdnn          = None,
                                         signal_quality = round(sqi, 2),
                                         window_type   = "hr_10s")
                            hr_samples_for_hrv.append(hr)
                            print(f"[HR]  {round(hr, 1):5.1f} BPM  "
                                  f"SQI: {sqi:.2f}")
                        else:
                            print(f"[HR] rejected — implausible value: {hr:.1f} BPM")
                    else:
                        print("[rPPG] Not enough signal yet "
                              f"(need {HR_WINDOW_SECONDS}s of face data).")

                    hr_window_start = now

                # ── HRV reading every 5 min ───────────────────────────────────
                if now - hrv_window_start >= HRV_WINDOW_SECONDS:
                    result = physio.get_hrv(HRV_WINDOW_SECONDS)

                    if result and result.get("hr"):
                        hr  = float(result["hr"])
                        sqi = float(result.get("SQI") or 0.0)
                        hrv = result.get("hrv") or {}   # empty if SQI ≤ 0.5

                        rmssd_raw = hrv.get("rmssd")
                        sdnn_raw  = hrv.get("sdnn")
                        lf_hf     = hrv.get("LF/HF")

                        # ── Cross-validate and apply plausibility bounds ──────
                        rmssd, sdnn = validate_hrv_metrics(
                            rmssd_raw, sdnn_raw, hr, hr_samples_for_hrv
                        )

                        # For the console report we still need avg_10s_hr / hr_delta
                        if hr_samples_for_hrv:
                            avg_10s_hr = sum(hr_samples_for_hrv) / len(hr_samples_for_hrv)
                            hr_delta   = abs(hr - avg_10s_hr)
                        else:
                            avg_10s_hr = hr
                            hr_delta   = 0.0

                        if 40.0 <= hr <= 200.0:
                            save_reading(
                                conn,
                                heart_rate    = round(hr, 1),
                                rmssd         = rmssd,
                                lf_hf         = round(lf_hf, 2) if lf_hf else None,
                                sdnn          = sdnn,
                                signal_quality = round(sqi, 2),
                                window_type   = "hrv_5min",
                            )
                            # Report raw vs. accepted values so discards are visible
                            rmssd_str = (
                                f"{rmssd:.1f} ms" if rmssd else
                                f"-- (raw={rmssd_raw:.1f} ms, rejected)"
                                if rmssd_raw else "-- (SQI too low)"
                            )
                            print(f"\n{'-'*52}")
                            print(f"  [HRV 5-min window]")
                            print(f"  Heart Rate    : {hr:.1f} BPM  "
                                  f"(10s avg {avg_10s_hr:.1f}, d={hr_delta:.1f})")
                            print(f"  RMSSD         : {rmssd_str}")
                            print(f"  SDNN          : "
                                  f"{sdnn:.1f} ms" if sdnn else
                                  f"  SDNN          : --")
                            print(f"  LF/HF         : "
                                  f"{lf_hf:.2f}" if lf_hf else
                                  f"  LF/HF         : --")
                            print(f"  Signal quality: {sqi:.2f} / 1.00")
                            if hr_delta > _MAX_HR_DELTA:
                                print(f"  *** HR inconsistency ({hr_delta:.1f} BPM) - HRV discarded")
                            print(f"{'-'*52}\n")
                        else:
                            print(f"[HR] rejected — implausible value: {hr:.1f} BPM")
                    else:
                        print("[rPPG] HRV window: no signal available yet.")

                    # Reset the HR sample buffer for the next HRV window
                    hr_samples_for_hrv.clear()
                    hrv_window_start = now

    except KeyboardInterrupt:
        print("\n[rPPG] Stopped by Ctrl+C.")

    finally:
        cv2.destroyAllWindows()
        conn.close()
        print("[rPPG] Clean shutdown complete.")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[rPPG] Standalone mode — creating shared camera feed.")
    with SharedCameraFeed(camera_index=CAMERA_INDEX, fps=CAMERA_FPS) as feed:
        run_monitor(feed)
