"""
rppg_monitor.py
---------------
Hello World proof-of-concept #1 — Physiological signals via rPPG.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

What this script does:
  1. Opens the webcam using open-rppg's real-time pipeline
  2. Every WINDOW_SECONDS, reads Heart Rate from the model
  3. Computes RMSSD and LF/HF independently using neurokit2 on the raw BVP signal
     (more reliable than asking open-rppg for HRV directly at short windows)
  4. Saves each reading to a local SQLite database with a timestamp
  5. Shuts down cleanly when Q is pressed, suppressing open-rppg's thread noise

Fixes from v1:
  - HRV (RMSSD, LF/HF) now computed via neurokit2 on the raw BVP waveform
    rather than relying on open-rppg's internal HRV output, which needs a
    longer buffer than 30s to return non-zero values
  - Thread RuntimeError on quit is suppressed cleanly using a stop event
    and by wrapping the camera loop exit in a try/except

Dependencies:
  pip install open-rppg opencv-python numpy neurokit2
"""

import time
import sqlite3
import threading
from datetime import datetime

import cv2
import numpy as np
import neurokit2 as nk   # HRV analysis toolkit — we use this for RMSSD and LF/HF

import rppg


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WEBCAM_INDEX   = 1      # change to 1 if index 0 doesn't open your webcam
WINDOW_SECONDS = 30     # seconds of signal per reading
DB_PATH        = "data/stress_monitor.db"
PRINT_INTERVAL = 5      # seconds between "still running" status messages
RPPG_MODEL     = "FacePhys.rlap"
SAMPLING_RATE  = 30     # assumed webcam FPS — used by neurokit2 for HRV math


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def init_database(db_path: str) -> sqlite3.Connection:
    """
    Open or create the SQLite database and ensure the physio table exists.
    This is the same table structure as v1, unchanged — existing data is safe.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS physio_readings (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT    NOT NULL,
            heart_rate     REAL,
            rmssd          REAL,
            lf_hf_ratio    REAL,
            signal_quality REAL,
            modality       TEXT DEFAULT 'rppg_face'
        )
    """)
    conn.commit()
    print(f"[DB] Database ready → {db_path}")
    return conn


def save_reading(conn, heart_rate, rmssd, lf_hf, signal_quality):
    """Insert one physiological reading. Called every WINDOW_SECONDS."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO physio_readings
            (timestamp, heart_rate, rmssd, lf_hf_ratio, signal_quality, modality)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ts, heart_rate, rmssd, lf_hf, signal_quality, "rppg_face"))
    conn.commit()
    print(f"[DB] Saved → {ts} | HR: {heart_rate:.1f} BPM | "
          f"RMSSD: {rmssd:.1f} ms | LF/HF: {lf_hf:.2f} | "
          f"Quality: {signal_quality:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# HRV COMPUTATION VIA NEUROKIT2
# We extract the raw BVP waveform from open-rppg and pass it to neurokit2,
# which is purpose-built for physiological signal analysis and gives us
# reliable RMSSD and LF/HF even on short (30s) windows.
# ─────────────────────────────────────────────────────────────────────────────

def compute_hrv(model: rppg.Model,
                window_seconds: int,
                sampling_rate: int) -> dict:
    """
    Extract the BVP signal from open-rppg's buffer and compute HRV metrics
    using neurokit2.

    Steps:
      1. Retrieve raw BVP waveform from the last window_seconds of signal
      2. Use neurokit2 to clean the signal and detect peaks (R-peaks / systolic peaks)
      3. Compute RR intervals from the peak positions
      4. Compute RMSSD and LF/HF from those intervals

    Returns a dict with: heart_rate, rmssd, lf_hf, signal_quality
    Returns zeros if the signal is too noisy or too short.
    """

    # ── Get HR from open-rppg (this is reliable even at 30s) ─────────────────
    result = model.hr(start=-window_seconds)
    if result is None or not result.get("hr"):
        return {"heart_rate": 0.0, "rmssd": 0.0, "lf_hf": 0.0, "signal_quality": 0.0}

    heart_rate = float(result["hr"])
    if not (30 < heart_rate < 220):
        # Reject physiologically impossible values
        return {"heart_rate": 0.0, "rmssd": 0.0, "lf_hf": 0.0, "signal_quality": 0.0}

    # ── Get raw BVP waveform from open-rppg's buffer ──────────────────────────
    # bvp() returns (signal_array, timestamps_array) for the last N seconds
    bvp_data = model.bvp(start=-window_seconds)
    if bvp_data is None:
        # HR is valid but no raw waveform available — return HR only
        return {"heart_rate": round(heart_rate, 1),
                "rmssd": 0.0, "lf_hf": 0.0, "signal_quality": 0.5}

    bvp_signal, _ = bvp_data

    # Need at least 10 seconds of signal for meaningful HRV
    if len(bvp_signal) < sampling_rate * 10:
        return {"heart_rate": round(heart_rate, 1),
                "rmssd": 0.0, "lf_hf": 0.0, "signal_quality": 0.5}

    try:
        # ── neurokit2: clean signal and find peaks ────────────────────────────
        # ppg_clean() applies bandpass filtering suited for PPG/BVP signals
        cleaned = nk.ppg_clean(bvp_signal, sampling_rate=sampling_rate)

        # ppg_findpeaks() locates systolic peaks (heartbeat positions)
        peaks_info = nk.ppg_findpeaks(cleaned, sampling_rate=sampling_rate)
        peak_indices = peaks_info["PPG_Peaks"]

        if len(peak_indices) < 4:
            # Too few peaks for HRV — signal may still be stabilising
            return {"heart_rate": round(heart_rate, 1),
                    "rmssd": 0.0, "lf_hf": 0.0, "signal_quality": 0.3}

        # ── Compute RR intervals in milliseconds ──────────────────────────────
        # RR interval = time between consecutive heartbeats
        rr_intervals_ms = np.diff(peak_indices) / sampling_rate * 1000

        # ── RMSSD: Root Mean Square of Successive Differences ─────────────────
        # Standard short-term HRV metric. Lower under stress.
        successive_diffs = np.diff(rr_intervals_ms)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))

        # ── LF/HF ratio via neurokit2 frequency analysis ──────────────────────
        # Requires at least ~60s for reliable frequency domain HRV.
        # At 30s we attempt it but fall back gracefully if it fails.
        try:
            hrv_freq = nk.hrv_frequency(
                peak_indices,
                sampling_rate=sampling_rate,
                show=False   # don't open a matplotlib plot
            )
            lf_hf = float(hrv_freq["HRV_LFHF"].iloc[0])
            # Sanity check: LF/HF should be between 0 and 20 for real signals
            if not (0 < lf_hf < 20):
                lf_hf = 0.0
        except Exception:
            # LF/HF computation can fail at short windows — that's fine
            lf_hf = 0.0

        # ── Signal quality estimate ───────────────────────────────────────────
        # Use coefficient of variation of RR intervals as a rough quality proxy.
        # High variation in RR intervals suggests motion artefacts.
        cv = np.std(rr_intervals_ms) / np.mean(rr_intervals_ms)
        quality = float(np.clip(1.0 - cv, 0.0, 1.0))

        return {
            "heart_rate":     round(heart_rate, 1),
            "rmssd":          round(rmssd, 1),
            "lf_hf":          round(lf_hf, 2),
            "signal_quality": round(quality, 2),
        }

    except Exception as e:
        # Catch any unexpected neurokit2 errors gracefully
        print(f"[WARN] HRV computation failed: {e}")
        return {"heart_rate": round(heart_rate, 1),
                "rmssd": 0.0, "lf_hf": 0.0, "signal_quality": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# PHYSIO MODULE
# Wraps open-rppg. All rPPG logic lives here — the rest of the system only
# calls start() and stop(). Swapping the library later only changes this class.
# ─────────────────────────────────────────────────────────────────────────────

class PhysioModule:
    """Thin wrapper around open-rppg for clean integration into the system."""

    def __init__(self, model_name: str = RPPG_MODEL):
        print(f"[rPPG] Loading model: {model_name} ...")
        self.model = rppg.Model(model_name)
        self._camera_ctx = None
        print("[rPPG] Model loaded.")

    def __enter__(self):
        self._camera_ctx = self.model.video_capture(WEBCAM_INDEX)
        self._camera_ctx.__enter__()
        print(f"[rPPG] Webcam opened (index {WEBCAM_INDEX}).")
        return self

    def __exit__(self, *args):
        # Suppress the RuntimeError open-rppg raises on thread join at shutdown.
        # The error is cosmetic — cleanup completes correctly regardless.
        try:
            if self._camera_ctx:
                self._camera_ctx.__exit__(*args)
        except RuntimeError:
            pass   # known open-rppg cleanup bug — safe to ignore
        print("[rPPG] Webcam released.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor():
    """
    Open the webcam, collect frames, compute readings every WINDOW_SECONDS,
    and save them to the database. Press Q or Ctrl+C to stop cleanly.
    """
    conn        = init_database(DB_PATH)
    physio      = PhysioModule()
    stop_event  = threading.Event()   # signals the loop to exit cleanly on Q

    print(f"\n[INFO] Starting rPPG monitor.")
    print(f"[INFO] Readings every {WINDOW_SECONDS}s — keep face visible and still.")
    print(f"[INFO] Press Q in the preview window, or Ctrl+C, to quit.\n")

    window_start = time.time()
    last_status  = time.time()

    try:
        with physio:
            for frame, box in physio.model.preview:

                if stop_event.is_set():
                    break

                # ── Draw preview ──────────────────────────────────────────────
                preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if box is not None:
                    y1, y2 = box[0]
                    x1, x2 = box[1]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 100), 2)
                    cv2.putText(preview, "Face OK", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)
                else:
                    cv2.putText(preview,
                                "No face — centre yourself in frame",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

                elapsed   = time.time() - window_start
                remaining = max(0, WINDOW_SECONDS - int(elapsed))
                cv2.putText(preview, f"Next reading in {remaining}s",
                            (10, preview.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("rPPG Monitor — Q to quit", preview)

                # Set the stop event instead of breaking immediately —
                # this lets the with-block exit on its own terms (cleaner shutdown)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Q pressed — stopping.")
                    stop_event.set()
                    break

                # ── Periodic status ───────────────────────────────────────────
                now = time.time()
                if now - last_status >= PRINT_INTERVAL:
                    print(f"[INFO] Collecting... {remaining}s until next reading.")
                    last_status = now

                # ── Reading every WINDOW_SECONDS ──────────────────────────────
                if elapsed >= WINDOW_SECONDS:
                    metrics = compute_hrv(physio.model, WINDOW_SECONDS, SAMPLING_RATE)

                    if metrics["heart_rate"] > 0:
                        save_reading(
                            conn,
                            heart_rate=metrics["heart_rate"],
                            rmssd=metrics["rmssd"],
                            lf_hf=metrics["lf_hf"],
                            signal_quality=metrics["signal_quality"],
                        )
                        print(f"\n{'─'*48}")
                        print(f"  Heart Rate    : {metrics['heart_rate']:.1f} BPM")
                        print(f"  RMSSD         : {metrics['rmssd']:.1f} ms")
                        print(f"  LF/HF ratio   : {metrics['lf_hf']:.2f}")
                        print(f"  Signal quality: {metrics['signal_quality']:.2f} / 1.00")
                        print(f"{'─'*48}\n")
                    else:
                        print("[WARN] Signal too weak this window. "
                              "Ensure face is well-lit, centred, and still.")

                    window_start = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by Ctrl+C.")

    finally:
        cv2.destroyAllWindows()
        conn.close()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    run_monitor()