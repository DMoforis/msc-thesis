"""
baseline.py
-----------
Personal physiological baseline calibrator.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

A 2-minute resting session records the user's normal HR, RMSSD, EAR,
blink rate, valence, and arousal. LateFusion uses these personal values
instead of population means, making stress scoring individually calibrated.

Usage
-----
  python src/utils/baseline.py --calibrate   # run a 2-minute session
  python src/utils/baseline.py --show        # print current stored values

  # In other modules:
  from src.utils.baseline import get_baseline
  bl = get_baseline()   # dict: hr, rmssd, ear, blink_rate, valence, arousal
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import DB_PATH, BASELINE_DURATION_SECONDS
from src.utils.db import open_db, ensure_baselines_table


# ── Population defaults — used when no personal baseline exists ───────────────
_POPULATION_DEFAULTS: dict = {
    "hr":         70.0,
    "rmssd":      42.0,
    "ear":        0.30,
    "blink_rate": 15.0,
    "valence":    0.0,
    "arousal":    0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_baseline(db_path: str = DB_PATH) -> dict:
    """
    Return the most recent personal baseline from the database.
    Falls back to population means for any field that is NULL or missing.

    Returns
    -------
    dict with keys: hr, rmssd, ear, blink_rate, valence, arousal
    """
    try:
        conn = open_db(db_path)
        ensure_baselines_table(conn)
        row = conn.execute(
            "SELECT hr, rmssd, ear, blink_rate, valence, arousal "
            "FROM baselines ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        if row is None:
            return dict(_POPULATION_DEFAULTS)

        result: dict = {}
        # Positive-only fields: fall back to default if NULL or non-positive
        for key in ("hr", "rmssd", "ear", "blink_rate"):
            val = row[key]
            result[key] = float(val) if (val is not None and val > 0) else _POPULATION_DEFAULTS[key]
        # Affective fields can legitimately be 0 or negative
        for key in ("valence", "arousal"):
            val = row[key]
            result[key] = float(val) if val is not None else _POPULATION_DEFAULTS[key]

        return result

    except Exception as exc:
        print(f"[Baseline] DB read failed ({exc}), using population defaults.")
        return dict(_POPULATION_DEFAULTS)


# ─────────────────────────────────────────────────────────────────────────────
# DB WRITE HELPER (private)
# ─────────────────────────────────────────────────────────────────────────────

def _save_baseline(
    conn: sqlite3.Connection,
    hr:         float | None,
    rmssd:      float | None,
    ear:        float | None,
    blink_rate: float | None,
    valence:    float | None,
    arousal:    float | None,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO baselines (timestamp, hr, rmssd, ear, blink_rate, valence, arousal) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, hr, rmssd, ear, blink_rate, valence, arousal),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(db_path: str = DB_PATH) -> dict | None:
    """
    Collect BASELINE_DURATION_SECONDS of resting physiological data from the
    webcam, compute personal averages, and store them in the baselines table.

    Returns the saved baseline dict, or None if aborted or data insufficient.
    Heavy imports (cv2, rppg, mediapipe) are deferred to this function so that
    get_baseline() can be called without those packages installed.
    """
    import time
    import numpy as np
    import cv2

    from src.camera.shared_feed import SharedCameraFeed
    from src.physio.rppg_monitor import PhysioModule
    from src.face.face_monitor import FaceModule
    from src.utils.config import CAMERA_INDEX, CAMERA_FPS, HR_WINDOW_SECONDS

    print("\n[Calibration] ── Personal Baseline Calibration ───────────────────")
    print("[Calibration] Sit comfortably, face the camera, and relax.")
    print(f"[Calibration] Duration: {BASELINE_DURATION_SECONDS} seconds (2 minutes)")
    print("[Calibration] Press Q to abort.\n")

    try:
        feed = SharedCameraFeed(camera_index=CAMERA_INDEX, fps=CAMERA_FPS)
        feed.start()
    except RuntimeError as exc:
        print(f"[Calibration] Camera error: {exc}")
        return None

    physio = PhysioModule()
    face   = FaceModule()

    hr_samples: list[float] = []
    rmssd_val: float | None = None
    elapsed  = 0.0
    aborted  = False

    start_t  = time.monotonic()
    hr_timer = start_t

    try:
        with physio.model:

            while True:
                now      = time.monotonic()
                elapsed  = now - start_t
                remaining = BASELINE_DURATION_SECONDS - elapsed

                if remaining <= 0:
                    break

                frame = feed.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                physio.push_frame(frame)
                annotated = face.process_frame(frame.copy())

                # Terminal countdown — overwrites the same line
                print(
                    f"\r[Calibration] Collecting resting data… {int(remaining):3d}s remaining  ",
                    end="", flush=True,
                )

                # Progress bar + label on the camera window
                pct   = min(1.0, elapsed / BASELINE_DURATION_SECONDS)
                bar_w, bar_h = 400, 14
                bar_x = 10
                bar_y = annotated.shape[0] - 16
                cv2.rectangle(
                    annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                    (70, 70, 70), -1,
                )
                cv2.rectangle(
                    annotated, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h),
                    (0, 200, 100), -1,
                )
                cv2.putText(
                    annotated,
                    f"CALIBRATION  {int(remaining)}s remaining — stay relaxed",
                    (10, annotated.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
                )

                cv2.imshow("Baseline Calibration — Q to abort", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break

                # Collect HR every HR_WINDOW_SECONDS
                if now - hr_timer >= HR_WINDOW_SECONDS:
                    result = physio.get_hr(HR_WINDOW_SECONDS)
                    if result and result.get("hr"):
                        hr_val = float(result["hr"])
                        if 30.0 < hr_val < 220.0:
                            hr_samples.append(hr_val)
                    hr_timer = now

            # Still inside `with physio.model:` — get HRV from the full window
            if not aborted and elapsed >= 10:
                hrv_result = physio.get_hrv(max(10, int(elapsed)))
                if hrv_result and isinstance(hrv_result.get("hrv"), dict):
                    rmssd_val = hrv_result["hrv"].get("rmssd")

    finally:
        cv2.destroyAllWindows()
        feed.stop()

    print()  # newline after the carriage-return countdown

    if aborted:
        print("[Calibration] Aborted — no data saved.")
        return None

    # Face reading covers the entire calibration window
    face_reading = face.get_reading(elapsed_seconds=max(1.0, elapsed))

    # Round and coerce
    bl_hr         = round(float(np.mean(hr_samples)), 1) if hr_samples else None
    bl_rmssd      = round(float(rmssd_val), 1)            if rmssd_val  else None
    bl_ear        = round(face_reading["mean_ear"],   3)  if face_reading else None
    bl_blink_rate = round(face_reading["blink_rate"], 1)  if face_reading else None
    bl_valence    = (
        round(face_reading["valence"], 3)
        if (face_reading and face_reading.get("valence") is not None)
        else None
    )
    bl_arousal    = (
        round(face_reading["arousal"], 3)
        if (face_reading and face_reading.get("arousal") is not None)
        else None
    )

    # Persist
    try:
        conn = open_db(db_path)
        ensure_baselines_table(conn)
        _save_baseline(conn, bl_hr, bl_rmssd, bl_ear, bl_blink_rate, bl_valence, bl_arousal)
        conn.close()
    except Exception as exc:
        print(f"[Calibration] DB save failed: {exc}")
        return None

    print("[Calibration] ── Results ────────────────────────────────────────────")
    print(f"  Heart Rate   : {bl_hr:.1f} BPM"         if bl_hr         is not None else "  Heart Rate   : — (no rPPG signal)")
    print(f"  RMSSD        : {bl_rmssd:.1f} ms"        if bl_rmssd      is not None else "  RMSSD        : — (insufficient signal for HRV)")
    print(f"  Mean EAR     : {bl_ear:.3f}"              if bl_ear        is not None else "  Mean EAR     : — (no face detected)")
    print(f"  Blink rate   : {bl_blink_rate:.1f} /min"  if bl_blink_rate is not None else "  Blink rate   : —")
    print(f"  Valence      : {bl_valence:+.3f}"         if bl_valence    is not None else "  Valence      : — (VA model unavailable)")
    print(f"  Arousal      : {bl_arousal:+.3f}"         if bl_arousal    is not None else "  Arousal      : —")
    print("[Calibration] Saved. LateFusion will use these values going forward.\n")

    return get_baseline(db_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personal physiological baseline calibration."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--calibrate",
        action="store_true",
        help=f"Run a {BASELINE_DURATION_SECONDS}-second resting calibration session.",
    )
    group.add_argument(
        "--show",
        action="store_true",
        help="Print the current stored baseline and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.show:
        bl = get_baseline()
        src = "DB" if bl != _POPULATION_DEFAULTS else "population defaults (no calibration stored)"
        print(f"\n[Baseline] Current values ({src}):")
        for k, v in bl.items():
            print(f"  {k:<12}: {v}")
        sys.exit(0)

    if args.calibrate:
        result = run_calibration()
        sys.exit(0 if result is not None else 1)
