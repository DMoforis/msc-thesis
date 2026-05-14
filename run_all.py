"""
run_all.py
----------
Phase 2 unified launcher for the Multimodal Stress Detection System.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Architecture
------------
Single-process, shared camera feed.

  SharedCameraFeed   One OpenCV capture thread; all consumers call get_frame().
  PhysioModule       Runs on the main thread. push_frame() queues BGR frames to
                     the rPPG inference thread started by 'with physio.model:'.
  FaceModule         Runs on the main thread. process_frame() executes MediaPipe
                     synchronously; VA inference is guarded inside get_reading().
  Aggregator         Daemon thread. Wakes every 5 min, reads raw tables, scores,
                     writes to aggregated_windows.
  Flutter desktop    Subprocess. Desktop context monitor (active window, idle,
                     activity %) that writes to desktop_readings.

Why PhysioModule and FaceModule run on the main thread:
  cv2.imshow / cv2.waitKey must be called from the main thread on Windows.
  push_frame() and process_frame() are fast enough to run inline — the heavy
  rPPG inference is already off-loaded to an internal thread by open-rppg.

Stop conditions:
  Q key in the monitor window  OR  Ctrl+C in the terminal.
"""

import os
import sys
import time
import signal
import threading
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from src.camera.shared_feed import SharedCameraFeed
from src.physio.rppg_monitor import (
    PhysioModule,
    init_database as _init_physio_db,
    save_reading  as _save_physio,
)
from src.face.face_monitor import (
    FaceModule,
    init_database as _init_face_db,
    save_reading  as _save_face,
)
from src.fusion.aggregator import Aggregator
from src.utils.config import (
    CAMERA_INDEX, CAMERA_FPS,
    HR_WINDOW_SECONDS, HRV_WINDOW_SECONDS, FACE_WINDOW_SECONDS,
    AGGREGATION_MINUTES,
    DB_PATH, RPPG_MODEL,
)

_FLUTTER_EXE = os.path.join(
    _ROOT, "src", "desktop",
    "build", "windows", "x64", "runner", "Release",
    "desktop_monitor.exe",
)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCY PROBES
# Each returns a short status string used in the startup summary.
# ─────────────────────────────────────────────────────────────────────────────

def _probe_ollama() -> str:
    try:
        import ollama  # noqa: F401
        return "READY"
    except ImportError:
        return "fallback  (pip install ollama)"


def _probe_plyer() -> str:
    try:
        import plyer  # noqa: F401
        return "READY"
    except ImportError:
        return "fallback  (pip install plyer)"


def _probe_torch() -> str:
    try:
        import torch  # noqa: F401
        suffix = f" (CUDA: {torch.cuda.get_device_name(0)})" \
                 if torch.cuda.is_available() else " (CPU only)"
        return f"READY{suffix}"
    except ImportError:
        return "fallback  VA disabled (pip install torch torchvision)"


# ─────────────────────────────────────────────────────────────────────────────
# FLUTTER SUBPROCESS LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────

def _launch_flutter() -> tuple[subprocess.Popen | None, str]:
    """Launch the pre-built Flutter desktop app. Return (proc, status_string)."""
    if not os.path.exists(_FLUTTER_EXE):
        hint = "cd src/desktop && flutter build windows --release"
        return None, f"NOT FOUND  (build: {hint})"
    try:
        proc = subprocess.Popen([_FLUTTER_EXE], cwd=_ROOT)
        return proc, f"READY  PID {proc.pid}"
    except Exception as exc:
        return None, f"FAILED  {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

_BANNER_WIDTH = 62

def _print_banner() -> None:
    print("=" * _BANNER_WIDTH)
    print("  Multimodal Stress Detection System  --  Phase 2")
    print("  Dimitris Moforis, University of Piraeus")
    print("  Department of Digital Systems")
    print("=" * _BANNER_WIDTH)
    print()


def _print_startup_summary(rows: list[tuple[str, str]]) -> None:
    """Print a formatted component-status table, then usage hint."""
    col = 22
    print(f"  {'Component':<{col}}  Status")
    print(f"  {'-' * col}  {'-' * 36}")
    for label, status in rows:
        ok = status.startswith("READY")
        marker = "[OK]" if ok else "[--]"
        print(f"  {marker} {label:<{col - 5}}  {status}")
    print()
    print(f"  Database:  {DB_PATH}")
    print()
    print("  Press Q in the monitor window or Ctrl+C to stop.")
    print("=" * _BANNER_WIDTH)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# FRAME OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_rppg_box(frame: np.ndarray, physio: PhysioModule) -> None:
    """Draw the rPPG face detection bounding box (orange) if available."""
    box = physio.model.box
    if box is None:
        return
    try:
        y1, y2 = int(box[0][0]), int(box[0][1])
        x1, x2 = int(box[1][0]), int(box[1][1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, "rPPG", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
    except (TypeError, IndexError, ValueError):
        pass   # box geometry can be None or malformed before first face lock


def _draw_status_overlay(
    frame:       np.ndarray,
    last_hr:     float | None,
    last_stress: float | None,
) -> None:
    """Render HR and stress index in the bottom-left corner."""
    h   = frame.shape[0]
    hr_txt = f"HR: {last_hr:.0f} BPM" if last_hr is not None else "HR: --"
    si_txt = f"Stress idx: {last_stress:.3f}" \
             if last_stress is not None else "Stress idx: -- (5-min window pending)"
    cv2.putText(frame, hr_txt,
                (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)
    cv2.putText(frame, si_txt,
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)


# ─────────────────────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _query_last_stress(conn) -> float | None:
    """Return the most recent stress_index from aggregated_windows, or None."""
    try:
        row = conn.execute(
            "SELECT stress_index FROM aggregated_windows "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# Must be called from the main thread (cv2.imshow / waitKey requirement).
# ─────────────────────────────────────────────────────────────────────────────

def _run_loop(
    feed:        SharedCameraFeed,
    physio:      PhysioModule,
    face:        FaceModule,
    conn_physio,
    conn_face,
    stop_ev:     threading.Event,
) -> None:
    hr_t     = time.monotonic()
    hrv_t    = time.monotonic()
    face_t   = time.monotonic()
    stress_t = time.monotonic()

    last_hr:     float | None = None
    last_stress: float | None = None

    _STRESS_REFRESH = 30.0   # seconds between stress-index DB queries

    print("[Monitor] Running. Press Q in the window or Ctrl+C to stop.\n")

    with physio.model:   # starts rPPG inference thread
        while not stop_ev.is_set():

            frame = feed.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Feed rPPG model ───────────────────────────────────────────────
            physio.push_frame(frame)

            # ── Face processing + mesh/EAR/pose annotations ───────────────────
            annotated = face.process_frame(frame)   # draws on a copy

            # ── rPPG bounding box ─────────────────────────────────────────────
            _draw_rppg_box(annotated, physio)

            # ── HR + stress index status line ─────────────────────────────────
            _draw_status_overlay(annotated, last_hr, last_stress)

            # ── Display ───────────────────────────────────────────────────────
            cv2.imshow("Stress Monitor -- Q to quit", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:   # Q or Escape
                print("[Monitor] Quit key pressed.")
                stop_ev.set()
                break

            now = time.monotonic()

            # ── HR reading every 10 s ─────────────────────────────────────────
            if now - hr_t >= HR_WINDOW_SECONDS:
                result = physio.get_hr(HR_WINDOW_SECONDS)
                if result and result.get("hr"):
                    hr  = float(result["hr"])
                    sqi = float(result.get("SQI") or 0.0)
                    if 30.0 < hr < 220.0:
                        _save_physio(
                            conn_physio,
                            heart_rate    = round(hr, 1),
                            rmssd         = None,
                            lf_hf         = None,
                            sdnn          = None,
                            signal_quality = round(sqi, 2),
                            window_type   = "hr_10s",
                        )
                        last_hr = round(hr, 1)
                        print(f"[HR]  {last_hr:5.1f} BPM   SQI: {sqi:.2f}")
                    else:
                        print(f"[rPPG] HR {hr:.0f} BPM out of valid range — discarded.")
                else:
                    print("[rPPG] Accumulating signal — keep face visible.")
                hr_t = now

            # ── HRV reading every 5 min ───────────────────────────────────────
            if now - hrv_t >= HRV_WINDOW_SECONDS:
                result = physio.get_hrv(HRV_WINDOW_SECONDS)
                if result and result.get("hr"):
                    hr  = float(result["hr"])
                    sqi = float(result.get("SQI") or 0.0)
                    hrv = result.get("hrv") or {}
                    rmssd = hrv.get("rmssd")
                    sdnn  = hrv.get("sdnn")
                    lf_hf = hrv.get("LF/HF")
                    if 30.0 < hr < 220.0:
                        _save_physio(
                            conn_physio,
                            heart_rate    = round(hr, 1),
                            rmssd         = round(rmssd, 1) if rmssd else None,
                            lf_hf         = round(lf_hf, 2) if lf_hf else None,
                            sdnn          = round(sdnn,  1) if sdnn  else None,
                            signal_quality = round(sqi, 2),
                            window_type   = "hrv_5min",
                        )
                        r_s = f"{rmssd:.1f}ms" if rmssd else "--"
                        s_s = f"{sdnn:.1f}ms"  if sdnn  else "--"
                        l_s = f"{lf_hf:.2f}"   if lf_hf else "--"
                        print(f"\n[HRV]  HR={round(hr,1):.1f} BPM  "
                              f"RMSSD={r_s}  SDNN={s_s}  LF/HF={l_s}  "
                              f"SQI={sqi:.2f}\n")
                    else:
                        print(f"[rPPG] HRV window: HR {hr:.0f} BPM out of range.")
                else:
                    print("[rPPG] HRV window: no signal available yet.")
                hrv_t = now

            # ── Face reading every 30 s ───────────────────────────────────────
            if now - face_t >= FACE_WINDOW_SECONDS:
                elapsed = now - face_t
                reading = face.get_reading(elapsed_seconds=elapsed)
                if reading:
                    _save_face(
                        conn_face,
                        blink_rate = reading["blink_rate"],
                        mean_ear   = reading["mean_ear"],
                        pitch      = reading["pitch"],
                        yaw        = reading["yaw"],
                        roll       = reading["roll"],
                        face_pct   = reading["face_pct"],
                        valence    = reading["valence"],
                        arousal    = reading["arousal"],
                    )
                    v = f"{reading['valence']:+.3f}" if reading["valence"] is not None else "--"
                    a = f"{reading['arousal']:+.3f}" if reading["arousal"] is not None else "--"
                    print(f"[Face] EAR={reading['mean_ear']:.3f}  "
                          f"Blinks={reading['blink_rate']:.1f}/min  "
                          f"V={v}  A={a}")
                else:
                    print("[Face] Too few frames with face detected in this window.")
                face.reset_window()
                face_t = now

            # ── Refresh stress index from DB (every 30 s) ─────────────────────
            if now - stress_t >= _STRESS_REFRESH:
                last_stress = _query_last_stress(conn_physio)
                stress_t = now


# ─────────────────────────────────────────────────────────────────────────────
# SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup(
    feed:         SharedCameraFeed,
    aggregator:   Aggregator,
    flutter_proc: subprocess.Popen | None,
    conn_physio,
    conn_face,
) -> None:
    print("\n[run_all] Shutting down...")

    cv2.destroyAllWindows()

    aggregator.stop()

    feed.stop()

    if flutter_proc and flutter_proc.poll() is None:
        print("[run_all] Stopping Flutter process...")
        flutter_proc.terminate()
        try:
            flutter_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            flutter_proc.kill()

    for conn in (conn_physio, conn_face):
        try:
            conn.close()
        except Exception:
            pass

    print("[run_all] Clean shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _print_banner()
    print("  Initialising modules...\n")

    # ── Optional dependency probes ────────────────────────────────────────────
    ollama_status = _probe_ollama()
    plyer_status  = _probe_plyer()
    torch_status  = _probe_torch()

    # ── Camera ────────────────────────────────────────────────────────────────
    feed = SharedCameraFeed(camera_index=CAMERA_INDEX, fps=CAMERA_FPS)
    try:
        feed.start()
        cam_status = f"READY  {feed.width}x{feed.height} @ {CAMERA_FPS} fps"
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)

    # ── Database connections ──────────────────────────────────────────────────
    conn_physio = _init_physio_db()
    conn_face   = _init_face_db()

    # ── Physio module ─────────────────────────────────────────────────────────
    try:
        physio = PhysioModule()
        physio_status = (f"READY  {RPPG_MODEL} | "
                         f"HR every {HR_WINDOW_SECONDS}s, "
                         f"HRV every {HRV_WINDOW_SECONDS}s")
    except Exception as exc:
        print(f"[FATAL] Physio module failed to load: {exc}")
        feed.stop()
        sys.exit(1)

    # ── Face module ───────────────────────────────────────────────────────────
    try:
        face = FaceModule()
        face_status = f"READY  MediaPipe FaceMesh, reading every {FACE_WINDOW_SECONDS}s"
    except Exception as exc:
        print(f"[FATAL] Face module failed to load: {exc}")
        feed.stop()
        sys.exit(1)

    # ── Aggregator ────────────────────────────────────────────────────────────
    aggregator = Aggregator()
    aggregator.start()
    agg_status = f"READY  {AGGREGATION_MINUTES}-min windows, background thread"

    # ── Flutter subprocess ────────────────────────────────────────────────────
    flutter_proc, flutter_status = _launch_flutter()

    # ── Startup summary ───────────────────────────────────────────────────────
    _print_startup_summary([
        ("Camera",            cam_status),
        ("Physio (rPPG/HRV)", physio_status),
        ("Face + MediaPipe",  face_status),
        ("VA model (EmoNet)", torch_status),
        ("Aggregator",        agg_status),
        ("LLM classifier",    ollama_status),
        ("Notifier",          plyer_status),
        ("Flutter desktop",   flutter_status),
    ])

    # ── Signal handler + stop event ───────────────────────────────────────────
    stop_ev = threading.Event()

    def _sigint(sig, frame):
        print("\n[run_all] Ctrl+C received.")
        stop_ev.set()

    signal.signal(signal.SIGINT, _sigint)

    # ── Main processing loop ──────────────────────────────────────────────────
    # Runs on the main thread — required for cv2.imshow on Windows.
    try:
        _run_loop(feed, physio, face, conn_physio, conn_face, stop_ev)
    finally:
        _cleanup(feed, aggregator, flutter_proc, conn_physio, conn_face)


if __name__ == "__main__":
    main()
