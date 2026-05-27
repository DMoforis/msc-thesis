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
    _FRAME_PATH,
    _FRAME_TMP,
)
from src.fusion.aggregator import Aggregator
from src.physio.hrv_processor import validate_hrv_metrics
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


def _probe_windows_toasts() -> str:
    try:
        from windows_toasts import WindowsToaster  # noqa: F401
        return "READY"
    except ImportError:
        return "fallback  (pip install windows-toasts)"


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

def _hide_flutter_window(pid: int, timeout_s: float = 10.0) -> None:
    """
    Daemon thread: poll EnumWindows every 50 ms until the Flutter GUI window
    becomes visible, then immediately call ShowWindow(SW_HIDE) to suppress it.

    Why not CREATE_NO_WINDOW or STARTUPINFO?
    -----------------------------------------
    CREATE_NO_WINDOW only prevents *console* windows from being allocated.
    Flutter desktop is a Win32 GUI application (WinMain subsystem) — it uses
    CreateWindow (without WS_VISIBLE) and later calls ShowWindow(SW_SHOW)
    explicitly from its SetNextFrameCallback, ignoring any process-creation
    flags or STARTUPINFO.wShowWindow hints entirely.

    The EnumWindows approach catches the ShowWindow(SW_SHOW) call within ≤50 ms
    and immediately reverses it, keeping the window permanently hidden while
    Flutter continues writing desktop context data to the database.
    """
    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32

    # Callback type for EnumWindows: (HWND, LPARAM) → BOOL
    _WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM,
    )

    found: list[int] = []

    def _enum_cb(hwnd: int, _lparam: int) -> bool:
        """Return False to stop enumeration once the target window is found."""
        pid_buf = ctypes.wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buf))
        if pid_buf.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False   # stop — found what we need
        return True        # continue

    # Keep a persistent Python reference so the GC doesn't collect the callback
    # while EnumWindows is executing.
    cb = _WNDENUMPROC(_enum_cb)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.05)   # poll every 50 ms
        found.clear()
        user32.EnumWindows(cb, 0)
        if found:
            hwnd = found[0]
            user32.ShowWindow(hwnd, 0)   # 0 = SW_HIDE
            print(f"[Flutter] Window suppressed (PID {pid}, hwnd {hwnd})")
            return

    print(
        f"[Flutter] Warning: no visible window found for PID {pid} "
        f"within {timeout_s:.0f} s — Flutter may not have started correctly."
    )


def _launch_flutter(no_ui: bool = False) -> tuple[subprocess.Popen | None, str]:
    """
    Launch the pre-built Flutter desktop app. Return (proc, status_string).

    When *no_ui* is True a daemon thread (_hide_flutter_window) suppresses
    the Flutter GUI window as soon as it appears.  The process keeps running
    and continues writing desktop context data to desktop_readings.

    Note: CREATE_NO_WINDOW is intentionally NOT used here — see
    _hide_flutter_window docstring for why it cannot suppress a GUI window.
    """
    print(f"[Flutter] launching headless={no_ui}")

    if not os.path.exists(_FLUTTER_EXE):
        msg = (
            f"NOT FOUND  ({_FLUTTER_EXE})\n"
            "  Build with: cd src/desktop && flutter build windows --release"
        )
        return None, msg

    try:
        proc = subprocess.Popen([_FLUTTER_EXE], cwd=_ROOT)
    except Exception as exc:
        return None, f"FAILED  {exc}"

    if no_ui:
        t = threading.Thread(
            target=_hide_flutter_window,
            args=(proc.pid,),
            daemon=True,
            name="flutter-hide",
        )
        t.start()
        return proc, f"READY headless  PID {proc.pid}"

    return proc, f"READY  PID {proc.pid}"


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
    no_ui:       bool = False,
) -> None:
    hr_t     = time.monotonic()
    hrv_t    = time.monotonic()
    face_t   = time.monotonic()
    stress_t = time.monotonic()

    last_hr:     float | None = None
    last_stress: float | None = None

    # Collect 10-second HR readings for HRV cross-validation.
    # Cleared after each HRV window so validate_hrv_metrics() compares
    # the HRV-window HR against readings from the same 5-minute span.
    hr_samples: list[float] = []

    _STRESS_REFRESH = 30.0   # seconds between stress-index DB queries

    if no_ui:
        print("[Monitor] Running headless (--no-ui). Ctrl+C to stop.\n")
    else:
        print("[Monitor] Running. Press Q in the window or Ctrl+C to stop.\n")

    with physio.model:   # starts rPPG inference thread
        while not stop_ev.is_set():

            frame = feed.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Feed rPPG model ───────────────────────────────────────────────
            physio.push_frame(frame)

            # ── Face processing (always needed for landmark state) ────────────
            annotated = face.process_frame(frame)   # draws on a copy
            # face.process_frame() saves every 5th frame to _FRAME_PATH with
            # face-mesh + EAR + blink + pose overlays.  We overwrite it below
            # with the more complete version (rPPG box + HR/stress) so the
            # dashboard always sees all overlays regardless of UI mode.

            # ── Always draw rPPG box + status (both UI and headless modes) ───
            _draw_rppg_box(annotated, physio)
            _draw_status_overlay(annotated, last_hr, last_stress)

            # ── Overwrite shared frame with the complete annotated version ────
            # Atomic write: temp file → os.replace → final path.
            if face._frame_counter % 5 == 0:
                try:
                    cv2.imwrite(_FRAME_TMP, annotated,
                                [cv2.IMWRITE_JPEG_QUALITY, 70])
                    os.replace(_FRAME_TMP, _FRAME_PATH)
                except Exception:
                    pass

            if no_ui:
                # Headless mode: no window, no key polling, yield CPU briefly
                time.sleep(0.001)
            else:
                # ── Display ────────────────────────────────────────────────────
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
                        hr_samples.append(last_hr)
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
                    rmssd_raw = hrv.get("rmssd")
                    sdnn_raw  = hrv.get("sdnn")
                    lf_hf     = hrv.get("LF/HF")
                    if 30.0 < hr < 220.0:
                        # Validate RMSSD/SDNN against plausibility bounds and
                        # cross-check HRV-window HR against the 10 s readings.
                        # validate_hrv_metrics() returns None for out-of-range
                        # values, keeping the DB free of physiologically
                        # implausible rPPG artefacts.
                        rmssd, sdnn = validate_hrv_metrics(
                            rmssd_raw, sdnn_raw, hr, hr_samples
                        )
                        _save_physio(
                            conn_physio,
                            heart_rate     = round(hr, 1),
                            rmssd          = rmssd,   # already rounded (or None)
                            lf_hf          = round(lf_hf, 2) if lf_hf else None,
                            sdnn           = sdnn,    # already rounded (or None)
                            signal_quality = round(sqi, 2),
                            window_type    = "hrv_5min",
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
                # Reset HR samples for the next HRV window regardless of outcome.
                hr_samples.clear()
                hrv_t = now

            # ── Face reading every 30 s ───────────────────────────────────────
            if now - face_t >= FACE_WINDOW_SECONDS:
                elapsed = now - face_t
                reading = face.get_reading(elapsed_seconds=elapsed)
                if reading:
                    _save_face(
                        conn_face,
                        blink_rate    = reading["blink_rate"],
                        mean_ear      = reading["mean_ear"],
                        pitch         = reading["pitch"],
                        yaw           = reading["yaw"],
                        roll          = reading["roll"],
                        face_pct      = reading["face_pct"],
                        valence       = reading["valence"],
                        arousal       = reading["arousal"],
                        emotion_label = reading.get("emotion_label"),
                    )
                    v = f"{reading['valence']:+.3f}" if reading["valence"] is not None else "--"
                    a = f"{reading['arousal']:+.3f}" if reading["arousal"] is not None else "--"
                    emo = reading.get("emotion_label") or "--"
                    print(f"[Face] EAR={reading['mean_ear']:.3f}  "
                          f"Blinks={reading['blink_rate']:.1f}/min  "
                          f"V={v}  A={a}  Emo={emo}")
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

    # Remove the shared frame files so the dashboard shows the placeholder
    # rather than a stale frozen frame after the backend stops.
    for _f in (_FRAME_PATH, _FRAME_TMP):
        try:
            if os.path.exists(_f):
                os.remove(_f)
        except Exception:
            pass
    print(f"[run_all] Removed shared frame file(s)")

    print("[run_all] Clean shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Stress Detection Monitor")
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run headless (no cv2 display window) — used when the PyQt6 dashboard "
             "is the primary interface.",
    )
    args, _ = parser.parse_known_args()   # ignore unknown args (e.g. pytest flags)

    _print_banner()
    print("  Initialising modules...\n")

    # ── Optional dependency probes ────────────────────────────────────────────
    ollama_status = _probe_ollama()
    notifier_status = _probe_windows_toasts()
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
    flutter_proc, flutter_status = _launch_flutter(no_ui=args.no_ui)

    # ── Startup summary ───────────────────────────────────────────────────────
    _print_startup_summary([
        ("Camera",            cam_status),
        ("Physio (rPPG/HRV)", physio_status),
        ("Face + MediaPipe",  face_status),
        ("VA model (EmoNet)", torch_status),
        ("Aggregator",        agg_status),
        ("LLM classifier",    ollama_status),
        ("Notifier",          notifier_status),
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
        _run_loop(feed, physio, face, conn_physio, conn_face, stop_ev,
                  no_ui=args.no_ui)
    finally:
        _cleanup(feed, aggregator, flutter_proc, conn_physio, conn_face)


if __name__ == "__main__":
    main()
