"""
run_all.py
----------
Unified launcher for the Multimodal Stress Detection System.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

What this script does:
  Launches all three PoC modules as separate processes so they run
  simultaneously — exactly as they would in the final integrated system.

  PoC #1 — src/physio/rppg_monitor.py   : Heart rate + HRV via rPPG
  PoC #2 — src/face/face_monitor.py     : Facial landmarks, EAR, head pose
  PoC #3 — src/desktop/                 : Desktop context via Flutter app

Usage:
  python run_all.py              # launch all three modules
  python run_all.py --poc 1      # launch only PoC #1
  python run_all.py --poc 2      # launch only PoC #2
  python run_all.py --poc 3      # launch only PoC #3

Notes:
  - Each module opens its own preview window
  - All modules write to the same data/stress_monitor.db
  - Press Ctrl+C in this terminal to stop all modules cleanly
  - The Flutter desktop app (PoC #3) must be built first:
      cd src/desktop && flutter build windows
"""

import subprocess
import sys
import os
import argparse
import signal
import time

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Build absolute paths so the script works regardless of where it is called from
# ─────────────────────────────────────────────────────────────────────────────

# Root of the project (directory containing this script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the Python interpreter inside our venv
PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")

# Paths to each PoC script
POC1_SCRIPT = os.path.join(PROJECT_ROOT, "src", "physio", "rppg_monitor.py")
POC2_SCRIPT = os.path.join(PROJECT_ROOT, "src", "face",   "face_monitor.py")

# Path to the pre-built Flutter desktop executable
POC3_EXE    = os.path.join(
    PROJECT_ROOT, "src", "desktop",
    "build", "windows", "x64", "runner", "Release",
    "desktop_monitor.exe"
)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS MANAGEMENT
# We launch each module as a subprocess so they run in parallel.
# All processes are tracked so we can terminate them cleanly on Ctrl+C.
# ─────────────────────────────────────────────────────────────────────────────

# List of all running child processes — populated as modules are launched
running_processes = []


def launch_python_module(name: str, script_path: str) -> subprocess.Popen | None:
    """
    Launch a Python script as a subprocess using the venv interpreter.
    Returns the Popen object so we can terminate it later, or None on error.
    """
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return None

    if not os.path.exists(PYTHON):
        print(f"[ERROR] Python interpreter not found at: {PYTHON}")
        print("[ERROR] Make sure the venv is set up correctly.")
        return None

    print(f"[LAUNCH] Starting {name}...")
    # creationflags=CREATE_NEW_CONSOLE opens each module in its own terminal
    # window, making it easier to see each module's output separately
    proc = subprocess.Popen(
        [PYTHON, script_path],
        cwd=PROJECT_ROOT,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    print(f"[OK] {name} running (PID {proc.pid})")
    return proc


def launch_flutter_module(name: str, exe_path: str) -> subprocess.Popen | None:
    """
    Launch the pre-built Flutter desktop app as a subprocess.
    Note: Flutter apps must be built before running via this launcher.
    Run 'cd src/desktop && flutter build windows --release' first.
    """
    if not os.path.exists(exe_path):
        print(f"[WARN] Flutter executable not found at:\n  {exe_path}")
        print("[WARN] To build it, run:")
        print("         cd src/desktop")
        print("         flutter build windows --release")
        print("[WARN] Skipping PoC #3 for now.")
        return None

    print(f"[LAUNCH] Starting {name}...")
    proc = subprocess.Popen([exe_path], cwd=PROJECT_ROOT)
    print(f"[OK] {name} running (PID {proc.pid})")
    return proc


def stop_all():
    """Terminate all running child processes cleanly."""
    print("\n[INFO] Stopping all modules...")
    for proc in running_processes:
        if proc and proc.poll() is None:   # poll() returns None if still running
            proc.terminate()
            try:
                proc.wait(timeout=5)       # wait up to 5s for clean exit
            except subprocess.TimeoutExpired:
                proc.kill()                # force kill if it doesn't stop
    print("[INFO] All modules stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Parse optional --poc argument to launch a specific module only
    parser = argparse.ArgumentParser(
        description="Launch Multimodal Stress Detection PoC modules"
    )
    parser.add_argument(
        "--poc",
        type=int,
        choices=[1, 2, 3],
        help="Launch only a specific PoC (1, 2, or 3). Omit to launch all."
    )
    args = parser.parse_args()

    print("=" * 56)
    print("  Multimodal Stress Detection System")
    print("  MSc Thesis — Dimitris Moforis, Univ. of Piraeus")
    print("=" * 56)

    launch_all = args.poc is None

    # ── Launch selected modules ───────────────────────────────────────────────
    if launch_all or args.poc == 1:
        proc = launch_python_module("PoC #1 — rPPG Heart Rate Monitor", POC1_SCRIPT)
        if proc:
            running_processes.append(proc)
        time.sleep(1)   # small delay so windows don't all open simultaneously

    if launch_all or args.poc == 2:
        proc = launch_python_module("PoC #2 — Facial Landmark Monitor", POC2_SCRIPT)
        if proc:
            running_processes.append(proc)
        time.sleep(1)

    if launch_all or args.poc == 3:
        proc = launch_flutter_module("PoC #3 — Desktop Context Monitor", POC3_EXE)
        if proc:
            running_processes.append(proc)

    if not running_processes:
        print("[ERROR] No modules were launched successfully. Check paths above.")
        sys.exit(1)

    print(f"\n[INFO] {len(running_processes)} module(s) running.")
    print("[INFO] Press Ctrl+C to stop all modules.\n")

    # ── Register Ctrl+C handler ───────────────────────────────────────────────
    # When the user presses Ctrl+C, stop_all() is called before exiting
    signal.signal(signal.SIGINT, lambda sig, frame: (stop_all(), sys.exit(0)))

    # ── Keep this script alive while modules run ──────────────────────────────
    # We poll every second to detect if any module crashed unexpectedly
    # Track which PIDs we have already warned about so we only print once
    warned_pids = set()

    try:
        while True:
            time.sleep(1)
            # Check if any process has exited — warn once per process
            for proc in running_processes:
                if proc.poll() is not None and proc.pid not in warned_pids:
                    warned_pids.add(proc.pid)
                    print(f"[INFO] Module (PID {proc.pid}) has stopped.")

            # If ALL modules have stopped, exit cleanly instead of looping forever
            if all(proc.poll() is not None for proc in running_processes):
                print("[INFO] All modules have stopped. Exiting.")
                break
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()