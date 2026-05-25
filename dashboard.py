"""
dashboard.py
------------
Root launcher for the Stress Monitor Well-being Dashboard.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Usage
-----
  python dashboard.py                  # dashboard only (monitoring runs separately)
  python dashboard.py --start-backend  # also start run_all.py --no-ui in background

When --start-backend is given, run_all.py is launched as a background subprocess
and terminated automatically when the dashboard window is closed.

Press Q in the run_all.py camera window (if open) or Ctrl+C in the terminal to
stop the monitoring backend independently.
"""

import os
import sys
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    # Import PyQt6 before anything else so errors surface immediately.
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        from PyQt6.QtCore import QTimer
    except ImportError:
        print(
            "[dashboard] PyQt6 is not installed.\n"
            "  pip install PyQt6 pyqtgraph"
        )
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("StressMonitor")
    app.setOrganizationName("StressMonitor")

    backend_proc: subprocess.Popen | None = None

    # ── Optionally launch the monitoring backend ──────────────────────────────
    if "--start-backend" in sys.argv:
        run_all_py = os.path.join(_ROOT, "run_all.py")
        if os.path.exists(run_all_py):
            try:
                backend_proc = subprocess.Popen(
                    [sys.executable, run_all_py, "--no-ui"],
                    cwd=_ROOT,
                )
                print(
                    f"[Dashboard] Monitoring backend started — PID {backend_proc.pid}\n"
                    "  (Ctrl+C in this terminal, or close the dashboard, to stop it.)"
                )
            except Exception as exc:
                QMessageBox.warning(
                    None,
                    "Backend Error",
                    f"Could not start monitoring backend:\n{exc}\n\n"
                    "The dashboard will still open, but no new data will be collected.",
                )
        else:
            QMessageBox.information(
                None,
                "Backend Not Found",
                "run_all.py was not found. The dashboard will open in read-only mode.",
            )

    # ── Show the dashboard window ─────────────────────────────────────────────
    from src.ui.dashboard import DashboardWindow
    win = DashboardWindow()
    win.show()

    exit_code = app.exec()

    # ── Stop the backend if we started it ────────────────────────────────────
    if backend_proc and backend_proc.poll() is None:
        print("[Dashboard] Stopping monitoring backend...")
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            print("[Dashboard] Sending SIGKILL...")
            backend_proc.kill()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
