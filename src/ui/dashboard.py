"""
src/ui/dashboard.py
-------------------
PyQt6 unified well-being dashboard for the Multimodal Stress Detection System.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Layout
------
  Left  (280 px) : Stress gauge · HR · HRV · Valence-Arousal scatter
                   Desktop context card · system status dot
  Centre (flex)  : 60-min stress trend (pyqtgraph) · session stats bar
  Right (260 px) : Intervention log · Export · Settings

Refresh     : QTimer every 30 s (configurable via Settings)
Persistence : QSettings("StressMonitor", "Dashboard")
Calibration : subprocess cmd.exe window → src/utils/baseline.py --calibrate
"""

import os
import sys
import subprocess
import time as _time
from datetime import datetime, timedelta

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Shared annotated-frame file written by face_monitor.py / run_all.py.
# Polled by CameraFeedWidget in non-standalone (--start-backend) mode.
_LATEST_FRAME_PATH: str = os.path.normpath(
    os.path.join(_ROOT, "data", "latest_frame.jpg")
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QListWidget, QListWidgetItem,
    QDialog, QDialogButtonBox, QSlider, QGroupBox, QFormLayout,
    QCheckBox, QMessageBox, QFrame, QSizePolicy, QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer, QSettings, QSize, QRectF, QPointF, pyqtSignal, QThread
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QImage, QPixmap,
)

try:
    import pyqtgraph as pg
    import numpy as np
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None       # type: ignore[assignment]
    _HAS_CV2 = False

from src.utils.db import open_db
from src.utils.config import DB_PATH, STRESS_TRIGGER_THRESHOLD, CAMERA_INDEX


# ─────────────────────────────────────────────────────────────────────────────
# THEME SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

_THEMES: dict[str, dict[str, str]] = {
    "dark": dict(
        bg="#1e1e2e", panel="#2a2a3e", card="#313150", border="#44446a",
        text="#cdd6f4", sub="#a6adc8", accent="#89b4fa",
        green="#a6e3a1", yellow="#f9e2af", orange="#fab387", red="#f38ba8",
    ),
    "light": dict(
        bg="#eff1f5", panel="#e6e9ef", card="#dce0e8", border="#bcc0cc",
        text="#4c4f69", sub="#6c6f85", accent="#1e66f5",
        green="#40a02b", yellow="#df8e1d", orange="#fe640b", red="#d20f39",
    ),
}

_T: dict[str, str] = dict(_THEMES["dark"])   # active palette — mutated by _apply_theme()


def _apply_theme(name: str) -> None:
    """Switch active palette. Affects all c() calls after this point."""
    _T.clear()
    _T.update(_THEMES.get(name, _THEMES["dark"]))


def c(key: str) -> str:
    """Return the hex colour string for *key* from the active theme."""
    return _T[key]


def stress_color(s: float | None) -> str:
    """Map a stress_index to the appropriate theme colour."""
    if s is None:
        return c("sub")
    if s < 0.30:
        return c("green")
    if s < 0.55:
        return c("yellow")
    if s < 0.70:
        return c("orange")
    return c("red")


_TRIGGER_LABELS: dict[str, str] = {
    "high_stress":     "High Stress",
    "disengagement":   "Disengaged",
    "negative_affect": "Negative Affect",
    "eye_strain":      "Eye Strain",
    "prolonged_idle":  "Prolonged Idle",
    "positive_flow":   "Flow State",
}

_TRIGGER_COLORS: dict[str, str] = {
    "high_stress":     "red",
    "disengagement":   "yellow",
    "negative_affect": "orange",
    "eye_strain":      "yellow",
    "prolonged_idle":  "sub",
    "positive_flow":   "green",
}

# App-category → theme colour key for the DesktopCard badge
_CATEGORY_COLORS: dict[str, str] = {
    "Software Development": "accent",
    "Academic Work":        "green",
    "Document Editing":     "accent",
    "Communication":        "sub",
    "Social Media":         "orange",
    "Entertainment":        "yellow",
    "Web Browsing":         "sub",
    "System/Utilities":     "sub",
    "Other":                "sub",
}


def _stylesheet() -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {c('bg')};
    color: {c('text')};
    font-family: 'Segoe UI', sans-serif;
}}
QLabel {{ color: {c('text')}; background: transparent; }}
QPushButton {{
    background-color: {c('card')};
    color: {c('text')};
    border: 1px solid {c('border')};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: {c('accent')}; color: {c('bg')}; border-color: {c('accent')}; }}
QPushButton:pressed {{ background-color: {c('border')}; }}
QListWidget {{
    background-color: {c('panel')};
    border: 1px solid {c('border')};
    border-radius: 6px;
    font-size: 11px;
    outline: none;
}}
QListWidget::item {{
    padding: 3px 4px;
    border-bottom: 1px solid {c('border')};
}}
QListWidget::item:selected {{
    background-color: {c('accent')};
    color: {c('bg')};
}}
QScrollBar:vertical {{
    background: {c('panel')};
    width: 6px;
    border-radius: 3px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c('border')};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QDialog {{ background-color: {c('bg')}; }}
QGroupBox {{
    color: {c('sub')};
    border: 1px solid {c('border')};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {c('sub')};
}}
QSlider::groove:horizontal {{
    background: {c('border')};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {c('accent')};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QCheckBox {{ color: {c('text')}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {c('border')};
    border-radius: 3px;
    background: {c('card')};
}}
QCheckBox::indicator:checked {{ background: {c('accent')}; border-color: {c('accent')}; }}
QDialogButtonBox QPushButton {{ min-width: 70px; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _db_connect(db_path: str = DB_PATH):
    """Open a DB connection; return None on failure."""
    try:
        return open_db(db_path)
    except Exception:
        return None


def _fetch_latest(conn) -> dict | None:
    """Return the most recent aggregated window as a plain dict."""
    try:
        row = conn.execute("""
            SELECT window_end, avg_hr, avg_rmssd, avg_lf_hf,
                   avg_blink_rate, avg_valence, avg_arousal,
                   stress_index, dominant_category, avg_activity_pct
            FROM aggregated_windows
            ORDER BY id DESC LIMIT 1
        """).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def _fetch_trend(conn, minutes: int = 60) -> tuple[list[float], list[float]]:
    """Return (unix-timestamps, stress_index values) for the last *minutes*."""
    try:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
        rows = conn.execute("""
            SELECT window_end, stress_index
            FROM aggregated_windows
            WHERE window_end >= ? AND stress_index IS NOT NULL
            ORDER BY window_end
        """, (cutoff,)).fetchall()
        times, values = [], []
        for row in rows:
            try:
                ts = datetime.strptime(row["window_end"], "%Y-%m-%d %H:%M:%S")
                times.append(ts.timestamp())
                values.append(float(row["stress_index"]))
            except (ValueError, TypeError):
                pass
        return times, values
    except Exception:
        return [], []


def _fetch_interventions(conn, limit: int = 15) -> list[dict]:
    """Return the *limit* most recent interventions, newest first."""
    try:
        rows = conn.execute("""
            SELECT timestamp, trigger_reason, message
            FROM interventions
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _fetch_session_stats(conn) -> dict:
    """Return today's aggregate session stats."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        row = conn.execute("""
            SELECT
                MIN(window_start)  AS session_start,
                COUNT(*)           AS n_windows,
                AVG(stress_index)  AS avg_stress,
                MAX(stress_index)  AS max_stress,
                SUM(CASE WHEN intervention_triggered = 1 THEN 1 ELSE 0 END) AS n_notifs
            FROM aggregated_windows
            WHERE window_start >= ?
        """, (f"{today} 00:00:00",)).fetchone()

        if row and row["n_windows"]:
            try:
                start_dt = datetime.strptime(row["session_start"], "%Y-%m-%d %H:%M:%S")
                duration = int((datetime.now() - start_dt).total_seconds() / 60)
            except (ValueError, TypeError):
                duration = 0
            return {
                "duration":   duration,
                "n_windows":  int(row["n_windows"] or 0),
                "avg_stress": row["avg_stress"],
                "max_stress": row["max_stress"],
                "n_notifs":   int(row["n_notifs"] or 0),
            }
    except Exception:
        pass
    return {"duration": 0, "n_windows": 0, "avg_stress": None, "max_stress": None, "n_notifs": 0}


def _is_system_active(conn) -> bool:
    """
    True if the monitoring backend is currently running.

    Checks whether any physio_readings OR face_readings row was written in
    the last 2 minutes, giving near-instant feedback rather than waiting for
    the 5-minute aggregation cycle to complete.

    Each table is queried independently so that a missing table (first launch,
    before init_database() has run) doesn't suppress the result from the other.
    """
    cutoff = (datetime.now() - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
    for table in ("physio_readings", "face_readings"):
        try:
            row = conn.execute(
                f"SELECT 1 FROM {table} WHERE timestamp >= ? LIMIT 1",  # noqa: S608
                (cutoff,),
            ).fetchone()
            if row is not None:
                return True
        except Exception:
            pass
    return False


def _has_baseline(conn) -> bool:
    """True if the baselines table contains at least one row."""
    try:
        row = conn.execute("SELECT COUNT(*) FROM baselines").fetchone()
        return row is not None and row[0] > 0
    except Exception:
        return True   # table missing → don't show splash


def _fetch_latest_desktop(conn) -> dict | None:
    """
    Return the most recent raw desktop_readings row as a plain dict.

    Normalises column names: Flutter may write 'active_window', 'window_title',
    or 'window' — all are tried in order, first non-None value wins.
    Returns None if the table is empty or doesn't exist.
    """
    try:
        row = conn.execute(
            "SELECT * FROM desktop_readings ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        # Active window title — try common column names
        active_window = (
            d.get("active_window")
            or d.get("window_title")
            or d.get("window")
        )
        return {
            "active_window":   active_window,
            "app_category":    d.get("app_category") or d.get("category"),
            "activity_pct":    d.get("activity_pct"),
            "idle_seconds":    d.get("idle_seconds"),
            "window_switches": d.get("window_switches"),
            "timestamp":       d.get("timestamp"),
        }
    except Exception:
        return None


def _fetch_latest_face(conn) -> dict | None:
    """Return the most recent face_readings row for camera feed overlay."""
    try:
        row = conn.execute("""
            SELECT timestamp, ear, blink_rate, face_pct, valence, arousal
            FROM face_readings
            ORDER BY id DESC LIMIT 1
        """).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def _update_config_constant(name: str, value: int) -> bool:
    """
    Rewrite a single numeric constant in src/utils/config.py.

    Replaces the integer literal on the line that starts with *name*, leaving
    any trailing comment intact.  Returns True on success, False on any error.
    Changes take effect on the **next** Python session (the running process
    has already imported the old value).
    """
    import re
    config_path = os.path.join(_ROOT, "src", "utils", "config.py")
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        # Match:  NAME = <integer>  <optional trailing comment>
        pattern     = rf"^({re.escape(name)}\s*=\s*)\d+(.*)$"
        replacement = rf"\g<1>{value}\g<2>"
        new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if n == 0:
            print(f"[Dashboard] Config: pattern for '{name}' not found.")
            return False
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(new_content)
        return True
    except Exception as exc:
        print(f"[Dashboard] Config update error for '{name}': {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PYQTGRAPH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_PG:
    class TimeAxisItem(pg.AxisItem):
        """
        AxisItem subclass that formats Unix timestamps as HH:MM tick labels.

        ROOT CAUSE OF THE x1e+09 BUG
        ──────────────────────────────
        pyqtgraph's AxisItem normally applies SI-prefix auto-scaling.  For an
        axis range of ~1.7e9 (Unix epoch seconds) it chooses scale = 1e-9
        (Giga prefix) and passes values ÷ 1e9 ≈ 1.7 to tickStrings.
        datetime.fromtimestamp(1.7)  →  1970-01-01, not the real time.

        Fix: call enableAutoSIPrefix(False) in __init__ so that scale is
        always 1.0 and values passed to tickStrings are true Unix timestamps.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enableAutoSIPrefix(False)
            self.setLabel(text="Time", units=None)

        def tickStrings(self, values, scale, spacing):   # noqa: N802
            result = []
            for v in values:
                try:
                    result.append(datetime.fromtimestamp(float(v)).strftime("%H:%M"))
                except Exception:
                    result.append("")
            return result


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA CAPTURE THREAD + FEED WIDGET
# ─────────────────────────────────────────────────────────────────────────────

def _va_to_emotion(valence: float | None, arousal: float | None) -> str:
    """Map Valence-Arousal coordinates to a human-readable quadrant label."""
    if valence is None or arousal is None:
        return "—"
    if valence >= 0 and arousal >= 0.2:
        return "Excited / Alert"
    if valence >= 0 and arousal < 0.2:
        return "Calm / Content"
    if valence < 0 and arousal >= 0.2:
        return "Tense / Stressed"
    return "Sad / Fatigued"

class CameraThread(QThread):
    """
    Background QThread that reads frames from an OpenCV VideoCapture and
    emits them via *frame_ready* at ~15 FPS (66 ms interval).

    Camera index selection (caller's responsibility):
      • system active  → CAMERA_INDEX + 1  (secondary cam; don't steal main feed)
      • system idle    → CAMERA_INDEX       (primary cam is free)
    """
    frame_ready = pyqtSignal(object)   # numpy BGR array

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self._index   = camera_index
        self._stop    = False
        self._paused  = False

    def run(self) -> None:
        if not _HAS_CV2:
            return
        cap = _cv2.VideoCapture(self._index, _cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Silently fall back — feed widget shows "No Camera"
            return
        cap.set(_cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 240)
        while not self._stop:
            if self._paused:
                self.msleep(100)
                continue
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(66)   # ~15 FPS
        cap.release()

    def request_stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False


class CameraFeedWidget(QWidget):
    """
    Left-panel widget that shows a 240×180 camera feed in two modes:

    is_standalone=True  (launched without --start-backend)
        OpenCV VideoCapture on CAMERA_INDEX+1 (system active) or CAMERA_INDEX
        (system idle).  BGR→RGB→QPixmap at ~15 FPS via QThread.
        Green border = face detected in the last 30 s.

    is_standalone=False (--start-backend: run_all.py already owns the camera)
        No VideoCapture is opened — avoids the "[FATAL] Camera opened but no
        frames arrived" conflict.  Instead a QTimer fires every 200 ms and
        reads data/latest_frame.jpg, which run_all.py writes with the fully
        annotated frame (face mesh + EAR + blinks + head pose + rPPG box +
        HR/stress overlay).  File freshness < 5 s is required; otherwise the
        widget shows a "Waiting for backend…" placeholder.
    """

    # Max age of latest_frame.jpg before it's considered stale
    _FRAME_MAX_AGE_S: float = 5.0

    def __init__(self, is_standalone: bool = True, parent=None):
        super().__init__(parent)
        self._is_standalone = is_standalone

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(3)

        hdr_text = "Live Camera" if is_standalone else "Live Feed (shared)"
        hdr = QLabel(hdr_text)
        hdr.setStyleSheet(f"color: {c('sub')}; font-size: 10px; font-weight: bold;")
        lay.addWidget(hdr)

        self._cam_lbl = QLabel("No Camera" if is_standalone else "Waiting for backend…")
        self._cam_lbl.setFixedSize(240, 180)
        self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_lbl.setStyleSheet(
            f"background-color: {c('card')}; border: 2px solid {c('border')};"
            f" border-radius: 4px; color: {c('sub')}; font-size: 10px;"
        )
        lay.addWidget(self._cam_lbl)
        # Note: no EAR/blinks label — those values are overlaid on the frame
        # itself by face_monitor.py, so a separate label is redundant.

        self._thread:      CameraThread | None = None
        self._face_recent: bool                = False

        # Non-standalone: start QTimer that polls the shared frame file
        if not is_standalone:
            self._file_timer: QTimer | None = QTimer(self)
            self._file_timer.timeout.connect(self._poll_frame_file)
            self._file_timer.start(200)   # 5 FPS display rate
        else:
            self._file_timer = None

    # ── Live-camera frame slot (standalone mode only) ─────────────────────────

    def _on_frame(self, frame) -> None:
        if frame is None or not _HAS_CV2:
            return
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        rgb = _cv2.resize(rgb, (240, 180))
        h, w, ch = rgb.shape
        img  = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(img)
        self._cam_lbl.setPixmap(pix)
        border = c("green") if self._face_recent else c("border")
        self._cam_lbl.setStyleSheet(
            f"background-color: {c('card')}; border: 2px solid {border};"
            f" border-radius: 4px;"
        )

    # ── Shared-file frame display (non-standalone mode) ────────────────────────

    def _poll_frame_file(self) -> None:
        """
        QTimer slot — fires every 200 ms in non-standalone mode.

        Loads data/latest_frame.jpg if it exists and was written within the
        last 5 seconds; scales it to 240×180 keeping aspect ratio; displays it
        with a coloured border (green = face recent, grey = no face data).
        Falls back to a text placeholder when the file is missing or stale.
        """
        try:
            if not os.path.exists(_LATEST_FRAME_PATH):
                self._show_placeholder("Waiting for backend…")
                return

            mtime = os.path.getmtime(_LATEST_FRAME_PATH)
            if _time.time() - mtime > self._FRAME_MAX_AGE_S:
                self._show_placeholder("No face data")
                return

            pix = QPixmap(_LATEST_FRAME_PATH)
            if pix.isNull():
                return   # file partially written — skip this tick

            scaled = pix.scaled(
                240, 180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            border = c("green") if self._face_recent else c("border")
            self._cam_lbl.setStyleSheet(
                f"background-color: {c('card')}; border: 2px solid {border};"
                f" border-radius: 4px;"
            )
            self._cam_lbl.setPixmap(scaled)

        except Exception:
            pass   # I/O race during write — next tick will recover

    def _show_placeholder(self, msg: str) -> None:
        """Show a text placeholder on the dark card background."""
        self._cam_lbl.setPixmap(QPixmap())
        self._cam_lbl.setText(msg)
        self._cam_lbl.setStyleSheet(
            f"background-color: {c('card')}; border: 2px solid {c('border')};"
            f" border-radius: 4px; color: {c('sub')}; font-size: 10px;"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_face_info(self, face_raw: dict | None) -> None:
        """
        Update the face-recency flag used for the border colour.
        Called every refresh cycle from DashboardWindow._refresh.
        Frame display (non-standalone) is handled by the _poll_frame_file
        QTimer; EAR/blink values are overlaid on the frame by face_monitor.py
        and are not duplicated in a separate label.
        """
        if face_raw is None:
            self._face_recent = False
            return
        ts = face_raw.get("timestamp", "")
        try:
            ts_clean = str(ts).replace("T", " ")
            dt = datetime.strptime(ts_clean[:19], "%Y-%m-%d %H:%M:%S")
            self._face_recent = (datetime.now() - dt).total_seconds() < 30
        except (ValueError, TypeError):
            self._face_recent = False

    def start_capture(self, system_active: bool) -> None:
        """Start (or restart) live camera.  No-op in non-standalone mode."""
        if not self._is_standalone:
            return   # camera is held by run_all.py — do not compete
        self._stop_thread()
        cam_idx = (CAMERA_INDEX + 1) if system_active else CAMERA_INDEX
        self._thread = CameraThread(cam_idx)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.start()

    def stop_capture(self) -> None:
        if self._file_timer is not None:
            self._file_timer.stop()
        self._stop_thread()
        self._cam_lbl.clear()
        self._cam_lbl.setText("No Camera" if self._is_standalone else "Waiting for data…")

    def pause_capture(self) -> None:
        if self._file_timer is not None:
            self._file_timer.stop()
        if self._thread:
            self._thread.pause()

    def resume_capture(self) -> None:
        if self._file_timer is not None:
            self._file_timer.start(200)
        if self._thread:
            self._thread.resume()

    def _stop_thread(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.request_stop()
            self._thread.wait(2000)
        self._thread = None


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM WIDGETS
# ─────────────────────────────────────────────────────────────────────────────

class StressGauge(QWidget):
    """
    Arc-style gauge (240° sweep) showing stress_index 0–1.
    Filled arc is colour-coded green → yellow → orange → red.
    Numeric value and label are drawn in the centre.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value: float | None = None
        self.setFixedSize(160, 160)

    def set_value(self, v: float | None) -> None:
        self._value = v
        self.update()

    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        m = 14
        rect = QRectF(m, m, w - 2 * m, h - 2 * m)

        # Track arc
        pen_track = QPen(QColor(c("border")), 10, Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap)
        p.setPen(pen_track)
        p.drawArc(rect, 210 * 16, -240 * 16)

        # Value arc
        if self._value is not None:
            span = int(-240 * 16 * max(0.0, min(1.0, self._value)))
            pen_val = QPen(QColor(stress_color(self._value)), 10,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen_val)
            p.drawArc(rect, 210 * 16, span)

        # Central value text
        cx, cy = w / 2, h / 2
        if self._value is not None:
            p.setPen(QPen(QColor(stress_color(self._value))))
            val_str = f"{self._value:.2f}"
        else:
            p.setPen(QPen(QColor(c("sub"))))
            val_str = "—"

        font_big = QFont("Segoe UI", 22, QFont.Weight.Bold)
        p.setFont(font_big)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(val_str)
        p.drawText(int(cx - tw / 2), int(cy + fm.ascent() / 2 - 6), val_str)

        # "Stress Index" label below
        font_sm = QFont("Segoe UI", 9)
        p.setFont(font_sm)
        p.setPen(QPen(QColor(c("sub"))))
        lbl = "Stress Index"
        fm2 = p.fontMetrics()
        tw2 = fm2.horizontalAdvance(lbl)
        p.drawText(int(cx - tw2 / 2), int(cy + 26), lbl)

        p.end()


class MetricCard(QFrame):
    """Compact labelled metric card with value + unit."""

    def __init__(self, label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self._label = label
        self._unit  = unit
        self._refresh_style()
        self.setFixedHeight(56)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(f"color: {c('sub')}; font-size: 11px;")
        lay.addWidget(self._lbl)
        lay.addStretch()

        right = QHBoxLayout()
        right.setSpacing(3)
        self._val_lbl = QLabel("—")
        self._val_lbl.setStyleSheet(
            f"color: {c('text')}; font-size: 18px; font-weight: bold;"
        )
        right.addWidget(self._val_lbl)
        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setStyleSheet(
            f"color: {c('sub')}; font-size: 10px; padding-top: 5px;"
        )
        right.addWidget(self._unit_lbl)
        lay.addLayout(right)

    def _refresh_style(self) -> None:
        self.setStyleSheet(
            f"background-color: {c('card')}; border-radius: 8px;"
            f" border: 1px solid {c('border')};"
        )

    def set_value(self, val: float | None, color: str | None = None) -> None:
        if val is None:
            self._val_lbl.setText("—")
            self._val_lbl.setStyleSheet(
                f"color: {c('sub')}; font-size: 18px; font-weight: bold;"
            )
        else:
            fmt = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
            self._val_lbl.setText(fmt)
            col = color or c("text")
            self._val_lbl.setStyleSheet(
                f"color: {col}; font-size: 18px; font-weight: bold;"
            )

    def apply_theme(self) -> None:
        self._refresh_style()
        self._lbl.setStyleSheet(f"color: {c('sub')}; font-size: 11px;")
        self._unit_lbl.setStyleSheet(
            f"color: {c('sub')}; font-size: 10px; padding-top: 5px;"
        )


class VAWidget(QWidget):
    """
    Miniature Valence-Arousal 2D plot.
    Quadrant labels describe the emotional state in each corner.
    The current V/A position is shown as a coloured circle.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._valence: float | None = None
        self._arousal: float | None = None
        self.setFixedSize(130, 130)
        self.setToolTip(
            "Valence (x-axis): Negative ← → Positive\n"
            "Arousal (y-axis): Calm ↓ ↑ Excited"
        )

    def set_va(self, valence: float | None, arousal: float | None) -> None:
        self._valence = valence
        self._arousal = arousal
        self.update()

    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        m = 18

        # Background
        p.fillRect(0, 0, w, h, QColor(c("card")))

        # Border
        pen_border = QPen(QColor(c("border")), 1)
        p.setPen(pen_border)
        p.drawRect(m, m, w - 2 * m, h - 2 * m)

        # Axis cross
        cx, cy = w / 2, h / 2
        p.drawLine(int(m), int(cy), int(w - m), int(cy))
        p.drawLine(int(cx), int(m), int(cx), int(h - m))

        # Quadrant labels
        font_q = QFont("Segoe UI", 7)
        p.setFont(font_q)
        p.setPen(QPen(QColor(c("sub"))))
        p.drawText(QRectF(cx + 1, m, cx - m - 1, cy - m),
                   Qt.AlignmentFlag.AlignCenter, "Tense")
        p.drawText(QRectF(m, m, cx - m - 1, cy - m),
                   Qt.AlignmentFlag.AlignCenter, "Excited")
        p.drawText(QRectF(cx + 1, cy, cx - m - 1, cy - m),
                   Qt.AlignmentFlag.AlignCenter, "Relaxed")
        p.drawText(QRectF(m, cy, cx - m - 1, cy - m),
                   Qt.AlignmentFlag.AlignCenter, "Sad")

        # Axis end labels
        font_ax = QFont("Segoe UI", 7)
        p.setFont(font_ax)
        p.setPen(QPen(QColor(c("sub"))))
        p.drawText(int(m - 2), int(h - 3), "V")
        p.drawText(int(w - 13), int(cy - 2), "A")

        # Current position dot
        if self._valence is not None and self._arousal is not None:
            px = m + (self._valence + 1) / 2 * (w - 2 * m)
            py = (h - m) - (self._arousal + 1) / 2 * (h - 2 * m)
            dot_col = c("accent") if self._valence >= 0 else c("orange")
            p.setBrush(QBrush(QColor(dot_col)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(px, py), 5.5, 5.5)
        else:
            # Placeholder crosshair
            p.setPen(QPen(QColor(c("sub")), 1.5))
            p.drawLine(int(cx) - 4, int(cy), int(cx) + 4, int(cy))
            p.drawLine(int(cx), int(cy) - 4, int(cx), int(cy) + 4)

        p.end()


class DesktopCard(QFrame):
    """
    Expanded desktop context card showing:
      • LLM app-category badge (colour-coded pill)
      • Active window title (truncated to 35 chars)
      • Activity percentage bar (green fill)
      • Idle time and window-switch count
      • Last-updated timestamp
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._refresh_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(5)

        # Header
        hdr = QLabel("Desktop Context")
        hdr.setStyleSheet(f"color: {c('sub')}; font-size: 10px; font-weight: bold;")
        lay.addWidget(hdr)

        # Category badge — styled as a pill
        self._cat_badge = QLabel("—")
        self._cat_badge.setStyleSheet(self._badge_style(c("sub")))
        self._cat_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cat_badge.setFixedHeight(22)
        lay.addWidget(self._cat_badge)

        # Active window title
        self._win_lbl = QLabel("—")
        self._win_lbl.setStyleSheet(
            f"color: {c('sub')}; font-size: 10px; font-style: italic;"
        )
        self._win_lbl.setWordWrap(False)
        lay.addWidget(self._win_lbl)

        # Activity progress bar
        act_row = QHBoxLayout()
        act_row.setContentsMargins(0, 0, 0, 0)
        act_row.setSpacing(6)
        act_lbl = QLabel("Activity")
        act_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 10px;")
        act_lbl.setFixedWidth(48)
        act_row.addWidget(act_lbl)
        self._act_bar = QProgressBar()
        self._act_bar.setRange(0, 100)
        self._act_bar.setValue(0)
        self._act_bar.setFixedHeight(8)
        self._act_bar.setTextVisible(False)
        self._act_bar.setStyleSheet(self._bar_style())
        act_row.addWidget(self._act_bar, stretch=1)
        self._act_pct = QLabel("—")
        self._act_pct.setStyleSheet(f"color: {c('accent')}; font-size: 10px;")
        self._act_pct.setFixedWidth(34)
        self._act_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        act_row.addWidget(self._act_pct)
        lay.addLayout(act_row)

        # Bottom row: idle · switches · timestamp
        bot_row = QHBoxLayout()
        bot_row.setContentsMargins(0, 2, 0, 0)
        self._idle_lbl = QLabel("Idle: —")
        self._idle_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        bot_row.addWidget(self._idle_lbl)
        self._sw_lbl = QLabel("Sw: —")
        self._sw_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        bot_row.addWidget(self._sw_lbl)
        bot_row.addStretch()
        self._ts_lbl = QLabel("")
        self._ts_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        bot_row.addWidget(self._ts_lbl)
        lay.addLayout(bot_row)

    # ── Style helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _badge_style(bg: str) -> str:
        return (
            f"background-color: {bg}22; color: {bg}; "
            f"border: 1px solid {bg}66; border-radius: 10px; "
            f"font-size: 10px; font-weight: bold; padding: 0 6px;"
        )

    @staticmethod
    def _bar_style() -> str:
        return (
            f"QProgressBar {{ background-color: {c('border')}; border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background-color: {c('green')}; border-radius: 4px; }}"
        )

    def _refresh_style(self) -> None:
        self.setStyleSheet(
            f"background-color: {c('card')}; border-radius: 8px;"
            f" border: 1px solid {c('border')};"
        )

    # ── Data update ────────────────────────────────────────────────────────────

    def set_data(self, desktop_raw: dict | None) -> None:
        """Populate all fields from a *desktop_raw* dict (from _fetch_latest_desktop)."""
        if desktop_raw is None:
            self._cat_badge.setText("—")
            self._cat_badge.setStyleSheet(self._badge_style(c("sub")))
            self._win_lbl.setText("—")
            self._act_bar.setValue(0)
            self._act_pct.setText("—")
            self._idle_lbl.setText("Idle: —")
            self._sw_lbl.setText("Sw: —")
            self._ts_lbl.setText("")
            return

        # Category badge
        cat = desktop_raw.get("app_category") or "—"
        col_key = _CATEGORY_COLORS.get(cat, "sub")
        col = c(col_key)
        self._cat_badge.setText(cat)
        self._cat_badge.setStyleSheet(self._badge_style(col))

        # Window title (truncated)
        win = desktop_raw.get("active_window") or ""
        self._win_lbl.setText((win[:35] + "…") if len(win) > 35 else win or "—")

        # Activity bar
        act = desktop_raw.get("activity_pct")
        if act is not None:
            self._act_bar.setValue(int(act))
            self._act_pct.setText(f"{act:.0f}%")
        else:
            self._act_bar.setValue(0)
            self._act_pct.setText("—")

        # Idle time
        idle = desktop_raw.get("idle_seconds")
        self._idle_lbl.setText(
            f"Idle: {int(idle)} s" if idle is not None else "Idle: —"
        )

        # Window switches
        sw = desktop_raw.get("window_switches")
        self._sw_lbl.setText(
            f"Sw: {int(sw)}" if sw is not None else "Sw: —"
        )

        # Timestamp
        ts = desktop_raw.get("timestamp", "")
        if ts:
            try:
                # Handle both space and T separator
                ts_clean = ts.replace("T", " ")
                dt = datetime.strptime(ts_clean[:19], "%Y-%m-%d %H:%M:%S")
                self._ts_lbl.setText(dt.strftime("%H:%M:%S"))
            except (ValueError, TypeError):
                self._ts_lbl.setText("")

    def apply_theme(self) -> None:
        self._refresh_style()
        self._act_bar.setStyleSheet(self._bar_style())


class StatusDot(QWidget):
    """Green/grey dot indicating whether the monitoring system is active."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self.setFixedSize(12, 12)

    def set_active(self, v: bool) -> None:
        self._active = v
        self.update()

    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = c("green") if self._active else c("sub")
        p.setBrush(QBrush(QColor(col)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(1, 1, 10, 10)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL
# ─────────────────────────────────────────────────────────────────────────────

class LeftPanel(QWidget):
    def __init__(self, standalone: bool = True, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)

        # Outer layout: scroll area + pinned status row
        outer_lay = QVBoxLayout(self)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        # ── Scrollable content ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(12, 12, 6, 12)
        lay.setSpacing(8)

        # Stress gauge (centred)
        gauge_row = QHBoxLayout()
        gauge_row.addStretch()
        self.gauge = StressGauge()
        gauge_row.addWidget(self.gauge)
        gauge_row.addStretch()
        lay.addLayout(gauge_row)

        # Metric cards
        self.hr_card  = MetricCard("♥  Heart Rate", unit="BPM")
        self.hrv_card = MetricCard("〜  RMSSD",       unit="ms")
        lay.addWidget(self.hr_card)
        lay.addWidget(self.hrv_card)

        # Valence-Arousal scatter (centred)
        va_lbl = QLabel("Valence — Arousal")
        va_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 10px;")
        va_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(va_lbl)
        va_row = QHBoxLayout()
        va_row.addStretch()
        self.va_widget = VAWidget()
        va_row.addWidget(self.va_widget)
        va_row.addStretch()
        lay.addLayout(va_row)

        # Desktop context
        self.desktop_card = DesktopCard()
        lay.addWidget(self.desktop_card)

        # Live camera feed (requires cv2); in non-standalone mode shows DB metrics
        if _HAS_CV2:
            self.camera_widget: CameraFeedWidget | None = CameraFeedWidget(
                is_standalone=standalone
            )
            lay.addWidget(self.camera_widget)
        else:
            self.camera_widget = None

        lay.addStretch()

        scroll.setWidget(content)
        outer_lay.addWidget(scroll, stretch=1)

        # ── Status row — pinned below scroll, always visible ──────────────
        status_row = QHBoxLayout()
        status_row.setContentsMargins(12, 4, 6, 8)
        self.status_dot = StatusDot()
        status_row.addWidget(self.status_dot)
        self.status_lbl = QLabel("System inactive")
        self.status_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 10px;")
        status_row.addWidget(self.status_lbl)
        status_row.addStretch()
        self._ts_lbl = QLabel("")
        self._ts_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        status_row.addWidget(self._ts_lbl)
        outer_lay.addLayout(status_row)

    def refresh(
        self,
        data:        dict | None,
        active:      bool,
        desktop_raw: dict | None = None,
        face_raw:    dict | None = None,
    ) -> None:
        if data:
            self.gauge.set_value(data.get("stress_index"))
            self.hr_card.set_value(data.get("avg_hr"))
            self.hrv_card.set_value(data.get("avg_rmssd"))
            self.va_widget.set_va(data.get("avg_valence"), data.get("avg_arousal"))
            ts = data.get("window_end", "")
            if ts:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    self._ts_lbl.setText(dt.strftime("%H:%M"))
                except ValueError:
                    self._ts_lbl.setText("")
        else:
            self.gauge.set_value(None)
            self.hr_card.set_value(None)
            self.hrv_card.set_value(None)
            self.va_widget.set_va(None, None)
            self._ts_lbl.setText("")

        # Desktop card uses the raw desktop_readings data for richer detail
        self.desktop_card.set_data(desktop_raw)

        # Camera widget: pass latest face data for overlay
        if self.camera_widget is not None:
            self.camera_widget.set_face_info(face_raw)

        self.status_dot.set_active(active)
        lbl_text = "System active" if active else "System inactive"
        col = c("green") if active else c("sub")
        self.status_lbl.setText(lbl_text)
        self.status_lbl.setStyleSheet(f"color: {col}; font-size: 10px;")


# ─────────────────────────────────────────────────────────────────────────────
# CENTRE PANEL
# ─────────────────────────────────────────────────────────────────────────────

class CenterPanel(QWidget):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 12, 6, 12)
        lay.setSpacing(10)

        # Section header
        trend_hdr = QLabel("Stress Index — Last 60 minutes")
        trend_hdr.setStyleSheet(
            f"color: {c('sub')}; font-size: 11px; font-weight: bold;"
        )
        lay.addWidget(trend_hdr)

        # Trend plot
        if _HAS_PG:
            self.plot = pg.PlotWidget(
                axisItems={"bottom": TimeAxisItem(orientation="bottom")}
            )
            self.plot.setBackground(c("panel"))
            self.plot.showGrid(x=True, y=True, alpha=0.2)
            self.plot.setYRange(0, 1.0, padding=0)
            self.plot.setMinimumHeight(220)
            self.plot.setLabel("left",   "Stress Index", color=c("sub"), size="10pt")
            self.plot.setLabel("bottom", "Time",         color=c("sub"), size="10pt")
            self.plot.setMouseEnabled(x=False, y=False)
            self.plot.getAxis("left").setTextPen(c("sub"))
            self.plot.getAxis("bottom").setTextPen(c("sub"))

            # Threshold line
            self._thresh_line = pg.InfiniteLine(
                pos=STRESS_TRIGGER_THRESHOLD,
                angle=0,
                pen=pg.mkPen(color=c("red"), style=Qt.PenStyle.DashLine, width=1),
                label=f"Threshold {STRESS_TRIGGER_THRESHOLD}",
                labelOpts={"color": c("red"), "position": 0.05},
            )
            self.plot.addItem(self._thresh_line)

            # Stress curve + fill
            self._curve = self.plot.plot(
                pen=pg.mkPen(color=c("accent"), width=2),
                fillLevel=0,
                brush=pg.mkBrush(color=(*bytes.fromhex(c("accent")[1:]), 40)),
            )
            lay.addWidget(self.plot, stretch=1)

            # No-data placeholder — shown when < 2 points in window, hidden otherwise
            self._no_data_lbl = QLabel(
                "No data in the last 60 minutes\n— monitoring in progress —"
            )
            self._no_data_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._no_data_lbl.setStyleSheet(
                f"color: {c('sub')}; font-size: 11px;"
            )
            self._no_data_lbl.setMinimumHeight(220)
            lay.addWidget(self._no_data_lbl, stretch=1)
            self._no_data_lbl.hide()          # plot visible by default

        else:
            no_pg = QLabel("pyqtgraph not installed\npip install pyqtgraph")
            no_pg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_pg.setStyleSheet(f"color: {c('sub')}; font-size: 12px;")
            lay.addWidget(no_pg, stretch=1)
            self._no_data_lbl = None          # type: ignore[assignment]

        # Date label (shows "Today, 26 May 2026") — updated on every refresh
        self._date_lbl = QLabel("")
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._date_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        lay.addWidget(self._date_lbl)

        # Session stats bar
        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            f"background-color: {c('panel')}; border-radius: 8px;"
            f" border: 1px solid {c('border')};"
        )
        stats_lay = QHBoxLayout(stats_frame)
        stats_lay.setContentsMargins(16, 8, 16, 8)

        self._stat_dur    = self._make_stat("Session Duration", "—")
        self._stat_avg    = self._make_stat("Avg Stress",       "—")
        self._stat_max    = self._make_stat("Peak Stress",      "—")
        self._stat_notifs = self._make_stat("Interventions",    "—")

        all_stats = [self._stat_dur, self._stat_avg, self._stat_max, self._stat_notifs]
        for i, sw in enumerate(all_stats):
            stats_lay.addWidget(sw)
            if i < len(all_stats) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color: {c('border')};")
                stats_lay.addWidget(sep)

        lay.addWidget(stats_frame)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_stat(self, label: str, default: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 0, 8, 0)
        v.setSpacing(2)
        lbl_w = QLabel(label)
        lbl_w.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        lbl_w.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_w = QLabel(default)
        val_w.setStyleSheet(
            f"color: {c('text')}; font-size: 14px; font-weight: bold;"
        )
        val_w.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl_w)
        v.addWidget(val_w)
        w._val_lbl = val_w   # type: ignore[attr-defined]
        return w

    # ── Public update methods ─────────────────────────────────────────────────

    def refresh_trend(self, times: list[float], values: list[float]) -> None:
        if not _HAS_PG:
            return

        now_ts = datetime.now().timestamp()

        # Update date label on every refresh
        today = datetime.now()
        self._date_lbl.setText(
            f"Today, {today.day} {today.strftime('%B %Y')}"
        )

        # Pin both axes — pyqtgraph resets ranges after setData without this
        self.plot.setXRange(now_ts - 3600, now_ts, padding=0)
        self.plot.setYRange(0, 1.0, padding=0)

        if len(times) < 2:
            # Not enough data — show placeholder, clear curve
            self._curve.clear()
            if self._no_data_lbl is not None:
                self.plot.hide()
                self._no_data_lbl.show()
            return

        # Enough data — show plot, hide placeholder
        if self._no_data_lbl is not None:
            self._no_data_lbl.hide()
            self.plot.show()

        x = np.array(times, dtype=float)
        y = np.array(values, dtype=float)

        self._curve.setData(x, y)
        # TimeAxisItem.enableAutoSIPrefix(False) ensures scale=1.0,
        # so tickStrings receives true Unix timestamps → HH:MM labels

    def refresh_stats(self, stats: dict) -> None:
        dur = stats.get("duration", 0)
        h, m = divmod(int(dur), 60)
        self._stat_dur._val_lbl.setText(  # type: ignore[attr-defined]
            f"{h}h {m:02d}m" if h else f"{m}m"
        )

        avg = stats.get("avg_stress")
        self._stat_avg._val_lbl.setText(  # type: ignore[attr-defined]
            f"{avg:.2f}" if avg is not None else "—"
        )
        if avg is not None:
            self._stat_avg._val_lbl.setStyleSheet(  # type: ignore[attr-defined]
                f"color: {stress_color(avg)}; font-size: 14px; font-weight: bold;"
            )

        mx = stats.get("max_stress")
        self._stat_max._val_lbl.setText(  # type: ignore[attr-defined]
            f"{mx:.2f}" if mx is not None else "—"
        )
        if mx is not None:
            self._stat_max._val_lbl.setStyleSheet(  # type: ignore[attr-defined]
                f"color: {stress_color(mx)}; font-size: 14px; font-weight: bold;"
            )

        n = stats.get("n_notifs", 0)
        self._stat_notifs._val_lbl.setText(str(n))  # type: ignore[attr-defined]

    def apply_theme(self) -> None:
        """Re-apply theme colours to the pyqtgraph plot and stats bar."""
        if _HAS_PG:
            self.plot.setBackground(c("panel"))
            self.plot.getAxis("left").setTextPen(c("sub"))
            self.plot.getAxis("bottom").setTextPen(c("sub"))
            self._curve.setPen(pg.mkPen(color=c("accent"), width=2))
            self._thresh_line.setPen(
                pg.mkPen(color=c("red"), style=Qt.PenStyle.DashLine, width=1)
            )


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL
# ─────────────────────────────────────────────────────────────────────────────

class _InterventionItem(QWidget):
    """Single row in the intervention log list."""

    def __init__(self, time_s: str, label: str, message: str, color: str,
                 parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        t_lbl = QLabel(time_s)
        t_lbl.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        top_row.addWidget(t_lbl)
        top_row.addStretch()
        r_lbl = QLabel(label)
        r_lbl.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
        top_row.addWidget(r_lbl)
        lay.addLayout(top_row)

        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet(f"color: {c('text')}; font-size: 10px;")
        msg_lbl.setWordWrap(True)
        lay.addWidget(msg_lbl)


class RightPanel(QWidget):
    export_clicked   = pyqtSignal()
    settings_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 12, 12, 12)
        lay.setSpacing(8)

        hdr = QLabel("Interventions")
        hdr.setStyleSheet(f"color: {c('sub')}; font-size: 11px; font-weight: bold;")
        lay.addWidget(hdr)

        self.list_w = QListWidget()
        self.list_w.setWordWrap(True)
        self.list_w.setSpacing(1)
        self.list_w.setUniformItemSizes(False)
        lay.addWidget(self.list_w, stretch=1)

        self._export_btn = QPushButton("📊  Export to Excel")
        self._export_btn.setToolTip("Run export_excel.py and open the output file")
        self._export_btn.clicked.connect(self.export_clicked)
        lay.addWidget(self._export_btn)

        self._settings_btn = QPushButton("⚙  Settings")
        self._settings_btn.clicked.connect(self.settings_clicked)
        lay.addWidget(self._settings_btn)

    def refresh(self, interventions: list[dict]) -> None:
        self.list_w.clear()
        for item in interventions:
            ts     = item.get("timestamp", "")
            reason = item.get("trigger_reason") or "default"
            msg    = item.get("message", "") or ""

            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                time_str = dt.strftime("%H:%M")
            except (ValueError, TypeError):
                time_str = ts[:5] if ts else "—"

            label = _TRIGGER_LABELS.get(reason, reason.replace("_", " ").title())
            col   = c(_TRIGGER_COLORS.get(reason, "sub"))
            short_msg = (msg[:68] + "…") if len(msg) > 68 else msg

            li = QListWidgetItem()
            widget = _InterventionItem(time_str, label, short_msg, col)
            widget.setMinimumHeight(48)
            li.setSizeHint(QSize(240, widget.sizeHint().height()))
            self.list_w.addItem(li)
            self.list_w.setItemWidget(li, widget)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    theme_changed = pyqtSignal(str)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(340)
        self.setModal(True)

        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        # Appearance
        app_grp = QGroupBox("Appearance")
        app_lay = QFormLayout(app_grp)
        self._dark_cb = QCheckBox("Dark mode")
        self._dark_cb.setChecked(settings.value("theme", "dark") == "dark")
        app_lay.addRow(self._dark_cb)
        lay.addWidget(app_grp)

        # Dashboard refresh interval
        ref_grp = QGroupBox("Dashboard refresh interval")
        ref_lay = QFormLayout(ref_grp)
        self._refresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._refresh_slider.setRange(10, 120)
        self._refresh_slider.setSingleStep(5)
        self._refresh_slider.setValue(int(settings.value("refresh_interval", 30)))
        self._refresh_lbl = QLabel(f"{self._refresh_slider.value()} s")
        self._refresh_lbl.setStyleSheet(f"color: {c('accent')};")
        self._refresh_slider.valueChanged.connect(
            lambda v: self._refresh_lbl.setText(f"{v} s")
        )
        ref_lay.addRow("Interval:", self._refresh_slider)
        ref_lay.addRow("",          self._refresh_lbl)
        lay.addWidget(ref_grp)

        # Minimum time between notifications
        notif_grp = QGroupBox("Minimum time between notifications")
        notif_lay = QFormLayout(notif_grp)
        from src.utils.config import MIN_MINUTES_BETWEEN_NOTIFS as _default_notif
        self._notif_slider = QSlider(Qt.Orientation.Horizontal)
        self._notif_slider.setRange(5, 60)
        self._notif_slider.setSingleStep(5)
        self._notif_slider.setPageStep(5)
        saved_notif = int(settings.value("min_notif_minutes", _default_notif))
        self._notif_slider.setValue(saved_notif)
        self._notif_lbl = QLabel(f"{saved_notif} minutes")
        self._notif_lbl.setStyleSheet(f"color: {c('accent')};")
        self._notif_slider.valueChanged.connect(
            lambda v: self._notif_lbl.setText(f"{v} minutes")
        )
        notif_info = QLabel(
            "Changes take effect on the next monitoring session\n"
            "(the backend reads this value at startup)."
        )
        notif_info.setStyleSheet(f"color: {c('sub')}; font-size: 9px;")
        notif_lay.addRow("Cooldown:", self._notif_slider)
        notif_lay.addRow("",          self._notif_lbl)
        notif_lay.addRow(notif_info)
        lay.addWidget(notif_grp)

        # Baseline calibration
        cal_grp = QGroupBox("Personal Baseline Calibration")
        cal_lay = QVBoxLayout(cal_grp)
        cal_info = QLabel(
            "Run a 2-minute resting calibration to personalise stress\n"
            "scoring. Keep your face visible to the camera throughout."
        )
        cal_info.setStyleSheet(f"color: {c('sub')}; font-size: 10px;")
        cal_info.setWordWrap(True)
        cal_lay.addWidget(cal_info)
        cal_btn = QPushButton("Run Calibration")
        cal_btn.clicked.connect(self._run_calibration)
        cal_lay.addWidget(cal_btn)
        lay.addWidget(cal_grp)

        # OK / Cancel
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _save(self) -> None:
        theme = "dark" if self._dark_cb.isChecked() else "light"
        self._settings.setValue("theme", theme)
        self._settings.setValue("refresh_interval", self._refresh_slider.value())

        # Persist notification cooldown to QSettings and write to config.py
        notif_val = self._notif_slider.value()
        self._settings.setValue("min_notif_minutes", notif_val)
        ok = _update_config_constant("MIN_MINUTES_BETWEEN_NOTIFS", notif_val)
        if not ok:
            QMessageBox.warning(
                self,
                "Settings",
                "Could not update MIN_MINUTES_BETWEEN_NOTIFS in config.py.\n"
                "The dashboard setting was saved — it will take effect next session.",
            )

        self.theme_changed.emit(theme)
        self.accept()

    def _run_calibration(self) -> None:
        script = os.path.join(_ROOT, "src", "utils", "baseline.py")
        cmd = (
            f'start cmd.exe /k "cd /d "{_ROOT}" && '
            f'python "{script}" --calibrate"'
        )
        subprocess.Popen(cmd, shell=True)
        QMessageBox.information(
            self,
            "Calibration Started",
            "A terminal window is opening for the calibration session.\n"
            "Follow the on-screen instructions.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SPLASH DIALOG  (shown on first launch when no baseline exists)
# ─────────────────────────────────────────────────────────────────────────────

class SplashDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Stress Monitor")
        self.setMinimumWidth(420)
        self.setModal(True)

        lay = QVBoxLayout(self)
        lay.setSpacing(14)

        title = QLabel("Welcome to the Stress Monitor")
        title.setStyleSheet(
            f"color: {c('text')}; font-size: 16px; font-weight: bold;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        info = QLabel(
            "No personal baseline calibration was found.\n\n"
            "For accurate, individually calibrated stress scoring, "
            "run a 2-minute resting calibration before your first session. "
            "Sit comfortably, face the camera, and relax.\n\n"
            "You can skip this now and calibrate later via ⚙ Settings."
        )
        info.setStyleSheet(f"color: {c('sub')}; font-size: 12px;")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(info)

        cal_btn = QPushButton("Run Calibration Now")
        cal_btn.setStyleSheet(
            f"background-color: {c('accent')}; color: {c('bg')}; "
            f"font-size: 13px; padding: 10px; border-radius: 6px; border: none;"
        )
        cal_btn.clicked.connect(self._run_calibration)
        lay.addWidget(cal_btn)

        skip_btn = QPushButton("Skip — I'll calibrate later")
        skip_btn.clicked.connect(self.accept)
        lay.addWidget(skip_btn)

    def _run_calibration(self) -> None:
        script = os.path.join(_ROOT, "src", "utils", "baseline.py")
        cmd = (
            f'start cmd.exe /k "cd /d "{_ROOT}" && '
            f'python "{script}" --calibrate"'
        )
        subprocess.Popen(cmd, shell=True)
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class DashboardWindow(QMainWindow):
    """
    Root application window.

    Panels are laid out left→centre→right with QFrame separators.
    A QTimer drives non-blocking DB reads every *refresh_interval* seconds.
    """

    def __init__(self, db_path: str = DB_PATH, standalone: bool = True):
        super().__init__()
        self._db_path  = db_path
        self._settings = QSettings("StressMonitor", "Dashboard")
        self._standalone = standalone

        # Apply saved theme before building any widgets
        theme_name = str(self._settings.value("theme", "dark"))
        _apply_theme(theme_name)

        if _HAS_PG:
            pg.setConfigOption("background", c("panel"))
            pg.setConfigOption("foreground", c("sub"))

        self.setWindowTitle("Stress Monitor — Well-being Dashboard")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 760)
        self.setStyleSheet(_stylesheet())

        # ── Build three-panel layout ──────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        self._left   = LeftPanel(standalone=standalone)
        self._center = CenterPanel(self._settings)
        self._right  = RightPanel()

        def _vsep() -> QFrame:
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setStyleSheet(f"color: {c('border')};")
            return f

        main_lay.addWidget(self._left)
        main_lay.addWidget(_vsep())
        main_lay.addWidget(self._center, stretch=1)
        main_lay.addWidget(_vsep())
        main_lay.addWidget(self._right)

        # ── Wire signals ──────────────────────────────────────────────────
        self._right.export_clicked.connect(self._run_export)
        self._right.settings_clicked.connect(self._open_settings)

        # ── Database connection ───────────────────────────────────────────
        self._conn = _db_connect(self._db_path)

        # Track last known active state to detect transitions for camera restart
        self._last_active: bool | None = None

        # ── Refresh timer ─────────────────────────────────────────────────
        interval_s = int(self._settings.value("refresh_interval", 30))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(interval_s * 1000)

        # Initial data load (starts camera as a side effect)
        self._refresh()

        # ── Splash screen on first launch (no baseline) ───────────────────
        if self._conn and not _has_baseline(self._conn):
            QTimer.singleShot(300, self._show_splash)

    # ── Refresh slot ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._conn is None:
            self._conn = _db_connect(self._db_path)
        if self._conn is None:
            return

        try:
            data          = _fetch_latest(self._conn)
            active        = _is_system_active(self._conn)
            desktop_raw   = _fetch_latest_desktop(self._conn)
            face_raw      = _fetch_latest_face(self._conn)
            times, values = _fetch_trend(self._conn, minutes=60)
            interventions = _fetch_interventions(self._conn)
            stats         = _fetch_session_stats(self._conn)

            self._left.refresh(data, active, desktop_raw=desktop_raw, face_raw=face_raw)
            self._center.refresh_trend(times, values)
            self._center.refresh_stats(stats)
            self._right.refresh(interventions)

            # ── Camera lifecycle ──────────────────────────────────────────
            cam = self._left.camera_widget
            if cam is not None:
                if self._last_active is None or active != self._last_active:
                    # Active state changed — restart camera on the right index
                    cam.start_capture(system_active=active)
                self._last_active = active

        except Exception as exc:
            print(f"[Dashboard] Refresh error: {exc}")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = _db_connect(self._db_path)

    # ── Action slots ──────────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._settings, parent=self)
        dlg.theme_changed.connect(self._apply_new_theme)
        dlg.exec()
        interval_s = int(self._settings.value("refresh_interval", 30))
        self._timer.setInterval(interval_s * 1000)

    def _apply_new_theme(self, name: str) -> None:
        _apply_theme(name)
        self.setStyleSheet(_stylesheet())
        self._center.apply_theme()

    def _run_export(self) -> None:
        script = os.path.join(_ROOT, "export_excel.py")
        if not os.path.exists(script):
            QMessageBox.warning(self, "Export", "export_excel.py not found.")
            return
        try:
            cmd = (
                f'start cmd.exe /k "cd /d "{_ROOT}" && '
                f'python "{script}" && pause"'
            )
            subprocess.Popen(cmd, shell=True)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _show_splash(self) -> None:
        SplashDialog(parent=self).exec()

    # ── Window lifecycle ──────────────────────────────────────────────────────

    def changeEvent(self, event) -> None:  # noqa: N802
        """Pause camera when window is minimised; resume when restored."""
        from PyQt6.QtCore import QEvent
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            cam = self._left.camera_widget
            if cam is not None:
                if self.isMinimized():
                    cam.pause_capture()
                else:
                    cam.resume_capture()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        # Stop camera thread before closing
        cam = self._left.camera_widget
        if cam is not None:
            cam.stop_capture()
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard(db_path: str = DB_PATH) -> None:
    """Convenience entry point — creates QApplication if needed and shows window."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = DashboardWindow(db_path=db_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_dashboard()
