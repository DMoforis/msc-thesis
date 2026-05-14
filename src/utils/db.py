"""
db.py
-----
Shared SQLite database helpers.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

All modules that need database access should open connections via open_db()
so WAL journal mode is enabled consistently. WAL allows concurrent reads
while a single writer is active — essential because physio, face, desktop,
and aggregator all touch the same database simultaneously.
"""

import os
import sys
import sqlite3
from datetime import datetime

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import DB_PATH


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def open_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the shared SQLite database.

    WAL journal mode: concurrent readers do not block a writer and vice versa.
    Row factory: columns accessible by name (row['heart_rate']) as well as index.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# TABLE CREATION
# ─────────────────────────────────────────────────────────────────────────────

def ensure_baselines_table(conn: sqlite3.Connection) -> None:
    """Create the baselines table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS baselines (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            hr          REAL,
            rmssd       REAL,
            ear         REAL,
            blink_rate  REAL,
            valence     REAL,
            arousal     REAL
        )
    """)
    conn.commit()


def ensure_aggregated_windows_table(conn: sqlite3.Connection) -> None:
    """Create aggregated_windows table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aggregated_windows (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            window_start           TEXT NOT NULL,
            window_end             TEXT NOT NULL,
            -- Physio
            avg_hr                 REAL,
            avg_rmssd              REAL,
            avg_lf_hf              REAL,
            -- Face
            avg_blink_rate         REAL,
            avg_ear                REAL,
            avg_valence            REAL,
            avg_arousal            REAL,
            avg_pitch              REAL,
            avg_yaw                REAL,
            -- Desktop
            dominant_category      TEXT,
            avg_activity_pct       REAL,
            avg_idle_seconds       REAL,
            total_window_switches  INTEGER,
            -- Fusion output
            stress_index           REAL,
            -- Intervention flag
            intervention_triggered INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def ensure_interventions_table(conn: sqlite3.Connection) -> None:
    """Create interventions and user_feedback tables if they do not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT NOT NULL,
            trigger_reason TEXT,
            message        TEXT,
            stress_index   REAL,
            valence        REAL,
            arousal        REAL,
            delivered      INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            intervention_id INTEGER REFERENCES interventions(id),
            timestamp       TEXT NOT NULL,
            rating          INTEGER CHECK(rating BETWEEN 1 AND 5),
            comment         TEXT
        )
    """)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW FETCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_physio_window(
    conn: sqlite3.Connection,
    start: datetime,
    end: datetime,
) -> list:
    """Return physio_readings rows in the half-open interval [start, end)."""
    return conn.execute("""
        SELECT * FROM physio_readings
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (_fmt(start), _fmt(end))).fetchall()


def fetch_face_window(
    conn: sqlite3.Connection,
    start: datetime,
    end: datetime,
) -> list:
    """Return face_readings rows in the half-open interval [start, end)."""
    return conn.execute("""
        SELECT * FROM face_readings
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (_fmt(start), _fmt(end))).fetchall()


def fetch_desktop_window(
    conn: sqlite3.Connection,
    start: datetime,
    end: datetime,
) -> list:
    """Return desktop_readings rows in the half-open interval [start, end)."""
    return conn.execute("""
        SELECT * FROM desktop_readings
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (_fmt(start), _fmt(end))).fetchall()


# ─────────────────────────────────────────────────────────────────────────────

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")
