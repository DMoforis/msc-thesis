"""
test_db.py
----------
Tests for shared SQLite database helpers in src/utils/db.py.
"""
from datetime import datetime

import pytest

from src.utils.db import (
    open_db,
    ensure_aggregated_windows_table,
    ensure_interventions_table,
    ensure_baselines_table,
    fetch_physio_window,
    fetch_desktop_window,
    data_health_check,
)


def test_tables_created(tmp_db):
    """ensure_* helpers should create the expected tables."""
    conn = open_db(tmp_db)
    ensure_aggregated_windows_table(conn)
    ensure_interventions_table(conn)
    ensure_baselines_table(conn)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "aggregated_windows" in tables
    assert "interventions"       in tables
    assert "user_feedback"       in tables
    assert "baselines"           in tables


def test_save_physio_reading(tmp_db):
    """save_reading() should persist a physio row with the correct values."""
    from src.physio.rppg_monitor import init_database, save_reading

    conn = init_database(tmp_db)
    save_reading(conn,
                 heart_rate=72.5, rmssd=None, lf_hf=None,
                 sdnn=None, signal_quality=0.82, window_type="hr_10s")

    rows = conn.execute(
        "SELECT heart_rate, signal_quality, window_type FROM physio_readings"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == 72.5
    assert abs(rows[0][1] - 0.82) < 0.001
    assert rows[0][2] == "hr_10s"


def test_save_face_reading(tmp_db):
    """A face_readings row inserted directly should be retrievable."""
    conn = open_db(tmp_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            blink_rate REAL, mean_ear REAL,
            pitch_deg REAL, yaw_deg REAL,
            valence REAL, arousal REAL
        )
    """)
    conn.commit()
    conn.execute(
        "INSERT INTO face_readings "
        "(timestamp, blink_rate, mean_ear, pitch_deg, yaw_deg, valence, arousal) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("2026-05-18 10:00:00", 14.5, 0.30, 2.5, -1.0, 0.20, 0.10),
    )
    conn.commit()

    rows = conn.execute("SELECT * FROM face_readings").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["mean_ear"] == pytest.approx(0.30, abs=0.001)
    assert rows[0]["valence"]  == pytest.approx(0.20, abs=0.001)


def test_timestamp_normalisation(tmp_db):
    """fetch_desktop_window() must find rows with ISO 8601 T-separator timestamps
    written by Flutter (e.g. '2026-05-18T10:02:30')."""
    conn = open_db(tmp_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS desktop_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, active_window TEXT, app_category TEXT,
            idle_seconds REAL, activity_pct REAL, window_switches INTEGER
        )
    """)
    conn.commit()
    conn.execute(
        "INSERT INTO desktop_readings "
        "(timestamp, app_category, activity_pct, idle_seconds, window_switches) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-05-18T10:02:30", "IDE/Terminal", 75.0, 10.0, 5),
    )
    conn.commit()

    start = datetime(2026, 5, 18, 10, 0, 0)
    end   = datetime(2026, 5, 18, 10, 5, 0)
    rows  = fetch_desktop_window(conn, start, end)
    conn.close()

    assert len(rows) == 1
    assert rows[0]["app_category"] == "IDE/Terminal"


def test_timestamp_normalisation_excludes_outside(tmp_db):
    """T-format rows outside the window bounds must NOT be returned."""
    conn = open_db(tmp_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS desktop_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, active_window TEXT, app_category TEXT,
            idle_seconds REAL, activity_pct REAL, window_switches INTEGER
        )
    """)
    conn.commit()
    conn.execute(
        "INSERT INTO desktop_readings "
        "(timestamp, app_category, activity_pct, idle_seconds, window_switches) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-05-18T09:55:00", "Browser", 50.0, 5.0, 2),
    )
    conn.commit()

    start = datetime(2026, 5, 18, 10, 0, 0)
    end   = datetime(2026, 5, 18, 10, 5, 0)
    rows  = fetch_desktop_window(conn, start, end)
    conn.close()

    assert len(rows) == 0


def test_data_health_check(tmp_db, capsys):
    """data_health_check() should run without error and print the report header."""
    data_health_check(tmp_db)
    captured = capsys.readouterr()
    assert "Data Quality Report" in captured.out
    assert "physio_readings"     in captured.out
