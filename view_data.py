"""
view_data.py
------------
Quick database summary viewer for the Multimodal Stress Detection System.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

What this script does:
  Reads all three tables from stress_monitor.db and prints a formatted
  summary to the terminal — useful for verifying data collection and
  for sharing output with your thesis supervisor.

Usage:
  python view_data.py            # show last 5 rows from each table
  python view_data.py --rows 10  # show last 10 rows from each table
  python view_data.py --all      # show all rows
"""

import sqlite3
import os
import argparse
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(PROJECT_ROOT, "data", "stress_monitor.db")


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def print_table(rows: list, columns: list):
    """
    Print rows as a simple aligned text table.
    columns: list of (header, key, width) tuples
    """
    if not rows:
        print("  No data recorded yet.")
        return

    # Print column headers
    header = "  " + "  ".join(
        str(col[0]).ljust(col[2]) for col in columns
    )
    print(header)
    print("  " + "─" * (sum(c[2] for c in columns) + 2 * len(columns)))

    # Print each row
    for row in rows:
        line = "  " + "  ".join(
            str(row[col[1]] if row[col[1]] is not None else "—")
            .ljust(col[2])[:col[2]]   # truncate if too long
            for col in columns
        )
        print(line)


def print_stats(conn: sqlite3.Connection, table: str, numeric_cols: list):
    """
    Print basic statistics (mean, min, max) for numeric columns in a table.
    """
    cursor = conn.cursor()
    stats_parts = ", ".join(
        f"ROUND(AVG({c}),2), ROUND(MIN({c}),2), ROUND(MAX({c}),2)"
        for c in numeric_cols
    )
    cursor.execute(f"SELECT COUNT(*), {stats_parts} FROM {table}")
    row = cursor.fetchone()

    if not row or row[0] == 0:
        return

    print(f"\n  Statistics ({row[0]} readings):")
    idx = 1
    for col in numeric_cols:
        avg, mn, mx = row[idx], row[idx+1], row[idx+2]
        print(f"    {col:<25} avg={avg}  min={mn}  max={mx}")
        idx += 3


# ─────────────────────────────────────────────────────────────────────────────
# TABLE VIEWERS
# One function per table — each knows its own column layout
# ─────────────────────────────────────────────────────────────────────────────

def view_physio(conn: sqlite3.Connection, limit: int):
    """Display physiological readings from PoC #1 (rPPG heart rate module)."""
    print_header("PoC #1 — Physiological Readings (rPPG Heart Rate)")

    query = f"""
        SELECT timestamp, heart_rate, rmssd, lf_hf_ratio, signal_quality
        FROM physio_readings
        ORDER BY id DESC
        LIMIT {limit}
    """
    rows = [dict(row) for row in conn.execute(query).fetchall()]
    rows.reverse()   # show oldest first within the selection

    columns = [
        # (header,         db key,          display width)
        ("Timestamp",      "timestamp",      20),
        ("HR (BPM)",       "heart_rate",      9),
        ("RMSSD (ms)",     "rmssd",          11),
        ("LF/HF",          "lf_hf_ratio",     7),
        ("Quality",        "signal_quality",  8),
    ]
    print_table(rows, columns)
    print_stats(conn, "physio_readings",
                ["heart_rate", "rmssd", "lf_hf_ratio", "signal_quality"])


def view_face(conn: sqlite3.Connection, limit: int):
    """Display facial landmark readings from PoC #2 (face mesh module)."""
    print_header("PoC #2 — Facial Landmark Readings (EAR + Head Pose)")

    query = f"""
        SELECT timestamp, blink_rate, mean_ear,
               pitch_deg, yaw_deg, roll_deg, face_detected_pct
        FROM face_readings
        ORDER BY id DESC
        LIMIT {limit}
    """
    rows = [dict(row) for row in conn.execute(query).fetchall()]
    rows.reverse()

    columns = [
        ("Timestamp",      "timestamp",        20),
        ("Blinks/min",     "blink_rate",        10),
        ("Mean EAR",       "mean_ear",           9),
        ("Pitch°",         "pitch_deg",          7),
        ("Yaw°",           "yaw_deg",            6),
        ("Roll°",          "roll_deg",           6),
        ("Face %",         "face_detected_pct",  7),
    ]
    print_table(rows, columns)
    print_stats(conn, "face_readings",
                ["blink_rate", "mean_ear", "pitch_deg", "yaw_deg"])


def view_desktop(conn: sqlite3.Connection, limit: int):
    """Display desktop context readings from PoC #3 (Flutter desktop app)."""
    print_header("PoC #3 — Desktop Context Readings (Activity Monitor)")

    # Detect which schema version exists — old (keystrokes_per_min) or
    # new (activity_pct) — so the viewer works with either version
    cursor = conn.execute("PRAGMA table_info(desktop_readings)")
    col_names = [row[1] for row in cursor.fetchall()]
    has_activity_pct = "activity_pct" in col_names

    if has_activity_pct:
        select = "timestamp, active_window, app_category, idle_seconds, activity_pct, window_switches"
        columns = [
            ("Timestamp",  "timestamp",       20),
            ("Category",   "app_category",    14),
            ("Idle (s)",   "idle_seconds",     9),
            ("Active %",   "activity_pct",     9),
            ("Switches",   "window_switches",  9),
            ("Window",     "active_window",   25),
        ]
        stat_cols = ["idle_seconds", "activity_pct", "window_switches"]
    else:
        # Old schema with keystrokes/clicks columns
        select = "timestamp, active_window, app_category, keystrokes_per_min, clicks_per_min, idle_seconds, window_switches"
        columns = [
            ("Timestamp",  "timestamp",           20),
            ("Category",   "app_category",        14),
            ("Keys/min",   "keystrokes_per_min",  10),
            ("Clicks/min", "clicks_per_min",       9),
            ("Idle (s)",   "idle_seconds",         9),
            ("Switches",   "window_switches",      9),
            ("Window",     "active_window",       20),
        ]
        stat_cols = ["keystrokes_per_min", "clicks_per_min", "idle_seconds"]

    query = f"""
        SELECT {select}
        FROM desktop_readings
        ORDER BY id DESC
        LIMIT {limit}
    """
    rows = [dict(row) for row in conn.execute(query).fetchall()]
    rows.reverse()

    print_table(rows, columns)
    print_stats(conn, "desktop_readings", stat_cols)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="View stress_monitor.db data in the terminal"
    )
    parser.add_argument("--rows", type=int, default=5,
                        help="Number of most recent rows to show per table (default: 5)")
    parser.add_argument("--all",  action="store_true",
                        help="Show all rows (overrides --rows)")
    args = parser.parse_args()

    limit = 999999 if args.all else args.rows

    # Check database exists
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found at: {DB_PATH}")
        print("[ERROR] Run one of the PoC scripts first to create it.")
        return

    print(f"\n  Multimodal Stress Detection — Data Summary")
    print(f"  Database: {DB_PATH}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Use row_factory so we can access columns by name (like a dict)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # Check which tables exist — not all PoCs may have run yet
        tables = {
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

        if "physio_readings"  in tables: view_physio(conn,  limit)
        else: print("\n[INFO] physio_readings table not found — run PoC #1 first.")

        if "face_readings"    in tables: view_face(conn,    limit)
        else: print("\n[INFO] face_readings table not found — run PoC #2 first.")

        if "desktop_readings" in tables: view_desktop(conn, limit)
        else: print("\n[INFO] desktop_readings table not found — run PoC #3 first.")

    finally:
        conn.close()

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()