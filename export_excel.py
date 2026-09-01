"""
export_excel.py
---------------
Export stress monitoring data to Excel for statistical analysis.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Sheets produced
---------------
  Raw Physio      — all physio_readings rows (HR + HRV metrics)
  Raw Face        — all face_readings rows (EAR, blink rate, head pose, VA)
  Raw Desktop     — all desktop_readings rows (app category, idle, activity)
  5-Min Windows   — aggregated_windows (primary thesis analysis sheet)
  Interventions   — all recommendations shown + delivery status
  LLM Comparison  — side-by-side Llama vs Qwen responses for thesis evaluation
  Correlations    — Pearson correlation matrix of key numeric variables
  Summary Stats   — per-column descriptive statistics (mean, std, min, max, …)

Usage
-----
  python export_excel.py                    # export all data
  python export_excel.py --date 2026-05-01  # export one day
  python export_excel.py --last 7           # export last 7 days

Output: data/exports/stress_analysis_YYYY-MM-DD.xlsx
        (directory is created automatically)

Dependencies: pip install pandas openpyxl
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime, timedelta

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import DB_PATH, EXPORT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _require_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        print(
            "[Export] ERROR: pandas is not installed.\n"
            "[Export]   Install: pip install pandas openpyxl"
        )
        sys.exit(1)


def _load_table(
    conn: sqlite3.Connection,
    table: str,
    date_filter: str | None,
    last_days: int | None,
) -> "pd.DataFrame":
    """Load a table with optional date filtering into a DataFrame."""
    import pandas as pd

    where_clause = ""
    params: list = []

    if date_filter:
        where_clause = "WHERE timestamp >= ? AND timestamp < ?"
        params = [f"{date_filter} 00:00:00", f"{date_filter} 23:59:59"]
    elif last_days:
        cutoff = (datetime.now() - timedelta(days=last_days)).strftime("%Y-%m-%d %H:%M:%S")
        where_clause = "WHERE timestamp >= ?"
        params = [cutoff]

    # Check if table exists (optional tables may not yet be in the DB)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()

    if not exists:
        print(f"[Export] Table '{table}' does not exist — empty sheet will be created.")
        return pd.DataFrame()

    query = f"SELECT * FROM {table} {where_clause} ORDER BY timestamp"
    return pd.read_sql_query(query, conn, params=params)


def _load_aggregated(
    conn: sqlite3.Connection,
    date_filter: str | None,
    last_days: int | None,
) -> "pd.DataFrame":
    """Load aggregated_windows with optional date filtering (uses window_start)."""
    import pandas as pd

    table = "aggregated_windows"
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()

    if not exists:
        return pd.DataFrame()

    where_clause = ""
    params: list = []

    if date_filter:
        where_clause = "WHERE window_start >= ? AND window_start < ?"
        params = [f"{date_filter} 00:00:00", f"{date_filter} 23:59:59"]
    elif last_days:
        cutoff = (datetime.now() - timedelta(days=last_days)).strftime("%Y-%m-%d %H:%M:%S")
        where_clause = "WHERE window_start >= ?"
        params = [cutoff]

    query = f"SELECT * FROM {table} {where_clause} ORDER BY window_start"
    return pd.read_sql_query(query, conn, params=params)


def _load_llm_evaluation() -> "pd.DataFrame":
    """Load the most recent llm_evaluation_*.xlsx from data/exports/, if any.

    Returns an empty DataFrame silently when no evaluation file is found.
    The evaluation file is produced by src/evaluation/llm_evaluator.py.
    """
    import glob
    import pandas as pd

    pattern = os.path.join(EXPORT_DIR, "llm_evaluation_*.xlsx")
    files   = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    latest = files[-1]
    try:
        df = pd.read_excel(latest, engine="openpyxl")
        print(f"[Export] LLM evaluation loaded from: {os.path.basename(latest)} ({len(df)} rows)")
        return df
    except Exception as exc:
        print(f"[Export] Warning: could not load LLM evaluation file: {exc}")
        return pd.DataFrame()


def _load_llm_comparisons(
    conn: sqlite3.Connection,
    date_filter: str | None,
    last_days: int | None,
) -> "pd.DataFrame":
    """Load llm_comparisons with the thesis-relevant columns only.

    Excludes the raw context_json column (stored for debugging, not analysis).
    """
    df = _load_table(conn, "llm_comparisons", date_filter, last_days)
    if df.empty:
        return df

    # Select the columns the thesis evaluation cares about; ignore id/context_json.
    display_cols = [
        "timestamp", "trigger_reason",
        "llama_response", "qwen_response",
        "llama_latency_ms", "qwen_latency_ms",
        "stress_index",
    ]
    available = [c for c in display_cols if c in df.columns]
    return df[available]


def _compute_correlations(agg_df: "pd.DataFrame") -> "pd.DataFrame":
    """Pearson correlation matrix of the key numeric columns in aggregated_windows."""
    key_cols = [
        "avg_hr", "avg_rmssd", "avg_lf_hf",
        "avg_blink_rate", "avg_ear",
        "avg_valence", "avg_arousal",
        "avg_pitch", "avg_yaw",
        "avg_activity_pct", "avg_idle_seconds", "total_window_switches",
        "stress_index",
    ]
    available = [c for c in key_cols if c in agg_df.columns]
    if len(available) < 2 or agg_df.empty:
        import pandas as pd
        return pd.DataFrame({"note": ["Not enough data for correlations"]})
    return agg_df[available].corr(method="pearson").round(3)


def _compute_summary(agg_df: "pd.DataFrame") -> "pd.DataFrame":
    """Descriptive statistics for the key numeric columns."""
    import pandas as pd

    key_cols = [
        "avg_hr", "avg_rmssd", "avg_lf_hf",
        "avg_blink_rate", "avg_ear",
        "avg_valence", "avg_arousal",
        "avg_activity_pct", "avg_idle_seconds", "total_window_switches",
        "stress_index",
    ]
    available = [c for c in key_cols if c in agg_df.columns]
    if not available or agg_df.empty:
        return pd.DataFrame({"note": ["No aggregated data found"]})

    return agg_df[available].describe().round(3)


def _style_and_write(
    writer: "pd.ExcelWriter",
    df: "pd.DataFrame",
    sheet_name: str,
) -> None:
    """Write a DataFrame to an Excel sheet with auto column widths."""
    df.to_excel(writer, sheet_name=sheet_name, index=(sheet_name in ("Correlations", "Summary Stats")))

    ws = writer.sheets[sheet_name]

    # Auto-fit column widths (openpyxl)
    for col_cells in ws.columns:
        max_len = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 40)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPORT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def export(
    date_filter: str | None = None,
    last_days:   int | None = None,
) -> str:
    """
    Export data to an Excel file and return the output path.

    Parameters
    ----------
    date_filter : str | None  — 'YYYY-MM-DD' to export a single day
    last_days   : int | None  — export last N calendar days
    """
    pd = _require_pandas()

    # Verify openpyxl is available
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[Export] ERROR: openpyxl is not installed.\n"
              "[Export]   Install: pip install openpyxl")
        sys.exit(1)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    date_tag = date_filter or datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(EXPORT_DIR, f"stress_analysis_{date_tag}.xlsx")

    conn = sqlite3.connect(DB_PATH)

    try:
        print(f"[Export] Reading data from {DB_PATH} ...")

        df_physio   = _load_table(conn, "physio_readings",  date_filter, last_days)
        df_face     = _load_table(conn, "face_readings",    date_filter, last_days)
        df_desktop  = _load_table(conn, "desktop_readings", date_filter, last_days)
        df_agg      = _load_aggregated(conn, date_filter, last_days)
        df_interv   = _load_table(conn, "interventions",    date_filter, last_days)
        df_llm_cmp  = _load_llm_comparisons(conn, date_filter, last_days)
        df_llm_eval = _load_llm_evaluation()
        df_corr     = _compute_correlations(df_agg)
        df_summary  = _compute_summary(df_agg)

        print(f"[Export] Rows: physio={len(df_physio)}, face={len(df_face)}, "
              f"desktop={len(df_desktop)}, windows={len(df_agg)}, "
              f"interventions={len(df_interv)}, llm_comparisons={len(df_llm_cmp)}, "
              f"llm_evaluation={len(df_llm_eval)}")

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            _style_and_write(writer, df_physio,   "Raw Physio")
            _style_and_write(writer, df_face,     "Raw Face")
            _style_and_write(writer, df_desktop,  "Raw Desktop")
            _style_and_write(writer, df_agg,      "5-Min Windows")
            _style_and_write(writer, df_interv,   "Interventions")
            _style_and_write(writer, df_llm_cmp,  "LLM Comparison")
            if not df_llm_eval.empty:
                _style_and_write(writer, df_llm_eval, "LLM Evaluation")
            _style_and_write(writer, df_corr,     "Correlations")
            _style_and_write(writer, df_summary,  "Summary Stats")

        print(f"[Export] Saved to {out_path}")
        return out_path

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export stress monitoring data to Excel."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Export data for a specific calendar day.",
    )
    group.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Export data from the last N calendar days.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = export(date_filter=args.date, last_days=args.last)
    print(f"\nDone — open with Excel: {path}")
