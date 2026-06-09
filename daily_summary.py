"""
daily_summary.py
----------------
Human-readable daily monitoring summary.

Usage:
  python daily_summary.py                      # today
  python daily_summary.py --date 2026-05-18    # specific date
"""

import os
import sys
import re
import argparse
from datetime import date, datetime
from io import StringIO

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.db import open_db
from src.utils.config import DB_PATH, EXPORT_DIR
from src.utils.emotion_labels import get_emotion_label

# ── Colorama ──────────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False
    class _Stub:
        def __getattr__(self, _): return ""
    Fore = Style = _Stub()

def _c(code, text): return f"{code}{text}{Style.RESET_ALL}" if _COLOR else str(text)
def green(t):  return _c(Fore.GREEN,   t)
def yellow(t): return _c(Fore.YELLOW,  t)
def red(t):    return _c(Fore.RED,     t)
def cyan(t):   return _c(Fore.CYAN,    t)
def bold(t):   return _c(Style.BRIGHT, t)
def dim(t):    return _c(Style.DIM,    t)

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# ── Dual-output writer ────────────────────────────────────────────────────────

class _Out:
    """Writes colored output to stdout and plain text to an internal buffer."""

    def __init__(self):
        self._buf = StringIO()

    def p(self, text: str = "") -> None:
        print(text)
        self._buf.write(_ANSI_RE.sub("", text) + "\n")

    def plain(self) -> str:
        return self._buf.getvalue()


# ── Formatting helpers ────────────────────────────────────────────────────────

_W = 65   # output line width

def _header(out: _Out, title: str) -> None:
    out.p(bold(f"\n {'='*_W}"))
    out.p(bold(f"  {title}"))
    out.p(bold(f" {'='*_W}"))

def _section(out: _Out, title: str) -> None:
    out.p(f"\n  {cyan(bold(title))}")
    out.p(f"  {'-'*(_W-2)}")

def _row(out: _Out, label: str, value: str, width: int = 22) -> None:
    out.p(f"  {label:<{width}}: {value}")

def _blank(out: _Out) -> None:
    out.p()


def _fmt_minutes(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} min"
    h, m = divmod(int(minutes), 60)
    return f"{h} h {m} min"


def _fmt_seconds(seconds: float) -> str:
    if seconds is None:
        return "--"
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} sec" if m else f"{int(s)} sec"


def _fmt_time(ts_str: str) -> str:
    """Extract HH:MM from a timestamp string."""
    try:
        return ts_str[11:16]
    except Exception:
        return ts_str


# ── Interpretation labels ─────────────────────────────────────────────────────

def _stress_label(idx: float) -> str:
    if idx < 0.30:  return green("Low")
    if idx < 0.60:  return yellow("Moderate")
    return red("High")

def _stress_val(idx: float) -> str:
    txt = f"{idx:.3f}"
    if idx < 0.30:  return green(txt)
    if idx < 0.60:  return yellow(txt)
    return red(txt)

def _hr_val(hr: float) -> str:
    txt = f"{hr:.1f} BPM"
    if 60.0 <= hr <= 90.0:  return green(txt)
    if hr > 100.0:          return red(txt)
    return yellow(txt)

def _sqi_label(sqi: float) -> str:
    txt = f"{sqi:.2f} / 1.00"
    if sqi >= 0.70:  return green(f"{txt}  [ Good ]")
    if sqi >= 0.40:  return yellow(f"{txt}  [ Fair ]")
    return red(f"{txt}  [ Poor — results less reliable ]")

def _rmssd_val(v: float) -> str:
    txt = f"{v:.1f} ms"
    if v >= 30.0:  return green(txt)
    if v >= 20.0:  return yellow(txt)
    return red(txt)

def _valence_label(v: float) -> str:
    if v >  0.20:  return green("Positive")
    if v < -0.20:  return red("Negative")
    return yellow("Neutral")

def _arousal_label(a: float) -> str:
    if   a >  0.30:  return yellow("High")
    elif a < -0.20:  return dim("Low")
    return green("Moderate")

def _quadrant_label(valence: float, arousal: float) -> str:
    v = "pos" if valence >  0.20 else ("neg" if valence < -0.20 else "neu")
    a = "hi"  if arousal >  0.30 else ("lo"  if arousal < -0.20 else "mid")
    table = {
        ("hi",  "pos"): (green,  "Engaged / excited"),
        ("hi",  "neu"): (yellow, "Alert / activated"),
        ("hi",  "neg"): (red,    "Stressed / anxious"),
        ("mid", "pos"): (green,  "Pleasantly focused"),
        ("mid", "neu"): (green,  "Balanced / attentive"),
        ("mid", "neg"): (yellow, "Mildly stressed"),
        ("lo",  "pos"): (green,  "Relaxed / calm"),
        ("lo",  "neu"): (dim,    "Neutral / detached"),
        ("lo",  "neg"): (red,    "Fatigued / disengaged"),
    }
    color_fn, label = table.get((a, v), (dim, "Unknown"))
    return color_fn(label)


# ── Data fetch ────────────────────────────────────────────────────────────────

def _fetch(conn, sql: str, params: tuple = ()):
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []

def _fetchone(conn, sql: str, params: tuple = ()):
    try:
        return conn.execute(sql, params).fetchone()
    except Exception:
        return None


def fetch_day_data(conn, date_str: str) -> dict:
    pat = f"{date_str}%"

    # ── Aggregated windows ────────────────────────────────────────────────────
    windows = _fetch(conn, """
        SELECT window_start, window_end, avg_hr, avg_rmssd, avg_lf_hf,
               avg_valence, avg_arousal, avg_ear,
               avg_activity_pct, total_window_switches, avg_idle_seconds,
               dominant_category, stress_index, intervention_triggered
        FROM aggregated_windows
        WHERE window_start LIKE ?
        ORDER BY window_start
    """, (pat,))

    # ── Session bookends from raw physio (fallback for monitoring time) ───────
    physio_span = _fetchone(conn, """
        SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
        FROM physio_readings WHERE timestamp LIKE ? AND window_type = 'hr_10s'
    """, (pat,))

    # ── Physio aggregates ─────────────────────────────────────────────────────
    physio_hr = _fetchone(conn, """
        SELECT AVG(heart_rate) AS avg_hr,
               AVG(signal_quality) AS avg_sqi,
               COUNT(*) AS n
        FROM physio_readings
        WHERE timestamp LIKE ? AND window_type = 'hr_10s'
    """, (pat,))

    physio_hrv = _fetchone(conn, """
        SELECT AVG(rmssd)  AS avg_rmssd,
               COUNT(*)    AS total_hrv,
               SUM(CASE WHEN rmssd IS NOT NULL THEN 1 ELSE 0 END) AS valid_rmssd
        FROM physio_readings
        WHERE timestamp LIKE ? AND window_type = 'hrv_5min'
    """, (pat,))

    # ── Interventions ─────────────────────────────────────────────────────────
    interventions = _fetch(conn, """
        SELECT timestamp, trigger_reason, message, stress_index
        FROM interventions
        WHERE timestamp LIKE ?
        ORDER BY timestamp
    """, (pat,))

    # ── Desktop: longest idle from raw readings ───────────────────────────────
    longest_idle = _fetchone(conn, """
        SELECT MAX(idle_seconds) AS max_idle
        FROM desktop_readings
        WHERE REPLACE(timestamp, 'T', ' ') LIKE ?
    """, (pat,))

    return {
        "date_str":     date_str,
        "windows":      windows,
        "physio_span":  physio_span,
        "physio_hr":    physio_hr,
        "physio_hrv":   physio_hrv,
        "interventions": interventions,
        "longest_idle": longest_idle,
    }


# ── Section printers ──────────────────────────────────────────────────────────

def print_session_overview(out: _Out, data: dict) -> None:
    _section(out, "SESSION OVERVIEW")

    windows = data["windows"]
    n_windows = len(windows)

    if n_windows == 0:
        _row(out, "5-min windows", red("0 — no data found for this date"))
        return

    # Monitoring time from aggregated window bookends
    first_ts = windows[0]["window_start"]
    last_ts  = windows[-1]["window_end"]
    try:
        t0 = datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(last_ts,  "%Y-%m-%d %H:%M:%S")
        duration_min = (t1 - t0).total_seconds() / 60.0
        span_str = f"{_fmt_minutes(duration_min)}  ({t0:%H:%M} to {t1:%H:%M})"
    except Exception:
        span_str = "--"

    n_triggered = sum(1 for w in windows if w["intervention_triggered"])
    notif_color = green if n_triggered == 0 else (yellow if n_triggered <= 2 else red)

    _row(out, "Monitoring time",   span_str)
    _row(out, "5-min windows",     str(n_windows))
    _row(out, "Interventions",     notif_color(str(n_triggered)))


def print_stress_summary(out: _Out, data: dict) -> None:
    _section(out, "STRESS SUMMARY")

    windows = data["windows"]
    scored  = [w for w in windows if w["stress_index"] is not None]

    if not scored:
        _row(out, "Stress index", dim("-- (no scored windows)"))
        return

    indices = [w["stress_index"] for w in scored]
    avg_idx = sum(indices) / len(indices)
    _row(out, "Average index",
         f"{_stress_val(avg_idx)}  [ {_stress_label(avg_idx)} ]")

    peak_w = max(scored, key=lambda w: w["stress_index"])
    low_w  = min(scored, key=lambda w: w["stress_index"])

    peak_cat    = peak_w["dominant_category"] or "Unknown"
    low_cat     = low_w["dominant_category"]  or "Unknown"
    peak_idx    = red(f"{peak_w['stress_index']:.3f}")
    low_idx     = green(f"{low_w['stress_index']:.3f}")

    _row(out, "Peak stress",
         f"{_fmt_time(peak_w['window_start'])}  |  "
         f"{peak_cat:<24}  (index: {peak_idx})")
    _row(out, "Most relaxed",
         f"{_fmt_time(low_w['window_start'])}  |  "
         f"{low_cat:<24}  (index: {low_idx})")


def print_physiological(out: _Out, data: dict) -> None:
    _section(out, "PHYSIOLOGICAL")

    hr_row = data["physio_hr"]
    hrv_row = data["physio_hrv"]

    if hr_row is None or hr_row["n"] == 0:
        _row(out, "Heart rate", dim("-- (no rPPG readings)"))
        _row(out, "Signal quality", dim("--"))
    else:
        avg_hr  = hr_row["avg_hr"]
        avg_sqi = hr_row["avg_sqi"]
        _row(out, "Average heart rate", _hr_val(avg_hr))
        _row(out, "Signal quality",     _sqi_label(avg_sqi))

    if hrv_row is None or hrv_row["total_hrv"] == 0:
        _row(out, "RMSSD", dim("-- (no HRV windows)"))
    else:
        valid = hrv_row["valid_rmssd"] or 0
        total = hrv_row["total_hrv"]
        if valid == 0:
            _row(out, "RMSSD",
                 yellow(f"-- (0 / {total} windows had sufficient signal)"))
        else:
            avg_rmssd = hrv_row["avg_rmssd"]
            coverage  = f"({valid} / {total} HRV windows valid)"
            _row(out, "RMSSD",
                 f"{_rmssd_val(avg_rmssd)}  {dim(coverage)}")


def print_emotional(out: _Out, data: dict) -> None:
    _section(out, "EMOTIONAL STATE")

    windows = data["windows"]
    vals = [w["avg_valence"] for w in windows if w["avg_valence"] is not None]
    arrs = [w["avg_arousal"] for w in windows if w["avg_arousal"] is not None]

    if not vals and not arrs:
        _row(out, "Valence / Arousal", dim("-- (face data unavailable)"))
        return

    avg_v = sum(vals) / len(vals) if vals else None
    avg_a = sum(arrs) / len(arrs) if arrs else None

    if avg_v is not None:
        _row(out, "Valence",
             f"{avg_v:+.3f}  [ {_valence_label(avg_v)} ]")
    else:
        _row(out, "Valence", dim("--"))

    if avg_a is not None:
        _row(out, "Arousal",
             f"{avg_a:+.3f}  [ {_arousal_label(avg_a)} ]")
    else:
        _row(out, "Arousal", dim("--"))

    if avg_v is not None and avg_a is not None:
        emo = get_emotion_label(avg_v, avg_a)
        _row(out, "Primary emotion",
             f"{bold(emo['primary'])}  ({emo['intensity']} intensity)")
        _row(out, "Predominant state", _quadrant_label(avg_v, avg_a))


def print_desktop(out: _Out, data: dict) -> None:
    _section(out, "DESKTOP ACTIVITY")

    windows = data["windows"]
    w_with_desktop = [w for w in windows if w["dominant_category"] is not None]

    if not w_with_desktop:
        _row(out, "App data", dim("-- (no desktop readings for this date)"))
        _row(out, "Tip", dim("ensure the Flutter desktop app was running"))
        return

    # Most used category + percentage
    from collections import Counter
    cat_counts = Counter(w["dominant_category"] for w in w_with_desktop)
    top_cat, top_n = cat_counts.most_common(1)[0]
    pct = round(top_n / len(w_with_desktop) * 100)
    _row(out, "Main activity", f"{top_cat}  ({pct}% of session)")

    # Activity level
    act_vals = [w["avg_activity_pct"] for w in windows if w["avg_activity_pct"] is not None]
    if act_vals:
        avg_act = sum(act_vals) / len(act_vals)
        act_color = green if avg_act >= 60 else (yellow if avg_act >= 35 else red)
        _row(out, "Activity level", act_color(f"{avg_act:.0f}%"))

    # Total window switches
    sw_vals = [w["total_window_switches"] for w in windows
               if w["total_window_switches"] is not None]
    if sw_vals:
        total_sw = sum(sw_vals)
        sw_rate  = total_sw / len(windows)   # per 5-min window
        sw_color = green if sw_rate <= 8 else (yellow if sw_rate <= 15 else red)
        _row(out, "Window switches", sw_color(f"{total_sw} total"))

    # Longest idle period from raw desktop_readings
    idle_row = data["longest_idle"]
    if idle_row and idle_row["max_idle"] is not None:
        idle_s = idle_row["max_idle"]
        idle_color = green if idle_s < 300 else (yellow if idle_s < 600 else red)
        _row(out, "Longest idle period", idle_color(_fmt_seconds(idle_s)))


def print_interventions(out: _Out, data: dict) -> None:
    interventions = data["interventions"]
    n = len(interventions)
    header = f"INTERVENTIONS  ({n} triggered)"
    _section(out, header)

    if n == 0:
        out.p(f"  {green('No interventions triggered - good session!')}")
        return

    for iv in interventions:
        ts      = _fmt_time(iv["timestamp"])
        reason  = iv["trigger_reason"] or "Unknown trigger"
        message = iv["message"] or ""

        # Shorten long messages to one line
        if len(message) > 70:
            message = message[:67] + "..."

        out.p(f"  {yellow(ts)}  |  {reason}")
        if message:
            out.p(f"           {dim(message)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily monitoring summary."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to summarise (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    args = parser.parse_args()

    date_str = args.date or date.today().isoformat()

    # Basic format validation
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"[Error] Invalid date format '{date_str}'. Use YYYY-MM-DD.")
        sys.exit(1)

    # Open database
    try:
        conn = open_db(args.db)
    except Exception as exc:
        print(f"[Error] Cannot open database: {exc}")
        sys.exit(1)

    out = _Out()

    _header(out, f"DAILY MONITORING SUMMARY  |  {date_str}")

    data = fetch_day_data(conn, date_str)
    conn.close()

    # Quick check: any data at all?
    has_data = (
        len(data["windows"]) > 0
        or (data["physio_hr"] and data["physio_hr"]["n"] > 0)
        or len(data["interventions"]) > 0
    )
    if not has_data:
        out.p(f"\n  {yellow('No monitoring data found for')} {bold(date_str)}")
        out.p(f"  {dim('Check that the system was running and the database path is correct.')}")
        out.p(f"  {dim(f'Database: {args.db}')}")
        _footer(out, date_str)
        return

    print_session_overview(out, data)
    print_stress_summary(out, data)
    print_physiological(out, data)
    print_emotional(out, data)
    print_desktop(out, data)
    print_interventions(out, data)
    _footer(out, date_str)

    _save_txt(out.plain(), date_str)


def _footer(out: _Out, date_str: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.p(bold(f"\n {'='*_W}"))
    out.p(dim(f"  Generated {now}"))
    out.p(bold(f" {'='*_W}\n"))


def _save_txt(plain_text: str, date_str: str) -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_path = os.path.join(EXPORT_DIR, f"daily_summary_{date_str}.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(plain_text)
        print(dim(f"  [Saved] {out_path}"))
    except Exception as exc:
        print(yellow(f"  [Warning] Could not save text file: {exc}"))


if __name__ == "__main__":
    main()
