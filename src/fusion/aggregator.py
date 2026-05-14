"""
aggregator.py
-------------
Five-minute rolling window aggregation across all three data streams.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Role
----
  - Background thread wakes every AGGREGATION_MINUTES (5 min)
  - Reads the last 5 minutes of raw rows from physio_readings, face_readings,
    and desktop_readings
  - Computes summary statistics (means, dominant category, total switches)
  - Calls LateFusion.score() to produce a single stress_index
  - Saves the result to aggregated_windows
  - Applies intervention trigger logic and sets intervention_triggered flag

Intervention trigger rules (from SPEC §5 + config.py):
  1. stress_index > STRESS_TRIGGER_THRESHOLD for 2 consecutive windows
  2. avg_valence < VALENCE_TRIGGER_THRESHOLD in the current window
  3. Cooldown: MIN_MINUTES_BETWEEN_NOTIFS between any two triggers

Usage
-----
  aggregator = Aggregator()
  aggregator.start()          # launches background thread
  ...
  aggregator.stop()           # waits up to 10 s for clean shutdown

  # Or use as context manager:
  with Aggregator() as agg:
      ...  # main loop
"""

import os
import sys
import time
import threading
from collections import Counter
from datetime import datetime, timedelta

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import (
    DB_PATH,
    AGGREGATION_MINUTES,
    STRESS_TRIGGER_THRESHOLD,
    VALENCE_TRIGGER_THRESHOLD,
    MIN_MINUTES_BETWEEN_NOTIFS,
)
from src.utils.db import (
    open_db,
    ensure_aggregated_windows_table,
    fetch_physio_window,
    fetch_face_window,
    fetch_desktop_window,
)
from src.fusion.late_fusion import LateFusion


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class Aggregator:
    """
    Five-minute aggregation thread.

    Creates one aggregated_windows row every AGGREGATION_MINUTES by summarising
    all raw readings that fell within the preceding window.

    Thread safety
    -------------
    All mutable state (_consecutive_stress, _last_intervention_time) is only
    accessed from the single background thread, so no locking is needed.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path   = db_path
        self._fusion    = LateFusion()
        self._stop      = threading.Event()
        self._thread: threading.Thread | None = None

        # Intervention trigger state (accessed only from background thread)
        self._consecutive_stress:    int              = 0
        self._last_intervention_time: datetime | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background aggregation thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="aggregator",
        )
        self._thread.start()
        print(f"[Aggregator] Started — window every {AGGREGATION_MINUTES} min.")

    def stop(self) -> None:
        """Signal the background thread to stop and wait up to 10 s."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        print("[Aggregator] Stopped.")

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        interval_s = AGGREGATION_MINUTES * 60
        next_run   = time.monotonic() + interval_s

        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_run:
                self._run_window()
                next_run = now + interval_s

            # Sleep until the next scheduled run or until stop is signalled.
            # Wake at most every 10 s so we react to stop() promptly.
            wait = max(1.0, min(10.0, next_run - time.monotonic()))
            self._stop.wait(timeout=wait)

    # ── One aggregation pass ──────────────────────────────────────────────────

    def _run_window(self) -> None:
        """Read, aggregate, score, and save one 5-minute window."""
        window_end   = datetime.now()
        window_start = window_end - timedelta(minutes=AGGREGATION_MINUTES)

        conn = open_db(self._db_path)
        try:
            ensure_aggregated_windows_table(conn)

            physio_rows  = fetch_physio_window(conn,  window_start, window_end)
            face_rows    = fetch_face_window(conn,    window_start, window_end)
            desktop_rows = fetch_desktop_window(conn, window_start, window_end)

            physio_agg  = self._aggregate_physio(physio_rows)
            face_agg    = self._aggregate_face(face_rows)
            desktop_agg = self._aggregate_desktop(desktop_rows)

            stress_index = self._fusion.score(physio_agg, face_agg, desktop_agg)
            triggered    = self._check_intervention(stress_index, face_agg)

            self._save_window(
                conn, window_start, window_end,
                physio_agg, face_agg, desktop_agg,
                stress_index, triggered,
            )
            self._print_summary(
                window_start, window_end,
                physio_agg, face_agg, desktop_agg,
                stress_index, triggered,
            )

        except Exception as exc:
            print(f"[Aggregator] ERROR in window pass: {exc}")

        finally:
            conn.close()

    # ── Aggregation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _aggregate_physio(rows: list) -> dict:
        hrs    = [r["heart_rate"] for r in rows if r["heart_rate"] is not None]
        rmssds = [r["rmssd"]      for r in rows if r["rmssd"]      is not None]
        lf_hfs = [r["lf_hf_ratio"]for r in rows if r["lf_hf_ratio"]is not None]
        return {
            "avg_hr":    _avg(hrs),
            "avg_rmssd": _avg(rmssds),
            "avg_lf_hf": _avg(lf_hfs),
        }

    @staticmethod
    def _aggregate_face(rows: list) -> dict:
        blinks   = [r["blink_rate"] for r in rows if r["blink_rate"] is not None]
        ears     = [r["mean_ear"]   for r in rows if r["mean_ear"]   is not None]
        valences = [r["valence"]    for r in rows if r["valence"]    is not None]
        arousals = [r["arousal"]    for r in rows if r["arousal"]    is not None]
        pitches  = [r["pitch_deg"]  for r in rows if r["pitch_deg"]  is not None]
        yaws     = [r["yaw_deg"]    for r in rows if r["yaw_deg"]    is not None]
        return {
            "avg_blink_rate": _avg(blinks),
            "avg_ear":        _avg(ears),
            "avg_valence":    _avg(valences),
            "avg_arousal":    _avg(arousals),
            "avg_pitch":      _avg(pitches),
            "avg_yaw":        _avg(yaws),
        }

    @staticmethod
    def _aggregate_desktop(rows: list) -> dict:
        activity = [r["activity_pct"]  for r in rows if r["activity_pct"]  is not None]
        idle     = [r["idle_seconds"]  for r in rows if r["idle_seconds"]  is not None]
        switches = [r["window_switches"]for r in rows if r["window_switches"]is not None]
        cats     = [r["app_category"]  for r in rows if r["app_category"]]

        dominant = Counter(cats).most_common(1)[0][0] if cats else None

        return {
            "avg_activity_pct":     _avg(activity),
            "avg_idle_seconds":     _avg(idle),
            "total_window_switches": sum(switches) if switches else None,
            "dominant_category":    dominant,
        }

    # ── Intervention trigger ──────────────────────────────────────────────────

    def _check_intervention(self, stress_index: float, face_agg: dict) -> bool:
        """
        Return True when an intervention should be triggered.

        Rules (evaluated in order):
          1. Cooldown: skip if MIN_MINUTES_BETWEEN_NOTIFS has not elapsed.
          2. stress_index above threshold for 2 consecutive windows.
          3. avg_valence below VALENCE_TRIGGER_THRESHOLD in this window.
        """
        # Rule 1 — cooldown
        if self._last_intervention_time is not None:
            elapsed_min = (
                datetime.now() - self._last_intervention_time
            ).total_seconds() / 60.0
            if elapsed_min < MIN_MINUTES_BETWEEN_NOTIFS:
                return False

        # Rule 2 — consecutive high stress
        if stress_index >= STRESS_TRIGGER_THRESHOLD:
            self._consecutive_stress += 1
        else:
            self._consecutive_stress = 0

        if self._consecutive_stress >= 2:
            self._consecutive_stress     = 0
            self._last_intervention_time = datetime.now()
            return True

        # Rule 3 — sustained negative valence
        valence = face_agg.get("avg_valence")
        if valence is not None and valence < VALENCE_TRIGGER_THRESHOLD:
            self._last_intervention_time = datetime.now()
            return True

        return False

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_window(
        conn,
        window_start: datetime,
        window_end:   datetime,
        physio:  dict,
        face:    dict,
        desktop: dict,
        stress_index: float,
        triggered:    bool,
    ) -> None:
        conn.execute("""
            INSERT INTO aggregated_windows (
                window_start,  window_end,
                avg_hr,        avg_rmssd,       avg_lf_hf,
                avg_blink_rate, avg_ear,
                avg_valence,   avg_arousal,
                avg_pitch,     avg_yaw,
                dominant_category,
                avg_activity_pct, avg_idle_seconds, total_window_switches,
                stress_index,  intervention_triggered
            ) VALUES (
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?,
                ?, ?, ?,
                ?, ?
            )
        """, (
            window_start.strftime("%Y-%m-%d %H:%M:%S"),
            window_end.strftime("%Y-%m-%d %H:%M:%S"),
            physio["avg_hr"],    physio["avg_rmssd"],   physio["avg_lf_hf"],
            face["avg_blink_rate"], face["avg_ear"],
            face["avg_valence"], face["avg_arousal"],
            face["avg_pitch"],   face["avg_yaw"],
            desktop["dominant_category"],
            desktop["avg_activity_pct"], desktop["avg_idle_seconds"],
            desktop["total_window_switches"],
            stress_index, int(triggered),
        ))
        conn.commit()

    # ── Console output ────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(
        t0: datetime, t1: datetime,
        physio: dict, face: dict, desktop: dict,
        stress_index: float,
        triggered: bool,
    ) -> None:
        hr_str = f"{physio['avg_hr']:.1f} BPM" if physio["avg_hr"] else "--"
        v_str  = f"{face['avg_valence']:+.2f}"  if face["avg_valence"] else "--"
        a_str  = f"{face['avg_arousal']:+.2f}"  if face["avg_arousal"] else "--"
        cat    = desktop["dominant_category"] or "--"

        bar_len = int(stress_index * 20)
        bar     = "#" * bar_len + "-" * (20 - bar_len)

        print(f"\n{'='*52}")
        print(f"  5-min window  {t0:%H:%M} – {t1:%H:%M}")
        print(f"{'='*52}")
        print(f"  HR:            {hr_str}")
        print(f"  Valence/Arousal: {v_str} / {a_str}")
        print(f"  Activity:      {desktop['avg_activity_pct']:.0f}%" if desktop["avg_activity_pct"] else "  Activity:      --")
        print(f"  App category:  {cat}")
        print(f"  Stress index:  [{bar}] {stress_index:.3f}")
        if triggered:
            print(f"  *** INTERVENTION TRIGGERED ***")
        print(f"{'='*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _avg(values: list) -> float | None:
    """Return the mean of a list of floats, or None if the list is empty."""
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the 5-minute aggregator standalone")
    parser.add_argument(
        "--once", action="store_true",
        help="Run one aggregation pass immediately and exit (for testing)",
    )
    args = parser.parse_args()

    agg = Aggregator()

    if args.once:
        print("[Aggregator] Running single aggregation pass (--once mode)...")
        agg._run_window()
        print("[Aggregator] Done.")
    else:
        print(f"[Aggregator] Running continuously. Press Ctrl+C to stop.")
        print(f"[Aggregator] Next window in {AGGREGATION_MINUTES} min.\n")
        try:
            with agg:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Aggregator] Stopped by Ctrl+C.")
