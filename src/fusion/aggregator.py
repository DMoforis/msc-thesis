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
  - Applies multi-condition trigger logic and fires notifications

Intervention trigger rules (OR logic — any single condition is sufficient):
  1. high_stress      — stress_index > threshold for 2 consecutive windows
  2. disengagement    — low valence + low arousal + low activity, 2 windows
  3. negative_affect  — low valence + elevated arousal, 2 windows
  4. eye_strain       — blink_rate < 8/min for 2 consecutive windows
  5. prolonged_idle   — activity_pct < 15% for 3 consecutive windows
  6. positive_flow    — calm + engaged + positive mood, 2 windows;
                        30-min cooldown, once per session maximum

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
    MIN_MINUTES_BETWEEN_NOTIFS,
    DISENGAGEMENT_VALENCE_THRESHOLD,
    DISENGAGEMENT_AROUSAL_THRESHOLD,
    DISENGAGEMENT_ACTIVITY_MAX,
    NEGATIVE_AFFECT_VALENCE_THRESHOLD,
    NEGATIVE_AFFECT_AROUSAL_MIN,
    EYE_STRAIN_BLINK_THRESHOLD,
    IDLE_ACTIVITY_THRESHOLD,
    FLOW_STRESS_CEILING,
    FLOW_ACTIVITY_FLOOR,
    FLOW_VALENCE_FLOOR,
    MIN_MINUTES_BETWEEN_FLOW_NOTIFS,
    MAX_FLOW_NOTIFS_PER_SESSION,
    AGGREGATOR_VERBOSE,
)
from src.utils.db import (
    open_db,
    ensure_aggregated_windows_table,
    fetch_physio_window,
    fetch_face_window,
    fetch_desktop_window,
)
from src.fusion.late_fusion import LateFusion
from src.wellbeing.notifier import WindowsNotifier
from src.llm.recommender import WellbeingRecommender


# ── Notification titles per trigger type ─────────────────────────────────────
_TRIGGER_TITLES: dict[str, str] = {
    "high_stress":     "Stress Check",
    "disengagement":   "Time to Re-engage",
    "negative_affect": "Taking a Breath",
    "eye_strain":      "Eye Strain Alert",
    "prolonged_idle":  "Still With Us?",
    "positive_flow":   "You Are in the Zone!",
}


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
        self._db_path        = db_path
        self._fusion         = LateFusion()
        self._stop           = threading.Event()
        self._thread: threading.Thread | None = None
        self._session_start  = datetime.now()

        # Consecutive-window counters (accessed only from background thread)
        self._consecutive_stress          = 0
        self._consecutive_disengagement   = 0
        self._consecutive_negative_affect = 0
        self._consecutive_eye_strain      = 0
        self._consecutive_idle            = 0
        self._consecutive_flow            = 0

        # Cooldown / session-cap state
        self._last_intervention_time: datetime | None = None
        self._last_flow_notif_time:   datetime | None = None
        self._flow_notifs_sent: int = 0

        # Notification components (graceful degradation handled internally)
        self._notifier    = WindowsNotifier(db_path=self._db_path)
        self._recommender = WellbeingRecommender()

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

            # Per-window diagnostic output — gated by AGGREGATOR_VERBOSE in config.py.
            # Set AGGREGATOR_VERBOSE = True when debugging desktop data ingestion;
            # leave False during pilot sessions to keep the terminal clean.
            if AGGREGATOR_VERBOSE:
                print(f"[Aggregator] window {window_start:%H:%M}-{window_end:%H:%M} | "
                      f"physio={len(physio_rows)} face={len(face_rows)} desktop={len(desktop_rows)}")
                if desktop_rows:
                    r0 = desktop_rows[0]
                    print(f"[Aggregator]   desktop sample: ts={r0['timestamp']} "
                          f"cat={r0['app_category']} act={r0['activity_pct']:.0f}%")
                else:
                    print("[Aggregator]   desktop: no rows in window — desktop columns will be NULL")

            physio_agg  = self._aggregate_physio(physio_rows)
            face_agg    = self._aggregate_face(face_rows)
            desktop_agg = self._aggregate_desktop(desktop_rows)

            stress_index   = self._fusion.score(physio_agg, face_agg, desktop_agg)
            trigger_reason = self._check_interventions(stress_index, face_agg, desktop_agg)

            # intervention_triggered=1 only when the notifier actually delivered.
            # positive_flow may be suppressed by its own cooldown even if
            # trigger_reason is not None, so we use the return value of
            # _fire_intervention() rather than the presence of trigger_reason.
            delivered = False
            if trigger_reason:
                delivered = self._fire_intervention(
                    trigger_reason, stress_index, face_agg, desktop_agg
                )

            self._save_window(
                conn, window_start, window_end,
                physio_agg, face_agg, desktop_agg,
                stress_index, delivered,
            )
            self._print_summary(
                window_start, window_end,
                physio_agg, face_agg, desktop_agg,
                stress_index, trigger_reason,
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

    def _check_interventions(
        self,
        stress_index: float,
        face_agg:     dict,
        desktop_agg:  dict,
    ) -> str | None:
        """
        Evaluate all six trigger conditions and return the trigger reason
        string of the first condition that fires, or None if nothing fires.

        All consecutive counters are updated every call so they reflect the
        true run of windows meeting each condition — including windows where
        the standard cooldown blocked firing. After cooldown expires, the
        counter is already at threshold, so the trigger fires on the first
        eligible window rather than requiring another two-window run.

        positive_flow uses its own 30-minute cooldown and a once-per-session
        cap, evaluated independently of the standard cooldown.
        """
        now = datetime.now()

        standard_cooldown_ok = (
            self._last_intervention_time is None
            or (now - self._last_intervention_time).total_seconds() / 60.0
               >= MIN_MINUTES_BETWEEN_NOTIFS
        )

        # ── Extract signals ───────────────────────────────────────────────
        valence  = face_agg.get("avg_valence")
        arousal  = face_agg.get("avg_arousal")
        blinks   = face_agg.get("avg_blink_rate")
        activity = desktop_agg.get("avg_activity_pct")

        # ── Update all consecutive counters ───────────────────────────────

        # (1) high_stress
        if stress_index >= STRESS_TRIGGER_THRESHOLD:
            self._consecutive_stress += 1
        else:
            self._consecutive_stress = 0

        # (2) disengagement: low mood + low energy + low activity
        if (valence  is not None and valence  < DISENGAGEMENT_VALENCE_THRESHOLD
                and arousal  is not None and arousal  < DISENGAGEMENT_AROUSAL_THRESHOLD
                and activity is not None and activity < DISENGAGEMENT_ACTIVITY_MAX):
            self._consecutive_disengagement += 1
        else:
            self._consecutive_disengagement = 0

        # (3) negative_affect: low valence + elevated arousal (tense/anxious)
        if (valence is not None and valence < NEGATIVE_AFFECT_VALENCE_THRESHOLD
                and arousal is not None and arousal > NEGATIVE_AFFECT_AROUSAL_MIN):
            self._consecutive_negative_affect += 1
        else:
            self._consecutive_negative_affect = 0

        # (4) eye_strain: blink rate below healthy threshold
        if blinks is not None and blinks < EYE_STRAIN_BLINK_THRESHOLD:
            self._consecutive_eye_strain += 1
        else:
            self._consecutive_eye_strain = 0

        # (5) prolonged_idle: near-zero activity (needs 3 windows = 15 min)
        if activity is not None and activity < IDLE_ACTIVITY_THRESHOLD:
            self._consecutive_idle += 1
        else:
            self._consecutive_idle = 0

        # (6) positive_flow: calm + engaged + positive mood
        if (stress_index < FLOW_STRESS_CEILING
                and activity is not None and activity > FLOW_ACTIVITY_FLOOR
                and valence  is not None and valence  > FLOW_VALENCE_FLOOR):
            self._consecutive_flow += 1
        else:
            self._consecutive_flow = 0

        # ── Evaluate standard-cooldown triggers (priority order) ──────────
        if standard_cooldown_ok:
            trigger = None
            if self._consecutive_stress >= 2:
                trigger = "high_stress"
                self._consecutive_stress = 0
            elif self._consecutive_disengagement >= 2:
                trigger = "disengagement"
                self._consecutive_disengagement = 0
            elif self._consecutive_negative_affect >= 2:
                trigger = "negative_affect"
                self._consecutive_negative_affect = 0
            elif self._consecutive_eye_strain >= 2:
                trigger = "eye_strain"
                self._consecutive_eye_strain = 0
            elif self._consecutive_idle >= 3:
                trigger = "prolonged_idle"
                self._consecutive_idle = 0

            if trigger:
                self._last_intervention_time = now
                return trigger

        # ── Positive flow: separate cooldown + once-per-session cap ───────
        flow_cooldown_ok = (
            self._last_flow_notif_time is None
            or (now - self._last_flow_notif_time).total_seconds() / 60.0
               >= MIN_MINUTES_BETWEEN_FLOW_NOTIFS
        )
        if (self._consecutive_flow >= 2
                and self._flow_notifs_sent < MAX_FLOW_NOTIFS_PER_SESSION
                and flow_cooldown_ok):
            self._consecutive_flow    = 0
            self._last_flow_notif_time = now
            self._flow_notifs_sent    += 1
            return "positive_flow"

        return None

    def _fire_intervention(
        self,
        trigger_reason: str,
        stress_index:   float,
        face_agg:       dict,
        desktop_agg:    dict,
    ) -> bool:
        """
        Build context, generate a recommendation, and send a notification.

        Returns
        -------
        True  — notification was delivered (notifier accepted and displayed it).
        False — delivery was suppressed by the notifier's own cooldown, or an
                exception prevented delivery.  intervention_triggered should
                only be set to 1 in the DB when this returns True.
        """
        try:
            session_min = int(
                (datetime.now() - self._session_start).total_seconds() / 60
            )
            context = {
                "trigger_reason":      trigger_reason,
                "app_category":        desktop_agg.get("dominant_category") or "Unknown",
                "stress_index":        stress_index,
                "valence":             face_agg.get("avg_valence"),
                "arousal":             face_agg.get("avg_arousal"),
                "session_minutes":     session_min,
                "minutes_since_break": None,
                "blink_rate":          face_agg.get("avg_blink_rate"),
                "avg_activity_pct":    desktop_agg.get("avg_activity_pct"),
            }
            message = self._recommender.generate(context)
            title   = _TRIGGER_TITLES.get(trigger_reason, "Well-being Check")
            delivered = self._notifier.send(
                title          = title,
                message        = message,
                trigger_reason = trigger_reason,
                stress_index   = stress_index,
                valence        = face_agg.get("avg_valence"),
                arousal        = face_agg.get("avg_arousal"),
            )
            return delivered
        except Exception as exc:
            print(f"[Aggregator] Intervention delivery error: {exc}")
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
        stress_index:   float,
        trigger_reason: str | None,
    ) -> None:
        hr_str = f"{physio['avg_hr']:.1f} BPM" if physio["avg_hr"] else "--"
        v_str  = f"{face['avg_valence']:+.2f}"  if face["avg_valence"] else "--"
        a_str  = f"{face['avg_arousal']:+.2f}"  if face["avg_arousal"] else "--"
        cat    = desktop["dominant_category"] or "--"

        bar_len = int(stress_index * 20)
        bar     = "#" * bar_len + "-" * (20 - bar_len)

        print(f"\n{'='*52}")
        print(f"  5-min window  {t0:%H:%M} - {t1:%H:%M}")
        print(f"{'='*52}")
        print(f"  HR:            {hr_str}")
        print(f"  Valence/Arousal: {v_str} / {a_str}")
        print(f"  Activity:      {desktop['avg_activity_pct']:.0f}%" if desktop["avg_activity_pct"] else "  Activity:      --")
        print(f"  App category:  {cat}")
        print(f"  Stress index:  [{bar}] {stress_index:.3f}")
        if trigger_reason:
            print(f"  *** INTERVENTION: {trigger_reason.upper()} ***")
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
