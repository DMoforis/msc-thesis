"""
late_fusion.py
--------------
Weighted late fusion of per-modality stress scores.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Method
------
Each modality independently produces a stress score in [0, 1]:
  0 = no stress indicators detected
  1 = maximum stress indicators

Final stress_index = (w_physio * s_physio + w_face * s_face + w_desktop * s_desktop)
                     / sum_of_present_weights

Weights (from config): physio=0.4, face=0.4, desktop=0.2
Desktop is weighted lower as it is a behavioural proxy, not physiological.
If a modality has no data in the window, its weight is dropped and the
remaining weights are renormalised.

Baselines
---------
Physiological baselines default to population means. They should be replaced
with personal baselines once a calibration session is available.
"""

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import W_PHYSIO, W_FACE, W_DESKTOP, EAR_THRESHOLD

# ── Population-mean physiological baselines ───────────────────────────────────
_BASELINE_HR     = 70.0    # BPM — typical resting HR for healthy adults
_BASELINE_RMSSD  = 40.0    # ms  — median RMSSD for healthy adults at rest
_MAX_HR_DELTA    = 30.0    # BPM deviation above baseline that maps to score 1.0

# Desktop score calibration
_HIGH_WINDOW_SWITCHES = 20  # switches per 5-min window → maps to score 1.0


class LateFusion:
    """
    Combines per-modality stress scores into a single stress_index.

    Usage
    -----
    fusion = LateFusion()
    stress_index = fusion.score(physio_agg, face_agg, desktop_agg)

    Each argument is a dict of averaged values from a 5-minute window
    (as produced by Aggregator._aggregate_*). Any key may be None if
    data was unavailable — the modality is then excluded and weights are
    renormalised over the remaining modalities.
    """

    def score(
        self,
        physio: dict,
        face: dict,
        desktop: dict,
    ) -> float:
        """
        Return stress_index in [0, 1].

        Parameters
        ----------
        physio  : dict  keys: avg_hr, avg_rmssd  (float | None)
        face    : dict  keys: avg_valence, avg_arousal, avg_ear  (float | None)
        desktop : dict  keys: avg_activity_pct, total_window_switches,
                              dominant_category  (float | int | str | None)
        """
        scored: list[tuple[float, float]] = []   # (score, weight)

        s_physio = self._physio_score(physio)
        if s_physio is not None:
            scored.append((s_physio, W_PHYSIO))

        s_face = self._face_score(face)
        if s_face is not None:
            scored.append((s_face, W_FACE))

        s_desktop = self._desktop_score(desktop)
        if s_desktop is not None:
            scored.append((s_desktop, W_DESKTOP))

        if not scored:
            return 0.0

        total_w = sum(w for _, w in scored)
        index   = sum(s * w for s, w in scored) / total_w
        return round(min(1.0, max(0.0, index)), 3)

    # ── Per-modality scoring ──────────────────────────────────────────────────

    def _physio_score(self, physio: dict) -> float | None:
        """
        Physio stress score in [0, 1].
        Combines HR elevation above baseline and RMSSD suppression below baseline.
        """
        hr    = physio.get("avg_hr")
        rmssd = physio.get("avg_rmssd")

        subscores: list[float] = []

        if hr is not None:
            # Linear ramp: no elevation → 0.0, +30 BPM above baseline → 1.0
            delta = max(0.0, hr - _BASELINE_HR)
            subscores.append(min(1.0, delta / _MAX_HR_DELTA))

        if rmssd is not None and rmssd > 0:
            # Low RMSSD = sympathetic dominance = stress.
            # Score 1.0 at RMSSD→0, score 0.0 at RMSSD ≥ 2×baseline (80 ms).
            subscores.append(max(0.0, 1.0 - rmssd / (_BASELINE_RMSSD * 2.0)))

        return sum(subscores) / len(subscores) if subscores else None

    def _face_score(self, face: dict) -> float | None:
        """
        Face stress score in [0, 1].
        Combines negative valence, arousal deviation from neutral, and low EAR.
        """
        valence = face.get("avg_valence")
        arousal = face.get("avg_arousal")
        ear     = face.get("avg_ear")

        subscores: list[float] = []
        weights:   list[float] = []

        if valence is not None:
            # valence in [-1, 1]: -1 → score 1.0, +1 → score 0.0
            subscores.append((1.0 - valence) / 2.0)
            weights.append(1.0)

        if arousal is not None:
            # High arousal (anxiety) AND very low arousal (disengagement) both
            # deviate from the neutral baseline → use absolute value.
            subscores.append(abs(arousal))
            weights.append(1.0)

        if ear is not None:
            # Sustained low EAR indicates eye strain or fatigue.
            # Only contributes when below the blink/closure threshold.
            ear_s = max(0.0, 1.0 - ear / EAR_THRESHOLD) if ear < EAR_THRESHOLD else 0.0
            subscores.append(ear_s)
            weights.append(0.5)   # lower weight — indirect indicator

        if not subscores:
            return None

        total_w = sum(weights)
        return sum(s * w for s, w in zip(subscores, weights)) / total_w

    def _desktop_score(self, desktop: dict) -> float | None:
        """
        Desktop stress score in [0, 1].
        Combines inactivity, fragmented attention (window switches), and app
        category (distracting apps during work hours).
        """
        activity_pct = desktop.get("avg_activity_pct")
        win_switches = desktop.get("total_window_switches")
        category     = desktop.get("dominant_category") or ""

        subscores: list[float] = []

        if activity_pct is not None:
            # Low activity → user is disengaged or avoiding work → stress signal
            subscores.append(max(0.0, 1.0 - activity_pct / 100.0))

        if win_switches is not None:
            # Frequent window switching → fragmented attention → stress signal
            subscores.append(min(1.0, win_switches / _HIGH_WINDOW_SWITCHES))

        if category in ("Social Media", "Entertainment", "Communication"):
            # Non-work context during a session indicates potential stress avoidance
            subscores.append(0.4)

        return sum(subscores) / len(subscores) if subscores else None


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    fusion = LateFusion()

    cases = [
        # (description, physio, face, desktop, expected_ballpark)
        ("All zeros — no stress",
         {"avg_hr": 70.0, "avg_rmssd": 40.0},
         {"avg_valence": 0.5, "avg_arousal": 0.1, "avg_ear": 0.28},
         {"avg_activity_pct": 80.0, "total_window_switches": 3, "dominant_category": "Academic Work"},
         "<0.25"),
        ("Elevated HR, low RMSSD, negative valence",
         {"avg_hr": 95.0, "avg_rmssd": 15.0},
         {"avg_valence": -0.6, "avg_arousal": 0.7, "avg_ear": 0.19},
         {"avg_activity_pct": 30.0, "total_window_switches": 18, "dominant_category": "Social Media"},
         ">0.65"),
        ("Missing physio — face + desktop only",
         {"avg_hr": None, "avg_rmssd": None},
         {"avg_valence": -0.3, "avg_arousal": 0.4, "avg_ear": 0.25},
         {"avg_activity_pct": 60.0, "total_window_switches": 8, "dominant_category": "Academic Work"},
         "0.2–0.5"),
        ("All missing",
         {"avg_hr": None, "avg_rmssd": None},
         {"avg_valence": None, "avg_arousal": None, "avg_ear": None},
         {"avg_activity_pct": None, "total_window_switches": None, "dominant_category": None},
         "0.0"),
    ]

    print(f"{'Description':<45} {'stress_index':>12}  {'Expected'}")
    print("-" * 75)
    for desc, physio, face, desktop, expected in cases:
        idx = fusion.score(physio, face, desktop)
        print(f"{desc:<45} {idx:>12.3f}  {expected}")
