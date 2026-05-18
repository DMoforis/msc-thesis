"""
emotion_labels.py
-----------------
Verbal emotion label mapping from continuous valence-arousal space.

Based on the Russell (1980) circumplex model of affect. Maps a (valence, arousal)
coordinate pair to a human-readable primary label, a quadrant description, and
an intensity level derived from the Euclidean distance from the origin.

Reference
---------
Russell, J.A. (1980). A circumplex model of affect. Journal of Personality
and Social Psychology, 39(6), 1161-1178.
"""

import math


def get_emotion_label(valence: float, arousal: float) -> dict:
    """
    Map a (valence, arousal) coordinate to a verbal emotion description.

    Parameters
    ----------
    valence : float  in [-1, 1]  negative = unpleasant, positive = pleasant
    arousal : float  in [-1, 1]  negative = drowsy/calm, positive = alert/activated

    Returns
    -------
    dict with keys:
        primary   : str  — single most likely emotion label
        quadrant  : str  — human-readable quadrant description
        intensity : str  — "low" | "moderate" | "high" based on Euclidean
                           distance from origin
    """
    return {
        "primary":   _primary_label(valence, arousal),
        "quadrant":  _quadrant_label(valence, arousal),
        "intensity": _intensity(valence, arousal),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _primary_label(v: float, a: float) -> str:
    """
    Return the primary emotion label.

    Sub-quadrant selection checks the most specific threshold pair first
    (both dimensions at the ±0.5 level) before falling back to the
    single-dimension ±0.5 checks and finally the ±0.2 defaults. This
    ensures that e.g. (v=0.8, a=0.8) → "Excited" rather than "Alert" or
    "Happy", without requiring explicit range checks for intermediate values.
    """
    # ── High arousal, positive valence ────────────────────────────────────────
    if a > 0.2 and v > 0.2:
        if v > 0.5 and a > 0.5:  return "Excited"
        if a > 0.5:               return "Alert"    # v in (0.2, 0.5]
        if v > 0.5:               return "Happy"    # a in (0.2, 0.5]
        return "Pleased"                             # both in (0.2, 0.5]

    # ── High arousal, negative valence ────────────────────────────────────────
    if a > 0.2 and v < -0.2:
        if v < -0.5 and a > 0.5: return "Angry"
        if a > 0.5:               return "Stressed"  # v in (-0.5, -0.2)
        if v < -0.5:              return "Upset"     # a in (0.2, 0.5]
        return "Tense"                               # both moderate

    # ── Low arousal, positive valence ─────────────────────────────────────────
    if a < -0.2 and v > 0.2:
        if v > 0.5 and a < -0.5: return "Calm"
        if v > 0.5:               return "Serene"   # a in [-0.5, -0.2)
        if a < -0.5:              return "Sleepy"   # v in (0.2, 0.5]
        return "Relaxed"                            # both moderate

    # ── Low arousal, negative valence ─────────────────────────────────────────
    if a < -0.2 and v < -0.2:
        if v < -0.5 and a < -0.5: return "Depressed"
        if a < -0.5:               return "Bored"   # v in (-0.5, -0.2)
        if v < -0.5:               return "Sad"     # a in [-0.5, -0.2)
        return "Fatigued"                           # both moderate

    # ── Neutral zone — both dimensions within ±0.2, or mixed axis ────────────
    return "Neutral"


def _quadrant_label(v: float, a: float) -> str:
    if a > 0.2 and v > 0.2:   return "High arousal, positive valence"
    if a > 0.2 and v < -0.2:  return "High arousal, negative valence"
    if a < -0.2 and v > 0.2:  return "Low arousal, positive valence"
    if a < -0.2 and v < -0.2: return "Low arousal, negative valence"
    return "Neutral zone"


def _intensity(v: float, a: float) -> str:
    distance = math.sqrt(v ** 2 + a ** 2)
    if distance < 0.3:  return "low"
    if distance < 0.6:  return "moderate"
    return "high"
