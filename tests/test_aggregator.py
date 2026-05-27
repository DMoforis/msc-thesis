"""
tests/test_aggregator.py
------------------------
Tests for Aggregator._check_interventions() multi-trigger logic.

All tests bypass _run_window() and call _check_interventions() directly.
No real database operations are performed — the aggregator is instantiated
with a temp path and only the in-memory trigger-state is exercised.
"""

import pytest
from datetime import datetime, timedelta

from src.fusion.aggregator import Aggregator
from src.utils.config import MIN_MINUTES_BETWEEN_NOTIFS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agg(tmp_path):
    """Fresh Aggregator with a temp DB path and zeroed trigger state."""
    return Aggregator(db_path=str(tmp_path / "test.db"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _face(valence=None, arousal=None, blink_rate=15.0):
    """Build a minimal face_agg dict."""
    return {
        "avg_valence":    valence,
        "avg_arousal":    arousal,
        "avg_blink_rate": blink_rate,
        "avg_ear":        None,
        "avg_pitch":      None,
        "avg_yaw":        None,
    }


def _desktop(activity=60.0, category="Software Development"):
    """Build a minimal desktop_agg dict."""
    return {
        "avg_activity_pct":      activity,
        "avg_idle_seconds":      None,
        "total_window_switches": None,
        "dominant_category":     category,
    }


# ── high_stress (base-case regression) ───────────────────────────────────────

def test_high_stress_trigger(agg):
    face    = _face()
    desktop = _desktop()
    assert agg._check_interventions(0.70, face, desktop) is None   # window 1
    assert agg._check_interventions(0.70, face, desktop) == "high_stress"


# ── disengagement ─────────────────────────────────────────────────────────────

def test_disengagement_trigger(agg):
    # Low valence, low arousal, low activity for 2 consecutive windows.
    face    = _face(valence=-0.40, arousal=-0.25, blink_rate=12.0)
    desktop = _desktop(activity=30.0)
    assert agg._check_interventions(0.30, face, desktop) is None
    assert agg._check_interventions(0.30, face, desktop) == "disengagement"


def test_disengagement_blocked_by_high_activity(agg):
    # Valence/arousal meet thresholds but activity is above the ceiling → no fire.
    face    = _face(valence=-0.40, arousal=-0.25)
    desktop = _desktop(activity=70.0)   # > DISENGAGEMENT_ACTIVITY_MAX (50%)
    agg._check_interventions(0.30, face, desktop)
    assert agg._check_interventions(0.30, face, desktop) is None


# ── negative_affect ───────────────────────────────────────────────────────────

def test_negative_affect_trigger(agg):
    # Low valence + elevated arousal (tense/anxious), no high stress.
    face    = _face(valence=-0.50, arousal=0.40)
    desktop = _desktop(activity=60.0)
    assert agg._check_interventions(0.30, face, desktop) is None
    assert agg._check_interventions(0.30, face, desktop) == "negative_affect"


def test_negative_affect_not_fired_when_arousal_low(agg):
    # Valence is negative but arousal is below the floor → no negative_affect.
    face    = _face(valence=-0.50, arousal=0.10)
    desktop = _desktop(activity=60.0)
    agg._check_interventions(0.30, face, desktop)
    assert agg._check_interventions(0.30, face, desktop) is None


# ── eye_strain ────────────────────────────────────────────────────────────────

def test_eye_strain_trigger(agg):
    face    = _face(valence=0.0, arousal=0.0, blink_rate=5.0)
    desktop = _desktop(activity=70.0)
    assert agg._check_interventions(0.30, face, desktop) is None
    assert agg._check_interventions(0.30, face, desktop) == "eye_strain"


# ── prolonged_idle ────────────────────────────────────────────────────────────

def test_prolonged_idle_trigger(agg):
    # Needs 3 consecutive windows (15 min of idle).
    face    = _face()
    desktop = _desktop(activity=10.0)
    assert agg._check_interventions(0.30, face, desktop) is None   # window 1
    assert agg._check_interventions(0.30, face, desktop) is None   # window 2
    assert agg._check_interventions(0.30, face, desktop) == "prolonged_idle"


def test_prolonged_idle_reset_on_activity(agg):
    # Counter resets when activity rises above threshold.
    face    = _face()
    desktop_idle   = _desktop(activity=10.0)
    desktop_active = _desktop(activity=60.0)
    agg._check_interventions(0.30, face, desktop_idle)    # window 1: idle counter = 1
    agg._check_interventions(0.30, face, desktop_active)  # window 2: counter resets to 0
    agg._check_interventions(0.30, face, desktop_idle)    # window 3: counter = 1 again
    assert agg._check_interventions(0.30, face, desktop_idle) is None  # counter = 2, need 3


# ── positive_flow ─────────────────────────────────────────────────────────────

def test_positive_flow_trigger(agg):
    face    = _face(valence=0.30, arousal=0.20, blink_rate=15.0)
    desktop = _desktop(activity=75.0)
    assert agg._check_interventions(0.10, face, desktop) is None
    assert agg._check_interventions(0.10, face, desktop) == "positive_flow"


def test_flow_fires_once_per_session(agg):
    face    = _face(valence=0.30, arousal=0.20, blink_rate=15.0)
    desktop = _desktop(activity=75.0)

    # Fire it the first (and only allowed) time.
    agg._check_interventions(0.10, face, desktop)
    r1 = agg._check_interventions(0.10, face, desktop)
    assert r1 == "positive_flow"

    # Simulate confirmed delivery — _run_window() calls _commit_delivery() here.
    agg._commit_delivery(r1)
    assert agg._flow_notifs_sent == 1

    # Fast-forward flow cooldown so it is no longer the limiting factor.
    agg._last_flow_notif_time = datetime.now() - timedelta(minutes=35)
    # Manually prime the consecutive counter to threshold.
    agg._consecutive_flow = 2

    # Session cap (MAX_FLOW_NOTIFS_PER_SESSION = 1) blocks the second fire.
    r2 = agg._check_interventions(0.10, face, desktop)
    assert r2 != "positive_flow"


def test_flow_not_fired_when_stress_too_high(agg):
    # Flow requires stress_index < FLOW_STRESS_CEILING (0.20).
    face    = _face(valence=0.30, arousal=0.20, blink_rate=15.0)
    desktop = _desktop(activity=75.0)
    agg._check_interventions(0.25, face, desktop)   # stress=0.25 > 0.20
    assert agg._check_interventions(0.25, face, desktop) is None


# ── cooldown ──────────────────────────────────────────────────────────────────

def test_cooldown_prevents_double_trigger(agg):
    face    = _face(valence=-0.50, arousal=0.50)
    desktop = _desktop(activity=60.0)

    # Two windows of high stress → fires.
    agg._check_interventions(0.70, face, desktop)
    r1 = agg._check_interventions(0.70, face, desktop)
    assert r1 == "high_stress"

    # Simulate confirmed delivery — _run_window() calls _commit_delivery() here.
    agg._commit_delivery(r1)
    assert agg._last_intervention_time is not None

    # Conditions still met, but standard cooldown is active.
    r2 = agg._check_interventions(0.70, face, desktop)
    assert r2 is None

    # Advance the intervention timestamp past the cooldown window.
    agg._last_intervention_time = (
        datetime.now() - timedelta(minutes=MIN_MINUTES_BETWEEN_NOTIFS + 1)
    )
    r3 = agg._check_interventions(0.70, face, desktop)
    assert r3 == "high_stress"


def test_flow_cooldown_independent_of_standard_cooldown(agg):
    # Trigger a standard intervention, then verify flow can still fire
    # if its own (separate) cooldown and session cap allow it.
    face_stress = _face(valence=-0.50, arousal=0.50)
    face_flow   = _face(valence=0.30, arousal=0.20, blink_rate=15.0)
    desktop     = _desktop(activity=75.0)

    # Trigger high_stress and simulate delivery to activate the standard cooldown.
    agg._check_interventions(0.70, face_stress, desktop)
    r_stress = agg._check_interventions(0.70, face_stress, desktop)   # fires high_stress
    assert r_stress == "high_stress"
    agg._commit_delivery(r_stress)   # sets _last_intervention_time → standard cooldown active

    # Standard cooldown is now active, but flow uses a separate one.
    # Prime the flow counter to threshold.
    agg._consecutive_flow = 2
    r = agg._check_interventions(0.10, face_flow, desktop)
    assert r == "positive_flow"   # flow fires despite standard cooldown being active


# ── Delivery-gated cooldown regression tests ──────────────────────────────────

def test_fire_intervention_updates_cooldown_only_on_delivery(agg):
    """
    _check_interventions() must NOT consume the standard cooldown.
    Only _commit_delivery() should set _last_intervention_time.
    If a trigger fires but delivery is suppressed, the cooldown must remain
    available so the next window can retry.
    """
    face    = _face(valence=-0.50, arousal=0.50)
    desktop = _desktop(activity=60.0)

    # Two windows → trigger detected.
    agg._check_interventions(0.70, face, desktop)
    r = agg._check_interventions(0.70, face, desktop)
    assert r == "high_stress"

    # Cooldown NOT yet consumed — _commit_delivery() has not been called.
    assert agg._last_intervention_time is None

    # Simulate delivery confirmed.
    agg._commit_delivery(r)

    # Now cooldown IS consumed.
    assert agg._last_intervention_time is not None

    # Next window is blocked by cooldown.
    r2 = agg._check_interventions(0.70, face, desktop)
    assert r2 is None


def test_positive_flow_session_cap_only_on_delivery(agg):
    """
    _flow_notifs_sent must NOT be incremented by _check_interventions().
    Only _commit_delivery() should increment it.
    A suppressed flow delivery must not consume the once-per-session cap.
    """
    face    = _face(valence=0.30, arousal=0.20, blink_rate=15.0)
    desktop = _desktop(activity=75.0)

    # Two windows → flow trigger detected.
    agg._check_interventions(0.10, face, desktop)
    r = agg._check_interventions(0.10, face, desktop)
    assert r == "positive_flow"

    # Session counter NOT yet incremented — delivery not confirmed.
    assert agg._flow_notifs_sent == 0

    # Simulate delivery confirmed.
    agg._commit_delivery(r)
    assert agg._flow_notifs_sent == 1

    # Reset flow cooldown so time is not the limiting factor.
    agg._last_flow_notif_time = datetime.now() - timedelta(minutes=35)
    agg._consecutive_flow = 2

    # Session cap (MAX_FLOW_NOTIFS_PER_SESSION = 1) now blocks a second fire.
    r2 = agg._check_interventions(0.10, face, desktop)
    assert r2 != "positive_flow"
