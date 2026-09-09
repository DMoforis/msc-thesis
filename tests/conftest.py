"""
conftest.py
-----------
Shared pytest fixtures for the stress detection test suite.
"""
import time as _real_time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.utils.db import open_db


# ── Threshold isolation fixture ───────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stable_config(monkeypatch):
    """
    Isolate every test from live config.py threshold values.

    config.py is edited during data-collection sessions (low thresholds fire
    notifications quickly for experimentation).  Without this fixture, those
    edits bleed into the test suite and cause non-high_stress trigger tests to
    receive 'high_stress' instead of their expected result.

    Two sets of patches are applied:
    1. src.fusion.aggregator.* — the aggregator uses `from … import` so the
       names are bound in the aggregator's own namespace; patching config alone
       would have no effect.
    2. src.llm.recommender.* — LLM_COMPARISON_MODE and the module-level `time`
       reference are patched so that (a) comparison mode is always enabled and
       (b) the 2-second GPU-settle sleep in _run_qwen_and_save is a no-op,
       allowing the background Qwen thread to complete within the short waits
       used in test_llm_comparison.py.

    All patches are reverted automatically by monkeypatch after each test.
    """
    import src.utils.config   as cfg
    import src.fusion.aggregator as agg
    import src.llm.recommender  as rec

    # ── 1. Aggregator trigger thresholds (production/baseline values) ─────────
    monkeypatch.setattr(agg, "STRESS_TRIGGER_THRESHOLD",          0.65)
    # MIN_MINUTES_BETWEEN_NOTIFS is NOT patched: config.py has 5 and
    # test_cooldown_prevents_double_trigger imports that value directly to
    # compute its time offset — keeping them in sync avoids a mismatch.
    monkeypatch.setattr(agg, "DISENGAGEMENT_VALENCE_THRESHOLD",   -0.30)
    monkeypatch.setattr(agg, "DISENGAGEMENT_AROUSAL_THRESHOLD",   -0.15)
    monkeypatch.setattr(agg, "DISENGAGEMENT_ACTIVITY_MAX",        50.0)
    monkeypatch.setattr(agg, "NEGATIVE_AFFECT_VALENCE_THRESHOLD", -0.35)
    monkeypatch.setattr(agg, "NEGATIVE_AFFECT_AROUSAL_MIN",       0.20)
    monkeypatch.setattr(agg, "EYE_STRAIN_BLINK_THRESHOLD",        8.0)
    monkeypatch.setattr(agg, "IDLE_ACTIVITY_THRESHOLD",           15.0)
    monkeypatch.setattr(agg, "FLOW_STRESS_CEILING",               0.20)
    monkeypatch.setattr(agg, "FLOW_ACTIVITY_FLOOR",               65.0)
    monkeypatch.setattr(agg, "FLOW_VALENCE_FLOOR",                0.10)

    # ── 2. LLM recommender ───────────────────────────────────────────────────
    # Ensure comparison mode is True regardless of what config.py says.
    monkeypatch.setattr(cfg, "LLM_COMPARISON_MODE", True)
    monkeypatch.setattr(rec, "LLM_COMPARISON_MODE", True)

    # Replace the module-level `time` in recommender with a thin mock so that
    # `time.sleep(2)` in _run_qwen_and_save is a no-op.  All other time
    # functions (monotonic etc.) delegate to the real module via wraps=.
    # Note: _timed_ollama_call uses a *local* `import time`, so it is unaffected.
    fake_time = MagicMock(wraps=_real_time)
    fake_time.sleep = MagicMock()           # no-op; LLM tests mock the call anyway
    monkeypatch.setattr(rec, "time", fake_time)


class MockLandmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


@pytest.fixture
def tmp_db(tmp_path):
    """Yield path to a fresh, empty SQLite database file."""
    db_file = str(tmp_path / "test_stress.db")
    conn = open_db(db_file)
    conn.close()
    return db_file


@pytest.fixture
def sample_landmarks():
    """
    468 MockLandmark objects where both eyes have EAR ~= 0.30 (open).

    Geometry (w = h = 100 px):
      Left eye  (LEFT_EYE = [362, 385, 387, 263, 373, 380]):
        pts[0]=lm[362] at (10,50), pts[3]=lm[263] at (40,50)  -> C=30 px
        pts[1]=lm[385] at (17,44), pts[5]=lm[380] at (17,53)  -> A=9 px
        pts[2]=lm[387] at (27,44), pts[4]=lm[373] at (27,53)  -> B=9 px
        EAR = (9+9) / (2*30) = 0.30
    """
    lms = [MockLandmark() for _ in range(468)]

    # Left eye
    lms[362] = MockLandmark(x=0.10, y=0.50)
    lms[263] = MockLandmark(x=0.40, y=0.50)
    lms[385] = MockLandmark(x=0.17, y=0.44)
    lms[380] = MockLandmark(x=0.17, y=0.53)
    lms[387] = MockLandmark(x=0.27, y=0.44)
    lms[373] = MockLandmark(x=0.27, y=0.53)

    # Right eye (RIGHT_EYE = [33, 160, 158, 133, 153, 144])
    lms[33]  = MockLandmark(x=0.60, y=0.50)
    lms[133] = MockLandmark(x=0.90, y=0.50)
    lms[160] = MockLandmark(x=0.67, y=0.44)
    lms[144] = MockLandmark(x=0.67, y=0.53)
    lms[158] = MockLandmark(x=0.77, y=0.44)
    lms[153] = MockLandmark(x=0.77, y=0.53)

    return lms


@pytest.fixture
def sample_bvp():
    """
    300-sample synthetic BVP at 30 fps representing 60 BPM.
    Period = 30 samples (1 Hz sine at 30 Hz sampling rate).
    """
    t = np.arange(300) / 30.0
    return np.sin(2 * np.pi * 1.0 * t)
