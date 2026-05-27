"""
test_fusion.py
--------------
Tests for LateFusion.score() — weighted late fusion of per-modality stress scores.
All tests mock get_baseline() to return population defaults so results are
deterministic regardless of whether a personal calibration has been run.
"""
from unittest.mock import patch

import pytest

# Population defaults (mirrors src/utils/baseline.py _POPULATION_DEFAULTS)
_POP = {
    "hr": 70.0, "rmssd": 42.0, "ear": 0.30,
    "blink_rate": 15.0, "valence": 0.0, "arousal": 0.0,
}

_HIGH_BASELINE = {
    "hr": 90.0, "rmssd": 60.0, "ear": 0.30,
    "blink_rate": 15.0, "valence": 0.0, "arousal": 0.0,
}


def _make_fusion(baseline: dict):
    with patch("src.fusion.late_fusion.get_baseline", return_value=dict(baseline)):
        from src.fusion.late_fusion import LateFusion
        return LateFusion()


@pytest.fixture
def fusion():
    return _make_fusion(_POP)


# ─────────────────────────────────────────────────────────────────────────────

def test_stress_index_range(fusion):
    """score() must always return a value in [0, 1]."""
    idx = fusion.score(
        {"avg_hr": 75.0, "avg_rmssd": 35.0},
        {"avg_valence": 0.1, "avg_arousal": 0.2, "avg_ear": 0.27},
        {"avg_activity_pct": 70.0, "total_window_switches": 5, "dominant_category": "Academic Work"},
    )
    assert 0.0 <= idx <= 1.0


def test_high_stress_inputs(fusion):
    """Clearly stressed inputs (elevated HR, low RMSSD, negative valence, fragmented attention)
    should produce a stress_index > 0.5."""
    idx = fusion.score(
        {"avg_hr": 100.0, "avg_rmssd": 10.0},
        {"avg_valence": -0.8, "avg_arousal": 0.7, "avg_ear": 0.15},
        {"avg_activity_pct": 20.0, "total_window_switches": 20, "dominant_category": "Social Media"},
    )
    assert idx > 0.5


def test_low_stress_inputs(fusion):
    """Relaxed inputs (HR below baseline, high RMSSD, positive valence, focused work)
    should produce a stress_index < 0.2."""
    idx = fusion.score(
        {"avg_hr": 65.0, "avg_rmssd": 50.0},
        {"avg_valence": 0.6, "avg_arousal": 0.1, "avg_ear": 0.28},
        {"avg_activity_pct": 90.0, "total_window_switches": 2, "dominant_category": "Academic Work"},
    )
    assert idx < 0.2


def test_missing_physio(fusion):
    """When physio data is absent the score should still be non-zero (face + desktop carry it)."""
    idx = fusion.score(
        {"avg_hr": None, "avg_rmssd": None},
        {"avg_valence": -0.3, "avg_arousal": 0.4, "avg_ear": None},
        {"avg_activity_pct": 60.0, "total_window_switches": 8, "dominant_category": "Academic Work"},
    )
    assert idx > 0.0


def test_missing_all_modalities(fusion):
    """All-None inputs must return exactly 0.0."""
    idx = fusion.score(
        {"avg_hr": None, "avg_rmssd": None},
        {"avg_valence": None, "avg_arousal": None, "avg_ear": None},
        {"avg_activity_pct": None, "total_window_switches": None, "dominant_category": None},
    )
    assert idx == 0.0


def test_baseline_personalisation():
    """A higher personal resting HR baseline should lower the stress score for the same inputs
    because the measured HR looks less elevated relative to the individual's normal."""
    inputs = (
        {"avg_hr": 80.0, "avg_rmssd": 30.0},
        {"avg_valence": 0.0, "avg_arousal": 0.2, "avg_ear": None},
        {"avg_activity_pct": 50.0, "total_window_switches": 5, "dominant_category": "Browser"},
    )

    fusion_pop  = _make_fusion(_POP)           # resting HR = 70 → 80 is +10 above baseline
    fusion_high = _make_fusion(_HIGH_BASELINE) # resting HR = 90 → 80 is below baseline

    score_pop  = fusion_pop.score(*inputs)
    score_high = fusion_high.score(*inputs)

    assert score_pop > score_high


def test_desktop_category_flutter_taxonomy(fusion):
    """IDE/Terminal must produce a higher overall stress_index than Media for
    identical physio/face inputs, because IDE work carries a higher cognitive
    load weight in the Flutter taxonomy (0.7 vs 0.1).

    Physio and face are set to None so the comparison is driven entirely by
    the desktop signal, making the test independent of baseline values.
    """
    no_physio = {"avg_hr": None, "avg_rmssd": None}
    no_face   = {"avg_valence": None, "avg_arousal": None, "avg_ear": None}
    shared_desktop = {"avg_activity_pct": 60.0, "total_window_switches": 5}

    idx_ide = fusion.score(
        no_physio, no_face,
        {**shared_desktop, "dominant_category": "IDE/Terminal"},
    )
    idx_media = fusion.score(
        no_physio, no_face,
        {**shared_desktop, "dominant_category": "Media"},
    )

    assert idx_ide > idx_media, (
        f"Expected IDE/Terminal ({idx_ide:.3f}) > Media ({idx_media:.3f})"
    )


def test_desktop_category_unknown(fusion):
    """Category labels not in the Flutter taxonomy (e.g. legacy 'Academic Work'
    from old test data) must not crash and must return a valid stress_index
    in [0, 1], falling back to the 'Other' weight (0.2).
    """
    idx = fusion.score(
        {"avg_hr": None, "avg_rmssd": None},
        {"avg_valence": None, "avg_arousal": None, "avg_ear": None},
        {
            "avg_activity_pct": 50.0,
            "total_window_switches": 5,
            "dominant_category": "Academic Work",   # not in Flutter taxonomy
        },
    )
    assert 0.0 <= idx <= 1.0, f"score out of range: {idx}"
    # With Other weight=0.2, activity=50%→0.5, switches=5/20=0.25 → mean≈0.317
    assert idx > 0.0, "score should be non-zero when desktop data is present"
