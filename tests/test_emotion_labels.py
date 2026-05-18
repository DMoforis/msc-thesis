"""
tests/test_emotion_labels.py
-----------------------------
Tests for the Russell (1980) circumplex emotion label mapper.
"""

import math
import pytest

from src.utils.emotion_labels import get_emotion_label


# ─────────────────────────────────────────────────────────────────────────────
# Return-value contract
# ─────────────────────────────────────────────────────────────────────────────

def test_return_keys():
    result = get_emotion_label(0.0, 0.0)
    assert set(result.keys()) == {"primary", "quadrant", "intensity"}


# ─────────────────────────────────────────────────────────────────────────────
# Neutral zone
# ─────────────────────────────────────────────────────────────────────────────

def test_neutral_center():
    assert get_emotion_label(0.0, 0.0)["primary"] == "Neutral"

def test_neutral_boundary_positive():
    # Exactly ±0.2 uses strict inequality (> 0.2), so it falls to Neutral.
    assert get_emotion_label(0.2, 0.2)["primary"] == "Neutral"

def test_neutral_boundary_negative():
    assert get_emotion_label(-0.2, -0.2)["primary"] == "Neutral"

def test_neutral_mixed_extreme_valence_only():
    # One dimension extreme, other in neutral band → Neutral.
    assert get_emotion_label(0.9, 0.0)["primary"] == "Neutral"

def test_neutral_mixed_extreme_arousal_only():
    assert get_emotion_label(0.0, 0.9)["primary"] == "Neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant 1: high arousal, positive valence  (a > 0.2, v > 0.2)
# ─────────────────────────────────────────────────────────────────────────────

def test_excited():
    # Both dimensions > 0.5 → Excited (most restrictive sub-quadrant).
    assert get_emotion_label(0.8, 0.8)["primary"] == "Excited"

def test_alert():
    # arousal > 0.5, valence in (0.2, 0.5] → Alert.
    assert get_emotion_label(0.3, 0.7)["primary"] == "Alert"

def test_happy():
    # valence > 0.5, arousal in (0.2, 0.5] → Happy.
    assert get_emotion_label(0.7, 0.3)["primary"] == "Happy"

def test_pleased():
    # Both in (0.2, 0.5] → Pleased.
    assert get_emotion_label(0.3, 0.3)["primary"] == "Pleased"


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant 2: high arousal, negative valence  (a > 0.2, v < -0.2)
# ─────────────────────────────────────────────────────────────────────────────

def test_angry():
    # valence < -0.5 and arousal > 0.5 → Angry.
    assert get_emotion_label(-0.7, 0.7)["primary"] == "Angry"

def test_stressed():
    # arousal > 0.5, valence in (-0.5, -0.2) → Stressed.
    assert get_emotion_label(-0.3, 0.7)["primary"] == "Stressed"

def test_upset():
    # valence < -0.5, arousal in (0.2, 0.5] → Upset.
    assert get_emotion_label(-0.7, 0.3)["primary"] == "Upset"

def test_tense():
    # Both moderate negatives → Tense.
    assert get_emotion_label(-0.3, 0.3)["primary"] == "Tense"


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant 3: low arousal, positive valence  (a < -0.2, v > 0.2)
# ─────────────────────────────────────────────────────────────────────────────

def test_calm():
    # valence > 0.5 and arousal < -0.5 → Calm.
    assert get_emotion_label(0.7, -0.7)["primary"] == "Calm"

def test_serene():
    # valence > 0.5, arousal in [-0.5, -0.2) → Serene.
    assert get_emotion_label(0.7, -0.3)["primary"] == "Serene"

def test_sleepy():
    # arousal < -0.5, valence in (0.2, 0.5] → Sleepy.
    assert get_emotion_label(0.3, -0.7)["primary"] == "Sleepy"

def test_relaxed():
    # Both moderate in this quadrant → Relaxed.
    assert get_emotion_label(0.3, -0.3)["primary"] == "Relaxed"


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant 4: low arousal, negative valence  (a < -0.2, v < -0.2)
# ─────────────────────────────────────────────────────────────────────────────

def test_depressed():
    # Both < -0.5 → Depressed.
    assert get_emotion_label(-0.7, -0.7)["primary"] == "Depressed"

def test_bored():
    # arousal < -0.5, valence in (-0.5, -0.2) → Bored.
    assert get_emotion_label(-0.3, -0.7)["primary"] == "Bored"

def test_sad():
    # valence < -0.5, arousal in [-0.5, -0.2) → Sad.
    assert get_emotion_label(-0.7, -0.3)["primary"] == "Sad"

def test_fatigued():
    # Both moderate negatives → Fatigued.
    assert get_emotion_label(-0.3, -0.3)["primary"] == "Fatigued"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary: just past ±0.2 enters a quadrant
# ─────────────────────────────────────────────────────────────────────────────

def test_boundary_just_above_0_2():
    result = get_emotion_label(0.21, 0.21)
    assert result["primary"] != "Neutral"

def test_boundary_just_below_minus_0_2():
    result = get_emotion_label(-0.21, -0.21)
    assert result["primary"] != "Neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Intensity: low / moderate / high
# ─────────────────────────────────────────────────────────────────────────────

def test_intensity_low():
    # distance = sqrt(0.1² + 0.1²) ≈ 0.141 < 0.3
    assert get_emotion_label(0.1, 0.1)["intensity"] == "low"

def test_intensity_moderate():
    # distance = sqrt(0.3² + 0.3²) ≈ 0.424, in [0.3, 0.6)
    assert get_emotion_label(0.3, 0.3)["intensity"] == "moderate"

def test_intensity_high():
    # distance = sqrt(0.6² + 0.6²) ≈ 0.849 >= 0.6
    assert get_emotion_label(0.6, 0.6)["intensity"] == "high"

def test_intensity_just_above_0_3():
    # distance = sqrt(0.22² + 0.22²) ≈ 0.311, just above the 0.3 threshold → moderate
    assert get_emotion_label(0.22, 0.22)["intensity"] == "moderate"

def test_intensity_just_above_0_6():
    # distance = sqrt(0.43² + 0.43²) ≈ 0.608, just above the 0.6 threshold → high
    assert get_emotion_label(0.43, 0.43)["intensity"] == "high"


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant labels
# ─────────────────────────────────────────────────────────────────────────────

def test_quadrant_neutral_zone():
    assert get_emotion_label(0.0, 0.0)["quadrant"] == "Neutral zone"

def test_quadrant_high_arousal_positive_valence():
    assert get_emotion_label(0.5, 0.5)["quadrant"] == "High arousal, positive valence"

def test_quadrant_high_arousal_negative_valence():
    assert get_emotion_label(-0.5, 0.5)["quadrant"] == "High arousal, negative valence"

def test_quadrant_low_arousal_positive_valence():
    assert get_emotion_label(0.5, -0.5)["quadrant"] == "Low arousal, positive valence"

def test_quadrant_low_arousal_negative_valence():
    assert get_emotion_label(-0.5, -0.5)["quadrant"] == "Low arousal, negative valence"
