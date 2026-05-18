"""
test_ear.py
-----------
Tests for the Eye Aspect Ratio (EAR) formula and blink-counting state machine.
Reference: Soukupova & Cech (2016)
"""
import pytest

from src.face.face_monitor import eye_aspect_ratio, LEFT_EYE, RIGHT_EYE
from src.utils.config import EAR_THRESHOLD, BLINK_MIN_FRAMES

W = H = 100   # normalised pixel dimensions used throughout


def test_ear_open_eye(sample_landmarks):
    """Open-eye geometry should produce EAR ~= 0.30."""
    ear = eye_aspect_ratio(sample_landmarks, LEFT_EYE, W, H)
    assert abs(ear - 0.30) < 0.01


def test_ear_closed_eye():
    """Near-zero vertical opening should give EAR well below EAR_THRESHOLD (0.21)."""
    from tests.conftest import MockLandmark

    lms = [MockLandmark() for _ in range(468)]

    # Same horizontal span (C = 30 px), but 1 px vertical opening (A = B = 1 px)
    lms[362] = MockLandmark(x=0.10, y=0.50)
    lms[263] = MockLandmark(x=0.40, y=0.50)
    lms[385] = MockLandmark(x=0.17, y=0.495)
    lms[380] = MockLandmark(x=0.17, y=0.505)
    lms[387] = MockLandmark(x=0.27, y=0.495)
    lms[373] = MockLandmark(x=0.27, y=0.505)

    ear = eye_aspect_ratio(lms, LEFT_EYE, W, H)
    assert ear < EAR_THRESHOLD


def test_blink_detection():
    """Two consecutive sub-threshold EAR frames should register exactly one blink."""
    eye_closed_frames = 0
    blink_count = 0

    # open → blink (2 closed frames) → open
    ear_sequence = [0.30, 0.30, 0.05, 0.05, 0.30, 0.30]

    for ear in ear_sequence:
        if ear < EAR_THRESHOLD:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= BLINK_MIN_FRAMES:
                blink_count += 1
            eye_closed_frames = 0

    assert blink_count == 1


def test_no_blink_on_brief_closure():
    """A single sub-threshold frame (BLINK_MIN_FRAMES = 2) must NOT count as a blink."""
    eye_closed_frames = 0
    blink_count = 0

    # one closed frame only
    ear_sequence = [0.30, 0.30, 0.05, 0.30, 0.30]

    for ear in ear_sequence:
        if ear < EAR_THRESHOLD:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= BLINK_MIN_FRAMES:
                blink_count += 1
            eye_closed_frames = 0

    assert blink_count == 0
