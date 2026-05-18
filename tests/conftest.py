"""
conftest.py
-----------
Shared pytest fixtures for the stress detection test suite.
"""
import numpy as np
import pytest

from src.utils.db import open_db


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
