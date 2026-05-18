"""
test_hrv.py
-----------
Tests for RMSSD computation and the HRV validation pipeline in hrv_processor.py.
"""
import math
import pytest

from src.physio.hrv_processor import compute_rmssd, validate_hrv_metrics


def test_rmssd_known_value():
    """RMSSD for a hand-computed RR sequence must match the analytic result."""
    rr = [800.0, 810.0, 790.0, 820.0, 800.0]
    # diffs = [10, -20, 30, -20]
    # mean(squares) = (100 + 400 + 900 + 400) / 4 = 450
    # RMSSD = sqrt(450) = 21.213...
    result = compute_rmssd(rr)
    assert abs(result - math.sqrt(450)) < 0.01


def test_rmssd_single_interval():
    """Fewer than two RR intervals should return 0.0 (no successive differences)."""
    assert compute_rmssd([800.0]) == 0.0
    assert compute_rmssd([]) == 0.0


def test_rmssd_upper_bound():
    """RMSSD above 80 ms must be rejected (None) by validate_hrv_metrics."""
    rmssd, _ = validate_hrv_metrics(85.0, None, 70.0, [70.0, 71.0])
    assert rmssd is None


def test_rmssd_lower_bound():
    """RMSSD below 5 ms must be rejected (likely artefact)."""
    rmssd, _ = validate_hrv_metrics(3.0, None, 70.0, [70.0])
    assert rmssd is None


def test_rmssd_valid_range():
    """RMSSD within 5–80 ms with consistent HR should be accepted."""
    rmssd, _ = validate_hrv_metrics(30.0, None, 70.0, [70.0, 72.0])
    assert rmssd == 30.0


def test_hr_cross_validation():
    """HRV window HR deviating > 20 BPM from 10 s readings must reject both metrics."""
    # HRV window HR = 70 BPM, but 10 s readings average ~95.5 BPM → delta > 20
    rmssd, sdnn = validate_hrv_metrics(30.0, 40.0, 70.0, [95.0, 96.0])
    assert rmssd is None
    assert sdnn is None


def test_hr_cross_validation_accepted():
    """HRV window HR within 20 BPM of 10 s readings must NOT reject valid metrics."""
    rmssd, sdnn = validate_hrv_metrics(30.0, 45.0, 72.0, [70.0, 71.0])
    assert rmssd == 30.0
    assert sdnn == 45.0


def test_no_samples_accepts_tentatively():
    """Without 10 s HR samples the validator should accept valid-range metrics."""
    rmssd, _ = validate_hrv_metrics(25.0, None, 75.0, [])
    assert rmssd == 25.0
