"""
hrv_processor.py
----------------
HRV computation and validation helpers.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Extracted from rppg_monitor.py so that the pure math functions can be tested
independently without starting a camera or the rPPG inference thread.
"""

import numpy as np


# Plausibility bounds for rPPG-derived HRV (tighter than ECG due to peak jitter)
_RMSSD_LOWER: float = 5.0     # ms — artefact floor
_RMSSD_UPPER: float = 80.0    # ms — rPPG ceiling (ECG would allow ~120 ms)
_SDNN_LOWER:  float = 5.0     # ms
_SDNN_UPPER:  float = 150.0   # ms

# HR cross-validation gate
_MAX_HR_DELTA: float = 20.0   # BPM — max tolerable deviation between HRV-window HR and 10 s readings


def compute_rmssd(rr_intervals_ms: list[float]) -> float:
    """
    Root mean square of successive RR interval differences.

    Parameters
    ----------
    rr_intervals_ms : sequence of RR intervals in milliseconds

    Returns
    -------
    RMSSD in milliseconds, or 0.0 if fewer than 2 intervals are supplied.
    """
    arr = np.asarray(rr_intervals_ms, dtype=float)
    if len(arr) < 2:
        return 0.0
    diffs = np.diff(arr)
    return float(np.sqrt(np.mean(diffs ** 2)))


def validate_hrv_metrics(
    rmssd_raw:  float | None,
    sdnn_raw:   float | None,
    hrv_hr:     float,
    hr_samples: list[float],
) -> tuple[float | None, float | None]:
    """
    Apply plausibility bounds and HR cross-validation to rPPG HRV metrics.

    Parameters
    ----------
    rmssd_raw   : RMSSD in ms from the model (None if SQI was too low)
    sdnn_raw    : SDNN in ms from the model (None if SQI was too low)
    hrv_hr      : HR estimated from the HRV window's BVP peak series (BPM)
    hr_samples  : 10-second HR readings collected during the same window

    Returns
    -------
    (rmssd, sdnn) — validated, rounded to 1 dp, or None if rejected
    """
    if hr_samples:
        avg_hr = sum(hr_samples) / len(hr_samples)
        hr_consistent = abs(hrv_hr - avg_hr) <= _MAX_HR_DELTA
    else:
        hr_consistent = True   # no 10 s data yet — accept tentatively

    # Open interval: values exactly at a boundary (e.g. 5.0 ms, 80.0 ms) are
    # rejected.  In practice rPPG signals never land exactly on these limits.
    rmssd = (
        round(rmssd_raw, 1)
        if (rmssd_raw is not None and _RMSSD_LOWER < rmssd_raw < _RMSSD_UPPER and hr_consistent)
        else None
    )
    sdnn = (
        round(sdnn_raw, 1)
        if (sdnn_raw is not None and _SDNN_LOWER < sdnn_raw < _SDNN_UPPER and hr_consistent)
        else None
    )
    return rmssd, sdnn
