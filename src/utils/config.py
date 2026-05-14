"""
config.py
---------
Single source of truth for all system-wide configuration constants.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Rule: no module file hardcodes numeric constants or paths. Everything lives here.
"""

import os

# ── Path anchors ──────────────────────────────────────────────────────────────
# BASE_DIR resolves to stress_detection/ regardless of where Python is invoked.
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EXPORT_DIR = os.path.join(BASE_DIR, "data", "exports")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(DATA_DIR, "stress_monitor.db")

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1
CAMERA_FPS   = 30

# ── Timing windows ────────────────────────────────────────────────────────────
HR_WINDOW_SECONDS   = 10    # HR reading frequency (seconds)
HRV_WINDOW_SECONDS  = 300   # 5-minute rolling window for HRV computation
FACE_WINDOW_SECONDS = 30    # facial feature aggregation window
AGGREGATION_MINUTES = 5     # fusion aggregation window (minutes)

# ── rPPG model ────────────────────────────────────────────────────────────────
RPPG_MODEL    = "FacePhys.rlap"
SAMPLING_RATE = 30   # assumed camera FPS for HRV calculations

# ── Fusion weights ────────────────────────────────────────────────────────────
# Desktop weighted lower because it is a behavioural proxy, not physiological.
W_PHYSIO  = 0.4
W_FACE    = 0.4
W_DESKTOP = 0.2

# ── Intervention thresholds ───────────────────────────────────────────────────
STRESS_TRIGGER_THRESHOLD   = 0.65   # stress_index > this for 2 consecutive windows
VALENCE_TRIGGER_THRESHOLD  = -0.4   # negative valence sustained ≥ 10 min
AROUSAL_LOW_THRESHOLD      = 0.2    # disengagement sustained ≥ 15 min
MIN_MINUTES_BETWEEN_NOTIFS = 15     # cooldown between notifications

# ── Ollama LLM ────────────────────────────────────────────────────────────────
OLLAMA_MODEL   = "llama3.2:8b"
OLLAMA_TIMEOUT = 2   # seconds; fall back to keyword matching if exceeded

# ── Face landmark detection ───────────────────────────────────────────────────
EAR_THRESHOLD    = 0.21   # below this → eye considered closed (Soukupova & Cech 2016)
BLINK_MIN_FRAMES = 2      # minimum consecutive closed frames to count as a blink
