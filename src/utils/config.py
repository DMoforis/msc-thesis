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
CAMERA_INDEX = 0
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
STRESS_TRIGGER_THRESHOLD   = 0.10   # stress_index > this for 2 consecutive windows
VALENCE_TRIGGER_THRESHOLD  = -0.4   # kept for backwards compatibility
AROUSAL_LOW_THRESHOLD      = 0.2    # kept for backwards compatibility
MIN_MINUTES_BETWEEN_NOTIFS = 5     # cooldown between notifications (minutes)

# Per-condition thresholds (multi-trigger system)
DISENGAGEMENT_VALENCE_THRESHOLD   = -0.10  # avg_valence below this → low mood
DISENGAGEMENT_AROUSAL_THRESHOLD   = -0.15  # avg_arousal below this → low energy
DISENGAGEMENT_ACTIVITY_MAX        = 50.0   # avg_activity_pct must also be below this
NEGATIVE_AFFECT_VALENCE_THRESHOLD = -0.10  # avg_valence below this → tense/anxious
NEGATIVE_AFFECT_AROUSAL_MIN       = 0.20   # avg_arousal above this → activated
EYE_STRAIN_BLINK_THRESHOLD        = 20.0    # blinks/min below this → screen fixation
IDLE_ACTIVITY_THRESHOLD           = 15.0   # avg_activity_pct below this → near idle
FLOW_STRESS_CEILING               = 0.20   # stress_index must be below this
FLOW_ACTIVITY_FLOOR               = 65.0   # avg_activity_pct must be above this
FLOW_VALENCE_FLOOR                = 0.10   # avg_valence must be above this
MIN_MINUTES_BETWEEN_FLOW_NOTIFS   = 30     # separate cooldown for positive_flow
MAX_FLOW_NOTIFS_PER_SESSION       = 1      # positive_flow fires at most once

# ── Ollama LLM ────────────────────────────────────────────────────────────────
# Primary model drives live notifications; secondary runs in parallel for
# thesis comparison only and is never on the critical path.
OLLAMA_MODEL_PRIMARY   = "llama3.1:8b"
OLLAMA_MODEL_SECONDARY = "qwen3.5:4b"
OLLAMA_MODEL           = OLLAMA_MODEL_PRIMARY   # alias used by WindowClassifier

# When True, both models are called for every recommendation; results are
# saved side-by-side in the llm_comparisons table for thesis evaluation.
# Set False to revert to single-model operation.
LLM_COMPARISON_MODE = True

OLLAMA_TIMEOUT           = 30   # seconds; primary falls back to template if exceeded
OLLAMA_SECONDARY_TIMEOUT = 60   # seconds; Qwen runs in background — longer wait
                                # does not delay the notification

# Independent judge model used by src/evaluation/llm_evaluator.py to score
# Llama vs Qwen responses with DeepEval metrics.  Not used during live
# monitoring — only needed when running the evaluator post-hoc.
OLLAMA_MODEL_JUDGE = "gemma4:12b"

# ── Baseline calibration ─────────────────────────────────────────────────────
BASELINE_DURATION_SECONDS = 120   # 2-minute resting calibration session

# ── Diagnostics ───────────────────────────────────────────────────────────────
# Set True to print per-window aggregation detail (desktop sample rows, row
# counts) to the terminal.  Keep False during normal use to reduce noise.
AGGREGATOR_VERBOSE = False

# ── Face landmark detection ───────────────────────────────────────────────────
EAR_THRESHOLD    = 0.21   # below this → eye considered closed (Soukupova & Cech 2016)
BLINK_MIN_FRAMES = 2      # minimum consecutive closed frames to count as a blink
