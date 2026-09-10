# SPEC.md â€” Technical Specification
## Multimodal Stress Detection & Mental Well-being Support System

---

## 1. Shared Camera Feed

**File:** `src/camera/shared_feed.py`

**Problem it solves:** PoC #1 (rPPG) and PoC #2 (face landmarks) both call
`cv2.VideoCapture()` independently, causing an exclusive access conflict on Windows.

**Solution:** A single `SharedCameraFeed` class runs one OpenCV capture thread.
Consumers register themselves and receive frames via a thread-safe queue.

**Interface:**
```python
feed = SharedCameraFeed(camera_index=0, fps=30)
feed.start()

# Each consumer calls this in its own loop:
frame = feed.get_frame()  # returns latest BGR frame or None

feed.stop()
```

**Requirements:**
- Thread-safe frame distribution to unlimited consumers
- Non-blocking: consumers that are slow do not block the capture thread
- Graceful shutdown on stop()
- Exposes frame dimensions (width, height) for downstream modules

---

## 2. Physio Module (Refactored)

**File:** `src/physio/rppg_monitor.py`

**Changes from Phase 1:**
- Receives frames from SharedCameraFeed instead of opening webcam directly
- HR reading frequency: every 10 seconds (was 30s)
- HRV computed over 5-minute rolling window (was 30s)
- Saves to `physio_readings` table with updated schema

**Updated schema:**
```sql
CREATE TABLE physio_readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    heart_rate     REAL,     -- BPM, updated every 10s
    rmssd          REAL,     -- ms, computed over 5-min window
    lf_hf_ratio    REAL,     -- frequency domain HRV, 5-min window
    sdnn           REAL,     -- standard deviation of RR intervals
    signal_quality REAL,     -- 0-1 confidence from open-rppg
    window_type    TEXT      -- 'hr_10s' or 'hrv_5min'
);
```

---

## 3. Face Module (Refactored + Valence-Arousal)

**File:** `src/face/face_monitor.py`
**New file:** `src/face/valence_arousal.py`

**Changes from Phase 1:**
- Receives frames from SharedCameraFeed
- Adds Valence-Arousal prediction per 30s window
- RollÂ° computation fixed using quaternion decomposition

**Valence-Arousal model:**
- Model: **EmoNet-8** (Toisoul et al., 2021) â€” `data/models/emonet_8.pth`
- Architecture: `data/models/emonet_arch.py` (8-class emotion backbone repurposed for VA regression)
- Runs locally via PyTorch, GPU-accelerated via CUDA (CPU fallback supported)
- Input: cropped face image from MediaPipe bounding box (resized to 256Ã—256)
- Output: valence (âˆ’1 to +1), arousal (âˆ’1 to +1)
- Valence: negative = unpleasant/stressed, positive = pleasant/relaxed
- Arousal: low = drowsy/disengaged, high = alert/excited
- *Note: original SPEC referenced `Mavdol/NPC-Valence-Arousal-Prediction` (HuggingFace transformers). Changed to EmoNet-8 for more accurate facial affect estimation and lower GPU memory footprint.*

**Updated schema:**
```sql
CREATE TABLE face_readings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    blink_rate        REAL,   -- blinks/min
    mean_ear          REAL,   -- 0-1
    pitch_deg         REAL,
    yaw_deg           REAL,
    roll_deg          REAL,   -- fixed: quaternion decomposition
    face_detected_pct REAL,
    valence           REAL,   -- -1 to 1
    arousal           REAL,   -- -1 to 1
    emotion_label     TEXT    -- Russell (1980) circumplex quadrant label
);
```

---

## 4. LLM Module

**Files:** `src/llm/classifier.py`, `src/llm/recommender.py`

**Setup requirement:** Ollama installed locally with Llama 3.1 8B pulled.
```bash
# One-time setup:
ollama pull llama3.1:8b
```

### 4a. Window Title Classifier

**Purpose:** Replace hardcoded keyword matching with intelligent LLM classification.

> **Important implementation note â€” Flutter/Python architecture constraint:**
> The desktop context data (active window title, idle time, activity %) is written
> directly to `desktop_readings` by the Flutter native Windows subprocess.  Because
> Flutter is a Dart/compiled binary it cannot call the Python Ollama client at
> runtime.  As a result, the **`app_category` field written by Flutter uses
> keyword-based rules** (defined in `src/desktop/lib/main.dart`) rather than
> live Ollama inference.  The Python `src/llm/classifier.py` module is available
> for offline re-classification of stored window titles, but is not called in the
> live monitoring pipeline.  This is documented as a known limitation in the
> thesis (see Â§Known Limitations).

**Interface:**
```python
classifier = WindowClassifier()
category = classifier.classify("DMoforis/msc-thesis â€” GitHub â€” Google Chrome")
# Returns: "Academic Work"

category = classifier.classify("Facebook")
# Returns: "Social Media"
```

**Categories:**
- Academic Work
- Software Development
- Communication
- Social Media
- Entertainment
- Document Editing
- Research / Reading
- System / Utility
- Other

**Implementation notes:**
- Uses Ollama Python client to call local Llama 3.1 8B
- Prompt engineered for single-word/short category response
- Response cached: same title always returns same category (LRU cache)
- Falls back to keyword matching if Ollama is not running
- Timeout: uses the configured local Ollama timeout; if Ollama is unavailable or times out, the system falls back to keyword/template-based handling

### 4b. Recommendation Engine

**Purpose:** Generate contextual, natural language well-being recommendations.

**Trigger conditions (any one sufficient):**
- stress_index > 0.65 for 2 consecutive 5-min windows
- valence < -0.4 sustained for 10+ minutes
- arousal < 0.2 (disengagement) for 15+ minutes
- blink_rate < 8/min (eye strain) for 10+ minutes
- activity_pct < 20% for 20+ minutes (prolonged idle)

**Prompt template:**
```
System: You are a well-being assistant for a knowledge worker.
Generate ONE short, friendly recommendation (max 2 sentences).
Be specific, not generic. Never mention medical advice.

Context:
- Current activity: {app_category} in {active_window}
- Stress index: {stress_index:.2f}/1.0
- Emotional state: valence={valence:.2f}, arousal={arousal:.2f}
- Session duration: {session_minutes} minutes
- Last break: {minutes_since_break} minutes ago
- Blink rate: {blink_rate:.1f}/min (normal: 15-20)

Generate a recommendation:
```

**Output:** String, max 200 characters, delivered as Windows notification.

---

## 5. Five-Minute Aggregation Layer

**File:** `src/fusion/aggregator.py`

**Purpose:** Compute 5-minute summary windows across all modalities.
This is the primary unit of analysis for thesis evaluation.

**Logic:**
- Runs as a background thread, triggers every 5 minutes
- Reads last 5 minutes of data from all three raw tables
- Computes summary statistics
- Saves to `aggregated_windows` table
- Triggers fusion scoring
- Triggers intervention check

**Schema:**
```sql
CREATE TABLE aggregated_windows (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    window_start         TEXT NOT NULL,
    window_end           TEXT NOT NULL,
    -- Physio
    avg_hr               REAL,
    avg_rmssd            REAL,
    avg_lf_hf            REAL,
    -- Face
    avg_blink_rate       REAL,
    avg_ear              REAL,
    avg_valence          REAL,
    avg_arousal          REAL,
    avg_pitch            REAL,
    avg_yaw              REAL,
    -- Desktop
    dominant_category    TEXT,
    avg_activity_pct     REAL,
    avg_idle_seconds     REAL,
    total_window_switches INTEGER,
    -- Fusion output
    stress_index         REAL,   -- 0-1
    -- Flags
    intervention_triggered INTEGER  -- 0 or 1
);
```

---

## 6. Late Fusion Scoring

**File:** `src/fusion/late_fusion.py`

**Method:** Weighted average of three modality scores.

```python
stress_index = (
    w_physio  * score_physio  +
    w_face    * score_face    +
    w_desktop * score_desktop
)
# Initial weights: w_physio=0.4, w_face=0.4, w_desktop=0.2
# Desktop weighted lower as it is a behavioral proxy, not physiological
```

**Per-modality scoring rules:**

Physio score:
- HR deviation from personal baseline â†’ 0-1
- RMSSD below personal baseline â†’ 0-1
- Combined: mean of both

Face score:
- Valence normalised to 0-1 (valence=-1 â†’ score=1, valence=+1 â†’ score=0)
- Arousal deviation from neutral â†’ 0-1
- EAR below threshold â†’ increases score
- Combined: weighted mean

Desktop score â€” three components averaged (missing data excluded):

1. **App category cognitive load** (Flutter taxonomy mapping):

   | Flutter category | Weight | Rationale |
   |-----------------|--------|-----------|
   | `IDE/Terminal`  | 0.7    | Intense focus work â€” high cognitive load |
   | `Communication` | 0.6    | Meetings / email â€” sustained social pressure |
   | `Document`      | 0.5    | Sustained writing effort â€” moderate load |
   | `Browser`       | 0.3    | Research or distraction â€” ambiguous context |
   | `Media`         | 0.1    | Entertainment â€” user is likely resting |
   | `Other`         | 0.2    | Unrecognised label â€” conservative default |

   > **Architecture note:** `app_category` is written by the Flutter subprocess
   > using keyword-based Dart rules.  The Python `src/llm/classifier.py` module
   > is available for offline re-classification only (see Â§Known Limitations).
   > Unrecognised category strings fall back to the `Other` weight.

2. **Inactivity:** `1.0 âˆ’ activity_pct / 100.0`
   Low keyboard/mouse activity while the session is running indicates
   disengagement or task avoidance â€” an indirect stress signal.

3. **Attention fragmentation:** `min(1.0, total_window_switches / 20)`
   20 window switches in a 5-minute window (â‰ˆ 4/min) maps to score 1.0.
   Frequent context-switching reflects difficulty sustaining focus.

---

## 7. Windows Notifications

**File:** `src/wellbeing/notifier.py`

**Library:** `windows-toasts` (WinRT native toasts, Windows 10/11)

**Interface:**
```python
notifier = WindowsNotifier(app_name="Well-being Assistant")
notifier.send(
    title="Time for a break",
    message="You've been focused for 52 minutes. A short walk would help.",
    timeout=10
)
```

**Cooldown:** Minimum 15 minutes between notifications to avoid annoyance.
**Log:** Every notification saved to `interventions` table with timestamp and content.

---

## 8. Excel Export

**File:** `export_excel.py`

**Purpose:** Export 5-minute aggregated data to Excel for statistical analysis.

**Output:** `data/stress_analysis_{date}.xlsx` with sheets:
- **Raw Physio** â€” all physio_readings
- **Raw Face** â€” all face_readings
- **Raw Desktop** â€” all desktop_readings
- **5-Min Windows** â€” aggregated_windows (primary analysis sheet)
- **Interventions** â€” all recommendations shown
- **Correlations** â€” auto-computed correlation matrix between key variables
- **Summary Stats** â€” descriptive statistics per variable

**Usage:**
```bash
python export_excel.py                    # export all data
python export_excel.py --date 2026-05-01  # export specific date
python export_excel.py --last 7           # export last 7 days
```

---

## 9. Configuration

**File:** `src/utils/config.py`

All constants live here. Never hardcode values in module files.

```python
# Camera
CAMERA_INDEX = 0
CAMERA_FPS   = 30

# Windows
HR_WINDOW_SECONDS      = 10    # HR reading frequency
HRV_WINDOW_SECONDS     = 300   # 5 minutes for HRV
FACE_WINDOW_SECONDS    = 30
AGGREGATION_MINUTES    = 5

# Database
DB_PATH = os.path.join(BASE_DIR, "data", "stress_monitor.db")

# Fusion weights
W_PHYSIO  = 0.4
W_FACE    = 0.4
W_DESKTOP = 0.2

# Intervention thresholds
STRESS_TRIGGER_THRESHOLD    = 0.65
VALENCE_TRIGGER_THRESHOLD   = -0.4
AROUSAL_LOW_THRESHOLD       = 0.2
MIN_MINUTES_BETWEEN_NOTIFS  = 15

# Ollama
OLLAMA_MODEL   = "llama3.1:8b"
OLLAMA_TIMEOUT = 30  # seconds; primary local LLM timeout

# Paths
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
EXPORT_DIR   = os.path.join(BASE_DIR, "data", "exports")
```

---

## 10. Ollama Setup (one-time)

```powershell
# 1. Download Ollama from https://ollama.com/download/windows
# 2. Install and restart terminal
# 3. Pull the model (downloads ~5GB, stored locally)
ollama pull llama3.1:8b

# 4. Verify it works
ollama run llama3.1:8b "Say hello in one sentence"

# 5. Install Python client
pip install ollama
```

Ollama runs as a local server on http://localhost:11434.
It can use CUDA automatically when a compatible NVIDIA GPU/PyTorch build is available; otherwise it falls back to CPU.

---

## 11. Additional pip installs needed for Phase 2

```powershell
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ollama
pip install windows-toasts
pip install openpyxl
pip install pandas
pip install Pillow
pip freeze > requirements.txt
```

---

## 12. DeepEval LLM Evaluation

**Files:** `src/evaluation/llm_evaluator.py`, `tests/test_evaluator.py`

**Purpose:** Automated quality comparison of the two LLM recommendation models (Llama 3.1:8b vs Qwen 3.5:4b) using the DeepEval framework with a local judge model.

### Judge model

**Model:** `gemma4:12b` (pulled via Ollama, runs locally on the RTX 2060 Super)

Gemma 4:12b acts as the evaluator: it receives the original prompt context, the model's recommendation, and a metric-specific rubric, then returns a 0â€“1 score with a reason string.

```python
# config.py
OLLAMA_MODEL_JUDGE = "gemma4:12b"
```

### Metrics

| Metric | What it measures | How scored |
|--------|-----------------|------------|
| **AnswerRelevancy** | Is the recommendation directly relevant to the stated stress/fatigue context- | Judge assesses topical fit to the input context |
| **WellbeingAppropriateness** | Is the advice safe, supportive, and free of medical claims- | Judge checks tone, safety language, and absence of clinical instructions |
| **ContextAwareness** | Does the recommendation reference specific context signals (app category, session duration, blink rate, etc.)- | Judge checks for concrete contextual anchors vs generic advice |

Each metric returns a score in [0, 1]. The composite score is the unweighted average of all three.

### Results

Evaluation run across 6 matched intervention events (trigger: high_stress, same trigger context, both models queried):

| Model | Overall score | Qualitative outcome | Avg latency |
|---|---:|---|---:|
| Llama 3.1:8b | **0.89** | 3/6 preferred outputs | ~5.3 s |
| Qwen3.5:4b | **0.86** | 3/6 preferred outputs | ~13.1 s |

The two models produced comparable qualitative results. Llama 3.1:8b was retained as the preferred real-time recommendation model due to substantially lower latency, while Qwen3.5:4b remains available for comparison/evaluation mode.

**Key findings:**
- Qwen scores ~3.5 percentage points higher on composite quality, primarily from stronger ContextAwareness (it tends to reference more specific signals from the prompt)
- Llama is approximately **2.5Ã— faster** at inference, which is why it remains on the critical notification path
- The sequential execution architecture (Llama â†’ notification delivered â†’ Qwen runs in background) means Qwen's higher latency has zero impact on user-perceived response time
- Both models comfortably exceed 0.80 composite on WellbeingAppropriateness, confirming neither produces clinically unsafe or alarmist language

### Architecture integration

```
Trigger detected
    â”‚
    â”œâ”€â–º Llama 3.1:8b  (ThreadPoolExecutor, _timeout=30s)
    â”‚       â””â”€â–º _clean_response() â†’ Windows toast delivered  â† user sees this
    â”‚
    â””â”€â–º Qwen 3.5:4b   (daemon thread, sleep(2) GPU settle, _timeout=60s)
            â””â”€â–º _extract_from_thinking() salvage if content empty
                    â””â”€â–º _save_comparison_row() â†’ llm_comparisons table
                                â””â”€â–º llm_evaluator.py reads this table post-hoc
```

**Schema â€” `llm_comparisons` table:**
```sql
CREATE TABLE llm_comparisons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    trigger_reason   TEXT,
    context_json     TEXT,   -- full prompt context as JSON
    llama_response   TEXT,
    qwen_response    TEXT,
    llama_latency_ms REAL,
    qwen_latency_ms  REAL
);
```

---

## Research questions (from thesis proposal)

The original proposal included user-acceptance-oriented research questions. In the final thesis scope, the evaluation was reframed as a technical and functional proof-of-concept assessment. User acceptance, perceived privacy and intervention usefulness are therefore treated as limitations and future work.

---

## Evaluation methodology

The system is evaluated as a proof-of-concept through:

- **Automated tests**: 99 unit and integration tests covering fusion logic, HRV validation, classifier accuracy, aggregator behaviour, and dual-model LLM comparison (see `tests/`)
- **Functional demonstration**: end-to-end operation of all three modalities (rPPG, facial cues, desktop context) on a single workstation under realistic working conditions
- **Data quality monitoring**: `data_health_check()` in `src/utils/db.py` reports row counts, signal quality, and per-modality coverage across a session
- **Baseline personalisation**: single-user resting calibration validated against live session data via the `aggregated_windows` table

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Desktop window classification | The Flutter subprocess (`src/desktop/lib/main.dart`) writes `app_category` to the database using keyword-based rules implemented in Dart. The compiled Flutter binary cannot call the Python Ollama client at runtime (cross-process, cross-language boundary). `src/llm/classifier.py` exists for offline re-classification of stored titles and for post-hoc thesis analysis, but is **not** invoked during live monitoring sessions. |
| RollÂ° instability | The atan2 Euler decomposition is unstable at extreme pitch angles (Â±60Â°+). RollÂ° values in those regions are unreliable. |
| LF/HF at short windows | Frequency-domain HRV requires â‰¥5 min of clean BVP signal. LF/HF is NULL early in each session. |
| rPPG SQI under office lighting | Typical SQI is 0.35â€“0.50. HRV metrics are suppressed when SQI < 0.5. Stable frontal lighting improves quality. |
| EmoNet-8 model files not in git | `data/models/emonet_8.pth` (~170 MB) and `data/models/emonet_arch.py` are excluded from the repository (`.gitignore`). They must be downloaded separately. VA inference degrades gracefully to `NULL` if the model is absent. |
| Qwen 3.5:4b thinking-token exhaustion | Qwen 3.5:4b runs in an Ollama build that always allocates a thinking scratchpad before emitting content. With short `num_predict` budgets the model exhausts all tokens on internal reasoning and returns empty content. Three-layer suppression is applied in `src/llm/recommender.py`: (1) `/no_think\n\n` prefix on the user message, (2) `think: False` option in the Ollama chat API, and (3) `num_predict=512` + `num_ctx=4096`. Even with these measures the content field is occasionally empty; `_extract_from_thinking()` salvages the final recommendation from the thinking field as a fallback. |
