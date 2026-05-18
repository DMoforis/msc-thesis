# Multimodal Stress Detection & Mental Well-being Support System

**MSc Thesis — Phase 2 Integrated System**  
Dimitris Moforis · Department of Digital Systems · University of Piraeus  
Supervisor: Professor Andreas Menychtas  
Student ID: ME2457

---

## Overview

This system detects cognitive stress in real time by fusing three independent data streams — physiological signals extracted from the webcam, continuous facial expression analysis, and desktop behavioural context — and delivers personalised well-being recommendations using a locally hosted large language model.

Designed for knowledge workers (students, developers, analysts) who spend extended hours at a computer. All processing is fully local: no video, audio, or raw biometric data leaves the machine. Only extracted numerical features are stored.

When elevated stress or sustained negative affect is detected, the system generates contextual natural-language recommendations via Llama 3.1 8B (Ollama) and delivers them as Windows desktop toast notifications. Stress scoring is individually calibrated: a two-minute resting baseline session records the user's personal HR, RMSSD, EAR, blink rate, valence, and arousal, which the fusion layer uses instead of fixed population means.

**Core output:** a `stress_index` in [0, 1] computed every five minutes, together with continuous valence and arousal estimates that characterise the user's emotional state on the Russell (1980) circumplex model of affect.

| Modality | Data source | Key signals |
|----------|-------------|-------------|
| Physiological | Webcam (rPPG) | Heart rate (BPM), RMSSD, LF/HF ratio |
| Facial analysis | Webcam (MediaPipe + EmoNet-8) | Blink rate, EAR, head pose, valence, arousal |
| Desktop context | Windows OS (Flutter + win32) | Active window, LLM-classified category, idle time, activity %, window switches |

---

## System Architecture

```
                        SHARED CAMERA FEED
                   src/camera/shared_feed.py
                   (single OpenCV capture thread)
                              |
               +--------------+--------------+
               v              v              v
       +-------------+ +------------+ +-------------------+
       |   PHYSIO    | |    FACE    | |      DESKTOP      |
       | rppg_monitor| | face_monitor| |  Flutter + win32  |
       |             | |            | |  (separate build) |
       | HR  / 10 s  | | EAR, blink | |  active window    |
       | HRV / 5 min | | head pose  | |  LLM category     |
       |             | | valence    | |  idle time        |
       |             | | arousal    | |  activity %       |
       +------+------+ +-----+------+ +--------+----------+
              |              |                  |
              v              v                  v
       physio_readings   face_readings    desktop_readings
                  \           |            /
                   \          |           /
                    v         v          v
              +-------------------------------+
              |   AGGREGATOR  (every 5 min)   |
              |   src/fusion/aggregator.py    |
              +---------------+---------------+
                              |
                              v
                   +----------------------+
                   |    LATE FUSION       |
                   |  physio  x 0.40      |
                   |  face    x 0.40      |
                   |  desktop x 0.20      |
                   +----------+-----------+
                              |  stress_index
                              v
                   +----------------------+
                   |  INTERVENTION ENGINE |
                   |  Llama 3.1 8B        |
                   |  (Ollama, GPU-local) |
                   |  -> Windows toast    |
                   +----------------------+
```

---

## Modules

| Module | File | Purpose |
|--------|------|---------|
| Shared camera feed | `src/camera/shared_feed.py` | Single OpenCV capture thread; distributes frames to physio and face modules without exclusive-access conflicts |
| Physio monitor | `src/physio/rppg_monitor.py` | rPPG → BVP signal → HR every 10 s, full HRV every 5 min |
| HRV processor | `src/physio/hrv_processor.py` | RMSSD computation and plausibility validation (5–80 ms bounds, HR cross-validation) |
| Face monitor | `src/face/face_monitor.py` | MediaPipe 468-landmark mesh → EAR, blink rate, head pose, every 30 s |
| Valence-Arousal | `src/face/valence_arousal.py` | EmoNet-8 (Toisoul et al., 2021) → continuous valence and arousal from face crop |
| Desktop monitor | `src/desktop/lib/main.dart` | Flutter + win32 → active window title, LLM-classified category, idle time, activity %, window switches |
| LLM classifier | `src/llm/classifier.py` | Llama 3.1 8B classifies window titles into activity categories (Academic Work, Software Development, Social Media, etc.) |
| LLM recommender | `src/llm/recommender.py` | Generates contextual, natural-language well-being recommendations based on the current stress context |
| Aggregator | `src/fusion/aggregator.py` | Background thread that merges raw readings into 5-minute summary windows and triggers fusion scoring |
| Late fusion | `src/fusion/late_fusion.py` | Weighted combination of per-modality stress scores; uses personal baselines for physio and face scoring |
| Intervention engine | `src/wellbeing/interventions.py` | Trigger logic: stress threshold, sustained negative valence, cooldown enforcement |
| Notifier | `src/wellbeing/notifier.py` | Delivers recommendations as Windows desktop toast notifications |
| Baseline calibrator | `src/utils/baseline.py` | 2-minute resting session that records personal HR, RMSSD, EAR, blink rate, valence, and arousal |

---

## Prerequisites

### Python environment

- Python 3.12
- A CUDA-capable GPU is recommended (NVIDIA GTX 1060 or better); CPU fallback is supported for all models

```powershell
# Create and activate a virtual environment (recommended):
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

To install PyTorch with CUDA 12.1 support (RTX series GPUs):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Key packages installed via `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `open-rppg` | rPPG heart rate and BVP extraction |
| `mediapipe` | Real-time face landmark detection |
| `torch`, `torchvision` | EmoNet-8 valence-arousal inference |
| `ollama` | Local LLM client (Llama 3.1 8B) |
| `neurokit2` | HRV signal processing utilities |
| `opencv-python` | Camera capture and frame processing |
| `pandas`, `openpyxl` | Excel export |
| `windows-toasts` | Native Windows 10/11 toast notifications (WinRT) |
| `colorama` | Colour-coded terminal output for daily summaries |

### Ollama — local LLM runtime

The window-title classifier and recommendation engine require Ollama installed locally.

1. Download and install Ollama from the official site.
2. Pull the required model (approximately 5 GB, downloaded once):

```powershell
ollama pull llama3.1:8b
ollama run llama3.1:8b "Say hello in one sentence"   # verify
```

### Flutter — desktop context monitor

- Flutter SDK 3.x (stable channel)
- Visual Studio 2022 or later with the **Desktop development with C++** workload

```powershell
cd src/desktop
flutter pub get
flutter build windows --release
cd ../..
```

The compiled executable is placed in `src/desktop/build/windows/x64/runner/Release/`.

---

## Running the System

### Step 1 — Personal baseline calibration (first run only)

Collect two minutes of resting physiological data to personalise the stress scoring. Sit comfortably in front of the camera and remain relaxed.

```powershell
python src/utils/baseline.py --calibrate
```

The calibrator records your resting HR, RMSSD, EAR, blink rate, valence, and arousal, and stores them in the `baselines` table. The late fusion layer uses these personal values instead of fixed population means, making stress scoring individually calibrated. Re-run if your baseline conditions change significantly (e.g. after illness or a change of environment).

To inspect the currently stored baseline:

```powershell
python src/utils/baseline.py --show
```

### Step 2 — Start Ollama (separate terminal)

The LLM classifier and recommendation engine require the Ollama inference server to be running:

```powershell
ollama serve
```

This starts the local server on `http://localhost:11434`. The model is loaded into GPU VRAM on the first request. If Ollama is not running, the classifier falls back to keyword matching and the recommender uses template-based messages.

### Step 3 — Launch the unified system

```powershell
python run_all.py
```

This starts all Python modules (physio monitor, face monitor, aggregator, intervention engine) sharing a single camera feed, and launches the Flutter desktop context monitor. A combined preview window displays the face mesh overlay with live HR and stress index annotations.

Press **Q** in the preview window to stop all modules cleanly.

---

## Data Collection

### How the aggregation cycle works

The system operates on a five-minute rolling window cycle. During each window:

1. **Raw feature extraction** — the physio, face, and desktop modules write readings to the database continuously at their respective rates: HR every 10 s; face metrics (EAR, blink rate, valence, arousal, head pose) every 30 s; desktop metrics every 30 s.
2. **Window aggregation** — at the end of each five-minute interval, the Aggregator reads all raw rows from the preceding window and computes per-modality summary statistics (mean HR, mean RMSSD, mean valence and arousal, dominant app category, total window switches, etc.).
3. **Fusion scoring** — the `LateFusion` module combines the per-modality summaries into a single `stress_index` in [0, 1], weighted as physio × 0.40, face × 0.40, desktop × 0.20. If a modality produced no data in the window, its weight is dropped and the remaining weights are renormalised. Physio and face scores are computed as deviations from the user's personal resting baseline.
4. **Intervention check** — if the `stress_index` exceeds 0.65 for two consecutive windows, or if mean valence falls below −0.4 in the current window, the intervention engine requests a natural-language recommendation from Llama 3.1 8B and delivers it as a Windows toast notification. A 15-minute cooldown prevents notification overload.

All four steps are fully automatic once `run_all.py` is running. The `aggregated_windows` table is the primary analysis unit for the thesis.

### Monitoring data quality

A built-in data quality report can be printed at any time without stopping the system:

```powershell
python -c "from src.utils.db import data_health_check; data_health_check()"
```

This prints row counts for all tables, rPPG signal quality statistics, RMSSD plausibility metrics, desktop timestamp format distribution, and modality coverage per aggregated window.

### Session review

After a monitoring session, a human-readable daily summary can be generated for any date:

```powershell
python daily_summary.py                      # today
python daily_summary.py --date 2026-05-18    # specific date
```

The summary prints colour-coded statistics — stress index, heart rate, valence and arousal with quadrant interpretation, dominant app category, total window switches, and any interventions triggered — directly to the terminal. A plain-text copy is saved automatically to `data/exports/daily_summary_YYYY-MM-DD.txt`.

---

## Viewing and Exporting Data

### Terminal data viewer

```powershell
python view_data.py            # last 5 readings per module
python view_data.py --rows 10  # last N readings
python view_data.py --all      # full database dump
```

### Excel export

Exports all collected data to a multi-sheet workbook for descriptive statistics and correlation analysis (thesis evaluation).

```powershell
python export_excel.py                    # all data
python export_excel.py --date 2026-05-01  # specific date
python export_excel.py --last 7           # last 7 days
```

Output file: `data/exports/stress_analysis_<date>.xlsx`

| Sheet | Contents |
|-------|----------|
| Raw Physio | All `physio_readings` rows |
| Raw Face | All `face_readings` rows |
| Raw Desktop | All `desktop_readings` rows |
| 5-Min Windows | `aggregated_windows` — primary analysis sheet |
| Interventions | All recommendations delivered to the user |
| Correlations | Correlation matrix between key stress indicators |
| Summary Stats | Descriptive statistics per variable |

---

## Data Schema

All modules write to `data/stress_monitor.db` (SQLite, WAL journal mode):

```sql
-- Physiological readings (rPPG)
-- window_type: 'hr_10s' (every 10 s) or 'hrv_5min' (every 5 min)
CREATE TABLE physio_readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    heart_rate     REAL,   -- BPM
    rmssd          REAL,   -- ms  (hrv_5min rows only; validated 5-80 ms)
    lf_hf_ratio    REAL,   -- frequency-domain HRV (hrv_5min rows only)
    sdnn           REAL,   -- ms
    signal_quality REAL,   -- 0-1 (open-rppg SQI)
    window_type    TEXT
);

-- Facial analysis (MediaPipe + EmoNet-8)
CREATE TABLE face_readings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    blink_rate        REAL,   -- blinks/min
    mean_ear          REAL,   -- Eye Aspect Ratio 0-1
    pitch_deg         REAL,   -- head tilt up/down
    yaw_deg           REAL,   -- head turn left/right
    roll_deg          REAL,   -- head tilt left/right
    face_detected_pct REAL,   -- % of 30-s window with face visible
    valence           REAL,   -- -1 (unpleasant) to +1 (pleasant)
    arousal           REAL    -- -1 (drowsy) to +1 (alert)
);

-- Desktop context (Flutter + win32)
CREATE TABLE desktop_readings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    active_window    TEXT,
    app_category     TEXT,   -- LLM-classified activity type
    idle_seconds     REAL,
    activity_pct     REAL,   -- % of window with keyboard/mouse input
    window_switches  INTEGER
);

-- 5-minute aggregated windows (primary analysis unit)
CREATE TABLE aggregated_windows (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    window_start            TEXT NOT NULL,
    window_end              TEXT NOT NULL,
    avg_hr                  REAL,
    avg_rmssd               REAL,
    avg_lf_hf               REAL,
    avg_blink_rate          REAL,
    avg_ear                 REAL,
    avg_valence             REAL,
    avg_arousal             REAL,
    avg_pitch               REAL,
    avg_yaw                 REAL,
    dominant_category       TEXT,
    avg_activity_pct        REAL,
    avg_idle_seconds        REAL,
    total_window_switches   INTEGER,
    stress_index            REAL,   -- 0-1
    intervention_triggered  INTEGER -- 0 or 1
);

-- Personal resting baseline (calibrated per user)
CREATE TABLE baselines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    hr          REAL,   -- resting heart rate (BPM)
    rmssd       REAL,   -- resting RMSSD (ms)
    ear         REAL,   -- resting Eye Aspect Ratio
    blink_rate  REAL,   -- resting blink rate (blinks/min)
    valence     REAL,   -- resting valence (-1 to +1)
    arousal     REAL    -- resting arousal (-1 to +1)
);

-- Intervention and feedback log
CREATE TABLE interventions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    trigger_reason TEXT,
    message        TEXT,
    stress_index   REAL,
    valence        REAL,
    arousal        REAL,
    delivered      INTEGER DEFAULT 1
);

CREATE TABLE user_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id INTEGER REFERENCES interventions(id),
    timestamp       TEXT NOT NULL,
    rating          INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment         TEXT
);
```

---

## Design Principles

- **Privacy-by-design** — no video, audio, or screenshots are stored; only extracted numerical features
- **Local-first** — all inference (rPPG, MediaPipe, EmoNet-8, Llama 3.1) runs on-device; no network calls after the initial model download
- **Personal baseline** — stress scoring uses individual resting measurements rather than fixed population thresholds
- **Graceful degradation** — each modality is independent; if data is unavailable the modality is excluded and remaining weights are renormalised
- **Modularity** — each module has a single responsibility and is independently testable (see `tests/`)

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Single camera constraint | All camera-dependent modules (rPPG, face) share the `SharedCameraFeed` singleton. Direct `cv2.VideoCapture()` calls within individual modules are not permitted. |
| LF/HF at short windows | Frequency-domain HRV requires at least 5 minutes of clean signal. LF/HF values will be `NULL` in the early phase of each session. |
| Roll° instability | OpenCV `RQDecomp3x3` exhibits gimbal-lock artefacts at extreme pitch angles. Roll° readings beyond ±60° should be treated as unreliable. |
| rPPG signal quality | The rPPG SQI under typical office lighting is approximately 0.35–0.50. HRV metrics are suppressed when SQI < 0.5. Stable frontal lighting improves signal quality. |
| RMSSD range validation | RMSSD values outside 5–80 ms are rejected before storage. The upper bound is tighter than ECG-based guidelines (120 ms) because rPPG-derived peak detection produces greater inter-beat jitter, inflating RMSSD estimates under low-SQI conditions. |
| Desktop timestamp normalisation | The Flutter app writes ISO 8601 timestamps with a `T` separator; Python modules use a space separator. The database query layer normalises this automatically via `REPLACE(timestamp, 'T', ' ')`. |

---

## Repository Structure

```
stress_detection/
├── CLAUDE.md                          <- project context for Claude Code sessions
├── SPEC.md                            <- full technical specification
├── README.md                          <- this file
├── requirements.txt
├── run_all.py                         <- unified launcher (all modules)
├── view_data.py                       <- terminal database viewer
├── export_excel.py                    <- Excel export script
├── daily_summary.py                   <- colour-coded daily summary for participants
|
├── data/
|   ├── stress_monitor.db              <- shared SQLite database (WAL mode)
|   ├── exports/                       <- Excel workbooks and daily summary txt files
|   └── models/                        <- cached model weights (EmoNet-8)
|
├── docs/
|   └── screenshots/
|
├── src/
|   ├── camera/
|   |   └── shared_feed.py             <- single OpenCV capture thread
|   |
|   ├── physio/
|   |   ├── rppg_monitor.py            <- HR every 10 s, HRV every 5 min
|   |   └── hrv_processor.py           <- RMSSD computation and validation helpers
|   |
|   ├── face/
|   |   ├── face_monitor.py            <- EAR, blink rate, head pose (30 s)
|   |   ├── ear_blink.py               <- EAR formula + blink event detection
|   |   ├── head_pose.py               <- solvePnP head pose estimation
|   |   └── valence_arousal.py         <- EmoNet-8 continuous VA prediction
|   |
|   ├── desktop/
|   |   └── lib/main.dart              <- Flutter desktop context monitor
|   |
|   ├── llm/
|   |   ├── classifier.py              <- Ollama window-title classification
|   |   └── recommender.py             <- Ollama recommendation generation
|   |
|   ├── fusion/
|   |   ├── late_fusion.py             <- weighted per-modality score combination
|   |   └── aggregator.py              <- 5-minute aggregation background thread
|   |
|   ├── wellbeing/
|   |   ├── interventions.py           <- intervention trigger logic
|   |   └── notifier.py                <- Windows desktop toast notifications
|   |
|   └── utils/
|       ├── db.py                      <- shared SQLite helpers (WAL, row factory)
|       ├── config.py                  <- all configuration constants
|       ├── baseline.py                <- personal baseline calibration
|       └── logger.py                  <- structured logging
|
└── tests/
    ├── conftest.py                    <- shared fixtures (tmp_db, sample_landmarks, sample_bvp)
    ├── test_ear.py                    <- EAR formula and blink detection
    ├── test_hrv.py                    <- RMSSD computation and validation
    ├── test_fusion.py                 <- LateFusion scoring
    ├── test_db.py                     <- database helpers and timestamp normalisation
    └── test_llm.py                    <- LLM classifier smoke tests
```

---

## Screenshots

### PoC #1 — Physiological Signals (rPPG)

Heart rate and HRV extraction from the webcam using remote photoplethysmography. No wearable device required.

![PoC #1 Preview Window](docs/screenshots/PoC%231.1_rPPG_HeartRate_Preview.png)
![PoC #1 Terminal Output](docs/screenshots/PoC%231.2_rPPG_HeartRate_Terminal.png)

---

### PoC #2 — Facial Landmark Analysis

MediaPipe Face Mesh detecting 468 landmarks per frame. Extracts Eye Aspect Ratio, blink rate, and head pose.

![PoC #2 Preview Window](docs/screenshots/PoC%232.1_FaceMesh_EAR_Preview.png)
![PoC #2 Terminal Output](docs/screenshots/PoC%232.2_FaceMesh_EAR_Terminal.png)

---

### PoC #3 — Desktop Context Monitor

Flutter native Windows app monitoring active window, application category, idle time, and activity percentage.

![PoC #3 Desktop App](docs/screenshots/PoC%233.1_Desktop_ActivityMonitor.png)

---

### PoC #4 — Unified Launcher and Data Viewer

Combined run_all.py launcher with all PoC modules active, and the view_data.py summary table.

![PoC #4 Summary Table](docs/screenshots/PoC%234.1_SummaryTable.png)
![PoC #4 All Modules](docs/screenshots/PoC%234.2_AllModules.png)

---

## Full System Screenshots (Phase 2)

### Combined preview — face mesh, rPPG bounding box, HR, and stress index

The Phase 2 unified launcher (`run_all.py`) produces a single annotated camera window showing: the MediaPipe face mesh overlay, the rPPG bounding box (orange) used for BVP signal extraction, live HR reading (77 BPM) and stress index (0.000) in the bottom-left corner. The Flutter Desktop Context Monitor runs alongside, displaying the active window title, LLM-classified activity category, idle time, activity percentage, and window switch count.

![Phase 2 — Face Capture and Desktop Module](docs/screenshots/FullModule%231.1_FaceCapture%2BDesktopModule.png)

---

### Baseline calibration session

The two-minute resting calibration (`python src/utils/baseline.py --calibrate`) overlays real-time face mesh, EAR reading (0.335), blink count, head pose pitch and yaw angles, and a countdown timer at the bottom of the preview window (65 seconds remaining shown). The collected averages are saved to the `baselines` table and used by the fusion layer for all subsequent sessions.

![Phase 2 — Baseline Calibration](docs/screenshots/FullModule%231.2_Calibration.png)

---

## Academic References

1. **Soukupova & Cech (2016)** — Real-time eye blink detection using facial landmarks. *21st Computer Vision Winter Workshop.*
2. **Schmidt et al. (2018)** — Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *Proceedings of ICMI 2018.*
3. **Makowski et al. (2021)** — NeuroKit2: a Python toolbox for neurophysiological signal processing. *Behavior Research Methods, 53*, 1689–1178.
4. **Toisoul et al. (2021)** — Estimation of continuous valence and arousal levels from faces in naturalistic conditions. *Nature Machine Intelligence, 3*, 42–50.
5. **Russell (1980)** — A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178.

---

*University of Piraeus · Department of Digital Systems*  
*MSc Programme: Information Systems & Services*

