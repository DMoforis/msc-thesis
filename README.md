# Multimodal Stress Detection & Mental Well-being Support System

**MSc Thesis — Proof of Concept Deliverable**  
Dimitris Moforis · Department of Digital Systems · University of Piraeus  
Supervisor: Professor Andreas Menychtas  

---

## Overview

This repository contains three independent proof-of-concept modules that together form the foundation of a multimodal system for detecting cognitive load and supporting mental well-being in knowledge workers (students, developers, analysts) who spend extended hours at a computer.

The system fuses three data sources:

| Module | Source | Key Signals |
|--------|--------|-------------|
| PoC #1 — Physiological | Webcam (rPPG) | Heart rate (BPM), RMSSD, LF/HF ratio |
| PoC #2 — Facial Analysis | Webcam (MediaPipe) | Blink rate, Eye Aspect Ratio (EAR), Head pose |
| PoC #3 — Desktop Context | Windows OS (Flutter) | Active window, App category, Idle time, Activity % |

All data is stored locally in a single SQLite database (`data/stress_monitor.db`). No data leaves the machine.

---

## System Architecture

```
stress_detection/
├── src/
│   ├── physio/
│   │   └── rppg_monitor.py        ← PoC #1: Heart rate + HRV via rPPG
│   ├── face/
│   │   └── face_monitor.py        ← PoC #2: Facial landmarks + EAR
│   ├── desktop/                   ← PoC #3: Flutter desktop context app
│   │   └── lib/main.dart
│   ├── fusion/                    ← Late fusion layer (planned)
│   └── wellbeing/                 ← Intervention module (planned)
├── data/
│   └── stress_monitor.db          ← Shared SQLite database
├── docs/
│   └── screenshots/               ← PoC documentation screenshots
├── run_all.py                     ← Unified launcher for all modules
├── view_data.py                   ← Database summary viewer
└── requirements.txt
```

---

## Design Principles

- **Privacy-by-design** — no video, audio, or screenshots are stored; only extracted numerical features
- **Data minimisation** — one row per 30-second window per modality
- **Modularity** — each module is independently testable and replaceable
- **Local-first** — all processing and storage happens on-device
- **Late fusion** — each modality produces its own score; combination happens at the decision layer

---

## Prerequisites

### Python modules (PoC #1 and #2)

- Python 3.12
- Dependencies listed in `requirements.txt`

```powershell
# From the stress_detection/ folder with venv active:
pip install -r requirements.txt
```

### Flutter desktop app (PoC #3)

- Flutter SDK 3.x (stable channel)
- Visual Studio 2022+ with "Desktop development with C++" workload

```powershell
# Build the release executable once:
cd src/desktop
flutter pub get
flutter build windows --release
cd ../..
```

---

## Running the Modules

### Option A — Launch individually (recommended for PoC stage)

PoC #1 and PoC #2 both require exclusive webcam access. At this stage
they are run separately. A shared camera feed is planned for the
integrated system (Phase 2).

```powershell
# PoC #1 — rPPG Heart Rate + HRV (run alone — requires webcam)
python src/physio/rppg_monitor.py

# PoC #2 — Facial Landmark Analysis (run alone — requires webcam)
python src/face/face_monitor.py

# PoC #3 — Desktop Context Monitor (run alongside either of the above)
python run_all.py --poc 3
```

### Option B — Launch PoC #2 + PoC #3 together

The face module and desktop module can run simultaneously since only
the face module uses the webcam.

```powershell
python run_all.py --poc 2
# Then in a second terminal:
python run_all.py --poc 3
```

### Known limitation — shared camera feed

PoC #1 (rPPG) and PoC #2 (face landmarks) cannot run simultaneously
in this PoC configuration as both require exclusive webcam access of the same camera.
The integrated system will resolve this by implementing a single shared
OpenCV capture thread that distributes frames to both modules, eliminating
the conflict. This is a planned task before Phase 2 integration.
However, if more than one camera is available after the necessary changes in the 
rppg_monitor.py or face_monitor.py 'WEBCAM_INDEX' are made to accomodate each camera

### View collected data

```powershell
python view_data.py           # last 5 readings per module
python view_data.py --rows 10 # last 10 readings
python view_data.py --all     # all readings
```

---

## Module Details

### PoC #1 — Physiological Signals (rPPG)

The purpose of this module is to extract the heart rate and HRV from the webcam feed using
remote photoplethysmography (rPPG). No wearable device required at the moment.

**Algorithm:** FacePhys (open-rppg) with neurokit2 HRV post-processing  
**Window:** 30 seconds  
**Output saved to:** `physio_readings` table

| Signal | Description | Stress relevance |
|--------|-------------|-----------------|
| Heart rate | Beats per minute | Elevated HR correlates with stress |
| RMSSD | Root mean square of successive RR differences (ms) | Lower RMSSD indicates reduced HRV, associated with stress |
| LF/HF ratio | Low frequency / high frequency power ratio | Higher ratio associated with sympathetic dominance (stress) |

![PoC #1 Preview Window](docs/screenshots/PoC%231.1_rPPG_HeartRate_Preview.png)
![PoC #1 Terminal Output](docs/screenshots/PoC%231.2_rPPG_HeartRate_Terminal.png)

---

### PoC #2 — Facial Landmark Analysis

Detects 468 facial landmarks per frame using MediaPipe Face Mesh and extracts fatigue and attention indicators.

**Library:** MediaPipe Face Mesh (CPU, real-time)  
**Window:** 30 seconds  
**Output saved to:** `face_readings` table

| Signal | Description | Stress relevance |
|--------|-------------|-----------------|
| Blink rate | Blinks per minute | Elevated rate (>25/min) indicates fatigue |
| Mean EAR | Eye Aspect Ratio (0–1) | Lower values indicate drowsiness |
| Head pitch | Up/down head rotation (degrees) | Drooping head signals fatigue |
| Head yaw | Left/right head rotation (degrees) | Large values suggest distraction |

![PoC #2 Preview Window](docs/screenshots/PoC%232.1_FaceMesh_EAR_Preview.png)
![PoC #2 Terminal Output](docs/screenshots/PoC%232.2_FaceMesh_EAR_Terminal.png)

---

### PoC #3 — Desktop Context Monitor

A native Windows desktop app built with Flutter that monitors user activity using Windows APIs. Runs alongside the Python modules.

**Framework:** Flutter (Dart) with win32 Windows API bindings  
**Activity detection:** `GetLastInputInfo()` polling (system-wide, no hooks required)  
**Window:** 30 seconds  
**Output saved to:** `desktop_readings` table

| Signal | Description | Stress relevance |
|--------|-------------|-----------------|
| App category | IDE, Browser, Communication, Document, Media | Task type proxy for cognitive load |
| Idle seconds | Seconds with no keyboard/mouse input | Low activity may indicate fatigue or disengagement |
| Activity % | Percentage of window with active input | High sustained activity correlates with workload |
| Window switches | Number of application changes | Frequent switching indicates task-switching stress |

![PoC #3 Desktop App](docs/screenshots/PoC%233.1_Desktop_ActivityMonitor.png)

---

## Data Schema

All three modules write to `data/stress_monitor.db`:

```sql
-- PoC #1
CREATE TABLE physio_readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT,
    heart_rate     REAL,   -- BPM
    rmssd          REAL,   -- ms
    lf_hf_ratio    REAL,
    signal_quality REAL,   -- 0–1
    modality       TEXT    -- 'rppg_face'
);

-- PoC #2
CREATE TABLE face_readings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT,
    blink_rate        REAL,   -- blinks/min
    mean_ear          REAL,   -- 0–1
    pitch_deg         REAL,
    yaw_deg           REAL,
    roll_deg          REAL,
    face_detected_pct REAL    -- % of window
);

-- PoC #3
CREATE TABLE desktop_readings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT,
    active_window    TEXT,
    app_category     TEXT,
    idle_seconds     REAL,
    activity_pct     REAL,   -- % of window active
    window_switches  INTEGER
);
```

---

## Next Steps

- [ ] Late fusion layer — combine per-modality scores into a unified stress index
- [ ] Well-being intervention module — trigger micro-interventions based on stress score
- [ ] Pilot study — 10–20 knowledge workers, 1–2 weeks, self-reported stress validation
- [ ] Replace rPPG with smartwatch HRV when hardware becomes available

---

*University of Piraeus · Department of Digital Systems*  
*MSc Programme: Information Systems & Services*