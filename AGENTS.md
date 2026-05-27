# AGENTS.md — Multimodal Stress Detection System
## Project context for Codex sessions

---

## Who I am
Dimitris Moforis, MSc student, Department of Digital Systems, University of Piraeus.
Supervisor: Professor Andreas Menychtas (emobot.gr research group).
Student ID: ME2457

---

## What this project is
A multimodal stress detection and mental well-being support system for knowledge workers
(students, developers, analysts) who spend extended hours at a computer.

The system fuses three data streams in real time:
1. **Physiological signals** — Heart rate + HRV via rPPG from webcam
2. **Facial analysis** — Landmarks, EAR, blink rate, head pose, Valence-Arousal emotion model
3. **Desktop context** — Active window, app category (LLM-classified), idle time, activity %

All processing is **fully local** — no data leaves the machine. Privacy-by-design.

---

## Supervisor feedback (incorporated into all development)
- Use rPPG from webcam instead of smartwatch (avoids sync complexity)
- Take more frequent HR readings (every 10s, not 30s)
- Aggregate all data into **5-minute windows** for analysis
- Implement **Valence-Arousal model** for emotional state (not just binary stress)
- Use **local LLM (Llama via Ollama)** for intelligent window/tab classification
- Use **local LLM** to generate natural language well-being recommendations
- Export aggregated data to **Excel** for descriptive statistics and correlation analysis
- All modules must share **one camera feed** — no exclusive webcam access conflicts
- Keep everything local, modular, build one subsystem at a time

---

## Hardware (development machine)
- CPU: AMD Ryzen 7 3700X (8-core, 16 logical)
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 2060 Super (8GB VRAM)
- OS: Windows 11
- Python: 3.12.10
- IDE: VS Code

---

## Technology stack
| Component | Technology |
|---|---|
| Language (Python modules) | Python 3.12 |
| Language (desktop app) | Dart / Flutter |
| Face landmarks + mesh | MediaPipe Face Mesh |
| rPPG heart rate | open-rppg (FacePhys model) |
| HRV computation | neurokit2 |
| Valence-Arousal | EmoNet-8 (Toisoul et al. 2021), PyTorch, GPU via CUDA |
| Local LLM | Ollama + Llama 3.1 8B (GPU via CUDA) |
| Database | SQLite (shared, local) |
| Excel export | openpyxl + pandas |
| Windows notifications | windows-toasts (WinRT) |
| Desktop context | Flutter + win32 Windows API |
| Fusion layer | Python, weighted late fusion |

---

## Architecture — five layers

### Layer 1: Shared camera feed
Single OpenCV capture thread distributes frames to all consumers.
No module opens the camera directly — they all receive frames from the shared feed.
This resolves the exclusive webcam access conflict between rPPG and face modules.

### Layer 2: Feature extraction (per modality)
Each module processes frames independently:
- **PhysioModule**: rPPG → BVP signal → HR (every 10s) + HRV (every 5min)
- **FaceModule**: MediaPipe landmarks → EAR, blink rate, head pose, Valence-Arousal
- **DesktopModule**: Flutter app → active window, LLM category, idle time, activity %

### Layer 3: Storage
SQLite database at `data/stress_monitor.db` with tables:
- `physio_readings` — HR, RMSSD, LF/HF, signal quality (every 10s)
- `face_readings` — EAR, blink rate, pitch, yaw, valence, arousal (every 30s)
- `desktop_readings` — window, category, idle_seconds, activity_pct (every 30s)
- `aggregated_windows` — 5-minute summaries across all modalities
- `interventions` — log of all recommendations shown to user
- `user_feedback` — user ratings of interventions (1-5)

### Layer 4: Fusion + scoring
Late fusion: each modality produces a score (0-1), combined via weighted average.
Weights start equal (0.33 each), later tuned via pilot study data.
Output: `stress_index` (0-1) + `valence` (-1 to 1) + `arousal` (-1 to 1)

### Layer 5: Intervention engine
Local Llama 3.2 8B via Ollama generates natural language recommendations.
Triggered when stress_index > threshold OR valence < -0.3 OR arousal deviates from baseline.
Delivered as Windows desktop notifications.
Examples:
- "You've been in deep focus for 47 minutes with rising stress — a 5-minute break would help."
- "Your engagement seems to be drifting. Consider closing distracting tabs and refocusing."

---

## Repository structure
```
stress_detection/
├── AGENTS.md                          ← this file
├── SPEC.md                            ← full technical specification
├── README.md                          ← user-facing documentation
├── requirements.txt
├── run_all.py                         ← unified launcher
├── view_data.py                       ← terminal data viewer
├── export_excel.py                    ← Excel export script
├── .vscode/settings.json
├── .gitignore
│
├── data/
│   └── stress_monitor.db
│
├── docs/
│   └── screenshots/
│
├── src/
│   ├── camera/
│   │   └── shared_feed.py             ← single OpenCV capture thread
│   │
│   ├── physio/
│   │   ├── rppg_monitor.py            ← PoC #1 (refactored for shared feed)
│   │   └── hrv_processor.py           ← HRV computation via neurokit2
│   │
│   ├── face/
│   │   ├── face_monitor.py            ← PoC #2 (refactored for shared feed)
│   │   └── valence_arousal.py         ← EmoNet-8 VA model (Toisoul et al. 2021)
│   │
│   ├── desktop/
│   │   └── lib/main.dart              ← PoC #3 Flutter app (keyword classification)
│   │
│   ├── llm/
│   │   ├── classifier.py              ← Ollama window title classification (offline)
│   │   └── recommender.py             ← Ollama recommendation generation
│   │
│   ├── fusion/
│   │   ├── late_fusion.py             ← weighted score combination
│   │   └── aggregator.py              ← 5-minute aggregation + 6-trigger engine
│   │
│   ├── wellbeing/
│   │   └── notifier.py                ← Windows desktop notifications
│   │
│   ├── ui/
│   │   └── dashboard.py               ← PyQt6 three-panel dashboard
│   │
│   └── utils/
│       ├── db.py                      ← shared database helpers
│       ├── config.py                  ← all configuration constants
│       ├── baseline.py                ← 2-minute resting calibration
│       └── emotion_labels.py          ← Russell (1980) VA → quadrant label mapping
│
└── tests/                             ← 71 tests
    ├── conftest.py
    ├── test_ear.py
    ├── test_hrv.py
    ├── test_fusion.py
    ├── test_aggregator.py
    ├── test_db.py
    └── test_emotion_labels.py
```

---

## Development phases

### Phase 1 — COMPLETE ✅
- PoC #1: rPPG heart rate + HRV (open-rppg + neurokit2)
- PoC #2: Facial landmarks, EAR, blink rate, head pose (MediaPipe)
- PoC #3: Desktop context monitor (Flutter + Windows API)
- Shared SQLite database
- Unified launcher + data viewer + README

### Phase 2 — IN PROGRESS
- [ ] Shared camera feed (src/camera/shared_feed.py)
- [ ] Valence-Arousal model integration (src/face/valence_arousal.py)
- [ ] More frequent HR readings (every 10s)
- [ ] LLM window classification (src/llm/classifier.py)
- [ ] 5-minute aggregation layer (src/fusion/aggregator.py)
- [ ] Excel export (export_excel.py)
- [ ] LLM recommendation engine (src/llm/recommender.py)
- [ ] Windows notifications (src/wellbeing/notifier.py)

### Phase 3 — PLANNED
- [ ] Late fusion scoring layer
- [ ] Pilot study data collection (10-20 participants)
- [ ] Correlation analysis
- [ ] Thesis writing support

---

## Coding standards
- Python: snake_case, type hints, docstrings on all public functions
- Comments: in English, inline where non-obvious
- Each module has a single responsibility
- All file paths use os.path.join with absolute anchoring via __file__
- No hardcoded paths except in config.py
- All database access goes through src/utils/db.py
- Configuration constants live in src/utils/config.py only

---

## Known limitations from Phase 1 (to fix in Phase 2)
1. PoC #1 and #2 cannot run simultaneously — fixed by shared_feed.py
2. Roll° values unreliable due to gimbal lock in RQDecomp3x3 — fix with quaternions
3. LF/HF ratio always 0 at 30s windows — fixed by 5-minute aggregation
4. Window title classifier uses hardcoded keywords — fixed by LLM classifier
5. No Valence-Arousal emotional state — fixed by HuggingFace model

---

## Key academic references
- Soukupova & Cech (2016) — EAR blink detection
- Schmidt et al. (2018) — WESAD multimodal stress dataset
- Benedetto et al. (2011) — Eye blink and cognitive fatigue
- Makowski et al. (2021) — NeuroKit2
- Russell (1980) — Valence-Arousal circumplex model of affect
- Baltrusaitis et al. (2018) — OpenFace 2.0
