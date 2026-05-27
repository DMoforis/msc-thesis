# Installation Guide — Multimodal Stress Detection System

**MSc Thesis — Dimitris Moforis · University of Piraeus, Dept. of Digital Systems**

This guide walks a new user through setting up the system from scratch on a Windows 10/11 machine. Follow the steps in order. Estimated total time: 30–60 minutes (most of which is download time).

---

## Prerequisites

Install the following tools before beginning. All are free.

| Tool | Version | Download |
|------|---------|----------|
| **Python** | 3.12 (exact) | [python.org/downloads](https://www.python.org/downloads/) |
| **Git** | any recent | [git-scm.com/download/win](https://git-scm.com/download/win) |
| **Flutter SDK** | 3.x stable | [docs.flutter.dev/get-started/install/windows](https://docs.flutter.dev/get-started/install/windows) |
| **Visual Studio** | 2022 or later | [visualstudio.microsoft.com](https://visualstudio.microsoft.com/) — select the **"Desktop development with C++"** workload during installation |
| **Ollama** | latest | [ollama.com/download](https://ollama.com/download) |

> **GPU note:** An NVIDIA GPU (GTX 1060 or better) is strongly recommended. The EmoNet-8 valence-arousal model and Llama 3.1 8B both run on CPU if no CUDA-capable GPU is present, but inference will be noticeably slower. The rPPG module runs on CPU regardless.

---

## Step 1 — Clone the repository

Open **PowerShell** and run:

```powershell
git clone https://github.com/DMoforis/msc-thesis.git
cd msc-thesis
```

> If you received the project as a ZIP archive instead, extract it and `cd` into the extracted folder.

---

## Step 2 — Python environment setup

Create an isolated virtual environment and install all dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You should see `(venv)` at the start of your prompt, confirming the environment is active.

### CUDA-accelerated PyTorch (NVIDIA GPU only)

If you have an NVIDIA RTX/GTX card, replace the CPU-only PyTorch from `requirements.txt` with the CUDA 12.1 build:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Skip this step on machines without a CUDA-capable GPU — the CPU fallback installed by `requirements.txt` works correctly, just slower.

### Verify the installation

```powershell
python -c "import cv2, mediapipe, torch; print('OK — torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

Expected output (GPU machine): `OK — torch: 2.x.x | CUDA: True`  
Expected output (CPU machine): `OK — torch: 2.x.x | CUDA: False`

---

## Step 3 — Build the Flutter desktop module

The desktop context monitor is a native Windows application written in Flutter/Dart. It must be compiled once before first use; the compiled binary is not included in the repository.

**Requirements:** Flutter SDK on your `PATH` and Visual Studio 2022 with the **Desktop development with C++** workload installed (see Prerequisites).

```powershell
cd src\desktop
flutter pub get
flutter build windows --release
cd ..\..
```

The compiled executable will be placed at:

```
src\desktop\build\windows\x64\runner\Release\desktop_monitor.exe
```

> **Verify:** check that the `.exe` file exists at the path above before proceeding. If the build fails, confirm that Visual Studio is installed with the C++ workload, and that `flutter doctor` reports no issues.

> **If you skip this step:** the system will start without desktop context data. The physiological and facial modalities will continue collecting data normally, but `app_category`, `activity_pct`, and `window_switches` will be absent from all aggregated windows.

---

## Step 4 — Set up Ollama and download the LLM

Ollama manages the local large language model used for recommendation generation. Start it as a background service:

```powershell
ollama serve
```

Keep this terminal window open. Then, in a **separate** terminal window, pull the required model (approximately 5 GB — downloaded once):

```powershell
ollama pull llama3.1:8b
```

Verify the model responds correctly:

```powershell
ollama run llama3.1:8b "Say hello in one sentence"
```

> **If you skip this step or if Ollama is not running when the system starts:**
> - The window classifier falls back to keyword-based rules (no LLM call)
> - The recommendation engine falls back to template-based messages
> - Data collection and stress scoring continue normally
>
> Ollama is optional for data collection but required for intelligent recommendations.

---

## Step 5 — First-time baseline calibration

Before the first monitoring session, collect a two-minute resting baseline. This calibrates the stress scoring to your personal physiological resting state.

1. Sit comfortably in front of the camera in your normal working position.
2. Relax — do not type, move around, or speak.
3. Ensure your face is well-lit from the front.

```powershell
python src\utils\baseline.py --calibrate
```

The calibrator records your resting heart rate, RMSSD, EAR, blink rate, valence, and arousal over two minutes and stores them in the `baselines` table. The late fusion layer uses these values to score all future sessions relative to your personal baseline.

To view your stored baseline at any time:

```powershell
python src\utils\baseline.py --show
```

> **Re-calibration:** re-run `--calibrate` if your resting conditions change significantly (e.g. after illness, after a major change in your work environment, or if stress scores seem consistently too high or too low).

> **If you skip this step:** the system will prompt you to calibrate when the dashboard first opens. You can also start monitoring without a baseline; the fusion layer will use fixed population defaults until calibration is complete.

---

## Step 6 — Launch the system

### Terminal 1 — Ollama (keep running throughout the session)

```powershell
ollama serve
```

### Terminal 2 — System

```powershell
# Activate the virtual environment first (if not already active)
venv\Scripts\Activate.ps1

# Launch dashboard + all backend modules
python dashboard.py --start-backend
```

This single command:
1. Opens the **PyQt6 dashboard** (stress gauge, HR, valence-arousal scatter, trend graph, live camera feed, intervention log)
2. Starts `run_all.py` as a background subprocess — the three modalities begin collecting data immediately
3. Launches the Flutter desktop monitor headlessly in the background

To close the system: click the dashboard's close button, or press **Ctrl+C** in the terminal. All background processes are terminated automatically.

---

## Troubleshooting

### "Camera not found" or black camera feed

The system defaults to camera index `1` (external webcam). If your webcam is on a different index, open [src/utils/config.py](src/utils/config.py) and change:

```python
CAMERA_INDEX = 1   # change to 0 for built-in webcam
```

Verify your webcam index with:

```powershell
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"
```

### "Ollama not running" or template recommendations

If the system prints `[Recommender] Ollama error` or recommendations seem generic, Ollama is not reachable. Make sure `ollama serve` is running in a separate terminal before launching the system.

The system continues working without Ollama — it falls back to keyword classification and template-based recommendations automatically.

### Flutter window does not appear / no desktop data

The Flutter release build may be missing. Re-run Step 3:

```powershell
cd src\desktop
flutter build windows --release
cd ..\..
```

Then confirm the executable exists:

```powershell
Test-Path src\desktop\build\windows\x64\runner\Release\desktop_monitor.exe
```

### "No face data in dashboard" / RMSSD showing `—`

- **No face data:** ensure there is good, even frontal lighting on your face. Avoid backlighting (bright window behind you). The rPPG signal requires a visible face region in every frame.
- **RMSSD showing `—`:** normal for the first 5 minutes of each session. HRV metrics require a clean 5-minute rPPG signal. The `—` will be replaced with a value once the first HRV window completes.

### Stress score seems too high or too low

Re-run the baseline calibration (Step 5). The late fusion layer scores stress relative to your personal resting baseline — an incorrect baseline will shift all readings.

### PyQt6 import error or missing packages

Confirm the virtual environment is active (`(venv)` in the prompt) and reinstall:

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Optional — VS Code setup

The project is developed in VS Code. The following configuration gives the best experience.

### Recommended extensions

Install these from the VS Code Extensions panel (`Ctrl+Shift+X`):

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| Python | Microsoft | IntelliSense, linting, debugging |
| Pylance | Microsoft | Fast type checking |
| Dart | Dart Code | Dart/Flutter language support |
| Flutter | Dart Code | Flutter tooling |
| SQLite Viewer | Florian Klampfer | Inspect `stress_monitor.db` in-editor |

### Python interpreter

1. Open the Command Palette (`Ctrl+Shift+P`)
2. Type **"Python: Select Interpreter"**
3. Choose the interpreter inside `venv\Scripts\python.exe`

### Recommended `settings.json` additions

Open `.vscode/settings.json` (or create it) and add:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "editor.rulers": [100],
    "files.trimTrailingWhitespace": true
}
```

With `"python.terminal.activateEnvironment": true`, VS Code automatically activates the virtual environment whenever you open a new terminal inside the project — no manual `Activate.ps1` needed.

---

## Quick reference — key commands

```powershell
# Activate environment
venv\Scripts\Activate.ps1

# Launch everything
ollama serve                                  # Terminal 1
python dashboard.py --start-backend           # Terminal 2

# Calibration
python src\utils\baseline.py --calibrate      # record resting baseline
python src\utils\baseline.py --show           # view stored baseline

# Data review
python view_data.py                           # last 5 readings per table
python daily_summary.py                       # colour-coded daily summary
python export_excel.py                        # export all data to Excel

# Tests
pytest tests/ -v                              # run all 75 tests
```
