"""
valence_arousal.py
------------------
Continuous Valence-Arousal prediction from face images using EmoNet.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Model choice rationale
----------------------
The supervisor's originally specified model (Mavdol/NPC-Valence-Arousal-Prediction)
is a DistilBERT text classifier trained on fictional NPC dialogue — it cannot
process face images. After evaluating alternatives, EmoNet was selected:

  Toisoul A. et al. (2021). "Estimation of continuous valence and arousal levels
  from faces in naturalistic conditions." Nature Machine Intelligence, 3, 42–50.

Why EmoNet:
  - Trained on AffectNet (450k+ in-the-wild human face images with VA labels)
  - Only model available that performs DIRECT continuous VA regression from faces
    (alternatives on HuggingFace only produce discrete emotion categories)
  - Outputs valence and arousal as floats in [-1, 1], matching the VA circumplex
    model of affect (Russell, 1980), already cited in the thesis references
  - Architecture: ResNet-50 + Stacked Hourglass, published in Nature MI

Setup (first use only — performed automatically by this module):
  1. pip install torch torchvision  (see SPEC.md §11)
  2. EmoNet architecture source is downloaded from GitHub to data/models/
  3. Pretrained weights (~170 MB) are downloaded from GitHub to data/models/

After the first successful call, both files are cached locally and no network
access is needed.

Interface:
  predictor = ValenceArousalPredictor()

  # Call once per face window (not per-frame — inference takes ~15-50 ms on GPU)
  result = predictor.predict(face_bgr_image)  # BGR numpy array, any size
  if result is not None:
      valence, arousal = result   # both in [-1.0, 1.0]
  else:
      # model unavailable or image invalid
      ...

  predictor.available  # True once model is loaded successfully
"""

import os
import sys
import threading
import urllib.request

import cv2
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import DATA_DIR

# ── Remote sources ────────────────────────────────────────────────────────────
_EMONET_PY_URL  = (
    "https://raw.githubusercontent.com/face-analysis/emonet"
    "/master/emonet/models/emonet.py"
)
_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/face-analysis/emonet"
    "/master/pretrained/emonet_8.pth"
)

# ── Local cache paths ─────────────────────────────────────────────────────────
_MODELS_DIR    = os.path.join(DATA_DIR, "models")
_ARCH_FILE     = os.path.join(_MODELS_DIR, "emonet_arch.py")
_WEIGHTS_FILE  = os.path.join(_MODELS_DIR, "emonet_8.pth")

# EmoNet input size (fixed — matches training preprocessing)
_IMG_SIZE = 256


# ─────────────────────────────────────────────────────────────────────────────
# VALENCE-AROUSAL PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class ValenceArousalPredictor:
    """
    Predicts continuous valence (pleasantness) and arousal (activation) from
    a cropped BGR face image.

    Model: EmoNet-8 (Toisoul et al., 2021)
    Training data: AffectNet, 450k+ in-the-wild face images
    Output range: valence in [-1, 1], arousal in [-1, 1]

    The model and weights are downloaded automatically from GitHub on first use
    and cached in data/models/ for all subsequent calls.

    Thread safety: _load() is protected by a threading.Lock so it is safe to
    call predict() from multiple threads concurrently.
    """

    def __init__(self):
        self._model    = None
        self._device   = None
        self._lock     = threading.Lock()
        self._available: bool | None = None   # None = not yet attempted

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True once the model is loaded and ready for inference."""
        if self._available is None:
            self._load()
        return bool(self._available)

    def predict(self, face_bgr: np.ndarray) -> tuple[float, float] | None:
        """
        Predict valence and arousal from a BGR face crop.

        Parameters
        ----------
        face_bgr : np.ndarray
            BGR face image (any resolution — resized internally to 256×256).
            Should be a tight crop around the face with some padding.

        Returns
        -------
        (valence, arousal) : tuple[float, float]
            Both values clamped to [-1.0, 1.0].
            valence > 0  →  pleasant / relaxed
            valence < 0  →  unpleasant / stressed
            arousal > 0  →  alert / excited
            arousal < 0  →  drowsy / calm
        None
            If the model is unavailable or the image is invalid.
        """
        if self._available is None:
            self._load()
        if not self._available or self._model is None:
            return None

        # Guard against None, empty, or too-small crops
        if face_bgr is None or face_bgr.ndim < 3 or face_bgr.size == 0:
            return None
        h, w = face_bgr.shape[:2]
        if h < 16 or w < 16:
            return None

        try:
            import torch  # import guarded — fails gracefully if not installed

            # ── Preprocessing (matches EmoNet training pipeline) ──────────────
            # BGR → RGB, resize to 256×256, scale [0,1], add batch dim
            face_rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (_IMG_SIZE, _IMG_SIZE),
                                      interpolation=cv2.INTER_AREA)
            tensor = (
                torch.from_numpy(face_resized)
                .permute(2, 0, 1)          # HWC → CHW
                .float()
                .div(255.0)
                .unsqueeze(0)              # add batch dimension
                .to(self._device)
            )

            # ── Inference ─────────────────────────────────────────────────────
            # Do NOT pass reset_smoothing=True: the model is instantiated with
            # temporal_smoothing=False, so self.temporal_state is never
            # initialised and accessing it would raise AttributeError.
            with torch.no_grad():
                output = self._model(tensor)

            valence = float(output["valence"].squeeze().clamp(-1.0, 1.0).cpu().item())
            arousal = float(output["arousal"].squeeze().clamp(-1.0, 1.0).cpu().item())

            return round(valence, 3), round(arousal, 3)

        except Exception as exc:
            print(f"[VA] Prediction error: {exc}")
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self) -> bool:
        """Download (if needed) and load the EmoNet model. Thread-safe."""
        with self._lock:
            if self._available is not None:
                return self._available   # another thread already loaded it

            if not self._ensure_torch():
                self._available = False
                return False

            if not self._download_files():
                self._available = False
                return False

            self._available = self._init_model()
            return self._available

    def _ensure_torch(self) -> bool:
        """Check that PyTorch is importable; print install hint if not."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            print(
                "[VA] WARN: PyTorch is not installed — VA disabled.\n"
                "[VA]   Install: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu121"
            )
            return False

    def _download_files(self) -> bool:
        """Download EmoNet architecture source and weights if not cached."""
        os.makedirs(_MODELS_DIR, exist_ok=True)

        try:
            if not os.path.exists(_ARCH_FILE):
                print("[VA] Downloading EmoNet architecture source...")
                urllib.request.urlretrieve(_EMONET_PY_URL, _ARCH_FILE)
                print("[VA] Architecture source saved.")

            if not os.path.exists(_WEIGHTS_FILE):
                print("[VA] Downloading EmoNet weights (~170 MB) — one-time download...")

                def _progress(count, block_size, total_size):
                    if total_size > 0:
                        pct = count * block_size * 100 // total_size
                        # Print only at 25% increments to avoid spam
                        if pct % 25 == 0:
                            print(f"[VA]   {pct}% …")

                urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_FILE, _progress)
                print("[VA] Weights saved to", _WEIGHTS_FILE)

            return True

        except Exception as exc:
            print(
                f"[VA] WARN: Download failed: {exc}\n"
                f"[VA]   If you are offline, manually place emonet_8.pth at:\n"
                f"[VA]   {_WEIGHTS_FILE}"
            )
            return False

    def _init_model(self) -> bool:
        """Import the EmoNet class, load weights, move to device."""
        try:
            import torch
            import importlib.util

            # Dynamically import the vendored EmoNet architecture
            spec   = importlib.util.spec_from_file_location("emonet_arch", _ARCH_FILE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            EmoNet = module.EmoNet

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            dev_label = (
                f"CUDA ({torch.cuda.get_device_name(0)})"
                if torch.cuda.is_available() else "CPU"
            )
            print(f"[VA] Loading EmoNet-8 on {dev_label} ...")

            # Build model and load weights
            net = EmoNet(n_expression=8, temporal_smoothing=False)
            state_dict = torch.load(
                _WEIGHTS_FILE, map_location="cpu", weights_only=True
            )
            # Strip 'module.' prefix left by DataParallel training
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            net.load_state_dict(state_dict, strict=False)

            # Move to device first, then call eval() separately.
            # EmoNet overrides eval() without returning self, so chaining
            # net.to(device).eval() would assign None to self._model.
            net = net.to(self._device)
            net.eval()
            self._model = net
            print(f"[VA] EmoNet-8 ready — valence/arousal active.")
            return True

        except Exception as exc:
            print(f"[VA] WARN: Model initialisation failed: {exc}")
            return False


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("ValenceArousalPredictor smoke-test")
    print("  Opens webcam, runs VA on each frame. Press Q to quit.\n")

    # Import shared feed from the project tree
    from src.camera.shared_feed import SharedCameraFeed
    from src.utils.config import CAMERA_INDEX, CAMERA_FPS
    import mediapipe as mp

    predictor = ValenceArousalPredictor()
    if not predictor.available:
        print("[ERROR] Model unavailable — check install and network.")
        sys.exit(1)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    def crop_face(frame, lm, w, h):
        xs = [lm[i].x for i in range(len(lm))]
        ys = [lm[i].y for i in range(len(lm))]
        pad = 0.10
        x1 = max(0, int((min(xs) - pad) * w))
        y1 = max(0, int((min(ys) - pad) * h))
        x2 = min(w, int((max(xs) + pad) * w))
        y2 = min(h, int((max(ys) + pad) * h))
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    last_va  = None
    frame_no = 0

    with SharedCameraFeed(camera_index=CAMERA_INDEX, fps=CAMERA_FPS) as feed:
        while True:
            frame = feed.get_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm   = results.multi_face_landmarks[0].landmark
                crop = crop_face(frame, lm, w, h)
                # Run VA every 10 frames to reduce load
                if crop is not None and frame_no % 10 == 0:
                    va = predictor.predict(crop)
                    if va:
                        last_va = va

            if last_va:
                cv2.putText(frame,
                    f"V: {last_va[0]:+.2f}  A: {last_va[1]:+.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 130), 2)

            cv2.imshow("VA Smoke-test — Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_no += 1

    cv2.destroyAllWindows()
    face_mesh.close()
