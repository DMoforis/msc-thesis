"""
face_monitor.py
---------------
Facial landmark analysis — Phase 2 (shared-feed refactor).
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Changes from Phase 1 (PoC #2):
  - No longer opens the camera directly; receives frames from SharedCameraFeed.
  - face_readings schema updated: adds valence and arousal columns (NULL until
    src/face/valence_arousal.py is implemented in a later Phase 2 step).
  - Roll° computation fixed: replaced RQDecomp3x3 (gimbal-lock-prone) with a
    direct atan2 decomposition of the rotation matrix.
  - All config constants imported from src/utils/config.py.

Why these metrics?
  - Blink rate / EAR: established proxies for cognitive fatigue
    (Stern et al. 1994; Benedetto et al. 2011; Soukupova & Cech 2016).
  - Head pose: pitch (nodding), yaw (looking away) correlate with attention
    and fatigue in driver-drowsiness and knowledge-worker studies.
  - Valence / Arousal: Russell (1980) circumplex model of affect; planned via
    HuggingFace Mavdol/NPC-Valence-Arousal-Prediction model.

Dependencies (already installed via open-rppg):
  pip install mediapipe opencv-python numpy
"""

import sys
import os
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.camera.shared_feed import SharedCameraFeed
from src.face.valence_arousal import ValenceArousalPredictor
from src.utils.config import (
    DB_PATH, CAMERA_INDEX, CAMERA_FPS,
    FACE_WINDOW_SECONDS,
    EAR_THRESHOLD, BLINK_MIN_FRAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# Fixed 468-point topology. We use only the subset needed for each feature.
# ─────────────────────────────────────────────────────────────────────────────

LEFT_EYE       = [362, 385, 387, 263, 373, 380]   # 6 points around left eye
RIGHT_EYE      = [33,  160, 158, 133, 153, 144]   # 6 points around right eye
POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]       # nose tip, chin, eye corners, mouth corners


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def init_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open the shared database and ensure face_readings has the Phase 2 schema.

    New Phase 2 columns (valence, arousal) are added via ALTER TABLE if the
    table was created by Phase 1, so existing data is preserved.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_readings (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            blink_rate        REAL,
            mean_ear          REAL,
            pitch_deg         REAL,
            yaw_deg           REAL,
            roll_deg          REAL,
            face_detected_pct REAL,
            valence           REAL,
            arousal           REAL
        )
    """)
    conn.commit()

    # Add Phase 2 columns to tables created by Phase 1 schema (safe no-ops
    # if columns already exist — OperationalError is expected and swallowed).
    for col in ("valence REAL", "arousal REAL"):
        try:
            conn.execute(f"ALTER TABLE face_readings ADD COLUMN {col}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

    print(f"[DB] face_readings table ready → {db_path}")
    return conn


def save_reading(
    conn: sqlite3.Connection,
    blink_rate: float,
    mean_ear: float,
    pitch: float,
    yaw: float,
    roll: float,
    face_pct: float,
    valence: float | None = None,
    arousal: float | None = None,
) -> None:
    """Insert one facial feature reading for the current window."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO face_readings
            (timestamp, blink_rate, mean_ear,
             pitch_deg, yaw_deg, roll_deg, face_detected_pct,
             valence, arousal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, blink_rate, mean_ear, pitch, yaw, roll, face_pct, valence, arousal))
    conn.commit()
    print(f"[DB] Saved → {ts} | "
          f"Blinks/min: {blink_rate:.1f} | EAR: {mean_ear:.3f} | "
          f"Pitch: {pitch:.1f}° | Yaw: {yaw:.1f}° | Roll: {roll:.1f}° | "
          f"Face: {face_pct:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices: list[int], w: int, h: int) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)

    Values near 0 = fully closed; ~0.3 = open.
    Reference: Soukupova & Cech (2016).
    """
    pts = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in eye_indices
    ])
    A = np.linalg.norm(pts[1] - pts[5])   # p2–p6
    B = np.linalg.norm(pts[2] - pts[4])   # p3–p5
    C = np.linalg.norm(pts[0] - pts[3])   # p1–p4 (horizontal)
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


def _rotation_matrix_to_euler(rot_mat: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a 3×3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees.

    Uses a direct atan2 decomposition of the rotation matrix rows/columns,
    which is numerically stable and avoids the gimbal-lock artefact seen with
    cv2.RQDecomp3x3 on the roll axis (Phase 1 known limitation).

    The decomposition order is X→Y→Z (pitch→yaw→roll), matching the
    coordinate convention used by cv2.solvePnP with a frontal face model.

    Returns
    -------
    pitch : float  — up/down rotation in degrees (negative = head drooping)
    yaw   : float  — left/right rotation in degrees (large = looking away)
    roll  : float  — side-to-side tilt in degrees
    """
    # sy: length of the projection onto the XY plane — approaches 0 at ±90° pitch
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6   # true near gimbal-lock singularity

    if not singular:
        pitch = float(np.degrees(np.arctan2( rot_mat[2, 1],  rot_mat[2, 2])))
        yaw   = float(np.degrees(np.arctan2(-rot_mat[2, 0],  sy)))
        roll  = float(np.degrees(np.arctan2( rot_mat[1, 0],  rot_mat[0, 0])))
    else:
        # Near singularity: roll is indeterminate; assign 0 by convention.
        pitch = float(np.degrees(np.arctan2(-rot_mat[1, 2],  rot_mat[1, 1])))
        yaw   = float(np.degrees(np.arctan2(-rot_mat[2, 0],  sy)))
        roll  = 0.0

    return pitch, yaw, roll


def estimate_head_pose(landmarks, w: int, h: int) -> tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) in degrees using cv2.solvePnP.

    Matches 6 stable facial landmarks to a 3D canonical face model and solves
    for the rotation that maps the model onto the 2D image. The rotation vector
    is then converted to a stable Euler decomposition (see _rotation_matrix_to_euler).

    Returns (0, 0, 0) if solvePnP fails (e.g. degenerate landmark layout).
    """
    model_3d = np.array([
        [ 0.0,    0.0,    0.0  ],   # nose tip
        [ 0.0,  -63.6,  -12.5 ],   # chin
        [-43.3,  32.7,  -26.0 ],   # left eye outer corner
        [ 43.3,  32.7,  -26.0 ],   # right eye outer corner
        [-28.9, -28.9,  -24.1 ],   # left mouth corner
        [ 28.9, -28.9,  -24.1 ],   # right mouth corner
    ], dtype=np.float64)

    img_2d = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in POSE_LANDMARKS
    ], dtype=np.float64)

    focal   = float(w)
    cam_mat = np.array([
        [focal, 0,     w / 2.0],
        [0,     focal, h / 2.0],
        [0,     0,     1.0    ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        model_3d, img_2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    return _rotation_matrix_to_euler(rot_mat)


# ─────────────────────────────────────────────────────────────────────────────
# FACE MODULE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FaceModule:
    """
    Wraps MediaPipe Face Mesh and manages per-frame feature extraction.

    Accumulates blink count, EAR, and pose history over a window, then
    computes summary statistics when get_reading() is called.
    """

    def __init__(self):
        self.mp_face_mesh    = mp.solutions.face_mesh
        self.face_mesh       = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw         = mp.solutions.drawing_utils
        self.mp_draw_styles  = mp.solutions.drawing_styles

        self.blink_count       = 0
        self.eye_closed_frames = 0

        self.ear_history   : list[float] = []
        self.pitch_history : list[float] = []
        self.yaw_history   : list[float] = []
        self.roll_history  : list[float] = []
        self.frames_total  = 0
        self.frames_face   = 0

        # VA predictor — lazy-loaded on first get_reading() call
        self._va             = ValenceArousalPredictor()
        self._last_face_crop : np.ndarray | None = None

    def reset_window(self) -> None:
        """Clear all accumulators for the next window."""
        self.blink_count       = 0
        self.eye_closed_frames = 0
        self.ear_history.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.roll_history.clear()
        self.frames_total = 0
        self.frames_face  = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one BGR frame: run MediaPipe, compute EAR/blink/pose,
        draw annotations, and update per-window accumulators.

        Returns the annotated frame for display.
        """
        h, w = frame.shape[:2]
        self.frames_total += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face — centre yourself in frame",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
            return frame

        self.frames_face += 1
        lm = results.multi_face_landmarks[0].landmark

        # ── Face mesh overlay ─────────────────────────────────────────────────
        self.mp_draw.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                self.mp_draw_styles.get_default_face_mesh_tesselation_style()
            ),
        )

        # ── Eye Aspect Ratio + blink detection ───────────────────────────────
        ear_left  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_mean  = (ear_left + ear_right) / 2.0
        self.ear_history.append(ear_mean)

        if ear_mean < EAR_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= BLINK_MIN_FRAMES:
                self.blink_count += 1
            self.eye_closed_frames = 0

        # ── Head pose ─────────────────────────────────────────────────────────
        pitch, yaw, roll = estimate_head_pose(lm, w, h)
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.roll_history.append(roll)

        # ── Face crop for VA (updated every frame with a detected face) ────────
        self._last_face_crop = self._crop_face(frame, lm, w, h)

        # ── Metrics overlay ───────────────────────────────────────────────────
        for i, line in enumerate([
            f"EAR: {ear_mean:.3f}",
            f"Blinks: {self.blink_count}",
            f"P: {pitch:.1f}°  Y: {yaw:.1f}°  R: {roll:.1f}°",
        ]):
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)

        return frame

    @staticmethod
    def _crop_face(frame: np.ndarray, lm, w: int, h: int) -> np.ndarray | None:
        """Crop the face region from a frame using MediaPipe landmark bounds."""
        xs  = [lm[i].x for i in range(len(lm))]
        ys  = [lm[i].y for i in range(len(lm))]
        pad = 0.10   # 10% padding on each side
        x1  = max(0, int((min(xs) - pad) * w))
        y1  = max(0, int((min(ys) - pad) * h))
        x2  = min(w, int((max(xs) + pad) * w))
        y2  = min(h, int((max(ys) + pad) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def get_reading(self, elapsed_seconds: float) -> dict | None:
        """
        Compute and return summary statistics for the current window.
        Returns None if fewer than 10 frames had a detected face.
        """
        if self.frames_face < 10:
            return None

        face_pct   = (self.frames_face / self.frames_total * 100
                      if self.frames_total > 0 else 0.0)
        blink_rate = self.blink_count / elapsed_seconds * 60 if elapsed_seconds > 0 else 0.0
        mean_ear   = float(np.mean(self.ear_history))   if self.ear_history   else 0.0
        mean_pitch = float(np.mean(self.pitch_history)) if self.pitch_history else 0.0
        mean_yaw   = float(np.mean(self.yaw_history))   if self.yaw_history   else 0.0
        mean_roll  = float(np.mean(self.roll_history))  if self.roll_history  else 0.0

        # VA: run EmoNet on the most recent face crop from this window
        va = self._va.predict(self._last_face_crop) if self._last_face_crop is not None else None

        return {
            "blink_rate": round(blink_rate, 1),
            "mean_ear":   round(mean_ear,   3),
            "pitch":      round(mean_pitch, 1),
            "yaw":        round(mean_yaw,   1),
            "roll":       round(mean_roll,  1),
            "face_pct":   round(face_pct,   1),
            "valence":    va[0] if va else None,
            "arousal":    va[1] if va else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor(feed: SharedCameraFeed) -> None:
    """
    Process frames from the shared camera feed, compute facial features
    every FACE_WINDOW_SECONDS, and save readings to the database.

    Parameters
    ----------
    feed : SharedCameraFeed
        Running shared camera feed. Must already be started before calling.
    """
    conn   = init_database()
    face   = FaceModule()

    window_start = time.monotonic()
    last_status  = time.monotonic()
    STATUS_INTERVAL = 5.0

    print(f"\n[Face] Monitor started.")
    print(f"[Face] Readings every {FACE_WINDOW_SECONDS}s — keep face visible.")
    print(f"[Face] Press Q to quit.\n")

    try:
        while True:
            frame = feed.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            annotated = face.process_frame(frame)

            elapsed   = time.monotonic() - window_start
            remaining = max(0, FACE_WINDOW_SECONDS - int(elapsed))
            cv2.putText(annotated, f"Next reading in {remaining}s",
                        (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Face Monitor — Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Face] Q pressed — stopping.")
                break

            now = time.monotonic()
            if now - last_status >= STATUS_INTERVAL:
                print(f"[Face] Collecting… {remaining}s | "
                      f"Blinks so far: {face.blink_count}")
                last_status = now

            if elapsed >= FACE_WINDOW_SECONDS:
                reading = face.get_reading(elapsed_seconds=elapsed)

                if reading:
                    save_reading(
                        conn,
                        blink_rate = reading["blink_rate"],
                        mean_ear   = reading["mean_ear"],
                        pitch      = reading["pitch"],
                        yaw        = reading["yaw"],
                        roll       = reading["roll"],
                        face_pct   = reading["face_pct"],
                        valence    = reading["valence"],
                        arousal    = reading["arousal"],
                    )
                    v_str = f"{reading['valence']:+.3f}" if reading['valence'] is not None else "—"
                    a_str = f"{reading['arousal']:+.3f}" if reading['arousal'] is not None else "—"
                    print(f"\n{'─'*48}")
                    print(f"  Blink rate  : {reading['blink_rate']:.1f} blinks/min")
                    print(f"  Mean EAR    : {reading['mean_ear']:.3f}")
                    print(f"  Head pitch  : {reading['pitch']:.1f}°")
                    print(f"  Head yaw    : {reading['yaw']:.1f}°")
                    print(f"  Head roll   : {reading['roll']:.1f}°")
                    print(f"  Face visible: {reading['face_pct']:.0f}% of window")
                    print(f"  Valence     : {v_str}   Arousal: {a_str}")
                    print(f"{'─'*48}\n")
                else:
                    print("[Face] Too few frames with face detected. "
                          "Ensure face is centred and well-lit.")

                face.reset_window()
                window_start = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Face] Stopped by Ctrl+C.")

    finally:
        cv2.destroyAllWindows()
        conn.close()
        print("[Face] Clean shutdown complete.")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[Face] Standalone mode — creating shared camera feed.")
    with SharedCameraFeed(camera_index=CAMERA_INDEX, fps=CAMERA_FPS) as feed:
        run_monitor(feed)
