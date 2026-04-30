"""
face_monitor.py
---------------
Hello World proof-of-concept #2 — Facial landmark analysis.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

What this script does:
  1. Opens the webcam using OpenCV
  2. Detects 468 facial landmarks per frame using MediaPipe Face Mesh
  3. Every WINDOW_SECONDS computes:
       - Blink rate       : blinks per minute (fatigue indicator)
       - Eye Aspect Ratio : average eye openness (drowsiness indicator)
       - Head pose        : pitch, yaw, roll in degrees (attention indicator)
  4. Saves each reading to the face_readings table in stress_monitor.db
  5. Displays a live annotated preview with landmarks and metrics overlay

Why MediaPipe Face Mesh?
  - Runs fully on CPU, no GPU needed
  - 468 landmarks at 30+ FPS on a standard laptop
  - Already installed as a dependency of open-rppg so no new install needed
  - Used by your professor's emobot.gr project — good alignment with lab work
  - Privacy-safe: no images stored, only numerical features extracted

Why these three metrics?
  - Blink rate and EAR are established proxies for cognitive fatigue in the
    literature (Stern et al., 1994; Benedetto et al., 2011)
  - Head pose deviation from neutral is used in driver drowsiness detection
    and translates directly to knowledge worker fatigue contexts
  - All three feed naturally into the late fusion stress score later

Dependencies (already installed):
  pip install mediapipe opencv-python numpy
  (mediapipe was installed as part of open-rppg's dependencies)
"""

import time
import sqlite3
import math
from datetime import datetime
from collections import deque   # efficient fixed-length buffer for blink history

import cv2
import numpy as np
import mediapipe as mp          # Google's MediaPipe — landmark detection


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WEBCAM_INDEX   = 1
WINDOW_SECONDS = 30          # compute and save metrics every N seconds
DB_PATH        = "data/stress_monitor.db"
PRINT_INTERVAL = 5

# Eye Aspect Ratio threshold — below this value the eye is considered closed.
# 0.21 is a well-established threshold from the literature (Soukupova & Cech, 2016).
EAR_THRESHOLD  = 0.21

# Minimum number of consecutive frames with EAR < threshold to count as a blink.
# At 30 FPS, 2 frames = ~67ms. This filters out noise and camera artefacts.
BLINK_MIN_FRAMES = 2


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# MediaPipe Face Mesh uses a fixed 468-point topology.
# We define the specific indices we need for each feature here.
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ─────────────────────────────────────────────────────────────────────────────

# Left eye landmarks (6 points forming an ellipse around the eye)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmarks (6 points — mirror of left eye)
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Nose tip — used as the origin point for head pose estimation
NOSE_TIP  = 1

# Face outline points used to estimate head pose via solvePnP
# These correspond to: nose tip, chin, left eye corner,
# right eye corner, left mouth corner, right mouth corner
POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def init_database(db_path: str) -> sqlite3.Connection:
    """
    Open the shared stress_monitor.db and create the face_readings table
    if it does not already exist. The physio_readings table from PoC #1
    is untouched — we are just adding a new table to the same database.

    face_readings columns:
      timestamp   — ISO datetime string
      blink_rate  — blinks per minute (float). Normal ~15–20/min at rest.
                    Elevated (>25) suggests fatigue.
      mean_ear    — mean Eye Aspect Ratio across both eyes (float, 0–1).
                    Lower values = more closed eyes = more drowsy.
      pitch_deg   — head nodding up/down in degrees. Negative = head drooping.
      yaw_deg     — head turning left/right. Large values = distraction.
      roll_deg    — head tilting side to side.
      face_detected_pct — % of frames in the window where a face was found.
                          Low % means the user was away or occluded.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_readings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            blink_rate          REAL,
            mean_ear            REAL,
            pitch_deg           REAL,
            yaw_deg             REAL,
            roll_deg            REAL,
            face_detected_pct   REAL
        )
    """)
    conn.commit()
    print(f"[DB] face_readings table ready → {db_path}")
    return conn


def save_reading(conn, blink_rate, mean_ear,
                 pitch, yaw, roll, face_pct):
    """Insert one facial feature reading for the current window."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO face_readings
            (timestamp, blink_rate, mean_ear,
             pitch_deg, yaw_deg, roll_deg, face_detected_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ts, blink_rate, mean_ear, pitch, yaw, roll, face_pct))
    conn.commit()
    print(f"[DB] Saved → {ts} | Blinks/min: {blink_rate:.1f} | "
          f"EAR: {mean_ear:.3f} | Pitch: {pitch:.1f}° | "
          f"Yaw: {yaw:.1f}° | Roll: {roll:.1f}° | "
          f"Face: {face_pct:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices: list, w: int, h: int) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1–p6 are the six eye landmark points in order:
      p1, p4 = horizontal corners
      p2, p3, p5, p6 = vertical points

    A value near 0 means the eye is fully closed.
    A value around 0.3 means the eye is open.

    Parameters
    ----------
    landmarks  : MediaPipe normalised landmark list
    eye_indices: list of 6 landmark indices for this eye
    w, h       : frame width and height (for denormalising coordinates)
    """
    # Convert normalised [0,1] coordinates to pixel coordinates
    pts = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in eye_indices
    ])

    # Vertical distances (two pairs)
    A = np.linalg.norm(pts[1] - pts[5])   # p2 to p6
    B = np.linalg.norm(pts[2] - pts[4])   # p3 to p5

    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])   # p1 to p4

    ear = (A + B) / (2.0 * C) if C > 0 else 0.0
    return float(ear)


def estimate_head_pose(landmarks, w: int, h: int) -> tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) in degrees using cv2.solvePnP.

    solvePnP solves the Perspective-n-Point problem: given N 3D model points
    and their corresponding 2D image projections, find the rotation and
    translation that maps the 3D model to the image.

    We use 6 stable facial landmarks and match them to a 3D canonical face model.

    Returns (pitch, yaw, roll) in degrees.
      pitch: up/down rotation — negative = head dropping forward
      yaw  : left/right rotation — large values = looking away
      roll : tilt — large values = head tilted sideways
    """

    # 3D reference points from a canonical face model (in arbitrary units)
    # These correspond to: nose tip, chin, left eye corner,
    # right eye corner, left mouth corner, right mouth corner
    model_3d = np.array([
        [0.0,    0.0,    0.0  ],   # nose tip
        [0.0,   -63.6, -12.5 ],   # chin
        [-43.3,  32.7, -26.0 ],   # left eye outer corner
        [43.3,   32.7, -26.0 ],   # right eye outer corner
        [-28.9, -28.9, -24.1 ],   # left mouth corner
        [28.9,  -28.9, -24.1 ],   # right mouth corner
    ], dtype=np.float64)

    # Corresponding 2D image points from MediaPipe landmarks
    img_2d = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in POSE_LANDMARKS
    ], dtype=np.float64)

    # Camera intrinsic matrix — approximation using frame dimensions
    focal   = w
    cam_mat = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no lens distortion

    # Solve for rotation vector and translation vector
    success, rot_vec, _ = cv2.solvePnP(
        model_3d, img_2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rot_vec)

    # Decompose rotation matrix into Euler angles using RQ decomposition
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

    pitch = float(angles[0])
    yaw   = float(angles[1])
    roll  = float(angles[2])

    return pitch, yaw, roll


# ─────────────────────────────────────────────────────────────────────────────
# FACE MODULE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FaceModule:
    """
    Wraps MediaPipe Face Mesh and manages per-frame feature extraction.
    Accumulates blink count, EAR history, and pose history over a window,
    then computes summary statistics when get_reading() is called.

    Same swappable-module pattern as PhysioModule in rppg_monitor.py.
    """

    def __init__(self):
        # Initialise MediaPipe Face Mesh
        # max_num_faces=1: we only need one face (the user's)
        # refine_landmarks=True: adds iris landmarks for better EAR accuracy
        # min_detection_confidence: how confident MP must be before reporting a face
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # ── Blink detection state ─────────────────────────────────────────────
        self.blink_count       = 0      # total blinks in current window
        self.eye_closed_frames = 0      # consecutive frames where EAR < threshold

        # ── Per-window accumulators ───────────────────────────────────────────
        self.ear_history   = []   # EAR value per frame (for mean EAR)
        self.pitch_history = []   # head pose per frame
        self.yaw_history   = []
        self.roll_history  = []
        self.frames_total  = 0    # total frames processed this window
        self.frames_face   = 0    # frames where a face was detected

    def reset_window(self):
        """Clear all accumulators for the next 30-second window."""
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
        Process one webcam frame:
          - Run MediaPipe Face Mesh detection
          - Compute EAR and update blink counter
          - Estimate head pose
          - Draw landmarks and metrics on the frame for preview

        Returns the annotated frame for display.
        """
        h, w = frame.shape[:2]
        self.frames_total += 1

        # MediaPipe expects RGB — OpenCV gives us BGR by default
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # No face detected this frame — skip feature extraction
            cv2.putText(frame,
                        "No face — centre yourself in frame",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
            return frame

        self.frames_face += 1
        lm = results.multi_face_landmarks[0].landmark   # first (only) face

        # ── Draw face mesh (tesselation) ──────────────────────────────────────
        # Drawing the full mesh visually confirms landmark detection quality
        self.mp_draw.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_draw_styles
                .get_default_face_mesh_tesselation_style()
        )

        # ── Eye Aspect Ratio ──────────────────────────────────────────────────
        ear_left  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_mean  = (ear_left + ear_right) / 2.0
        self.ear_history.append(ear_mean)

        # ── Blink detection ───────────────────────────────────────────────────
        if ear_mean < EAR_THRESHOLD:
            # Eye is closed this frame — increment consecutive closed counter
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= BLINK_MIN_FRAMES:
                # Eye was closed for enough consecutive frames — count as blink
                self.blink_count += 1
            # Reset the consecutive closed counter
            self.eye_closed_frames = 0

        # ── Head pose ─────────────────────────────────────────────────────────
        pitch, yaw, roll = estimate_head_pose(lm, w, h)
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.roll_history.append(roll)

        # ── Overlay metrics on preview ────────────────────────────────────────
        overlay_lines = [
            f"EAR: {ear_mean:.3f}",
            f"Blinks: {self.blink_count}",
            f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line,
                        (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)

        return frame

    def get_reading(self, elapsed_seconds: float) -> dict | None:
        """
        Compute summary statistics for the current window and return them.
        Returns None if too few frames had a face detected to be meaningful.
        """
        if self.frames_face < 10:
            return None   # not enough data

        face_pct   = (self.frames_face / self.frames_total * 100
                      if self.frames_total > 0 else 0.0)

        # Blink rate: normalise count to per-minute
        blink_rate = self.blink_count / elapsed_seconds * 60

        mean_ear   = float(np.mean(self.ear_history))   if self.ear_history   else 0.0
        mean_pitch = float(np.mean(self.pitch_history)) if self.pitch_history else 0.0
        mean_yaw   = float(np.mean(self.yaw_history))   if self.yaw_history   else 0.0
        mean_roll  = float(np.mean(self.roll_history))  if self.roll_history  else 0.0

        return {
            "blink_rate": round(blink_rate, 1),
            "mean_ear":   round(mean_ear,   3),
            "pitch":      round(mean_pitch, 1),
            "yaw":        round(mean_yaw,   1),
            "roll":       round(mean_roll,  1),
            "face_pct":   round(face_pct,   1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor():
    """
    Open the webcam, process frames through FaceModule, save readings
    every WINDOW_SECONDS. Press Q or Ctrl+C to exit cleanly.
    """
    conn   = init_database(DB_PATH)
    face   = FaceModule()
    cap    = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {WEBCAM_INDEX}.")
        return

    print(f"\n[INFO] Starting facial landmark monitor.")
    print(f"[INFO] Readings every {WINDOW_SECONDS}s — keep face visible.")
    print(f"[INFO] Press Q in the preview window, or Ctrl+C, to quit.\n")

    window_start = time.time()
    last_status  = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame — retrying.")
                continue

            # Process frame and get annotated preview
            annotated = face.process_frame(frame)

            # Show time remaining in the window
            elapsed   = time.time() - window_start
            remaining = max(0, WINDOW_SECONDS - int(elapsed))
            cv2.putText(annotated, f"Next reading in {remaining}s",
                        (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Face Monitor — Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Q pressed — stopping.")
                break

            # Periodic status print
            now = time.time()
            if now - last_status >= PRINT_INTERVAL:
                print(f"[INFO] Collecting... {remaining}s | "
                      f"Blinks so far: {face.blink_count}")
                last_status = now

            # Every WINDOW_SECONDS: compute reading and save
            if elapsed >= WINDOW_SECONDS:
                reading = face.get_reading(elapsed_seconds=elapsed)

                if reading:
                    save_reading(
                        conn,
                        blink_rate=reading["blink_rate"],
                        mean_ear=reading["mean_ear"],
                        pitch=reading["pitch"],
                        yaw=reading["yaw"],
                        roll=reading["roll"],
                        face_pct=reading["face_pct"],
                    )
                    print(f"\n{'─'*48}")
                    print(f"  Blink rate  : {reading['blink_rate']:.1f} blinks/min")
                    print(f"  Mean EAR    : {reading['mean_ear']:.3f}")
                    print(f"  Head pitch  : {reading['pitch']:.1f}°")
                    print(f"  Head yaw    : {reading['yaw']:.1f}°")
                    print(f"  Head roll   : {reading['roll']:.1f}°")
                    print(f"  Face visible: {reading['face_pct']:.0f}% of window")
                    print(f"{'─'*48}\n")
                else:
                    print("[WARN] Too few frames with face detected. "
                          "Ensure face is centred and well-lit.")

                # Reset for next window
                face.reset_window()
                window_start = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by Ctrl+C.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    run_monitor()