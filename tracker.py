"""tracker.py — FaceTracker: runs mediapipe face tracking in a daemon thread."""
from __future__ import annotations

import math
import socket
import struct
import threading
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import Config

# ── Face mesh landmark indices ──────────────────────────────────────────────
LM_NOSE_TIP    = 4
LM_CHIN        = 152
LM_LEFT_EYE    = 33
LM_RIGHT_EYE   = 263
LM_LEFT_MOUTH  = 61
LM_RIGHT_MOUTH = 291

FACE_MODEL_3D = np.array([
    [ 0.000,  0.000,  0.000],
    [ 0.000, -3.300, -1.300],
    [-4.500,  2.500, -4.000],
    [ 4.500,  2.500, -4.000],
    [-2.000,  0.000, -2.200],
    [ 2.000,  0.000, -2.200],
], dtype=np.float64)
FACE_MODEL_IDX = [LM_NOSE_TIP, LM_CHIN, LM_LEFT_EYE, LM_RIGHT_EYE, LM_LEFT_MOUTH, LM_RIGHT_MOUTH]

# ── Algorithm constants ──────────────────────────────────────────────────────
PITCH_GIMBAL_THRESHOLD = 90.0
MIN_EYE_DIST_PX        = 1.0
ROT_SINGULARITY_EPS    = 1e-6
MEDIAPIPE_MIN_CONF     = 0.5
AXIS_DISPLAY_LEN_CM    = 5
DEBUG_PRINT_INTERVAL   = 30


# ── Utility functions (module-level so tests can import them) ────────────────

def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    return struct.pack("<6d", x, y, z, yaw, pitch, roll)


def get_cam_matrix(w: int, h: int, focal_px: float) -> np.ndarray:
    return np.array([[focal_px, 0, w / 2],
                     [0, focal_px, h / 2],
                     [0, 0, 1]], dtype=np.float64)


def rot_to_euler(R: np.ndarray):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > ROT_SINGULARITY_EPS:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return x, y, z


def sample_2d(landmarks, w: int, h: int) -> np.ndarray:
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_MODEL_IDX],
        dtype=np.float64,
    )


# ── SmoothFilter ─────────────────────────────────────────────────────────────

class SmoothFilter:
    def __init__(self, alpha: float, deadzone: float = 0.0):
        self.alpha = alpha
        self.deadzone = deadzone
        self.value: Optional[float] = None

    def update(self, new_val: float) -> float:
        if self.value is None:
            self.value = new_val
            return self.value
        if abs(new_val - self.value) < self.deadzone:
            return self.value
        self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ── FaceTracker ──────────────────────────────────────────────────────────────

class FaceTracker:
    """Runs mediapipe face tracking in a daemon thread.

    Usage:
        tracker = FaceTracker(cfg)
        tracker.start(preview=True)
        # read tracker.live for latest values
        tracker.stop()
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Shared live values dict; written by tracker thread, read by UI thread.
        self.live: dict = {
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "x": 0.0, "y": 0.0, "z": 0.0,
            "tracking": False, "error": None,
        }
        self._preview = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_config(self, cfg: Config) -> None:
        with self._lock:
            self._cfg = cfg

    def start(self, preview: bool = True) -> None:
        if self.running:
            return
        self._preview = preview
        self._stop_event.clear()
        self.live["error"] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None
        self.live["tracking"] = False

    def _cfg_snapshot(self) -> Config:
        with self._lock:
            return self._cfg

    def _run(self) -> None:
        cfg = self._cfg_snapshot()
        try:
            self._tracking_loop(cfg)
        except Exception as e:
            self.live["error"] = str(e)
            self.live["tracking"] = False
        finally:
            cv2.destroyAllWindows()

    def _tracking_loop(self, initial_cfg: Config) -> None:
        cfg = initial_cfg
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cap = cv2.VideoCapture(cfg.cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{cfg.cam_index}")

        mp_mesh = mp.solutions.face_mesh
        face_mesh = mp_mesh.FaceMesh(
            max_num_faces=cfg.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=MEDIAPIPE_MIN_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_CONF,
        )

        f_yaw   = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_rot)
        f_pitch = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_rot)
        f_roll  = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_rot)
        f_x     = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_pos)
        f_y     = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_pos)
        f_z     = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_pos)

        prev_rvec = None
        prev_tvec = None
        locked_eye_mid = None
        cam_mtx = None
        dist_cfs = np.zeros((4, 1))
        frame_count = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                if cam_mtx is None:
                    cam_mtx = get_cam_matrix(w, h, cfg.focal_length_px)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm, locked_eye_mid = self._select_face(
                        results.multi_face_landmarks, w, h, locked_eye_mid, cfg
                    )
                    img_pts = sample_2d(lm, w, h)
                    pnp = self._solve_pose(img_pts, w, h, cfg, prev_rvec, prev_tvec)

                    if pnp:
                        rvec, tvec = pnp
                        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

                        R, _ = cv2.Rodrigues(rvec)
                        ex, ey, ez = rot_to_euler(R)

                        raw_yaw   = math.degrees(ey)
                        raw_pitch = math.degrees(ex)
                        raw_roll  = math.degrees(ez)

                        if raw_pitch > PITCH_GIMBAL_THRESHOLD:
                            raw_pitch -= 180
                        elif raw_pitch < -PITCH_GIMBAL_THRESHOLD:
                            raw_pitch += 180

                        yaw   =  raw_yaw   * cfg.yaw_scale
                        pitch = -raw_pitch * cfg.pitch_scale
                        roll  =  raw_roll  * cfg.roll_scale

                        pos = self._estimate_position(lm, w, h, cfg)
                        if pos:
                            tx, ty, tz = pos[0] * cfg.x_scale, pos[1] * cfg.y_scale, pos[2] * cfg.z_scale
                        else:
                            tx, ty, tz = 0.0, 0.0, 0.0

                        yaw   = f_yaw.update(yaw)
                        pitch = f_pitch.update(pitch)
                        roll  = f_roll.update(roll)
                        tx    = f_x.update(tx)
                        ty    = f_y.update(ty)
                        tz    = f_z.update(tz)

                        packet = pack_opentrack(tx, ty, tz, yaw, pitch, roll)
                        sock.sendto(packet, (cfg.udp_host, cfg.udp_port))

                        self.live.update({
                            "yaw": yaw, "pitch": pitch, "roll": roll,
                            "x": tx, "y": ty, "z": tz, "tracking": True,
                        })

                        if self._preview:
                            self._draw_preview(frame, rvec, tvec, cam_mtx, dist_cfs,
                                               yaw, pitch, roll, tx, ty, tz)

                        frame_count += 1
                        if frame_count % DEBUG_PRINT_INTERVAL == 0:
                            print(f"  Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+5.1f}"
                                  f"  X:{tx:+5.1f}  Y:{ty:+5.1f}  Z:{tz:+5.1f}")
                else:
                    prev_rvec = None
                    prev_tvec = None
                    locked_eye_mid = None
                    self.live["tracking"] = False
                    if self._preview:
                        cv2.putText(frame, "No face / 未偵測到臉部",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self._preview:
                    cv2.imshow("Face Tracker Preview  (Q/ESC to close preview)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        self._preview = False
                        cv2.destroyAllWindows()
        finally:
            cap.release()
            sock.close()
            face_mesh.close()

    @staticmethod
    def _select_face(all_faces, w, h, prev_eye_mid, cfg: Config):
        def eye_mid(lm):
            ex = (lm[LM_LEFT_EYE].x + lm[LM_RIGHT_EYE].x) / 2.0 * w
            ey = (lm[LM_LEFT_EYE].y + lm[LM_RIGHT_EYE].y) / 2.0 * h
            return ex, ey

        def eye_dist(lm):
            lx = lm[LM_LEFT_EYE].x * w;  ly = lm[LM_LEFT_EYE].y * h
            rx = lm[LM_RIGHT_EYE].x * w; ry = lm[LM_RIGHT_EYE].y * h
            return math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

        candidates = [f.landmark for f in all_faces]

        if prev_eye_mid is not None:
            px, py = prev_eye_mid
            best, best_dist = None, float("inf")
            for lm in candidates:
                ex, ey = eye_mid(lm)
                d = math.sqrt((ex - px) ** 2 + (ey - py) ** 2)
                if d < best_dist:
                    best_dist, best = d, lm
            if best_dist < cfg.lock_snap_dist_px:
                return best, eye_mid(best)

        best = max(candidates, key=eye_dist)
        return best, eye_mid(best)

    @staticmethod
    def _solve_pose(image_pts, w, h, cfg: Config, prev_rvec, prev_tvec):
        cam = get_cam_matrix(w, h, cfg.focal_length_px)
        dist = np.zeros((4, 1))
        if prev_rvec is not None and prev_tvec is not None:
            ok, rv, tv = cv2.solvePnP(
                FACE_MODEL_3D, image_pts, cam, dist,
                rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rv, tv = cv2.solvePnP(
                FACE_MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_SQPNP
            )
        return (rv, tv) if ok else None

    @staticmethod
    def _estimate_position(landmarks, w, h, cfg: Config):
        lx = landmarks[LM_LEFT_EYE].x * w;  ly = landmarks[LM_LEFT_EYE].y * h
        rx = landmarks[LM_RIGHT_EYE].x * w; ry = landmarks[LM_RIGHT_EYE].y * h
        eye_dist_px = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
        if eye_dist_px < MIN_EYE_DIST_PX:
            return None
        focal = cfg.focal_length_px
        z_cm = (cfg.real_eye_dist_cm * focal) / eye_dist_px
        cx_px = (landmarks[LM_LEFT_EYE].x + landmarks[LM_RIGHT_EYE].x) / 2.0 * w
        cy_px = (landmarks[LM_LEFT_EYE].y + landmarks[LM_RIGHT_EYE].y) / 2.0 * h
        x_cm = (cx_px - w / 2.0) * z_cm / focal + cfg.cam_offset_x_cm
        y_cm = -((cy_px - h / 2.0) * z_cm / focal) + cfg.cam_offset_y_cm
        return x_cm, y_cm, z_cm

    @staticmethod
    def _draw_preview(frame, rvec, tvec, cam_mtx, dist_cfs,
                      yaw, pitch, roll, tx, ty, tz):
        _eye_o = np.float32([0.0, 2.5, -4.0])
        axes = np.float32([
            _eye_o + [AXIS_DISPLAY_LEN_CM, 0, 0],
            _eye_o + [0, -AXIS_DISPLAY_LEN_CM, 0],
            _eye_o + [0, 0, -AXIS_DISPLAY_LEN_CM],
            _eye_o,
        ])
        ap, _ = cv2.projectPoints(axes, rvec, tvec, cam_mtx, dist_cfs)
        o = tuple(ap[3].ravel().astype(int))
        cv2.line(frame, o, tuple(ap[0].ravel().astype(int)), (0, 0, 255), 2)
        cv2.line(frame, o, tuple(ap[1].ravel().astype(int)), (0, 255, 0), 2)
        cv2.line(frame, o, tuple(ap[2].ravel().astype(int)), (255, 0, 0), 2)
        cv2.putText(frame, f"Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+6.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"X:{tx:+6.1f}  Y:{ty:+6.1f}  Z:{tz:+6.1f} cm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
