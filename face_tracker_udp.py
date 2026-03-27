"""
face_tracker_udp.py  v2
========================
使用 MediaPipe 偵測臉部，計算 Yaw / Pitch / Roll 及頭部位移，
透過 UDP 以 OpenTrack 相容格式傳送給 UE5。

OpenTrack UDP 格式（48 bytes）：
  [X, Y, Z, Yaw, Pitch, Roll] — 6 個 little-endian double
  位置：cm；旋轉：degree

使用方式：
  python face_tracker_udp.py              # 預設 127.0.0.1:4242
  python face_tracker_udp.py --port 4243
  python face_tracker_udp.py --host 192.168.1.10
  python face_tracker_udp.py --no-preview   # 省效能，Ctrl+C 結束
"""

import cv2
import mediapipe as mp
import socket
import struct
import math
import argparse
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  🌐 ENVIRONMENT PROFILE  ← 換環境時只改這一區！
#  每次換相機、螢幕、電腦、或部署目標時，請更新以下設定。
#  Update ONLY this section when switching cameras, screens, or machines.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 🎥 攝影機 / Camera ────────────────────────────────────────────────────────
# 攝影機編號（0=內建、1=第一支外接，以此類推）
# Camera index (0=built-in, 1=first USB cam, etc.)
CAM_INDEX = 0

# 攝影機焦距（像素）— 執行 check_camera.py 查看建議值，或用 calibrate_camera.py 精確校正
# Focal length in pixels — run check_camera.py for suggestions, calibrate_camera.py for accuracy
# 640×480 @ FOV60° ≈ 554 | FOV70° ≈ 457 | FOV78° ≈ 395 | FOV90° ≈ 320
FOCAL_LENGTH_PX = 320

# MediaPipe 最多同時偵測幾張臉（越多越耗 CPU，建議 3~5）
# Max faces to detect simultaneously (more = more CPU, 3~5 recommended)
MAX_NUM_FACES = 5

# 跨幀臉部中心最大允許位移（pixel）— 超過此距離視為不同人，不繼續鎖定
# Max pixel distance between frames to still consider it the same face
# 640×480 畫面下 150px ≈ 畫面寬度 1/4
LOCK_SNAP_DIST_PX = 150

# ── 🖥️ 螢幕 / Screen ─────────────────────────────────────────────────────────
# 攝影機鏡頭相對螢幕中心的偏移（cm）
# Offset of camera lens from screen centre in cm
#   X: 靠右為正 / positive = camera is to the RIGHT of screen centre
#   Y: 靠上為正 / positive = camera is ABOVE screen centre
CAM_OFFSET_X_CM =  0.0   # 水平偏移（通常為 0，除非攝影機不在螢幕正中）
CAM_OFFSET_Y_CM = 16.2   # 垂直偏移（攝影機在螢幕上緣時約等於螢幕高度一半）

# ── 📡 UDP 目標 / UDP Target ──────────────────────────────────────────────────
# 接收端位址（UE5 / OpenTrack 所在的 IP 與 Port）
# Target address for UDP packets (UE5 / OpenTrack receiver)
UDP_HOST = "127.0.0.1"   # 本機="127.0.0.1"；遠端請填對方 IP
UDP_PORT = 4242

# ── 🧑 使用者 / User ──────────────────────────────────────────────────────────
# 使用者兩眼外角的真實距離（cm）— 成人平均約 9 cm
# Real distance between outer eye corners in cm — avg adult ~9 cm
REAL_EYE_DIST_CM = 9.0

# ═══════════════════════════════════════════════════════════════════════════════
#  ⚙️  調效參數 / TUNING PARAMETERS  ← 通常不必改，除非追蹤感覺不順
#  Fine-tuning params — usually leave as-is unless tracking feels off
# ═══════════════════════════════════════════════════════════════════════════════

# ── 卡爾曼濾波 / Kalman Filter ──────────────────────────────────────────────
# process_noise：越大 → 越信任量測值（靈敏但微抖），越小 → 越信任預測（穩但慢）
# measurement_noise：越大 → 越不信任量測值（更平滑），越小 → 越靈敏
# process_noise:  higher = trust measurement more (responsive but jittery)
# measurement_noise: higher = trust measurement less (smoother but slower)
KALMAN_PROCESS_NOISE_ROT   = 0.001   # 旋轉軸（Yaw/Pitch/Roll）— 越小越穩
KALMAN_MEASURE_NOISE_ROT   = 0.5     # 旋轉量測噪聲 — 越大越不信任抖動
KALMAN_PROCESS_NOISE_POS   = 0.001   # 位移軸（X/Y/Z）
KALMAN_MEASURE_NOISE_POS   = 0.3     # 位移量測噪聲

# 靈敏度倍率 / Sensitivity multiplier
YAW_SCALE   = 1.0
PITCH_SCALE = 1.0
ROLL_SCALE  = 1.0
X_SCALE     = 1.0
Y_SCALE     = 1.0
Z_SCALE     = 1.0

# ── 自適應噪聲 / Adaptive Noise ──────────────────────────────────────────────
# 靜止時自動提高 measurement noise（更穩），移動時恢復正常（低延遲）
# When still: increase measurement noise (smoother); when moving: normal (responsive)
ADAPTIVE_NOISE_MULTIPLIER = 8.0    # 靜止時 measurement noise 倍率
ADAPTIVE_VEL_THRESHOLD_ROT = 0.5   # 旋轉速度低於此值（°/幀）視為靜止
ADAPTIVE_VEL_THRESHOLD_POS = 0.3   # 位移速度低於此值（cm/幀）視為靜止

# 攝影機讀取連續失敗幾幀就自動停止（防止斷線後無限 loop）
# Auto-stop after this many consecutive read failures (prevents infinite loop)
MAX_READ_FAILURES = 30



# ─────────────────────────────────────────────────────────────────
#  MediaPipe 關鍵點索引 / Face mesh landmark indices
# ─────────────────────────────────────────────────────────────────
# 原始 6 點 / Original 6 points
LM_NOSE_TIP       = 4    # 鼻尖
LM_CHIN           = 152  # 下巴
LM_LEFT_EYE       = 33   # 左眼外角
LM_RIGHT_EYE      = 263  # 右眼外角
LM_LEFT_MOUTH     = 61   # 左嘴角
LM_RIGHT_MOUTH    = 291  # 右嘴角
LM_NOSE_BRIDGE    = 168  # 鼻樑中點（用於臉部中心位置）

# 新增 8 點：提升 solvePnP 穩定性 / 8 additional points for stability
LM_FOREHEAD       = 10   # 額頭頂部中心
LM_LEFT_EYE_IN    = 133  # 左眼內角
LM_RIGHT_EYE_IN   = 362  # 右眼內角
LM_LEFT_BROW_OUT  = 70   # 左眉外側
LM_RIGHT_BROW_OUT = 300  # 右眉外側
LM_LEFT_JAW       = 127  # 左顎
LM_RIGHT_JAW      = 356  # 右顎
LM_NOSE_BOTTOM    = 2    # 鼻底中心

# ─────────────────────────────────────────────────────────────────
#  演算法常數 / Algorithm constants
# ─────────────────────────────────────────────────────────────────

# pitch 超過此值表示旋轉矩陣已翻轉，需要折返修正
# If |pitch| exceeds this, the rotation matrix has flipped; subtract 180 to fix
PITCH_GIMBAL_THRESHOLD = 90.0

# 眼距低於此像素數視為無效（臉走出畫面或太暗時發生）
# Eye distance below this pixel count is considered invalid (face out of frame)
MIN_EYE_DIST_PX = 1.0

# 旋轉矩陣奇異點判定閾值（數學常數）
# Threshold for detecting singularity in rotation matrix decomposition
ROT_SINGULARITY_EPS = 1e-6

# MediaPipe 偵測 / 追蹤最低信心值
MEDIAPIPE_MIN_CONF = 0.5

# 預覽視窗 3D 座標軸的顯示長度（cm）
# Length of the 3D axes drawn on the preview window
AXIS_DISPLAY_LEN_CM = 5

# 每隔幾幀進行一次 terminal debug 輸出
DEBUG_PRINT_INTERVAL = 30

# ─────────────────────────────────────────────────────────────────
#  solvePnP 人臉 3D 模型參考點（單位：cm，鼻尖為原點）
#  14 個分佈廣的點，比 6 點更穩定（最小平方法平均掉噪聲）
#  14 well-distributed points for stable PnP (least-squares averages out noise)
# ─────────────────────────────────────────────────────────────────
FACE_MODEL_3D = np.array([
    # ── 原始 6 點 / Original 6 ──
    [ 0.000,  0.000,  0.000],  # (4)   鼻尖 / Nose tip
    [ 0.000, -3.300, -1.300],  # (152) 下巴 / Chin
    [-4.500,  2.500, -4.000],  # (33)  左眼外角 / Left eye outer
    [ 4.500,  2.500, -4.000],  # (263) 右眼外角 / Right eye outer
    [-2.000,  0.000, -2.200],  # (61)  左嘴角 / Left mouth
    [ 2.000,  0.000, -2.200],  # (291) 右嘴角 / Right mouth
    # ── 新增 8 點 / 8 new points ──
    [ 0.000,  6.500, -1.500],  # (10)  額頭 / Forehead top
    [-2.500,  2.700, -3.200],  # (133) 左眼內角 / Left eye inner
    [ 2.500,  2.700, -3.200],  # (362) 右眼內角 / Right eye inner
    [-4.200,  3.800, -3.000],  # (70)  左眉外側 / Left brow outer
    [ 4.200,  3.800, -3.000],  # (300) 右眉外側 / Right brow outer
    [-6.200, -1.500, -5.500],  # (127) 左顎 / Left jaw
    [ 6.200, -1.500, -5.500],  # (356) 右顎 / Right jaw
    [ 0.000, -1.000, -0.500],  # (2)   鼻底 / Nose bottom
], dtype=np.float64)
FACE_MODEL_IDX = [
    LM_NOSE_TIP, LM_CHIN, LM_LEFT_EYE, LM_RIGHT_EYE,
    LM_LEFT_MOUTH, LM_RIGHT_MOUTH,
    LM_FOREHEAD, LM_LEFT_EYE_IN, LM_RIGHT_EYE_IN,
    LM_LEFT_BROW_OUT, LM_RIGHT_BROW_OUT,
    LM_LEFT_JAW, LM_RIGHT_JAW, LM_NOSE_BOTTOM,
]


# ═══════════════════════════════════════════════════════════════════════════════
#  工具函式 / Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    """打包 OpenTrack UDP 封包 / Pack OpenTrack UDP packet (48 bytes)."""
    return struct.pack('<6d', x, y, z, yaw, pitch, roll)


def get_cam_matrix(w: int, h: int) -> np.ndarray:
    """攝影機內參矩陣 / Camera intrinsic matrix (no distortion assumed)."""
    f = FOCAL_LENGTH_PX
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0, 1  ]], dtype=np.float64)


def estimate_position_from_eyes(landmarks, w: int, h: int):
    """
    透過眼距三角測量法計算頭部位置（cm）
    Calculate head X/Y/Z in cm using inter-ocular distance triangulation.

    原理 / Principle:
      Z = (REAL_EYE_DIST_CM * focal_px) / eye_dist_px
      X = (face_center_x - image_cx) * Z / focal_px  + CAM_OFFSET_X_CM
      Y = (face_center_y - image_cy) * Z / focal_px  + CAM_OFFSET_Y_CM

    這樣計算的焦距估算對 X/Y 的比例沒有影響：
    X/Y/Z 都從同一個真實值推算，單位是公分。
    The focal length cancels out in the X/Y calculation,
    so accuracy only depends on REAL_EYE_DIST_CM.
    """
    # 左眼 / 右眼外角 pixel 坐標
    lx = landmarks[LM_LEFT_EYE].x * w
    ly = landmarks[LM_LEFT_EYE].y * h
    rx = landmarks[LM_RIGHT_EYE].x * w
    ry = landmarks[LM_RIGHT_EYE].y * h

    # 眼距（pixel）
    eye_dist_px = math.sqrt((rx - lx)**2 + (ry - ly)**2)
    if eye_dist_px < MIN_EYE_DIST_PX:
        return None

    # 從眼距推算 Z（cm）
    # Estimate Z from eye dist: Z = real_dist * focal / pixel_dist
    focal = FOCAL_LENGTH_PX
    z_cm = (REAL_EYE_DIST_CM * focal) / eye_dist_px

    # 臉部中心：雙眼外角中點（比鼻樑更接近眼睛視點位置）
    # Face centre: midpoint of outer eye corners (better represents eye/viewer position)
    cx_px = (landmarks[LM_LEFT_EYE].x + landmarks[LM_RIGHT_EYE].x) / 2.0 * w
    cy_px = (landmarks[LM_LEFT_EYE].y + landmarks[LM_RIGHT_EYE].y) / 2.0 * h

    # 影像中心到臉部中心的 pixel 偏量 → cm
    # Pixel offset of face centre from image centre → cm
    x_cm = (cx_px - w / 2.0) * z_cm / focal + CAM_OFFSET_X_CM
    # 注意：攝影機 Y 軸向下，取反得高度向上為正
    # Camera Y is downward; negate for "up = positive"
    y_cm = -((cy_px - h / 2.0) * z_cm / focal) + CAM_OFFSET_Y_CM

    return x_cm, y_cm, z_cm


def rot_to_euler(R: np.ndarray):
    """旋轉矩陣 → 歐拉角 (ZYX)，回傳 (x, y, z) radians。"""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > ROT_SINGULARITY_EPS:
        x = math.atan2( R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2( R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0.0
    return x, y, z


def sample_2d(landmarks, w: int, h: int) -> np.ndarray:
    """從 MediaPipe landmark 列表取出 2D 像素坐標。"""
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_MODEL_IDX],
        dtype=np.float64
    )



def select_face(all_faces, w: int, h: int, prev_eye_mid_px):
    """
    從 MediaPipe 回傳的多張臉中，選出要追蹤的那一張。

    邏輯 / Logic:
      - 若有鎖定位置（prev_eye_mid_px 不為 None）：
          找眼睛中點最靠近上一幀鎖定位置的臉。
          若最近距離 < LOCK_SNAP_DIST_PX，繼續追蹤（鎖定保持）。
          否則視為鎖定丟失，退回「選最顯眼」。
      - 若無鎖定 / 鎖定丟失：
          選眼距最大的臉（眼距大 = 最靠近鏡頭的人）。

    Args:
        all_faces: MediaPipe multi_face_landmarks 列表
        w, h     : 畫面寬高（pixel）
        prev_eye_mid_px: 上一幀鎖定臉的眼睛中點 (x, y)，None 表示尚未鎖定

    Returns:
        (landmark_list, new_eye_mid_px)
        new_eye_mid_px 為本幀選中臉的眼睛中點，供下一幀使用
    """
    def eye_mid(lm):
        """計算該臉的眼睛中點（pixel）/ Eye midpoint in pixels."""
        ex = (lm[LM_LEFT_EYE].x + lm[LM_RIGHT_EYE].x) / 2.0 * w
        ey = (lm[LM_LEFT_EYE].y + lm[LM_RIGHT_EYE].y) / 2.0 * h
        return ex, ey

    def eye_dist_px(lm):
        """計算兩眼外角的像素距離 / Eye distance in pixels."""
        lx = lm[LM_LEFT_EYE].x * w;  ly = lm[LM_LEFT_EYE].y * h
        rx = lm[LM_RIGHT_EYE].x * w; ry = lm[LM_RIGHT_EYE].y * h
        return math.sqrt((rx - lx)**2 + (ry - ly)**2)

    candidates = [face.landmark for face in all_faces]

    # ── 有鎖定位置：找最近的臉 ───────────────────────────────────────────────
    if prev_eye_mid_px is not None:
        px, py = prev_eye_mid_px
        best, best_dist = None, float('inf')
        for lm in candidates:
            ex, ey = eye_mid(lm)
            d = math.sqrt((ex - px)**2 + (ey - py)**2)
            if d < best_dist:
                best_dist = d
                best = lm

        # 距離夠近 → 繼續鎖定
        if best_dist < LOCK_SNAP_DIST_PX:
            return best, eye_mid(best)
        # 距離太遠（原主人離開）→ 鎖定丟失，往下繼續「選最顯眼」

    # ── 無鎖定 / 鎖定丟失：選眼距最大（最近）的臉 ─────────────────────────
    best = max(candidates, key=eye_dist_px)
    return best, eye_mid(best)


def solve_pose(image_pts: np.ndarray, w: int, h: int,
               dist_coeffs=None, prev_rvec=None, prev_tvec=None):
    """
    執行 solvePnP，回傳 (rvec, tvec) 或 None。
    使用 SQPNP（比 ITERATIVE 更穩定，不怕初始值）。
    若有上一帧的結果，改用 SOLVEPNP_ITERATIVE + useExtrinsicGuess 做 refine，
    這樣可以得到最穩定的效果。

    Args:
        dist_coeffs: 畸變校正係數，None 時使用零畸變
                     Distortion coefficients; None = no distortion
    """
    cam = get_cam_matrix(w, h)
    dist = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    if prev_rvec is not None and prev_tvec is not None:
        # 有上一帧：用它當初始猜測，做 iterative refine（最穩）
        ok, rv, tv = cv2.solvePnP(
            FACE_MODEL_3D, image_pts, cam, dist,
            rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    else:
        # 第一帧：用 SQPNP 找到好的全域解
        ok, rv, tv = cv2.solvePnP(
            FACE_MODEL_3D, image_pts, cam, dist,
            flags=cv2.SOLVEPNP_SQPNP
        )
    return (rv, tv) if ok else None


# ═══════════════════════════════════════════════════════════════════════════════
#  卡爾曼濾波器 / Kalman Filter (replaces EMA for lower latency)
#
#  原理 / How it works:
#    狀態 = [位置, 速度]，量測 = [位置]
#    State = [position, velocity], Measurement = [position]
#    每幀先用速度「預測」下一個位置（幾乎零延遲），
#    再用實際量測值「修正」預測（降低抖動）。
#    Each frame: predict next position using velocity (near-zero lag),
#    then correct prediction with actual measurement (reduces jitter).
# ═══════════════════════════════════════════════════════════════════════════════
class KalmanFilter1D:
    """
    1D 自適應卡爾曼濾波器，狀態=[位置, 速度]。
    1D Adaptive Kalman filter with state=[position, velocity].

    自適應機制 / Adaptive mechanism:
      當估計速度接近 0（靜止）時，自動提高 measurement noise，
      讓輸出更穩定；移動時恢復正常 noise，保持低延遲。
      When estimated velocity ≈ 0 (still), automatically increase
      measurement noise for stability; restore when moving.
    """
    def __init__(self, process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 vel_threshold: float = 1.0,
                 adaptive_multiplier: float = 1.0):
        # 2 個狀態（位置 + 速度），1 個量測（位置）
        # 2 states (pos + vel), 1 measurement (pos)
        self.kf = cv2.KalmanFilter(2, 1)

        # 狀態轉移矩陣：位置 = 位置 + 速度 × dt（dt=1 幀）
        # Transition: pos = pos + vel*dt (dt=1 frame)
        self.kf.transitionMatrix = np.array(
            [[1, 1],
             [0, 1]], dtype=np.float32)

        # 量測矩陣：只量測位置
        # Measurement: observe position only
        self.kf.measurementMatrix = np.array(
            [[1, 0]], dtype=np.float32)

        # 過程噪聲（模型不確定性）
        # Process noise (model uncertainty)
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise

        # 量測噪聲（感測器不確定性）— 基準值，自適應模式下會動態調整
        # Measurement noise (sensor uncertainty) — base value, dynamically adjusted
        self._base_meas_noise = measurement_noise
        self.kf.measurementNoiseCov = np.array(
            [[measurement_noise]], dtype=np.float32)

        # 自適應參數 / Adaptive parameters
        self._vel_threshold = vel_threshold       # 速度閾值（低於此視為靜止）
        self._adaptive_mult = adaptive_multiplier  # 靜止時 noise 倍率

        # 初始狀態後驗協方差（較大 = 前幾幀快速收斂）
        # Initial posterior covariance (large = converge fast in first frames)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)

        # 初始狀態
        self.kf.statePost = np.zeros((2, 1), dtype=np.float32)
        self._initialized = False

    def update(self, measurement: float) -> float:
        """
        輸入新量測值，回傳濾波後的估計值。
        Feed new measurement, return filtered estimate.
        """
        meas = np.array([[np.float32(measurement)]])

        if not self._initialized:
            # 第一次：直接把狀態設為量測值，速度為 0
            # First call: set state to measurement, velocity=0
            self.kf.statePost = np.array(
                [[np.float32(measurement)], [0.0]], dtype=np.float32)
            self._initialized = True
            return measurement

        # 預測 / Predict
        self.kf.predict()

        # 自適應：根據速度「漸變」measurement noise（不是開關式！）
        # Adaptive: GRADUALLY adjust noise based on velocity (no sudden switch!)
        # vel=0 → noise × multiplier（最穩）; vel≥threshold → noise × 1（最靈敏）
        est_vel = abs(float(self.kf.statePost[1, 0]))
        t = min(est_vel / self._vel_threshold, 1.0)  # 0=靜止, 1=移動中
        # 線性內插：靜止=base×mult, 移動=base×1
        adaptive_factor = self._adaptive_mult * (1.0 - t) + 1.0 * t
        self.kf.measurementNoiseCov[0, 0] = self._base_meas_noise * adaptive_factor

        # 修正 / Correct
        corrected = self.kf.correct(meas)
        return float(corrected[0, 0])

    def reset(self):
        """重置濾波器狀態 / Reset filter state."""
        self.kf.statePost = np.zeros((2, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        self.kf.measurementNoiseCov[0, 0] = self._base_meas_noise
        self._initialized = False


# ═══════════════════════════════════════════════════════════════════════════════
#  主程式 / Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='MediaPipe → OpenTrack UDP → UE5')
    # 預設值讀自 ENVIRONMENT PROFILE，也可用命令列參數覆蓋
    # Defaults come from ENVIRONMENT PROFILE above; CLI args override them
    parser.add_argument('--host', default=UDP_HOST)
    parser.add_argument('--port', default=UDP_PORT, type=int)
    parser.add_argument('--cam',  default=CAM_INDEX, type=int)
    parser.add_argument('--no-preview', action='store_true',
                        help='關閉預覽視窗（省 CPU）/ Disable preview window')
    parser.add_argument('--distortion', default=None, type=str,
                        help='鏡頭畸變校正 .npz 檔路徑 / Path to distortion .npz file')
    args = parser.parse_args()

    print(f"[INFO] 目標 / Target : {args.host}:{args.port}")
    print(f"[INFO] 攝影機 / Cam  : #{args.cam}")
    print("[INFO] 按 Q/ESC 結束 / Press Q or ESC to quit  (no-preview: Ctrl+C)")

    # ── 載入畸變校正 / Load distortion coefficients ─────────────────────────
    dist_coeffs = None
    if args.distortion:
        try:
            data = np.load(args.distortion)
            dist_coeffs = data['dist_coeffs']
            print(f"[INFO] 已載入畸變校正 / Loaded distortion from: {args.distortion}")
        except Exception as e:
            print(f"[WARN] 無法載入畸變校正 / Failed to load distortion: {e}")
            print("[WARN] 繼續使用零畸變 / Continuing with zero distortion")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap  = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機 / Cannot open camera")
        return

    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        max_num_faces=MAX_NUM_FACES,   # 允許偵測多張臉以支援 face lock
        refine_landmarks=True,
        min_detection_confidence=MEDIAPIPE_MIN_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_CONF
    )

    # 每軸一個卡爾曼濾波器，旋轉 / 位移使用不同噪聲與自適應參數
    # One Kalman filter per axis; rotation & position use different noise + adaptive params
    kf_yaw   = KalmanFilter1D(KALMAN_PROCESS_NOISE_ROT, KALMAN_MEASURE_NOISE_ROT,
                               ADAPTIVE_VEL_THRESHOLD_ROT, ADAPTIVE_NOISE_MULTIPLIER)
    kf_pitch = KalmanFilter1D(KALMAN_PROCESS_NOISE_ROT, KALMAN_MEASURE_NOISE_ROT,
                               ADAPTIVE_VEL_THRESHOLD_ROT, ADAPTIVE_NOISE_MULTIPLIER)
    kf_roll  = KalmanFilter1D(KALMAN_PROCESS_NOISE_ROT, KALMAN_MEASURE_NOISE_ROT,
                               ADAPTIVE_VEL_THRESHOLD_ROT, ADAPTIVE_NOISE_MULTIPLIER)
    kf_x     = KalmanFilter1D(KALMAN_PROCESS_NOISE_POS, KALMAN_MEASURE_NOISE_POS,
                               ADAPTIVE_VEL_THRESHOLD_POS, ADAPTIVE_NOISE_MULTIPLIER)
    kf_y     = KalmanFilter1D(KALMAN_PROCESS_NOISE_POS, KALMAN_MEASURE_NOISE_POS,
                               ADAPTIVE_VEL_THRESHOLD_POS, ADAPTIVE_NOISE_MULTIPLIER)
    kf_z     = KalmanFilter1D(KALMAN_PROCESS_NOISE_POS, KALMAN_MEASURE_NOISE_POS,
                               ADAPTIVE_VEL_THRESHOLD_POS, ADAPTIVE_NOISE_MULTIPLIER)

    # 儲存上一帧的 solvePnP 結果，用於增量 refine（更穩定）
    prev_rvec = None
    prev_tvec = None

    # Face lock：記錄上一帧鎖定臉的眼睛中點（pixel），None 表示尚未鎖定
    # Face lock: eye midpoint of locked face from last frame; None = unlocked
    locked_eye_mid = None

    cam_mtx  = None   # 第一帧後初始化
    dist_cfs = np.zeros((4, 1))

    # 攝影機讀取失敗計數器 / Camera read failure counter
    read_fail_count = 0
    # 幀數計數器（debug 輸出用）/ Frame counter for debug prints
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_fail_count += 1
                if read_fail_count >= MAX_READ_FAILURES:
                    print(f"[ERROR] 攝影機連續 {MAX_READ_FAILURES} 幀讀取失敗，自動停止")
                    print("[ERROR] Camera read failed continuously, auto-stopping")
                    break
                continue
            read_fail_count = 0  # 讀到了就歸零 / Reset on success

            frame = cv2.flip(frame, 1)     # 水平翻轉（鏡像）
            h, w  = frame.shape[:2]

            if cam_mtx is None:
                cam_mtx = get_cam_matrix(w, h)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                # 用 face lock 邏輯選出要追蹤的那張臉
                # Use face-lock logic to pick the target face
                lm, locked_eye_mid = select_face(
                    results.multi_face_landmarks, w, h, locked_eye_mid
                )
                img_pts = sample_2d(lm, w, h)

                pnp = solve_pose(img_pts, w, h, dist_coeffs, prev_rvec, prev_tvec)
                if pnp:
                    rvec, tvec = pnp
                    prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

                    # 旋轉向量 → 歐拉角
                    R, _ = cv2.Rodrigues(rvec)
                    ex, ey, ez = rot_to_euler(R)

                    # ── 座標軸修正 / Axis correction ────────────────────────
                    # solvePnP 假設臉部模型面向 +Z（相機後方），
                    # 所以臉正對鏡頭時 pitch ≈ ±180°。
                    # 減去 180° 把它歸零，再用 sign 讓方向符合 OpenTrack。
                    #
                    # solvePnP places the face 'behind' the camera model,
                    # resulting in pitch ≈ ±180 when facing camera directly.
                    # Subtract 180 to normalise, then fix signs to match
                    # OpenTrack convention (yaw: right=+, pitch: up=+)

                    raw_yaw   = math.degrees(ey)
                    raw_pitch = math.degrees(ex)
                    raw_roll  = math.degrees(ez)

                    # 把 pitch 從 ~±180° 空間移回 ~0° 附近
                    # Collapse pitch from ≈180 space back to ≈0
                    if raw_pitch > PITCH_GIMBAL_THRESHOLD:
                        raw_pitch -= 180
                    elif raw_pitch < -PITCH_GIMBAL_THRESHOLD:
                        raw_pitch += 180

                    # OpenTrack 慣例：右轉=正 Yaw，抬頭=負 Pitch（攝影機 Y 向下）
                    # OpenTrack convention: yaw right=+, pitch up=- (cam Y down)
                    yaw   =  raw_yaw   * YAW_SCALE
                    pitch = -raw_pitch * PITCH_SCALE   # 攝影機 Y 軸向下，取反
                    roll  =  raw_roll  * ROLL_SCALE

                    # 位移（cm）— 用眼距三角測量法，不依賴 solvePnP 的 tvec
                    # Position (cm) via eye-distance triangulation (more accurate)
                    pos = estimate_position_from_eyes(lm, w, h)
                    if pos:
                        tx, ty, tz = pos
                        tx *= X_SCALE
                        ty *= Y_SCALE
                        tz *= Z_SCALE
                    else:
                        tx, ty, tz = 0.0, 0.0, 0.0

                    # 卡爾曼濾波（預測 + 修正）/ Kalman filter (predict + correct)
                    yaw   = kf_yaw  .update(yaw)
                    pitch = kf_pitch.update(pitch)
                    roll  = kf_roll .update(roll)
                    tx    = kf_x    .update(tx)
                    ty    = kf_y    .update(ty)
                    tz    = kf_z    .update(tz)

                    # 送出 OpenTrack UDP 封包
                    packet = pack_opentrack(tx, ty, tz, yaw, pitch, roll)
                    sock.sendto(packet, (args.host, args.port))

                    # debug 輸出（每 N 帧印一次，不影響效能）
                    # Debug print every N frames
                    frame_count += 1
                    if frame_count % DEBUG_PRINT_INTERVAL == 0:
                        print(f"  Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+5.1f}  "
                              f"X:{tx:+5.1f}  Y:{ty:+5.1f}  Z:{tz:+5.1f}")

                    if not args.no_preview:
                        # 繪製 3D 坐標軸 / Draw 3D axes
                        # 原點設在雙眼中點（模型空間 Y=+2.5, Z=-4.0）而非鼻尖
                        # Origin at eye midpoint in model space, not nose tip
                        _eye_o = np.float32([0.0, 2.5, -4.0])
                        axes = np.float32([
                            _eye_o + [AXIS_DISPLAY_LEN_CM, 0, 0],
                            _eye_o + [0, -AXIS_DISPLAY_LEN_CM, 0],
                            _eye_o + [0, 0, -AXIS_DISPLAY_LEN_CM],
                            _eye_o
                        ])
                        ap, _ = cv2.projectPoints(axes, rvec, tvec, cam_mtx, dist_cfs)
                        o = tuple(ap[3].ravel().astype(int))
                        cv2.line(frame, o, tuple(ap[0].ravel().astype(int)), (0,0,255), 2)
                        cv2.line(frame, o, tuple(ap[1].ravel().astype(int)), (0,255,0), 2)
                        cv2.line(frame, o, tuple(ap[2].ravel().astype(int)), (255,0,0), 2)
                        # 文字 HUD
                        cv2.putText(frame,
                            f"Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+6.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(frame,
                            f"X:{tx:+6.1f}  Y:{ty:+6.1f}  Z:{tz:+6.1f} cm",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            else:
                # 臉部消失時，重置 solvePnP 初始猜測、face lock、卡爾曼濾波器
                # Reset solvePnP guess, face lock, and Kalman filters
                prev_rvec      = None
                prev_tvec      = None
                locked_eye_mid = None   # 下次有人出現時重新從最顯眼的臉開始鎖
                for kf in (kf_yaw, kf_pitch, kf_roll, kf_x, kf_y, kf_z):
                    kf.reset()
                if not args.no_preview:
                    cv2.putText(frame, "No face / 未偵測到臉部",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if not args.no_preview:
                cv2.imshow("MediaPipe → UE5  (Q/ESC to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 收到，結束中 / Stopping...")

    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    print("[INFO] 已結束 / Done")


if __name__ == "__main__":
    main()
