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
#  可調整參數 / TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# 平滑強度：越小越穩（但反應慢），越大越靈敏（但容易抖）
# Smoothing: lower = smoother/slower, higher = faster/jittery
SMOOTH_ALPHA = 0.25   # 建議範圍 0.05~0.25

# 死區：低於此變化量的微抖就直接忽略，單位：degree / cm
# Deadzone: ignore changes smaller than this threshold (deg / cm)
DEADZONE_ROT = 0.3    # 旋轉死區（度）
DEADZONE_POS = 0.15   # 位移死區（cm）

# 靈敏度倍率 / Sensitivity multiplier
YAW_SCALE   = 1.0
PITCH_SCALE = 1.0
ROLL_SCALE  = 1.0
X_SCALE     = 1.0
Y_SCALE     = 1.0
Z_SCALE     = 1.0

# 攝影機焦距估算（僅用於旋轉計算）/ Camera focal length (used only for rotation / solvePnP)
FOCAL_LENGTH_PX = 600.0

# ─────────────────────────────────────────────────────────────────
#  X/Y/Z 位置估算（三角測量法）
#  不依賴焦距猜測，直接用真實眼距估算。
#  Position estimation via triangulation (no focal length guessing)
# ─────────────────────────────────────────────────────────────────

# 兩眼外角的真實距離（cm）— 成人平均兩眼外角距離約 9 cm
# Real inter-ocular distance (outer eye corners, cm) — avg adult ~9 cm
REAL_EYE_DIST_CM = 9.0

# 攝影機安裝位置偏移（cm，相對螢幕中心）
# Offset of camera lens from screen center in cm (+ = camera is right/up of screen center)
# 如果攝影機就在螢幕正上方且若底部對齊，自行調整。
CAM_OFFSET_X_CM = 0.0   # 攝影機水平偏補（靠右為正）
CAM_OFFSET_Y_CM = 0.0   # 攝影機垂直偏補（靠上為正）

# ─────────────────────────────────────────────────────────────────
#  MediaPipe 關鍵點索引 / Face mesh landmark indices
# ─────────────────────────────────────────────────────────────────
LM_NOSE_TIP    = 4    # 鼻尖

LM_CHIN        = 152  # 下巴
LM_LEFT_EYE    = 33   # 左眼外角
LM_RIGHT_EYE   = 263  # 右眼外角
LM_LEFT_MOUTH  = 61   # 左嘴角
LM_RIGHT_MOUTH = 291  # 右嘴角
LM_NOSE_BRIDGE = 168  # 鼻樑中點（用於臉部中心位置）

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
# 選用分佈廣且穩定的關鍵點：鼻尖、下巴、雙眼外角、雙嘴角
# 3D face model reference points (cm, nose-tip as origin)
# ─────────────────────────────────────────────────────────────────
FACE_MODEL_3D = np.array([
    [ 0.000,  0.000,  0.000],  # LM_NOSE_TIP    (4)   鼻尖
    [ 0.000, -3.300, -1.300],  # LM_CHIN        (152) 下巴
    [-4.500,  2.500, -4.000],  # LM_LEFT_EYE    (33)  左眼外角
    [ 4.500,  2.500, -4.000],  # LM_RIGHT_EYE   (263) 右眼外角
    [-2.000,  0.000, -2.200],  # LM_LEFT_MOUTH  (61)  左嘴角
    [ 2.000,  0.000, -2.200],  # LM_RIGHT_MOUTH (291) 右嘴角
], dtype=np.float64)
FACE_MODEL_IDX = [LM_NOSE_TIP, LM_CHIN, LM_LEFT_EYE, LM_RIGHT_EYE, LM_LEFT_MOUTH, LM_RIGHT_MOUTH]


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

    # 臉部中心：鼻樑中點
    cx_px = landmarks[LM_NOSE_BRIDGE].x * w
    cy_px = landmarks[LM_NOSE_BRIDGE].y * h

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


def solve_pose(image_pts: np.ndarray, w: int, h: int,
               prev_rvec=None, prev_tvec=None):
    """
    執行 solvePnP，回傳 (rvec, tvec) 或 None。
    使用 SQPNP（比 ITERATIVE 更穩定，不怕初始值）。
    若有上一帧的結果，改用 SOLVEPNP_ITERATIVE + useExtrinsicGuess 做 refine，
    這樣可以得到最穩定的效果。
    """
    cam = get_cam_matrix(w, h)
    dist = np.zeros((4, 1))

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
#  EMA 低通濾波器 + 死區 / Low-pass filter with deadzone
# ═══════════════════════════════════════════════════════════════════════════════
class SmoothFilter:
    def __init__(self, alpha: float, deadzone: float = 0.0):
        self.alpha    = alpha     # 平滑係數
        self.deadzone = deadzone  # 死區閾值
        self.value    = None      # 目前輸出值

    def update(self, new_val: float) -> float:
        if self.value is None:
            self.value = new_val
            return self.value
        # 死區：若變化太小，直接沿用舊值，不更新
        if abs(new_val - self.value) < self.deadzone:
            return self.value
        # EMA 平滑
        self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value


# ═══════════════════════════════════════════════════════════════════════════════
#  主程式 / Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='MediaPipe → OpenTrack UDP → UE5')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=4242, type=int)
    parser.add_argument('--cam',  default=0,   type=int)
    parser.add_argument('--no-preview', action='store_true',
                        help='關閉預覽視窗（省 CPU）/ Disable preview window')
    args = parser.parse_args()

    print(f"[INFO] 目標 / Target : {args.host}:{args.port}")
    print(f"[INFO] 攝影機 / Cam  : #{args.cam}")
    print("[INFO] 按 Q/ESC 結束 / Press Q or ESC to quit  (no-preview: Ctrl+C)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap  = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機 / Cannot open camera")
        return

    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MEDIAPIPE_MIN_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_CONF
    )

    # 每軸一個濾波器，旋轉 / 位移分開設死區
    f_yaw   = SmoothFilter(SMOOTH_ALPHA, DEADZONE_ROT)
    f_pitch = SmoothFilter(SMOOTH_ALPHA, DEADZONE_ROT)
    f_roll  = SmoothFilter(SMOOTH_ALPHA, DEADZONE_ROT)
    f_x     = SmoothFilter(SMOOTH_ALPHA, DEADZONE_POS)
    f_y     = SmoothFilter(SMOOTH_ALPHA, DEADZONE_POS)
    f_z     = SmoothFilter(SMOOTH_ALPHA, DEADZONE_POS)

    # 儲存上一帧的 solvePnP 結果，用於增量 refine（更穩定）
    prev_rvec = None
    prev_tvec = None

    cam_mtx  = None   # 第一帧後初始化
    dist_cfs = np.zeros((4, 1))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)     # 水平翻轉（鏡像）
            h, w  = frame.shape[:2]

            if cam_mtx is None:
                cam_mtx = get_cam_matrix(w, h)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm      = results.multi_face_landmarks[0].landmark
                img_pts = sample_2d(lm, w, h)

                pnp = solve_pose(img_pts, w, h, prev_rvec, prev_tvec)
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

                    # 平滑 + 死區
                    yaw   = f_yaw  .update(yaw)
                    pitch = f_pitch.update(pitch)
                    roll  = f_roll .update(roll)
                    tx    = f_x    .update(tx)
                    ty    = f_y    .update(ty)
                    tz    = f_z    .update(tz)

                    # 送出 OpenTrack UDP 封包
                    packet = pack_opentrack(tx, ty, tz, yaw, pitch, roll)
                    sock.sendto(packet, (args.host, args.port))

                    # debug 輸出（每 30 帧印一次，不影響效能）
                    # Debug print every 30 frames
                    if not hasattr(main, '_fc'):
                        main._fc = 0
                    main._fc += 1
                    if main._fc % DEBUG_PRINT_INTERVAL == 0:
                        print(f"  Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+5.1f}  "
                              f"X:{tx:+5.1f}  Y:{ty:+5.1f}  Z:{tz:+5.1f}")

                    if not args.no_preview:
                        # 繪製 3D 坐標軸 / Draw 3D axes
                        axes = np.float32([
                            [AXIS_DISPLAY_LEN_CM, 0, 0],
                            [0, -AXIS_DISPLAY_LEN_CM, 0],
                            [0, 0, -AXIS_DISPLAY_LEN_CM],
                            [0, 0, 0]
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
                # 臉部消失時，重置 solvePnP 初始猜測，避免舊值污染下一次
                prev_rvec = None
                prev_tvec = None
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
