"""
kalman_tuner.py
===============
即時調整卡爾曼濾波參數的測試工具。
Real-time Kalman filter parameter tuner with OpenCV trackbar sliders.

用法 / Usage:
  python kalman_tuner.py
  python kalman_tuner.py --cam 1

功能 / Features:
  - 4 個滑桿即時調整 process_noise / measurement_noise（旋轉 & 位移）
  - 預覽視窗同時顯示 Raw（原始）& Filtered（濾波後）數值
  - 結束時印出最終參數，可直接複製到 face_tracker_udp.py
"""

import cv2
import mediapipe as mp
import socket
import struct
import math
import argparse
import numpy as np

# ── 從 face_tracker_udp.py 匯入共用常數與函式 ────────────────────────────────
# Import shared constants and functions from the main script
from face_tracker_udp import (
    CAM_INDEX, FOCAL_LENGTH_PX, MAX_NUM_FACES, LOCK_SNAP_DIST_PX,
    MEDIAPIPE_MIN_CONF, DEBUG_PRINT_INTERVAL, AXIS_DISPLAY_LEN_CM,
    YAW_SCALE, PITCH_SCALE, ROLL_SCALE, X_SCALE, Y_SCALE, Z_SCALE,
    PITCH_GIMBAL_THRESHOLD, UDP_HOST, UDP_PORT,
    LM_LEFT_EYE, LM_RIGHT_EYE,
    get_cam_matrix, estimate_position_from_eyes, rot_to_euler,
    sample_2d, select_face, solve_pose, KalmanFilter1D,
    pack_opentrack,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  滑桿控制 / Trackbar helpers
# ═══════════════════════════════════════════════════════════════════════════════

# 滑桿的範圍：0~1000，對應實際值 0.0001 ~ 1.0（對數刻度）
# Trackbar range: 0~1000, mapped to 0.0001~1.0 (logarithmic scale)
SLIDER_MAX = 1000
LOG_MIN = -4.0   # 10^(-4) = 0.0001
LOG_MAX =  0.0   # 10^(0)  = 1.0

def slider_to_value(pos: int) -> float:
    """滑桿位置 (0~1000) → 實際值 (0.0001~1.0)，對數刻度。"""
    # Logarithmic mapping for fine control at small values
    t = pos / SLIDER_MAX
    log_val = LOG_MIN + t * (LOG_MAX - LOG_MIN)
    return 10 ** log_val

def value_to_slider(val: float) -> int:
    """實際值 → 滑桿位置（反函數）。"""
    log_val = math.log10(max(val, 1e-5))
    t = (log_val - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return int(max(0, min(SLIDER_MAX, t * SLIDER_MAX)))


# 全域變數存放當前滑桿數值 / Global state for current slider values
params = {
    'proc_rot':  0.001,   # KALMAN_PROCESS_NOISE_ROT
    'meas_rot':  0.5,     # KALMAN_MEASURE_NOISE_ROT
    'proc_pos':  0.001,   # KALMAN_PROCESS_NOISE_POS
    'meas_pos':  0.3,     # KALMAN_MEASURE_NOISE_POS
}

def on_proc_rot(pos):
    params['proc_rot'] = slider_to_value(pos)
def on_meas_rot(pos):
    params['meas_rot'] = slider_to_value(pos)
def on_proc_pos(pos):
    params['proc_pos'] = slider_to_value(pos)
def on_meas_pos(pos):
    params['meas_pos'] = slider_to_value(pos)


def rebuild_filters():
    """用當前滑桿參數重建所有 Kalman filter。"""
    return {
        'yaw':   KalmanFilter1D(params['proc_rot'], params['meas_rot']),
        'pitch': KalmanFilter1D(params['proc_rot'], params['meas_rot']),
        'roll':  KalmanFilter1D(params['proc_rot'], params['meas_rot']),
        'x':     KalmanFilter1D(params['proc_pos'], params['meas_pos']),
        'y':     KalmanFilter1D(params['proc_pos'], params['meas_pos']),
        'z':     KalmanFilter1D(params['proc_pos'], params['meas_pos']),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  主程式 / Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Kalman Filter 即時調參工具（含 UDP）')
    parser.add_argument('--cam',  default=CAM_INDEX, type=int)
    parser.add_argument('--host', default=UDP_HOST)
    parser.add_argument('--port', default=UDP_PORT, type=int)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機 / Cannot open camera")
        return

    # ── 建立視窗 & 滑桿 ──────────────────────────────────────────────────────
    # ── UDP socket ────────────────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[INFO] UDP 目標 / Target : {args.host}:{args.port}")

    win = "Kalman Tuner (Q/ESC to quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # 建立 4 個滑桿 / Create 4 trackbars
    cv2.createTrackbar("ProcRot",  win, value_to_slider(params['proc_rot']),  SLIDER_MAX, on_proc_rot)
    cv2.createTrackbar("MeasRot",  win, value_to_slider(params['meas_rot']),  SLIDER_MAX, on_meas_rot)
    cv2.createTrackbar("ProcPos",  win, value_to_slider(params['proc_pos']),  SLIDER_MAX, on_proc_pos)
    cv2.createTrackbar("MeasPos",  win, value_to_slider(params['meas_pos']),  SLIDER_MAX, on_meas_pos)

    # ── MediaPipe 初始化 ─────────────────────────────────────────────────────
    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,
        min_detection_confidence=MEDIAPIPE_MIN_CONF,
        min_tracking_confidence=MEDIAPIPE_MIN_CONF,
    )

    # ── 濾波器 & 狀態 ───────────────────────────────────────────────────────
    filters = rebuild_filters()
    prev_rvec = None
    prev_tvec = None
    locked_eye_mid = None
    cam_mtx = None
    dist_cfs = np.zeros((4, 1))

    # 追蹤上一次的參數，偵測滑桿變動時重建 filter
    last_params = dict(params)

    print("=" * 60)
    print("  Kalman Tuner — 拖動滑桿即時調參")
    print("  ProcRot / MeasRot = 旋轉軸 process / measurement noise")
    print("  ProcPos / MeasPos = 位移軸 process / measurement noise")
    print("  按 Q / ESC 結束，結束時會印出最終參數")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if cam_mtx is None:
                cam_mtx = get_cam_matrix(w, h)

            # ── 偵測滑桿變動 → 重建 filter ──────────────────────────────────
            if params != last_params:
                filters = rebuild_filters()
                last_params = dict(params)
                prev_rvec = None
                prev_tvec = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # 預設值 / Defaults
            raw_yaw = raw_pitch = raw_roll = 0.0
            raw_tx = raw_ty = raw_tz = 0.0
            flt_yaw = flt_pitch = flt_roll = 0.0
            flt_tx = flt_ty = flt_tz = 0.0
            face_found = False

            if results.multi_face_landmarks:
                lm, locked_eye_mid = select_face(
                    results.multi_face_landmarks, w, h, locked_eye_mid
                )
                img_pts = sample_2d(lm, w, h)

                pnp = solve_pose(img_pts, w, h, None, prev_rvec, prev_tvec)
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

                    raw_yaw   =  raw_yaw   * YAW_SCALE
                    raw_pitch = -raw_pitch  * PITCH_SCALE
                    raw_roll  =  raw_roll   * ROLL_SCALE

                    pos = estimate_position_from_eyes(lm, w, h)
                    if pos:
                        raw_tx, raw_ty, raw_tz = pos
                        raw_tx *= X_SCALE
                        raw_ty *= Y_SCALE
                        raw_tz *= Z_SCALE

                    # 卡爾曼濾波 / Apply Kalman filters
                    flt_yaw   = filters['yaw']  .update(raw_yaw)
                    flt_pitch = filters['pitch'].update(raw_pitch)
                    flt_roll  = filters['roll'] .update(raw_roll)
                    flt_tx    = filters['x']    .update(raw_tx)
                    flt_ty    = filters['y']    .update(raw_ty)
                    flt_tz    = filters['z']    .update(raw_tz)
                    face_found = True

                    # 送出 UDP 封包 / Send UDP packet
                    packet = pack_opentrack(flt_tx, flt_ty, flt_tz,
                                           flt_yaw, flt_pitch, flt_roll)
                    sock.sendto(packet, (args.host, args.port))

                    # 繪製 3D 座標軸 / Draw axes
                    _eye_o = np.float32([0.0, 2.5, -4.0])
                    axes = np.float32([
                        _eye_o + [AXIS_DISPLAY_LEN_CM, 0, 0],
                        _eye_o + [0, -AXIS_DISPLAY_LEN_CM, 0],
                        _eye_o + [0, 0, -AXIS_DISPLAY_LEN_CM],
                        _eye_o,
                    ])
                    ap, _ = cv2.projectPoints(axes, rvec, tvec, cam_mtx, dist_cfs)
                    o = tuple(ap[3].ravel().astype(int))
                    cv2.line(frame, o, tuple(ap[0].ravel().astype(int)), (0,0,255), 2)
                    cv2.line(frame, o, tuple(ap[1].ravel().astype(int)), (0,255,0), 2)
                    cv2.line(frame, o, tuple(ap[2].ravel().astype(int)), (255,0,0), 2)
            else:
                prev_rvec = None
                prev_tvec = None
                locked_eye_mid = None
                for kf in filters.values():
                    kf.reset()

            # ── 繪製 HUD ────────────────────────────────────────────────────
            y0 = 25
            dy = 28

            # 當前參數值 / Current param values
            cv2.putText(frame,
                f"ProcRot:{params['proc_rot']:.4f}  MeasRot:{params['meas_rot']:.4f}",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame,
                f"ProcPos:{params['proc_pos']:.4f}  MeasPos:{params['meas_pos']:.4f}",
                (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            if face_found:
                # Raw 值（紅色）
                cv2.putText(frame,
                    f"RAW  Yaw:{raw_yaw:+6.1f} Pit:{raw_pitch:+6.1f} Rol:{raw_roll:+5.1f}  "
                    f"X:{raw_tx:+5.1f} Y:{raw_ty:+5.1f} Z:{raw_tz:+5.1f}",
                    (10, y0 + dy*2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,100,255), 1)

                # Filtered 值（綠色）
                cv2.putText(frame,
                    f"FILT Yaw:{flt_yaw:+6.1f} Pit:{flt_pitch:+6.1f} Rol:{flt_roll:+5.1f}  "
                    f"X:{flt_tx:+5.1f} Y:{flt_ty:+5.1f} Z:{flt_tz:+5.1f}",
                    (10, y0 + dy*3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
            else:
                cv2.putText(frame, "No face / 未偵測到臉部",
                    (10, y0 + dy*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

    # ── 結束時印出最終參數，方便複製 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  最終參數 / Final parameters")
    print("  複製以下內容到 face_tracker_udp.py 的調效參數區：")
    print("=" * 60)
    print(f"KALMAN_PROCESS_NOISE_ROT   = {params['proc_rot']:.4f}")
    print(f"KALMAN_MEASURE_NOISE_ROT   = {params['meas_rot']:.4f}")
    print(f"KALMAN_PROCESS_NOISE_POS   = {params['proc_pos']:.4f}")
    print(f"KALMAN_MEASURE_NOISE_POS   = {params['meas_pos']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
