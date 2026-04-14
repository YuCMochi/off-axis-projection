"""
calibrate_camera.py
===================
使用棋盤格校正攝影機，計算真實焦距（像素）。
Calibrate camera using a chessboard pattern to find accurate focal length.

使用方式 / How to use:
  1. 準備一張棋盤格（畫面或列印）/ Prepare a chessboard (on screen or printed)
  2. 執行此腳本 / Run this script:
       python calibrate_camera.py
  3. 對著鏡頭在不同角度、距離移動棋盤格，直到收集 20 張畫面
  4. 腳本自動計算焦距並輸出結果
  5. 把結果填入 face_tracker_udp.py 的 FOCAL_LENGTH_PX

棋盤格設定（根據你的棋盤格修改）
Chessboard settings (edit to match your chessboard)
"""

import cv2
import numpy as np
import time

# ── 棋盤格設定 / Chessboard settings ─────────────────────────────────────────
# 棋盤格 "內角點" 數量（格數 - 1）
# Number of INNER corners (squares - 1)
# 例如 9x7 格的棋盤格，內角點是 8x6
# e.g. a 9x7 chessboard has 8x6 inner corners
CHESSBOARD_COLS = 9   # 水平方向內角點數 / horizontal inner corners
CHESSBOARD_ROWS = 6   # 垂直方向內角點數 / vertical inner corners

# 棋盤格每一個小格的真實邊長（mm）
# Real size of each square in mm
SQUARE_SIZE_MM = 25.0   # 如果是在螢幕上顯示，量一量一格幾 mm

# 要拍幾張樣本 / How many sample images to collect
TARGET_SAMPLES = 20

# 攝影機編號 / Camera index
CAM_INDEX = 0
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # 準備棋盤格 3D 世界座標（假設 Z=0）
    # Prepare chessboard 3D object points (in the chessboard plane, Z=0)
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM   # 換算成真實 mm 單位 / Convert to real mm

    obj_points = []   # 3D 世界座標 / 3D world points
    img_points = []   # 2D 影像座標 / 2D image points

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機 / Cannot open camera")
        return

    print(f"[INFO] 開始校正！目標收集 {TARGET_SAMPLES} 張樣本")
    print("[INFO] 請把棋盤格對著鏡頭，在不同角度和距離移動")
    print("[INFO] 偵測到棋盤格時畫面會出現綠色角點，自動拍照")
    print("[INFO] 按 Q 提早結束並計算結果")

    count = 0
    last_capture_time = 0

    while count < TARGET_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            continue

        # 轉灰階偵測棋盤格 / Detect chessboard in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            None
        )

        display = frame.copy()

        if found:
            # 亞像素精確化 / Sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 畫出角點 / Draw corners
            cv2.drawChessboardCorners(display, (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners2, found)

            # 每 1.5 秒自動拍一張，讓你有時間移動棋盤格
            # Auto-capture every 1.5s so you have time to reposition
            now = time.time()
            if now - last_capture_time > 1.5:
                obj_points.append(objp)
                img_points.append(corners2)
                count += 1
                last_capture_time = now
                print(f"  ✅ 已收集 {count}/{TARGET_SAMPLES} 張")
        else:
            cv2.putText(display, "找不到棋盤格 / No chessboard detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 顯示進度 / Show progress
        cv2.putText(display, f"Samples: {count}/{TARGET_SAMPLES}",
                    (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Calibration - 按 Q 結束", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 5:
        print(f"[WARNING] 樣本不足（只有 {len(obj_points)} 張），無法校正")
        print("          請至少收集 5 張以上再試")
        return

    print(f"\n[INFO] 計算校正結果中...（共 {len(obj_points)} 張樣本）")

    h, w = gray.shape
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    # 焦距 / Focal length
    fx = mtx[0, 0]   # 水平焦距（像素）/ Horizontal focal length (px)
    fy = mtx[1, 1]   # 垂直焦距（像素）/ Vertical focal length (px)
    f_avg = (fx + fy) / 2.0

    # 主點（影像中心偏移）/ Principal point (image center offset)
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\n" + "="*50)
    print(f"  📐 解析度 / Resolution : {w} x {h}")
    print(f"  🎯 焦距 fx / Focal length fx : {fx:.1f} px")
    print(f"  🎯 焦距 fy / Focal length fy : {fy:.1f} px")
    print(f"  🎯 平均焦距 / Average       : {f_avg:.1f} px")
    print(f"  📍 主點 cx, cy / Principal  : ({cx:.1f}, {cy:.1f})")
    print(f"  📊 重投影誤差 / Reprojection error : {ret:.4f} px")
    print("="*50)

    print(f"""
✅ 請把以下數值填入 face_tracker_udp.py：

  FOCAL_LENGTH_PX = {f_avg:.1f}

（重投影誤差 < 0.5 表示校正非常準確）
（Reprojection error < 0.5 means excellent calibration）
""")


if __name__ == "__main__":
    main()
