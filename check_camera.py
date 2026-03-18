"""
check_camera.py
===============
自動讀取攝影機的規格（解析度、FPS），
並根據常見 FOV 估算建議的 FOCAL_LENGTH_PX 值。
Auto-detect camera specs and suggest FOCAL_LENGTH_PX values.

使用方式 / How to use:
  python check_camera.py
"""

import cv2

# ── 設定 / Settings ──────────────────────────────────────────────────────────
CAM_INDEX = 0   # 攝影機編號，通常是 0（內建）或 1（外接）
# ─────────────────────────────────────────────────────────────────────────────

def estimate_focal(width_px: float, fov_deg: float) -> float:
    """
    根據畫面寬度和水平視野角估算焦距（像素）
    Estimate focal length in pixels from image width and horizontal FOV.
    公式 / Formula: f = (W/2) / tan(FOV/2)
    """
    import math
    return (width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟攝影機 #{CAM_INDEX} / Cannot open camera #{CAM_INDEX}")
        return

    # 讀取攝影機回報的規格 / Read camera-reported specs
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # 畫面寬度（px）
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   # 畫面高度（px）
    fps    = cap.get(cv2.CAP_PROP_FPS)            # 幀率（FPS）

    cap.release()

    print("\n" + "="*55)
    print("  📷 攝影機規格 / Camera Specs")
    print("="*55)
    print(f"  解析度 / Resolution : {int(width)} x {int(height)} px")
    print(f"  幀率   / FPS        : {fps:.1f}")
    print("="*55)

    # 根據常見 FOV 估算建議焦距
    # Suggest focal lengths for common webcam FOVs
    print("\n  📐 建議 FOCAL_LENGTH_PX（依不同視野角估算）")
    print("  Suggested FOCAL_LENGTH_PX for common webcam FOVs:\n")

    fov_options = [60, 70, 78, 90]   # 常見 webcam 水平 FOV 範圍
    for fov in fov_options:
        f = estimate_focal(width, fov)
        print(f"    水平FOV {fov:>2}°  →  FOCAL_LENGTH_PX = {f:.1f}")

    print()
    print("  ⚠️  不知道 FOV？請查攝影機型號規格表（通常寫 'HFOV' 或 '視角'）")
    print("  ⚠️  If unsure, check your camera model spec sheet for HFOV.")
    print()
    print("  💡 最準確的方法：執行 calibrate_camera.py 用棋盤格校正")
    print("  💡 Most accurate: run calibrate_camera.py with a chessboard.")
    print()
    print("  👉 把你選定的值填入 face_tracker_udp.py 第 48 行：")
    print("     FOCAL_LENGTH_PX = <你選的值>")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
