"""
test_tracker.py
===============
face_tracker_udp.py 的自動化測試。
Automated tests for face_tracker_udp.py.

執行方式 / How to run:
  pip install pytest
  pytest test_tracker.py -v

這些測試只測「純計算函式」，不需要攝影機或 MediaPipe。
These tests only cover pure computation — no camera or MediaPipe needed.
"""

import struct
import math
import numpy as np
import pytest

# ── Import the functions we want to test ─────────────────────────────
# 中文: 從主程式匯入要測試的函式
from face_tracker_udp import (
    pack_opentrack,
    get_cam_matrix,
    estimate_position_from_eyes,
    rot_to_euler,
    sample_2d,
    solve_pose,
    KalmanFilter1D,
    select_face,
    FACE_MODEL_3D,
    FACE_MODEL_IDX,
    FOCAL_LENGTH_PX,
    REAL_EYE_DIST_CM,
    LM_LEFT_EYE,
    LM_RIGHT_EYE,
    LOCK_SNAP_DIST_PX,
    MIN_EYE_DIST_PX,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 建立假的 Landmark 物件（不需要 MediaPipe）
#  Helper: Create fake landmark objects (no MediaPipe needed)
# ═══════════════════════════════════════════════════════════════════════
class FakeLandmark:
    """模擬 MediaPipe 的 NormalizedLandmark / Mimics MediaPipe NormalizedLandmark."""
    def __init__(self, x, y, z=0.0):
        # x, y, z are normalized [0, 1] relative to image dimensions
        # 中文: x, y, z 是相對於畫面大小的 0~1 正規化座標
        self.x = x
        self.y = y
        self.z = z


class FakeFace:
    """模擬 MediaPipe 的 face_landmarks 回傳結構。"""
    def __init__(self, landmark_list):
        self.landmark = landmark_list


def make_centered_face(w=640, h=480, eye_dist_norm=0.1):
    """
    建立一張「正對鏡頭、在畫面正中央」的假臉，共 478 個 landmark。
    Create a fake face centered in the frame with 478 landmarks.

    Args:
        w, h: 畫面寬高（只用來設計合理的 normalized 座標）
        eye_dist_norm: 兩眼外角的 normalized x 距離
    """
    # 478 個 landmark，全部先放在畫面中央
    # 478 landmarks, all initially at image center
    landmarks = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(478)]

    # 設定關鍵的幾個 landmark 的位置
    # Set positions for key landmarks
    half_eye = eye_dist_norm / 2.0

    # 左眼外角 (index 33): 在中央偏左
    # Left eye outer (index 33): left of center
    landmarks[33] = FakeLandmark(0.5 - half_eye, 0.48, 0.0)

    # 右眼外角 (index 263): 在中央偏右
    # Right eye outer (index 263): right of center
    landmarks[263] = FakeLandmark(0.5 + half_eye, 0.48, 0.0)

    # 鼻尖 (index 4): 正中央偏下
    # Nose tip (index 4): center, slightly below eyes
    landmarks[4] = FakeLandmark(0.5, 0.55, 0.0)

    # 下巴 (index 152): 中央下方
    # Chin (index 152): below center
    landmarks[152] = FakeLandmark(0.5, 0.7, 0.0)

    # 左嘴角 (index 61) / 右嘴角 (index 291)
    landmarks[61] = FakeLandmark(0.5 - 0.03, 0.62, 0.0)
    landmarks[291] = FakeLandmark(0.5 + 0.03, 0.62, 0.0)

    # 鼻樑中點 (index 168)
    landmarks[168] = FakeLandmark(0.5, 0.50, 0.0)

    # 額頭 (index 10)
    landmarks[10] = FakeLandmark(0.5, 0.38, 0.0)

    # 左眼內角 (133) / 右眼內角 (362)
    landmarks[133] = FakeLandmark(0.5 - half_eye * 0.5, 0.48, 0.0)
    landmarks[362] = FakeLandmark(0.5 + half_eye * 0.5, 0.48, 0.0)

    # 左眉外側 (70) / 右眉外側 (300)
    landmarks[70] = FakeLandmark(0.5 - half_eye * 1.1, 0.42, 0.0)
    landmarks[300] = FakeLandmark(0.5 + half_eye * 1.1, 0.42, 0.0)

    # 左顎 (127) / 右顎 (356)
    landmarks[127] = FakeLandmark(0.5 - half_eye * 1.3, 0.65, 0.0)
    landmarks[356] = FakeLandmark(0.5 + half_eye * 1.3, 0.65, 0.0)

    # 鼻底 (index 2)
    landmarks[2] = FakeLandmark(0.5, 0.58, 0.0)

    return landmarks


# ═══════════════════════════════════════════════════════════════════════
#  TEST: pack_opentrack — OpenTrack UDP 封包格式
# ═══════════════════════════════════════════════════════════════════════
class TestPackOpentrack:
    """測試 OpenTrack UDP 封包的打包功能。"""

    def test_packet_size_is_48_bytes(self):
        """封包長度應該剛好是 48 bytes（6 × 8 bytes double）"""
        # English: 6 doubles × 8 bytes each = 48 bytes
        # 中文: 6 個 double × 每個 8 bytes = 48 bytes
        packet = pack_opentrack(1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
        assert len(packet) == 48

    def test_values_are_little_endian_doubles(self):
        """封包內的值應該能正確還原（little-endian double）"""
        # English: Pack and unpack should give back the same values
        # 中文: 封包再解包後，值應該完全一樣
        x, y, z, yaw, pitch, roll = 1.5, -2.3, 45.0, 10.0, -5.0, 3.0
        packet = pack_opentrack(x, y, z, yaw, pitch, roll)
        unpacked = struct.unpack('<6d', packet)
        assert unpacked == pytest.approx((x, y, z, yaw, pitch, roll))

    def test_zero_values(self):
        """全零值應該正常打包"""
        packet = pack_opentrack(0, 0, 0, 0, 0, 0)
        unpacked = struct.unpack('<6d', packet)
        assert all(v == 0.0 for v in unpacked)


# ═══════════════════════════════════════════════════════════════════════
#  TEST: get_cam_matrix — 攝影機內參矩陣
# ═══════════════════════════════════════════════════════════════════════
class TestCamMatrix:
    """測試攝影機內參矩陣的格式和值。"""

    def test_shape_is_3x3(self):
        """矩陣大小應該是 3×3"""
        mtx = get_cam_matrix(640, 480)
        assert mtx.shape == (3, 3)

    def test_focal_length_on_diagonal(self):
        """對角線上的焦距應等於 FOCAL_LENGTH_PX"""
        # English: fx and fy should both equal the configured focal length
        # 中文: fx 和 fy 應該都等於設定的焦距
        mtx = get_cam_matrix(640, 480)
        assert mtx[0, 0] == FOCAL_LENGTH_PX  # fx
        assert mtx[1, 1] == FOCAL_LENGTH_PX  # fy

    def test_principal_point_is_center(self):
        """主點（光軸中心）應該在畫面正中央"""
        # English: cx, cy should be at image center
        # 中文: cx, cy 應該在影像中心
        w, h = 640, 480
        mtx = get_cam_matrix(w, h)
        assert mtx[0, 2] == w / 2  # cx
        assert mtx[1, 2] == h / 2  # cy

    def test_different_resolutions(self):
        """不同解析度應該有不同的主點"""
        mtx1 = get_cam_matrix(640, 480)
        mtx2 = get_cam_matrix(1280, 720)
        # cx should be different
        assert mtx1[0, 2] != mtx2[0, 2]


# ═══════════════════════════════════════════════════════════════════════
#  TEST: rot_to_euler — 旋轉矩陣 → 歐拉角
# ═══════════════════════════════════════════════════════════════════════
class TestRotToEuler:
    """測試旋轉矩陣到歐拉角的轉換。"""

    def test_identity_gives_zero(self):
        """單位矩陣 → 歐拉角全部為 0"""
        # English: Identity matrix should give (0, 0, 0)
        # 中文: 單位矩陣應該回傳全零歐拉角
        R = np.eye(3)
        x, y, z = rot_to_euler(R)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z) < 1e-6

    def test_90_degree_yaw(self):
        """繞 Y 軸旋轉 90° → y ≈ π/2"""
        # English: 90° rotation around Y axis
        # 中文: 繞 Y 軸旋轉 90°
        angle = math.pi / 2
        R = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])
        x, y, z = rot_to_euler(R)
        assert abs(y - angle) < 1e-4

    def test_small_pitch(self):
        """繞 X 軸小角度旋轉 → x ≈ 該角度"""
        # English: Small rotation around X axis
        # 中文: 繞 X 軸旋轉小角度
        angle = 0.3  # ~17 degrees
        R = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle),  math.cos(angle)]
        ])
        x, y, z = rot_to_euler(R)
        assert abs(x - angle) < 1e-4

    def test_output_is_tuple_of_three(self):
        """回傳值應該是 3 個數字（x, y, z）"""
        result = rot_to_euler(np.eye(3))
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════
#  TEST: estimate_position_from_eyes — 眼距三角測量
# ═══════════════════════════════════════════════════════════════════════
class TestEstimatePosition:
    """測試用眼距三角測量法計算頭部 3D 位置。"""

    def test_centered_face_x_near_zero(self):
        """臉在正中央 → X 應接近 CAM_OFFSET（通常 0）"""
        # English: Face at center → X ≈ CAM_OFFSET_X_CM
        # 中文: 臉在畫面正中央 → X 應接近相機偏移量
        landmarks = make_centered_face()
        pos = estimate_position_from_eyes(landmarks, 640, 480)
        assert pos is not None
        x, y, z = pos
        # x should be near 0 (CAM_OFFSET_X_CM = 0 by default)
        assert abs(x) < 2.0  # within 2cm of center

    def test_z_is_positive(self):
        """Z（距離）應該是正數"""
        # English: Z (distance from camera) should be positive
        # 中文: 距離應該大於 0
        landmarks = make_centered_face()
        pos = estimate_position_from_eyes(landmarks, 640, 480)
        assert pos is not None
        _, _, z = pos
        assert z > 0

    def test_closer_face_has_smaller_z(self):
        """眼距大（臉近）→ Z 小；眼距小（臉遠）→ Z 大"""
        # English: Larger eye distance in pixels = closer = smaller Z
        # 中文: 眼距越大（像素）= 臉越近 = Z 越小
        close_face = make_centered_face(eye_dist_norm=0.15)  # 大眼距
        far_face = make_centered_face(eye_dist_norm=0.05)     # 小眼距

        pos_close = estimate_position_from_eyes(close_face, 640, 480)
        pos_far = estimate_position_from_eyes(far_face, 640, 480)

        assert pos_close is not None and pos_far is not None
        assert pos_close[2] < pos_far[2]  # 近的 Z 應該更小

    def test_face_on_right_has_positive_x(self):
        """臉偏畫面右邊 → X 應為正值"""
        # English: Face on the right side → positive X
        # 中文: 臉在畫面右側 → X 為正
        landmarks = make_centered_face()
        # 把全部 landmark 往右移（增加 x）
        for lm in landmarks:
            lm.x += 0.15
        pos = estimate_position_from_eyes(landmarks, 640, 480)
        assert pos is not None
        assert pos[0] > 0

    def test_invalid_eye_distance_returns_none(self):
        """兩眼距離太近（基本重疊）→ 應回傳 None"""
        # English: Eyes too close together → invalid, return None
        # 中文: 眼距 < MIN_EYE_DIST_PX → 回傳 None（無效）
        landmarks = make_centered_face(eye_dist_norm=0.0001)  # 幾乎重疊
        pos = estimate_position_from_eyes(landmarks, 640, 480)
        # 視眼距計算後是否 < MIN_EYE_DIST_PX，可能回傳 None
        # 如果 0.0001 * 640 = 0.064px < 1.0px → None
        assert pos is None


# ═══════════════════════════════════════════════════════════════════════
#  TEST: KalmanFilter1D — 卡爾曼濾波器
# ═══════════════════════════════════════════════════════════════════════
class TestKalmanFilter:
    """測試 1D 自適應卡爾曼濾波器。"""

    def test_converges_to_constant(self):
        """連續輸入同一個值 → 濾波器最終應收斂到該值"""
        # English: Input constant value → filter should converge to it
        # 中文: 連續輸入同一個常數 → 最終輸出應該等於該常數
        kf = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
        result = 0
        for _ in range(200):
            result = kf.update(42.0)
        assert abs(result - 42.0) < 0.5

    def test_first_output_equals_input(self):
        """第一次呼叫 → 回傳應等於輸入值（直接初始化）"""
        # English: First call should return the input value directly
        # 中文: 第一次呼叫應直接回傳輸入值
        kf = KalmanFilter1D()
        result = kf.update(10.0)
        assert result == 10.0

    def test_smooth_noisy_signal(self):
        """有雜訊的信號 → 濾波器的輸出標準差應比輸入小"""
        # English: Noisy input → output variance should be smaller
        # 中文: 有噪聲的輸入 → 輸出的標準差應該更小
        kf = KalmanFilter1D(process_noise=0.01, measurement_noise=0.5)
        np.random.seed(42)
        true_value = 5.0
        noisy_inputs = true_value + np.random.randn(200) * 1.0  # noise σ=1.0
        outputs = []
        for val in noisy_inputs:
            outputs.append(kf.update(float(val)))

        # 取最後 100 筆（已收斂）
        output_std = np.std(outputs[-100:])
        input_std = np.std(noisy_inputs[-100:])
        assert output_std < input_std  # 濾波後應該更平滑

    def test_reset_clears_state(self):
        """reset() 後 → 下一次輸入應直接回傳該值"""
        # English: After reset, next update should return the input value
        # 中文: 重置後，下一次 update 應直接回傳輸入值
        kf = KalmanFilter1D()
        kf.update(100.0)
        kf.update(100.0)
        kf.reset()
        result = kf.update(0.0)
        assert result == 0.0

    def test_responds_to_step_change(self):
        """值突然跳變 → 濾波器應在若干幀內跟上"""
        # English: After a step change, filter should catch up within frames
        # 中文: 值突然改變後，濾波器應在幾幀內跟上
        kf = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
        # 先穩定在 0
        for _ in range(50):
            kf.update(0.0)
        # 突變到 10
        result = 0
        for _ in range(50):
            result = kf.update(10.0)
        assert abs(result - 10.0) < 1.0  # 50 幀內應該夠接近

    def test_adaptive_still_vs_moving(self):
        """靜止時 vs 移動時 → 靜止應更穩定（輸出變化更小）"""
        # English: Still → less variation in output; Moving → more responsive
        # 中文: 靜止時輸出應該更穩定
        kf = KalmanFilter1D(
            process_noise=0.01, measurement_noise=0.1,
            vel_threshold=1.0, adaptive_multiplier=8.0
        )
        # 先穩定在 5.0（靜止）
        for _ in range(100):
            kf.update(5.0)
        # 加入小幅雜訊（靜止狀態）
        still_outputs = []
        for _ in range(50):
            still_outputs.append(kf.update(5.0 + np.random.randn() * 0.1))
        still_var = np.var(still_outputs)
        assert still_var < 0.1  # 靜止時輸出變化應該很小


# ═══════════════════════════════════════════════════════════════════════
#  TEST: sample_2d — 從 landmark 取 2D 座標
# ═══════════════════════════════════════════════════════════════════════
class TestSample2D:
    """測試從 landmark 列表取出 2D 像素坐標。"""

    def test_output_shape(self):
        """輸出形狀應該是 (14, 2) — 14 個點的 x, y"""
        landmarks = make_centered_face()
        pts = sample_2d(landmarks, 640, 480)
        assert pts.shape == (len(FACE_MODEL_IDX), 2)

    def test_values_in_pixel_range(self):
        """所有座標應該在 [0, w] 和 [0, h] 範圍內"""
        # English: All coordinates should be within image bounds
        # 中文: 所有座標應在畫面範圍內
        w, h = 640, 480
        landmarks = make_centered_face()
        pts = sample_2d(landmarks, w, h)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h)


# ═══════════════════════════════════════════════════════════════════════
#  TEST: solve_pose — solvePnP 封裝
# ═══════════════════════════════════════════════════════════════════════
class TestSolvePose:
    """測試 solvePnP 的基本回傳格式。"""

    def test_returns_tuple_or_none(self):
        """應回傳 (rvec, tvec) 或 None"""
        landmarks = make_centered_face()
        pts = sample_2d(landmarks, 640, 480)
        result = solve_pose(pts, 640, 480)
        if result is not None:
            rvec, tvec = result
            assert rvec.shape == (3, 1)
            assert tvec.shape == (3, 1)

    def test_with_previous_guess(self):
        """有上一帧的 rvec/tvec 時，用 ITERATIVE refine 不應爆炸"""
        landmarks = make_centered_face()
        pts = sample_2d(landmarks, 640, 480)
        # 第一次：無初始猜測
        result1 = solve_pose(pts, 640, 480)
        if result1 is not None:
            rvec, tvec = result1
            # 第二次：用上次結果當初始猜測
            result2 = solve_pose(pts, 640, 480,
                                 prev_rvec=rvec, prev_tvec=tvec)
            assert result2 is not None


# ═══════════════════════════════════════════════════════════════════════
#  TEST: select_face — 臉部選擇邏輯
# ═══════════════════════════════════════════════════════════════════════
class TestSelectFace:
    """測試多臉場景下的臉部選擇邏輯。"""

    def test_single_face(self):
        """只有一張臉 → 應該選到它"""
        face = FakeFace(make_centered_face())
        lm, eye_mid = select_face([face], 640, 480, None)
        assert lm is not None
        assert eye_mid is not None

    def test_locks_to_nearest_face(self):
        """有鎖定位置 → 選最近的臉"""
        # English: With a lock, should pick the closest face
        # 中文: 有鎖定位置時，應選距離最近的那張臉
        face1_lm = make_centered_face()  # 在中央
        face2_lm = make_centered_face()
        # 把 face2 移到右邊
        for lm in face2_lm:
            lm.x += 0.3

        face1 = FakeFace(face1_lm)
        face2 = FakeFace(face2_lm)

        # 上一帧鎖定在中央
        prev_mid = (320.0, 240.0)
        lm, _ = select_face([face1, face2], 640, 480, prev_mid)
        # 應該選到 face1（更近中央）
        assert lm is face1.landmark

    def test_picks_largest_when_no_lock(self):
        """無鎖定 → 選眼距最大（最近鏡頭）的臉"""
        # English: No lock → pick face with largest eye distance
        # 中文: 無鎖定時，選眼距最大的臉
        small_face_lm = make_centered_face(eye_dist_norm=0.05)  # 遠
        big_face_lm = make_centered_face(eye_dist_norm=0.15)    # 近
        # 把大臉移到旁邊避免重疊
        for lm in big_face_lm:
            lm.x += 0.2

        small_face = FakeFace(small_face_lm)
        big_face = FakeFace(big_face_lm)

        lm, _ = select_face([small_face, big_face], 640, 480, None)
        assert lm is big_face.landmark

    def test_lock_lost_when_too_far(self):
        """鎖定目標離開 → 鎖定丟失，退回選最顯眼"""
        # English: If locked face disappears, fall back to largest
        # 中文: 鎖定的臉離開畫面 → 退回選最顯眼的
        face_lm = make_centered_face()
        face = FakeFace(face_lm)

        # 上一帧鎖定在畫面極右邊（離現在的臉很遠）
        far_away_mid = (639.0, 10.0)
        lm, new_mid = select_face([face], 640, 480, far_away_mid)
        # 應該還是選到這張臉（因為只有一張）
        assert lm is not None


# ═══════════════════════════════════════════════════════════════════════
#  TEST: FACE_MODEL_3D — 3D 模型參考點
# ═══════════════════════════════════════════════════════════════════════
class TestFaceModel:
    """測試 3D 人臉模型常數的基本合理性。"""

    def test_model_has_14_points(self):
        """3D 模型應有 14 個點"""
        assert FACE_MODEL_3D.shape == (14, 3)

    def test_index_list_has_14_entries(self):
        """索引列表也應有 14 個"""
        assert len(FACE_MODEL_IDX) == 14

    def test_nose_tip_is_origin(self):
        """鼻尖（第一個點）應該在原點 (0, 0, 0)"""
        assert np.allclose(FACE_MODEL_3D[0], [0, 0, 0])

    def test_face_is_roughly_symmetric(self):
        """左右眼的 X 座標應該大致對稱"""
        # English: Left/right eye X should be roughly symmetric
        # 中文: 左右眼的 X 座標應有正負對稱
        left_eye = FACE_MODEL_3D[2]   # index 33
        right_eye = FACE_MODEL_3D[3]  # index 263
        assert abs(left_eye[0] + right_eye[0]) < 0.1  # X 座標相加接近 0
        assert abs(left_eye[1] - right_eye[1]) < 0.1  # Y 座標相同
