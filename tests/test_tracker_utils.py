import struct
import numpy as np
import pytest


def test_pack_opentrack_format():
    from sender import pack_opentrack
    data = pack_opentrack(1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
    assert len(data) == 48
    unpacked = struct.unpack("<6d", data)
    assert unpacked == (1.0, 2.0, 3.0, 10.0, 20.0, 30.0)


def test_smooth_filter_initial_value():
    from tracker import SmoothFilter
    f = SmoothFilter(alpha=0.5, deadzone=0.0)
    result = f.update(10.0)
    assert result == 10.0


def test_smooth_filter_ema():
    from tracker import SmoothFilter
    f = SmoothFilter(alpha=0.5, deadzone=0.0)
    f.update(0.0)
    result = f.update(10.0)
    assert result == pytest.approx(5.0)


def test_smooth_filter_deadzone_ignores_small_change():
    from tracker import SmoothFilter
    f = SmoothFilter(alpha=0.5, deadzone=1.0)
    f.update(5.0)
    result = f.update(5.5)   # change = 0.5 < deadzone 1.0
    assert result == 5.0


def test_smooth_filter_deadzone_passes_large_change():
    from tracker import SmoothFilter
    f = SmoothFilter(alpha=1.0, deadzone=1.0)
    f.update(5.0)
    result = f.update(8.0)   # change = 3.0 > deadzone 1.0
    assert result == 8.0


def test_get_cam_matrix_shape():
    from tracker import get_cam_matrix
    m = get_cam_matrix(w=640, h=480, focal_px=320.0)
    assert m.shape == (3, 3)
    assert m[0, 0] == 320.0   # fx
    assert m[1, 1] == 320.0   # fy
    assert m[0, 2] == 320.0   # cx = w/2
    assert m[1, 2] == 240.0   # cy = h/2


def test_rot_to_euler_identity():
    from tracker import rot_to_euler
    R = np.eye(3)
    x, y, z = rot_to_euler(R)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_solve_pose_accepts_cam_and_dist_params():
    from tracker import _solve_pose_with_cam, get_cam_matrix
    import numpy as np
    # Use a dummy set of 6 2D points that roughly correspond to the 3D model
    # viewed head-on; just verify the function signature accepts cam+dist.
    cam = get_cam_matrix(640, 480, 500.0)
    dist = np.zeros((4, 1))
    image_pts = np.array([
        [320.0, 240.0],
        [320.0, 300.0],
        [280.0, 220.0],
        [360.0, 220.0],
        [295.0, 245.0],
        [345.0, 245.0],
    ], dtype=np.float64)
    # Should not raise; ok may be False for degenerate input, that's fine
    result = _solve_pose_with_cam(image_pts, cam, dist, None, None)
    assert result is None or (len(result) == 2)


def test_estimate_position_uses_focal_px_param():
    from tracker import _estimate_position_with_focal
    from config import Config
    import pytest

    class _FakeLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    cfg = Config(real_eye_dist_cm=6.5, cam_offset_x_cm=0.0, cam_offset_y_cm=0.0)
    w, h = 640, 480
    # Eyes 100px apart, centred on frame
    landmarks = [None] * 478
    landmarks[33]  = _FakeLM(0.422, 0.5)   # left eye  ~ x=270
    landmarks[263] = _FakeLM(0.578, 0.5)   # right eye ~ x=370  → dist≈100px

    result = _estimate_position_with_focal(landmarks, w, h, cfg, focal_px=500.0)
    assert result is not None
    z_expected = (6.5 * 500.0) / 100.0   # = 32.5 cm
    assert result[2] == pytest.approx(z_expected, rel=0.05)
