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


def test_config_has_cam_hfov_deg():
    from config import Config
    cfg = Config()
    assert cfg.cam_hfov_deg == pytest.approx(70.0)
    assert not hasattr(cfg, "focal_length_px")
    assert not hasattr(cfg, "real_eye_dist_cm")


def test_hfov_to_focal_90deg():
    import math
    # 90° HFOV on 640-wide image → focal = 320.0 (tan 45° = 1.0)
    focal = (640 / 2.0) / math.tan(math.radians(90.0 / 2.0))
    assert focal == pytest.approx(320.0, rel=1e-6)
