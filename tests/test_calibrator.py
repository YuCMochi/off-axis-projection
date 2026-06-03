import json
import pytest


def test_load_calibration_missing_file(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "calibration.json")
    import tracker
    assert tracker.load_calibration() is None


def test_load_calibration_valid_file(tmp_path, monkeypatch):
    import config
    calib = {
        "camera_matrix": [[612.0, 0, 320.0], [0, 611.0, 240.0], [0, 0, 1]],
        "dist_coeffs": [0.1, -0.2, 0.0, 0.0, 0.05],
        "rms_error": 0.43,
        "image_size": [640, 480],
    }
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(calib), encoding="utf-8")
    monkeypatch.setattr(config, "CALIBRATION_PATH", p)
    import tracker
    result = tracker.load_calibration()
    assert result is not None
    assert result["rms_error"] == pytest.approx(0.43)
    assert result["camera_matrix"][0][0] == pytest.approx(612.0)


def test_load_calibration_malformed_file(tmp_path, monkeypatch):
    import config
    p = tmp_path / "calibration.json"
    p.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(config, "CALIBRATION_PATH", p)
    import tracker
    assert tracker.load_calibration() is None


def test_is_stable_true_when_corners_identical():
    import numpy as np
    from calibrator import Calibrator
    corners = np.random.rand(54, 1, 2).astype(np.float32)
    assert Calibrator._is_stable(corners, corners.copy()) is True


def test_is_stable_false_when_corners_move():
    import numpy as np
    from calibrator import Calibrator
    prev = np.zeros((54, 1, 2), dtype=np.float32)
    curr = prev.copy()
    curr[0, 0, 0] = 5.0  # one corner moved 5px
    assert Calibrator._is_stable(prev, curr) is False


def test_save_calibration_writes_json(tmp_path, monkeypatch):
    import json, numpy as np
    import config
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "calibration.json")
    from calibrator import Calibrator
    cam_mtx = np.array([[612.0, 0, 320.0], [0, 611.0, 240.0], [0, 0, 1.0]])
    dist = np.array([[0.1, -0.2, 0.0, 0.0, 0.05]])
    result = Calibrator._save_calibration(cam_mtx, dist, 0.43, (640, 480))
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["rms_error"] == pytest.approx(0.43)
    assert saved["camera_matrix"][0][0] == pytest.approx(612.0)
    assert len(saved["dist_coeffs"]) == 5
    assert result["image_size"] == [640, 480]
