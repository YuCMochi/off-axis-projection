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
