import json
import pytest
from pathlib import Path


def test_config_defaults(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load_config()
    assert cfg.cam_index == 0
    assert cfg.udp_host == "127.0.0.1"
    assert cfg.udp_port == 4242
    assert cfg.smooth_alpha == 0.25


def test_config_roundtrip(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    original = config.Config(cam_index=2, udp_port=9000, smooth_alpha=0.1, cam_offset_y_cm=20.0)
    config.save_config(original)
    loaded = config.load_config()
    assert loaded.cam_index == 2
    assert loaded.udp_port == 9000
    assert loaded.smooth_alpha == 0.1
    assert loaded.cam_offset_y_cm == 20.0


def test_config_load_invalid_json(tmp_path, monkeypatch):
    import config
    path = tmp_path / "config.json"
    path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    cfg = config.load_config()
    assert cfg.cam_index == 0   # falls back to defaults


def test_config_load_extra_keys_ignored(tmp_path, monkeypatch):
    import config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"cam_index": 1, "unknown_key": "value"}), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    cfg = config.load_config()
    assert cfg.cam_index == 1
