import json


def test_config_defaults(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load_config()
    assert cfg.cam_index == 0
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 40000
    assert cfg.output_protocol == "freed"
    assert cfg.smooth_alpha == 0.25


def test_config_default_constants():
    from config import DEFAULT_HOST, DEFAULT_PORTS
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORTS["freed"] == 40000
    assert DEFAULT_PORTS["opentrack"] == 4242


def test_config_roundtrip(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    original = config.Config(cam_index=2, port=9000, smooth_alpha=0.1, cam_offset_y_cm=20.0)
    config.save_config(original)
    loaded = config.load_config()
    assert loaded.cam_index == 2
    assert loaded.port == 9000
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


def test_config_old_udp_keys_ignored(tmp_path, monkeypatch):
    """Old configs with udp_host/udp_port/freed_host/freed_port are silently ignored."""
    import config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "udp_host": "192.168.1.1", "udp_port": 4242,
        "freed_host": "10.0.0.1", "freed_port": 40000,
        "cam_index": 3,
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    cfg = config.load_config()
    assert cfg.cam_index == 3
    assert cfg.host == "127.0.0.1"   # old keys ignored, falls back to default
    assert cfg.port == 40000


def test_config_invalid_protocol_falls_back_to_freed(tmp_path, monkeypatch):
    """Unknown output_protocol in config.json is replaced with 'freed' instead of crashing."""
    import config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_protocol": "livelink"}), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    cfg = config.load_config()
    assert cfg.output_protocol == "freed"


def test_config_valid_protocols_accepted(tmp_path, monkeypatch):
    """Valid output_protocol values are loaded without modification."""
    import config
    for proto in ("freed", "opentrack"):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"output_protocol": proto}), encoding="utf-8")
        monkeypatch.setattr(config, "CONFIG_PATH", path)
        cfg = config.load_config()
        assert cfg.output_protocol == proto
