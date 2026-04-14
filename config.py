"""config.py — Config dataclass + JSON persistence."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Resolve the directory next to the exe (frozen) or script (dev)
if getattr(sys, "frozen", False):
    _APP_DIR = Path(sys.executable).parent
else:
    _APP_DIR = Path(__file__).parent

CONFIG_PATH = _APP_DIR / "config.json"


@dataclass
class Config:
    # ── Environment Profile ──────────────────────────────────────────────────
    cam_index: int = 0
    focal_length_px: float = 320.0
    max_num_faces: int = 5
    lock_snap_dist_px: int = 150
    cam_offset_x_cm: float = 0.0
    cam_offset_y_cm: float = 16.2
    udp_host: str = "127.0.0.1"
    udp_port: int = 4242
    real_eye_dist_cm: float = 9.0
    # ── Tuning Parameters ────────────────────────────────────────────────────
    smooth_alpha: float = 0.25
    deadzone_rot: float = 0.3
    deadzone_pos: float = 0.15
    yaw_scale: float = 1.0
    pitch_scale: float = 1.0
    roll_scale: float = 1.0
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0


def load_config() -> Config:
    """Load config from CONFIG_PATH; return defaults on any error."""
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            fields = Config.__dataclass_fields__
            filtered = {k: v for k, v in data.items() if k in fields}
            return Config(**filtered)
        except Exception:
            pass
    return Config()


def save_config(cfg: Config) -> None:
    """Persist config to CONFIG_PATH as pretty JSON."""
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
