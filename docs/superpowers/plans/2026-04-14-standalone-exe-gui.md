# Standalone EXE + GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the face tracker as a folder-distributed standalone exe with a tkinter GUI for adjusting all parameters, persisted in config.json.

**Architecture:** A new `app.py` entry point creates a small tkinter control panel (MainWindow). The MainWindow starts/stops a `FaceTracker` (from `tracker.py`) that runs in a daemon thread and owns the cv2 preview window. A separate `SettingsWindow` Toplevel (opened by a button) exposes all parameters in two ttk.Notebook tabs, saving to `config.json` via `config.py`. PyInstaller --onedir bundles everything.

**Tech Stack:** Python 3.11, tkinter (ttk), cv2, mediapipe 0.10.9, PyInstaller 6.x, dataclasses, threading, json

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `config.py` | CREATE | Config dataclass + JSON load/save + APP_DIR resolution |
| `tracker.py` | CREATE | FaceTracker class: threading, tracking loop, SmoothFilter, all math |
| `settings_window.py` | CREATE | SettingsWindow Toplevel: two-tab param editor |
| `app.py` | CREATE | Entry point + MainWindow: status, HUD, start/stop, settings button |
| `off_axis_tracker.spec` | CREATE | PyInstaller onedir spec |
| `build.bat` | CREATE | Build helper script |
| `tests/test_config.py` | CREATE | Config unit tests |
| `tests/test_tracker_utils.py` | CREATE | SmoothFilter + math util tests |
| `face_tracker_udp.py` | KEEP | Unchanged — still works as standalone CLI |
| `requirements.txt` | MODIFY | Add pyinstaller |

---

## Task 1: Config Module

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1.1: Write failing tests for Config**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 1.2: Run tests — expect ImportError (module not yet created)**

```
cd "c:/Users/0/Desktop/my projects/off-axis-projection"
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 1.3: Create `config.py`**

```python
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
```

- [ ] **Step 1.4: Run tests — expect all pass**

```
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 4 passed

- [ ] **Step 1.5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add Config dataclass with JSON persistence"
```

---

## Task 2: Tracker Core

**Files:**
- Create: `tracker.py`
- Create: `tests/test_tracker_utils.py`

The tracker extracts all logic from `face_tracker_udp.py` into a `FaceTracker` class that runs in a daemon thread. The cv2 preview window is owned by the tracker thread. The main thread reads `tracker.live` (a plain dict) to display HUD values.

- [ ] **Step 2.1: Write failing tests for tracker utilities**

Create `tests/test_tracker_utils.py`:

```python
import math
import struct
import numpy as np
import pytest


def test_pack_opentrack_format():
    from tracker import pack_opentrack
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
```

- [ ] **Step 2.2: Run tests — expect ImportError**

```
.venv/Scripts/python -m pytest tests/test_tracker_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracker'`

- [ ] **Step 2.3: Create `tracker.py`**

```python
"""tracker.py — FaceTracker: runs mediapipe face tracking in a daemon thread."""
from __future__ import annotations

import math
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import Config

# ── Face mesh landmark indices ──────────────────────────────────────────────
LM_NOSE_TIP    = 4
LM_CHIN        = 152
LM_LEFT_EYE    = 33
LM_RIGHT_EYE   = 263
LM_LEFT_MOUTH  = 61
LM_RIGHT_MOUTH = 291

FACE_MODEL_3D = np.array([
    [ 0.000,  0.000,  0.000],
    [ 0.000, -3.300, -1.300],
    [-4.500,  2.500, -4.000],
    [ 4.500,  2.500, -4.000],
    [-2.000,  0.000, -2.200],
    [ 2.000,  0.000, -2.200],
], dtype=np.float64)
FACE_MODEL_IDX = [LM_NOSE_TIP, LM_CHIN, LM_LEFT_EYE, LM_RIGHT_EYE, LM_LEFT_MOUTH, LM_RIGHT_MOUTH]

# ── Algorithm constants ──────────────────────────────────────────────────────
PITCH_GIMBAL_THRESHOLD = 90.0
MIN_EYE_DIST_PX        = 1.0
ROT_SINGULARITY_EPS    = 1e-6
MEDIAPIPE_MIN_CONF     = 0.5
AXIS_DISPLAY_LEN_CM    = 5
DEBUG_PRINT_INTERVAL   = 30


# ── Utility functions (module-level so tests can import them) ────────────────

def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    return struct.pack("<6d", x, y, z, yaw, pitch, roll)


def get_cam_matrix(w: int, h: int, focal_px: float) -> np.ndarray:
    return np.array([[focal_px, 0, w / 2],
                     [0, focal_px, h / 2],
                     [0, 0, 1]], dtype=np.float64)


def rot_to_euler(R: np.ndarray):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > ROT_SINGULARITY_EPS:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return x, y, z


def sample_2d(landmarks, w: int, h: int) -> np.ndarray:
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_MODEL_IDX],
        dtype=np.float64,
    )


# ── SmoothFilter ─────────────────────────────────────────────────────────────

class SmoothFilter:
    def __init__(self, alpha: float, deadzone: float = 0.0):
        self.alpha = alpha
        self.deadzone = deadzone
        self.value: Optional[float] = None

    def update(self, new_val: float) -> float:
        if self.value is None:
            self.value = new_val
            return self.value
        if abs(new_val - self.value) < self.deadzone:
            return self.value
        self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ── FaceTracker ──────────────────────────────────────────────────────────────

class FaceTracker:
    """Runs mediapipe face tracking in a daemon thread.

    Usage:
        tracker = FaceTracker(cfg)
        tracker.start(preview=True)
        # read tracker.live for latest values
        tracker.stop()
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Shared live values dict; written by tracker thread, read by UI thread.
        self.live: dict = {
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "x": 0.0, "y": 0.0, "z": 0.0,
            "tracking": False, "error": None,
        }
        self._preview = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_config(self, cfg: Config) -> None:
        with self._lock:
            self._cfg = cfg

    def start(self, preview: bool = True) -> None:
        if self.running:
            return
        self._preview = preview
        self._stop_event.clear()
        self.live["error"] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None
        self.live["tracking"] = False

    def _cfg_snapshot(self) -> Config:
        with self._lock:
            return self._cfg

    def _run(self) -> None:
        cfg = self._cfg_snapshot()
        try:
            self._tracking_loop(cfg)
        except Exception as e:
            self.live["error"] = str(e)
            self.live["tracking"] = False
        finally:
            cv2.destroyAllWindows()

    def _tracking_loop(self, initial_cfg: Config) -> None:
        cfg = initial_cfg
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cap = cv2.VideoCapture(cfg.cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{cfg.cam_index}")

        mp_mesh = mp.solutions.face_mesh
        face_mesh = mp_mesh.FaceMesh(
            max_num_faces=cfg.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=MEDIAPIPE_MIN_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_CONF,
        )

        f_yaw   = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_rot)
        f_pitch = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_rot)
        f_roll  = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_rot)
        f_x     = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_pos)
        f_y     = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_pos)
        f_z     = SmoothFilter(cfg.smooth_alpha, cfg.deadzone_pos)

        prev_rvec = None
        prev_tvec = None
        locked_eye_mid = None
        cam_mtx = None
        dist_cfs = np.zeros((4, 1))
        frame_count = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                if cam_mtx is None:
                    cam_mtx = get_cam_matrix(w, h, cfg.focal_length_px)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm, locked_eye_mid = self._select_face(
                        results.multi_face_landmarks, w, h, locked_eye_mid, cfg
                    )
                    img_pts = sample_2d(lm, w, h)
                    pnp = self._solve_pose(img_pts, w, h, cfg, prev_rvec, prev_tvec)

                    if pnp:
                        rvec, tvec = pnp
                        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

                        R, _ = cv2.Rodrigues(rvec)
                        ex, ey, ez = rot_to_euler(R)

                        raw_yaw   = math.degrees(ey)
                        raw_pitch = math.degrees(ex)
                        raw_roll  = math.degrees(ez)

                        if raw_pitch > PITCH_GIMBAL_THRESHOLD:
                            raw_pitch -= 180
                        elif raw_pitch < -PITCH_GIMBAL_THRESHOLD:
                            raw_pitch += 180

                        yaw   =  raw_yaw   * cfg.yaw_scale
                        pitch = -raw_pitch * cfg.pitch_scale
                        roll  =  raw_roll  * cfg.roll_scale

                        pos = self._estimate_position(lm, w, h, cfg)
                        if pos:
                            tx, ty, tz = pos[0] * cfg.x_scale, pos[1] * cfg.y_scale, pos[2] * cfg.z_scale
                        else:
                            tx, ty, tz = 0.0, 0.0, 0.0

                        yaw   = f_yaw.update(yaw)
                        pitch = f_pitch.update(pitch)
                        roll  = f_roll.update(roll)
                        tx    = f_x.update(tx)
                        ty    = f_y.update(ty)
                        tz    = f_z.update(tz)

                        packet = pack_opentrack(tx, ty, tz, yaw, pitch, roll)
                        sock.sendto(packet, (cfg.udp_host, cfg.udp_port))

                        self.live.update({
                            "yaw": yaw, "pitch": pitch, "roll": roll,
                            "x": tx, "y": ty, "z": tz, "tracking": True,
                        })

                        if self._preview:
                            self._draw_preview(frame, rvec, tvec, cam_mtx, dist_cfs,
                                               yaw, pitch, roll, tx, ty, tz)

                        frame_count += 1
                        if frame_count % DEBUG_PRINT_INTERVAL == 0:
                            print(f"  Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+5.1f}"
                                  f"  X:{tx:+5.1f}  Y:{ty:+5.1f}  Z:{tz:+5.1f}")
                else:
                    prev_rvec = None
                    prev_tvec = None
                    locked_eye_mid = None
                    self.live["tracking"] = False
                    if self._preview:
                        cv2.putText(frame, "No face / 未偵測到臉部",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self._preview:
                    cv2.imshow("Face Tracker Preview  (Q/ESC to close preview)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        self._preview = False
                        cv2.destroyAllWindows()
        finally:
            cap.release()
            sock.close()
            face_mesh.close()

    @staticmethod
    def _select_face(all_faces, w, h, prev_eye_mid, cfg: Config):
        def eye_mid(lm):
            ex = (lm[LM_LEFT_EYE].x + lm[LM_RIGHT_EYE].x) / 2.0 * w
            ey = (lm[LM_LEFT_EYE].y + lm[LM_RIGHT_EYE].y) / 2.0 * h
            return ex, ey

        def eye_dist(lm):
            lx = lm[LM_LEFT_EYE].x * w;  ly = lm[LM_LEFT_EYE].y * h
            rx = lm[LM_RIGHT_EYE].x * w; ry = lm[LM_RIGHT_EYE].y * h
            return math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

        candidates = [f.landmark for f in all_faces]

        if prev_eye_mid is not None:
            px, py = prev_eye_mid
            best, best_dist = None, float("inf")
            for lm in candidates:
                ex, ey = eye_mid(lm)
                d = math.sqrt((ex - px) ** 2 + (ey - py) ** 2)
                if d < best_dist:
                    best_dist, best = d, lm
            if best_dist < cfg.lock_snap_dist_px:
                return best, eye_mid(best)

        best = max(candidates, key=eye_dist)
        return best, eye_mid(best)

    @staticmethod
    def _solve_pose(image_pts, w, h, cfg: Config, prev_rvec, prev_tvec):
        cam = get_cam_matrix(w, h, cfg.focal_length_px)
        dist = np.zeros((4, 1))
        if prev_rvec is not None and prev_tvec is not None:
            ok, rv, tv = cv2.solvePnP(
                FACE_MODEL_3D, image_pts, cam, dist,
                rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rv, tv = cv2.solvePnP(
                FACE_MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_SQPNP
            )
        return (rv, tv) if ok else None

    @staticmethod
    def _estimate_position(landmarks, w, h, cfg: Config):
        lx = landmarks[LM_LEFT_EYE].x * w;  ly = landmarks[LM_LEFT_EYE].y * h
        rx = landmarks[LM_RIGHT_EYE].x * w; ry = landmarks[LM_RIGHT_EYE].y * h
        eye_dist_px = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
        if eye_dist_px < MIN_EYE_DIST_PX:
            return None
        focal = cfg.focal_length_px
        z_cm = (cfg.real_eye_dist_cm * focal) / eye_dist_px
        cx_px = (landmarks[LM_LEFT_EYE].x + landmarks[LM_RIGHT_EYE].x) / 2.0 * w
        cy_px = (landmarks[LM_LEFT_EYE].y + landmarks[LM_RIGHT_EYE].y) / 2.0 * h
        x_cm = (cx_px - w / 2.0) * z_cm / focal + cfg.cam_offset_x_cm
        y_cm = -((cy_px - h / 2.0) * z_cm / focal) + cfg.cam_offset_y_cm
        return x_cm, y_cm, z_cm

    @staticmethod
    def _draw_preview(frame, rvec, tvec, cam_mtx, dist_cfs,
                      yaw, pitch, roll, tx, ty, tz):
        _eye_o = np.float32([0.0, 2.5, -4.0])
        axes = np.float32([
            _eye_o + [AXIS_DISPLAY_LEN_CM, 0, 0],
            _eye_o + [0, -AXIS_DISPLAY_LEN_CM, 0],
            _eye_o + [0, 0, -AXIS_DISPLAY_LEN_CM],
            _eye_o,
        ])
        ap, _ = cv2.projectPoints(axes, rvec, tvec, cam_mtx, dist_cfs)
        o = tuple(ap[3].ravel().astype(int))
        cv2.line(frame, o, tuple(ap[0].ravel().astype(int)), (0, 0, 255), 2)
        cv2.line(frame, o, tuple(ap[1].ravel().astype(int)), (0, 255, 0), 2)
        cv2.line(frame, o, tuple(ap[2].ravel().astype(int)), (255, 0, 0), 2)
        cv2.putText(frame, f"Yaw:{yaw:+6.1f}  Pitch:{pitch:+6.1f}  Roll:{roll:+6.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"X:{tx:+6.1f}  Y:{ty:+6.1f}  Z:{tz:+6.1f} cm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
```

- [ ] **Step 2.4: Run tracker utility tests — expect all pass**

```
.venv/Scripts/python -m pytest tests/test_tracker_utils.py -v
```

Expected: 7 passed

- [ ] **Step 2.5: Commit**

```bash
git add tracker.py tests/test_tracker_utils.py
git commit -m "feat: extract FaceTracker class with threading support"
```

---

## Task 3: Settings Window

**Files:**
- Create: `settings_window.py`

`SettingsWindow` is a `tk.Toplevel` with a `ttk.Notebook` containing two tabs. It reads the current `Config` on open and calls a callback with the new `Config` when the user clicks Apply/Save.

- [ ] **Step 3.1: Create `settings_window.py`**

```python
"""settings_window.py — Settings Toplevel with two tabs for all Config params."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from config import Config, save_config


class SettingsWindow(tk.Toplevel):
    """Settings editor.  Calls on_apply(cfg) when user clicks Apply or Save."""

    def __init__(self, parent: tk.Misc, cfg: Config, on_apply: Callable[[Config], None]):
        super().__init__(parent)
        self.title("設定 / Settings")
        self.resizable(False, False)
        self.grab_set()   # modal
        self._on_apply = on_apply

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        env_frame = ttk.Frame(notebook)
        tune_frame = ttk.Frame(notebook)
        notebook.add(env_frame,  text="🌐 環境設定 / Environment")
        notebook.add(tune_frame, text="⚙ 調效參數 / Tuning")

        self._vars: dict[str, tk.Variable] = {}
        self._build_env_tab(env_frame, cfg)
        self._build_tune_tab(tune_frame, cfg)

        # Buttons row
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Apply",      command=self._apply).pack(side="left",  padx=4)
        ttk.Button(btn_frame, text="Save",       command=self._save).pack(side="left",   padx=4)
        ttk.Button(btn_frame, text="Cancel",     command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Reset defaults", command=lambda: self._reset(cfg)).pack(side="right", padx=4)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_env_tab(self, parent: ttk.Frame, cfg: Config) -> None:
        rows = [
            ("cam_index",        "Camera Index",     "int",   0, 9,    1,    cfg.cam_index),
            ("focal_length_px",  "Focal Length (px)","float", 100, 1000, 1,  cfg.focal_length_px),
            ("max_num_faces",    "Max Faces",        "int",   1, 10,   1,    cfg.max_num_faces),
            ("lock_snap_dist_px","Lock Snap Dist (px)","int", 30, 500, 10,   cfg.lock_snap_dist_px),
            ("cam_offset_x_cm", "Cam Offset X (cm)","float",-30, 30,  0.5,  cfg.cam_offset_x_cm),
            ("cam_offset_y_cm", "Cam Offset Y (cm)","float",  0, 60,  0.5,  cfg.cam_offset_y_cm),
            ("real_eye_dist_cm","Eye Distance (cm)","float",  4, 15,  0.5,  cfg.real_eye_dist_cm),
        ]
        for r, (key, label, kind, lo, hi, res, default) in enumerate(rows):
            self._add_slider_row(parent, r, key, label, kind, lo, hi, res, default)

        # UDP Host — text entry
        r = len(rows)
        ttk.Label(parent, text="UDP Host", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        var = tk.StringVar(value=cfg.udp_host)
        self._vars["udp_host"] = var
        ttk.Entry(parent, textvariable=var, width=18).grid(row=r, column=1, columnspan=2, sticky="w", padx=6)

        # UDP Port — spinbox
        r += 1
        ttk.Label(parent, text="UDP Port", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        var2 = tk.IntVar(value=cfg.udp_port)
        self._vars["udp_port"] = var2
        ttk.Spinbox(parent, from_=1024, to=65535, textvariable=var2, width=7).grid(
            row=r, column=1, sticky="w", padx=6)

    def _build_tune_tab(self, parent: ttk.Frame, cfg: Config) -> None:
        rows = [
            ("smooth_alpha", "Smooth Alpha",    "float", 0.01, 1.0,  0.01, cfg.smooth_alpha),
            ("deadzone_rot", "Deadzone Rot (°)","float", 0.0,  10.0, 0.1,  cfg.deadzone_rot),
            ("deadzone_pos", "Deadzone Pos (cm)","float",0.0,  5.0,  0.05, cfg.deadzone_pos),
            ("yaw_scale",   "Yaw Scale",        "float", 0.1, 5.0,  0.1,  cfg.yaw_scale),
            ("pitch_scale", "Pitch Scale",      "float", 0.1, 5.0,  0.1,  cfg.pitch_scale),
            ("roll_scale",  "Roll Scale",       "float", 0.1, 5.0,  0.1,  cfg.roll_scale),
            ("x_scale",     "X Scale",          "float", 0.1, 5.0,  0.1,  cfg.x_scale),
            ("y_scale",     "Y Scale",          "float", 0.1, 5.0,  0.1,  cfg.y_scale),
            ("z_scale",     "Z Scale",          "float", 0.1, 5.0,  0.1,  cfg.z_scale),
        ]
        for r, (key, label, kind, lo, hi, res, default) in enumerate(rows):
            self._add_slider_row(parent, r, key, label, kind, lo, hi, res, default)

    def _add_slider_row(self, parent, row, key, label, kind, lo, hi, res, default):
        ttk.Label(parent, text=label, width=20, anchor="e").grid(row=row, column=0, padx=6, pady=3)
        var = tk.DoubleVar(value=float(default)) if kind == "float" else tk.IntVar(value=int(default))
        self._vars[key] = var
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", length=220)
        scale.grid(row=row, column=1, padx=4, pady=3)
        val_lbl = ttk.Label(parent, text=f"{default}", width=8)
        val_lbl.grid(row=row, column=2, padx=4)

        def _update_label(*_):
            v = var.get()
            val_lbl.config(text=f"{v:.2f}" if kind == "float" else str(int(v)))
        var.trace_add("write", _update_label)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _collect(self) -> Config:
        v = self._vars
        return Config(
            cam_index        = int(v["cam_index"].get()),
            focal_length_px  = float(v["focal_length_px"].get()),
            max_num_faces    = int(v["max_num_faces"].get()),
            lock_snap_dist_px= int(v["lock_snap_dist_px"].get()),
            cam_offset_x_cm  = float(v["cam_offset_x_cm"].get()),
            cam_offset_y_cm  = float(v["cam_offset_y_cm"].get()),
            udp_host         = v["udp_host"].get().strip(),
            udp_port         = int(v["udp_port"].get()),
            real_eye_dist_cm = float(v["real_eye_dist_cm"].get()),
            smooth_alpha     = float(v["smooth_alpha"].get()),
            deadzone_rot     = float(v["deadzone_rot"].get()),
            deadzone_pos     = float(v["deadzone_pos"].get()),
            yaw_scale        = float(v["yaw_scale"].get()),
            pitch_scale      = float(v["pitch_scale"].get()),
            roll_scale       = float(v["roll_scale"].get()),
            x_scale          = float(v["x_scale"].get()),
            y_scale          = float(v["y_scale"].get()),
            z_scale          = float(v["z_scale"].get()),
        )

    def _apply(self) -> None:
        cfg = self._collect()
        self._on_apply(cfg)

    def _save(self) -> None:
        cfg = self._collect()
        save_config(cfg)
        self._on_apply(cfg)
        messagebox.showinfo("Saved", "設定已儲存 / Settings saved to config.json", parent=self)
        self.destroy()

    def _reset(self, _original_cfg: Config) -> None:
        defaults = Config()
        for key, var in self._vars.items():
            val = getattr(defaults, key)
            var.set(val)
```

- [ ] **Step 3.2: Manual smoke test — verify settings window opens without error**

```
cd "c:/Users/0/Desktop/my projects/off-axis-projection"
.venv/Scripts/python -c "
import tkinter as tk
from config import Config
from settings_window import SettingsWindow
root = tk.Tk(); root.withdraw()
SettingsWindow(root, Config(), lambda cfg: print('apply:', cfg.smooth_alpha))
root.mainloop()
"
```

Expected: settings window appears with two tabs, sliders are draggable, Apply/Save/Cancel work, no exceptions.

- [ ] **Step 3.3: Commit**

```bash
git add settings_window.py
git commit -m "feat: add SettingsWindow with two-tab parameter editor"
```

---

## Task 4: Main Application Window

**Files:**
- Create: `app.py`

The main window is a compact tkinter panel (~320×220px). It polls `tracker.live` every 150ms via `after()` to update the HUD labels and status indicator.

- [ ] **Step 4.1: Create `app.py`**

```python
"""app.py — Entry point + MainWindow control panel."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from config import Config, load_config, save_config
from tracker import FaceTracker
from settings_window import SettingsWindow

POLL_MS = 150   # how often MainWindow refreshes HUD from tracker.live


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Off-Axis Face Tracker")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cfg = load_config()
        self._tracker = FaceTracker(self._cfg)
        self._settings_win: SettingsWindow | None = None

        self._build_ui()
        self._poll()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Status row ──────────────────────────────────────────────────────
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", **pad)

        self._status_dot = tk.Label(status_frame, text="●", fg="#555", font=("Consolas", 16))
        self._status_dot.pack(side="left")
        self._status_lbl = ttk.Label(status_frame, text="待機 / Idle", font=("Consolas", 11))
        self._status_lbl.pack(side="left", padx=6)
        self._cam_lbl = ttk.Label(status_frame,
                                   text=f"Cam #{self._cfg.cam_index}  →  {self._cfg.udp_host}:{self._cfg.udp_port}",
                                   font=("Consolas", 9), foreground="#888")
        self._cam_lbl.pack(side="right")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)

        # ── HUD labels ───────────────────────────────────────────────────────
        hud_frame = ttk.Frame(self.root)
        hud_frame.pack(fill="x", **pad)

        self._hud_rot = ttk.Label(hud_frame,
                                   text="Yaw:  +0.0°   Pitch:  +0.0°   Roll:  +0.0°",
                                   font=("Consolas", 10), foreground="#4fc")
        self._hud_rot.pack(anchor="w")
        self._hud_pos = ttk.Label(hud_frame,
                                   text="X: +0.0   Y: +0.0   Z: +0.0  cm",
                                   font=("Consolas", 10), foreground="#4cf")
        self._hud_pos.pack(anchor="w")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)

        self._start_btn = ttk.Button(btn_frame, text="▶  開始 / Start", width=18,
                                      command=self._toggle_tracker)
        self._start_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="⚙  設定 / Settings", width=18,
                   command=self._open_settings).pack(side="left")

        # ── Error label (hidden by default) ──────────────────────────────────
        self._err_lbl = ttk.Label(self.root, text="", foreground="red",
                                   font=("Consolas", 9), wraplength=300)
        self._err_lbl.pack(padx=10, pady=(0, 4))

    # ── Tracker control ───────────────────────────────────────────────────────

    def _toggle_tracker(self) -> None:
        if self._tracker.running:
            self._tracker.stop()
            self._start_btn.config(text="▶  開始 / Start")
        else:
            self._tracker = FaceTracker(self._cfg)
            self._tracker.start(preview=True)
            self._start_btn.config(text="■  停止 / Stop")

    def _open_settings(self) -> None:
        if self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return
        self._settings_win = SettingsWindow(
            self.root, self._cfg, on_apply=self._on_settings_apply
        )

    def _on_settings_apply(self, new_cfg: Config) -> None:
        self._cfg = new_cfg
        # Restart tracker so cam_index / max_num_faces / udp changes take effect
        was_running = self._tracker.running
        if was_running:
            self._tracker.stop()
        self._tracker = FaceTracker(self._cfg)
        if was_running:
            self._tracker.start(preview=True)
        # Update info label
        self._cam_lbl.config(
            text=f"Cam #{self._cfg.cam_index}  →  {self._cfg.udp_host}:{self._cfg.udp_port}"
        )

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        live = self._tracker.live
        tracking: bool = live["tracking"]
        error: str | None = live["error"]

        if error:
            self._status_dot.config(fg="#f55")
            self._status_lbl.config(text="錯誤 / Error")
            self._err_lbl.config(text=error)
        elif tracking:
            self._status_dot.config(fg="#4f4")
            self._status_lbl.config(text="追蹤中 / Tracking")
            self._err_lbl.config(text="")
            self._hud_rot.config(
                text=f"Yaw: {live['yaw']:+6.1f}°   Pitch: {live['pitch']:+6.1f}°   Roll: {live['roll']:+6.1f}°"
            )
            self._hud_pos.config(
                text=f"X: {live['x']:+6.1f}   Y: {live['y']:+6.1f}   Z: {live['z']:+6.1f}  cm"
            )
        elif self._tracker.running:
            self._status_dot.config(fg="#fa0")
            self._status_lbl.config(text="偵測中... / Detecting...")
            self._err_lbl.config(text="")
        else:
            self._status_dot.config(fg="#555")
            self._status_lbl.config(text="待機 / Idle")
            self._err_lbl.config(text="")

        self.root.after(POLL_MS, self._poll)

    def _on_close(self) -> None:
        if self._tracker.running:
            self._tracker.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Manual smoke test — launch the app**

```
cd "c:/Users/0/Desktop/my projects/off-axis-projection"
.venv/Scripts/python app.py
```

Verify:
- Control panel window appears (~320px wide)
- Status shows "待機 / Idle" with grey dot
- Click "▶ 開始 / Start" → camera preview window opens, status changes to yellow then green
- HUD numbers update while tracking
- Click "⚙ 設定" → settings window opens, both tabs work, sliders move, Apply restarts tracker
- Click "■ 停止 / Stop" → preview closes, status returns to Idle
- Close main window → app exits cleanly

- [ ] **Step 4.3: Commit**

```bash
git add app.py
git commit -m "feat: add MainWindow control panel with live HUD and settings integration"
```

---

## Task 5: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 5.1: Install PyInstaller and update requirements**

```
.venv/Scripts/pip install pyinstaller
.venv/Scripts/pip freeze | grep -i pyinstaller >> requirements.txt
```

Then open `requirements.txt` and verify `pyinstaller` appears (check version, typically `pyinstaller==6.x.x`). Remove the duplicate line if the package was already listed.

- [ ] **Step 5.2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add pyinstaller to requirements"
```

---

## Task 6: PyInstaller Packaging

**Files:**
- Create: `off_axis_tracker.spec`
- Create: `build.bat`

- [ ] **Step 6.1: Create `off_axis_tracker.spec`**

```python
# off_axis_tracker.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collect mediapipe bundled model files and OpenCV data
mp_datas   = collect_data_files("mediapipe")
cv2_datas  = collect_data_files("cv2")

# Include face_landmarker.task from project root (bundled into _internal/)
extra_datas = [("face_landmarker.task", ".")]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=mp_datas + cv2_datas + extra_datas,
    hiddenimports=[
        "mediapipe",
        "mediapipe.python",
        "mediapipe.python.solutions",
        "mediapipe.python.solutions.face_mesh",
        "cv2",
        "numpy",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OffAxisTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # no console window (set True if you want debug console)
    icon=None,           # add path to .ico here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OffAxisTracker",
)
```

- [ ] **Step 6.2: Create `build.bat`**

```bat
@echo off
echo Building Off-Axis Face Tracker...
.venv\Scripts\pyinstaller off_axis_tracker.spec --clean --noconfirm
echo.
echo Done. Output: dist\OffAxisTracker\OffAxisTracker.exe
pause
```

- [ ] **Step 6.3: Run the build**

```
cd "c:/Users/0/Desktop/my projects/off-axis-projection"
build.bat
```

Expected: build completes without error, `dist\OffAxisTracker\` folder created containing `OffAxisTracker.exe`.

- [ ] **Step 6.4: Smoke test the built exe**

Double-click `dist\OffAxisTracker\OffAxisTracker.exe` (or run from terminal):

```
.\dist\OffAxisTracker\OffAxisTracker.exe
```

Verify:
- App launches without console error
- Control panel opens
- Start tracking — camera preview appears, HUD updates
- Open Settings — both tabs work
- Change a value, click Save → `config.json` appears next to `OffAxisTracker.exe` in `dist\OffAxisTracker\`
- Close and reopen exe → saved settings are loaded (verify changed value persists)

If build fails with import errors, add missing modules to `hiddenimports` in the spec and rerun `build.bat`.

- [ ] **Step 6.5: Commit**

```bash
git add off_axis_tracker.spec build.bat
git commit -m "feat: add PyInstaller spec and build script for onedir exe"
```

---

## Manual Test Checklist (no automated test possible — camera required)

Run through these after Task 6.4 with the built exe:

- [ ] Launches with no Python installed on machine (test on a clean machine or VM if possible)
- [ ] `config.json` is created next to exe on first Save
- [ ] Relaunch loads saved values into Settings UI
- [ ] Changing CAM_INDEX to an invalid index shows error message in red in the control panel, app doesn't crash
- [ ] Closing settings without saving discards changes (Apply updates tracker but does not write file)
- [ ] Changing UDP host/port while tracker is running restarts tracker with new address
- [ ] Q/ESC in the camera preview window closes only the preview, not the whole app

---

## Add to .gitignore

- [ ] Add these lines to `.gitignore` (create it if absent):

```
dist/
build/
__pycache__/
*.pyc
.superpowers/
config.json
```

```bash
git add .gitignore
git commit -m "chore: add gitignore for build artifacts and local config"
```
