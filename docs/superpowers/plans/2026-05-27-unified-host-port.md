# Unified Host/Port Settings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four separate network config fields (`udp_host`, `udp_port`, `freed_host`, `freed_port`) with a single `host`/`port` pair; when the protocol dropdown changes, both fields auto-reset to that protocol's defaults.

**Architecture:** Update `config.py` first (adds `host`/`port`, removes old fields, exports `DEFAULT_HOST`/`DEFAULT_PORTS` constants); then update `tracker.py` to use the unified fields; then simplify `settings_window.py` to one host/port row with an auto-reset callback; finally update `app.py` to drop the now-unnecessary `_endpoint_str` helper.

**Tech Stack:** Python 3, dataclasses, tkinter/ttk, pytest

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Modify | `config.py` | Remove 4 old fields, add `host`/`port`, add `DEFAULT_HOST`/`DEFAULT_PORTS` |
| Modify | `tests/test_config.py` | Update field-name assertions, add constants test, add old-key-ignored test |
| Modify | `tracker.py` | Use `cfg.host`/`cfg.port` in dispatch block |
| Modify | `settings_window.py` | Remove FreeD rows + toggle callback; rename UDP→Host/Port; add reset callback |
| Modify | `app.py` | Remove `_endpoint_str`; use `cfg.host`/`cfg.port` directly |

---

## Task 1: Update config.py

**Files:**
- Modify: `config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_config.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_config.py -v
```

Expected: multiple FAIL — `Config` has no field `host`, no `DEFAULT_HOST`, etc.

- [ ] **Step 3: Update config.py**

Replace the full `Config` dataclass and add module-level constants. The file should become:

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

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORTS: dict[str, int] = {"freed": 40000, "opentrack": 4242}


@dataclass
class Config:
    # ── Environment Profile ──────────────────────────────────────────────────
    cam_index: int = 0
    focal_length_px: float = 320.0
    max_num_faces: int = 5
    lock_snap_dist_px: int = 150
    cam_offset_x_cm: float = 0.0
    cam_offset_y_cm: float = 16.2
    # ── Output Protocol ──────────────────────────────────────────────────────
    output_protocol: str = "freed"       # "freed" | "opentrack"
    host: str = "127.0.0.1"
    port: int = 40000
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

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_config.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "refactor: replace dual host/port fields with single host/port + DEFAULT_PORTS"
```

---

## Task 2: Update tracker.py

**Files:**
- Modify: `tracker.py`

- [ ] **Step 1: Update the dispatch block in tracker.py**

Find the protocol dispatch block (currently lines ~305–311):

```python
                        if cfg.output_protocol == "freed":
                            data = sender.pack_freed(tx, ty, tz, yaw, pitch, roll)
                            sender.send(sock, cfg.freed_host, cfg.freed_port, data)
                        elif cfg.output_protocol == "opentrack":
                            data = sender.pack_opentrack(tx, ty, tz, yaw, pitch, roll)
                            sender.send(sock, cfg.udp_host, cfg.udp_port, data)
                        else:
                            raise ValueError(f"Unknown output_protocol: {cfg.output_protocol!r}")
```

Replace it with:

```python
                        if cfg.output_protocol == "freed":
                            data = sender.pack_freed(tx, ty, tz, yaw, pitch, roll)
                        elif cfg.output_protocol == "opentrack":
                            data = sender.pack_opentrack(tx, ty, tz, yaw, pitch, roll)
                        else:
                            raise ValueError(f"Unknown output_protocol: {cfg.output_protocol!r}")
                        sender.send(sock, cfg.host, cfg.port, data)
```

- [ ] **Step 2: Run the full test suite**

```
python -m pytest tests/ -v
```

Expected: all 22 tests PASS (tracker tests don't exercise config fields directly)

- [ ] **Step 3: Commit**

```bash
git add tracker.py
git commit -m "refactor: use cfg.host/cfg.port in tracker dispatch"
```

---

## Task 3: Update settings_window.py

**Files:**
- Modify: `settings_window.py`

- [ ] **Step 1: Replace _build_env_tab entirely**

Replace the entire `_build_env_tab` method (lines 43–119) with:

```python
    def _build_env_tab(self, parent: ttk.Frame, cfg: Config) -> None:
        rows = [
            ("cam_index",        "Camera Index",       "int",   0,   9,    1,    cfg.cam_index),
            ("focal_length_px",  "Focal Length (px)",  "float", 100, 1000, 1,    cfg.focal_length_px),
            ("max_num_faces",    "Max Faces",          "int",   1,   10,   1,    cfg.max_num_faces),
            ("lock_snap_dist_px","Lock Snap Dist (px)","int",   30,  500,  10,   cfg.lock_snap_dist_px),
            ("cam_offset_x_cm", "Cam Offset X (cm)",  "float", -30, 30,   0.5,  cfg.cam_offset_x_cm),
            ("cam_offset_y_cm", "Cam Offset Y (cm)",  "float",  0,  60,   0.5,  cfg.cam_offset_y_cm),
            ("real_eye_dist_cm","Eye Distance (cm)",  "float",  4,  15,   0.5,  cfg.real_eye_dist_cm),
        ]
        for r, (key, label, kind, lo, hi, res, default) in enumerate(rows):
            self._add_slider_row(parent, r, key, label, kind, lo, hi, res, default)

        # Host — text entry
        r = len(rows)
        ttk.Label(parent, text="Host", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        host_var = tk.StringVar(value=cfg.host)
        self._vars["host"] = host_var
        ttk.Entry(parent, textvariable=host_var, width=18).grid(
            row=r, column=1, columnspan=2, sticky="w", padx=6)

        # Port — spinbox
        r += 1
        ttk.Label(parent, text="Port", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        port_var = tk.IntVar(value=cfg.port)
        self._vars["port"] = port_var
        ttk.Spinbox(parent, from_=1024, to=65535, textvariable=port_var, width=7).grid(
            row=r, column=1, sticky="w", padx=6)

        # Output Protocol — dropdown
        r += 1
        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=6)

        r += 1
        ttk.Label(parent, text="Output Protocol", width=20, anchor="e").grid(
            row=r, column=0, padx=6, pady=4)
        proto_var = tk.StringVar(value=cfg.output_protocol)
        self._vars["output_protocol"] = proto_var
        ttk.Combobox(
            parent, textvariable=proto_var,
            values=["freed", "opentrack"],
            state="readonly", width=15,
        ).grid(row=r, column=1, sticky="w", padx=6)

        def _on_protocol_change(*_):
            host_var.set(DEFAULT_HOST)
            port_var.set(DEFAULT_PORTS[proto_var.get()])

        proto_var.trace_add("write", _on_protocol_change)
```

- [ ] **Step 2: Update the import line at the top of settings_window.py**

Change line 8:

```python
# Before:
from config import Config, save_config

# After:
from config import Config, save_config, DEFAULT_HOST, DEFAULT_PORTS
```

- [ ] **Step 3: Replace _collect() entirely**

Replace the `_collect` method (lines 152–176) with:

```python
    def _collect(self) -> Config:
        v = self._vars
        return Config(
            cam_index         = int(v["cam_index"].get()),
            focal_length_px   = float(v["focal_length_px"].get()),
            max_num_faces     = int(v["max_num_faces"].get()),
            lock_snap_dist_px = int(v["lock_snap_dist_px"].get()),
            cam_offset_x_cm   = float(v["cam_offset_x_cm"].get()),
            cam_offset_y_cm   = float(v["cam_offset_y_cm"].get()),
            output_protocol   = v["output_protocol"].get(),
            host              = v["host"].get().strip(),
            port              = int(v["port"].get()),
            real_eye_dist_cm  = float(v["real_eye_dist_cm"].get()),
            smooth_alpha      = float(v["smooth_alpha"].get()),
            deadzone_rot      = float(v["deadzone_rot"].get()),
            deadzone_pos      = float(v["deadzone_pos"].get()),
            yaw_scale         = float(v["yaw_scale"].get()),
            pitch_scale       = float(v["pitch_scale"].get()),
            roll_scale        = float(v["roll_scale"].get()),
            x_scale           = float(v["x_scale"].get()),
            y_scale           = float(v["y_scale"].get()),
            z_scale           = float(v["z_scale"].get()),
        )
```

- [ ] **Step 4: Run the full test suite**

```
python -m pytest tests/ -v
```

Expected: all 22 tests PASS

- [ ] **Step 5: Commit**

```bash
git add settings_window.py
git commit -m "refactor: unify host/port UI, auto-reset on protocol change"
```

---

## Task 4: Update app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Remove _endpoint_str and update status bar initialization**

In `_build_ui`, replace lines 39–43:

```python
        # Before:
        _h, _p = self._endpoint_str(self._cfg)
        self._cam_lbl = ttk.Label(status_frame,
                                   text=f"Cam #{self._cfg.cam_index}  ->  {_h}:{_p}  [{self._cfg.output_protocol}]",
                                   font=("Consolas", 9), foreground="#888")

        # After:
        self._cam_lbl = ttk.Label(status_frame,
                                   text=f"Cam #{self._cfg.cam_index}  ->  {self._cfg.host}:{self._cfg.port}  [{self._cfg.output_protocol}]",
                                   font=("Consolas", 9), foreground="#888")
```

- [ ] **Step 2: Remove the _endpoint_str static method**

Delete these 4 lines entirely (lines 79–83):

```python
    @staticmethod
    def _endpoint_str(cfg) -> tuple:
        if cfg.output_protocol == "freed":
            return cfg.freed_host, cfg.freed_port
        return cfg.udp_host, cfg.udp_port
```

- [ ] **Step 3: Update _on_settings_apply**

Replace lines 112–115:

```python
        # Before:
        host, port = self._endpoint_str(new_cfg)
        self._cam_lbl.config(
            text=f"Cam #{new_cfg.cam_index}  ->  {host}:{port}  [{new_cfg.output_protocol}]"
        )

        # After:
        self._cam_lbl.config(
            text=f"Cam #{new_cfg.cam_index}  ->  {new_cfg.host}:{new_cfg.port}  [{new_cfg.output_protocol}]"
        )
```

- [ ] **Step 4: Run the full test suite**

```
python -m pytest tests/ -v
```

Expected: all 22 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "refactor: use cfg.host/cfg.port in app status bar, remove _endpoint_str"
```
