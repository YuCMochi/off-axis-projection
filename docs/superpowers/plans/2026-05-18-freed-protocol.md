# FreeD LiveLink Protocol Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FreeD LiveLink UDP output as a selectable alternative to OpenTrack, with FreeD as the default, by extracting packet logic into a new `sender.py` module.

**Architecture:** Create `sender.py` to own all packet-building and UDP dispatch; update `tracker.py` to delegate to it based on `cfg.output_protocol`; add three new Config fields; add FreeD protocol UI to `settings_window.py`.

**Tech Stack:** Python 3, standard library (`socket`, `struct`), tkinter/ttk, pytest

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Create | `sender.py` | `pack_opentrack`, `pack_freed`, `send` |
| Create | `tests/test_sender.py` | Unit tests for sender functions |
| Modify | `tracker.py` | Remove `pack_opentrack`, remove `import struct`, add `import sender`, switch send call |
| Modify | `tests/test_tracker_utils.py` | Update `pack_opentrack` import path |
| Modify | `config.py` | Add `output_protocol`, `freed_host`, `freed_port` fields |
| Modify | `tests/test_config.py` | Assert new field defaults |
| Modify | `settings_window.py` | Protocol dropdown + FreeD host/port fields + `_collect()` update |

---

## Task 1: Create sender.py

**Files:**
- Create: `sender.py`
- Create: `tests/test_sender.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sender.py`:

```python
import struct
import pytest


def test_pack_opentrack_length_and_values():
    from sender import pack_opentrack
    data = pack_opentrack(1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
    assert len(data) == 48
    assert struct.unpack("<6d", data) == (1.0, 2.0, 3.0, 10.0, 20.0, 30.0)


def test_pack_freed_length():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 0, 0, 0)
    assert len(data) == 29


def test_pack_freed_message_type_and_camera_id():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 0, 0, 0)
    assert data[0] == 0xD1
    assert data[1] == 0x00


def test_pack_freed_zero_checksum_value():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 0, 0, 0)
    # first 28 bytes sum: 0xD1=209, rest zero -> checksum = (256-209)%256 = 47
    assert data[28] == 47


def test_pack_freed_checksum_makes_total_zero_mod256():
    from sender import pack_freed
    data = pack_freed(10.0, 20.0, 150.0, 45.0, -10.0, 5.0)
    assert sum(data) % 256 == 0


def test_pack_freed_checksum_nonzero_rotation():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 90.0, 0, 0)
    assert sum(data) % 256 == 0


def test_pack_freed_clamps_overflow_without_raising():
    from sender import pack_freed
    data = pack_freed(99999, 99999, 99999, 999, 999, 999)
    assert len(data) == 29
    assert sum(data) % 256 == 0


def test_pack_freed_clamps_underflow_without_raising():
    from sender import pack_freed
    data = pack_freed(-99999, -99999, -99999, -999, -999, -999)
    assert len(data) == 29
    assert sum(data) % 256 == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_sender.py -v
```

Expected: `ModuleNotFoundError: No module named 'sender'`

- [ ] **Step 3: Create sender.py**

```python
"""sender.py — UDP packet builders and send helper."""
from __future__ import annotations

import socket
import struct


def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    return struct.pack("<6d", x, y, z, yaw, pitch, roll)


def _to_24bit(value: float) -> bytes:
    val_int = int(value)
    val_int = max(-8388608, min(8388607, val_int))
    return val_int.to_bytes(3, byteorder="big", signed=True)


def pack_freed(x_cm: float, y_cm: float, z_cm: float,
               yaw: float, pitch: float, roll: float) -> bytes:
    packet = bytearray()
    packet.append(0xD1)
    packet.append(0x00)
    packet.extend(_to_24bit(yaw   * 32768))
    packet.extend(_to_24bit(pitch * 32768))
    packet.extend(_to_24bit(roll  * 32768))
    packet.extend(_to_24bit(x_cm * 10 * 64))
    packet.extend(_to_24bit(y_cm * 10 * 64))
    packet.extend(_to_24bit(z_cm * 10 * 64))
    packet.extend(_to_24bit(0))
    packet.extend(_to_24bit(0))
    packet.extend(bytes([0x00, 0x00]))
    checksum = (256 - (sum(packet) % 256)) % 256
    packet.append(checksum)
    return bytes(packet)


def send(sock: socket.socket, host: str, port: int, data: bytes) -> None:
    sock.sendto(data, (host, port))
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_sender.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sender.py tests/test_sender.py
git commit -m "feat: add sender.py with pack_opentrack, pack_freed, send"
```

---

## Task 2: Update tracker.py to use sender

**Files:**
- Modify: `tracker.py`
- Modify: `tests/test_tracker_utils.py`

- [ ] **Step 1: Update the import in test_tracker_utils.py**

In `tests/test_tracker_utils.py`, change line 7:

```python
# Before:
from tracker import pack_opentrack

# After:
from sender import pack_opentrack
```

- [ ] **Step 2: Run the updated test to verify it still passes**

```
pytest tests/test_tracker_utils.py::test_pack_opentrack_format -v
```

Expected: PASS (now importing from sender)

- [ ] **Step 3: Update tracker.py**

Make these three changes to `tracker.py`:

**a) Remove `import struct` from the imports block** (line 7 — it's only used by `pack_opentrack`)

**b) Add `import sender` after the existing local imports** (after `from config import Config`):

```python
import sender
```

**c) Remove the `pack_opentrack` function** (lines 110–111):

```python
# Remove this entire function:
def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    return struct.pack("<6d", x, y, z, yaw, pitch, roll)
```

**d) Replace the send call in `_tracking_loop`** (currently line 309–310):

```python
# Before:
packet = pack_opentrack(tx, ty, tz, yaw, pitch, roll)
sock.sendto(packet, (cfg.udp_host, cfg.udp_port))

# After:
if cfg.output_protocol == "freed":
    data = sender.pack_freed(tx, ty, tz, yaw, pitch, roll)
    sender.send(sock, cfg.freed_host, cfg.freed_port, data)
else:
    data = sender.pack_opentrack(tx, ty, tz, yaw, pitch, roll)
    sender.send(sock, cfg.udp_host, cfg.udp_port, data)
```

- [ ] **Step 4: Run the full test suite**

```
pytest tests/ -v
```

Expected: all existing tests PASS (no references to `pack_opentrack` in tracker anymore)

- [ ] **Step 5: Commit**

```bash
git add tracker.py tests/test_tracker_utils.py
git commit -m "refactor: move pack_opentrack to sender, use sender for UDP dispatch"
```

---

## Task 3: Add FreeD fields to Config

**Files:**
- Modify: `config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_config.py`:

```python
def test_config_freed_defaults(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load_config()
    assert cfg.output_protocol == "freed"
    assert cfg.freed_host == "127.0.0.1"
    assert cfg.freed_port == 40000


def test_config_freed_roundtrip(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    original = config.Config(output_protocol="opentrack", freed_host="192.168.1.5", freed_port=9000)
    config.save_config(original)
    loaded = config.load_config()
    assert loaded.output_protocol == "opentrack"
    assert loaded.freed_host == "192.168.1.5"
    assert loaded.freed_port == 9000
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_config_freed_defaults tests/test_config.py::test_config_freed_roundtrip -v
```

Expected: FAIL — `Config` has no field `output_protocol`

- [ ] **Step 3: Add fields to config.py**

In `config.py`, add three fields to the `Config` dataclass after the existing `udp_port` line:

```python
    udp_host: str = "127.0.0.1"
    udp_port: int = 4242
    # ── Output Protocol ──────────────────────────────────────────────────────
    output_protocol: str = "freed"       # "freed" | "opentrack"
    freed_host: str = "127.0.0.1"
    freed_port: int = 40000
    real_eye_dist_cm: float = 9.0
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_config.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add output_protocol, freed_host, freed_port to Config"
```

---

## Task 4: Update settings_window.py

**Files:**
- Modify: `settings_window.py`

No pure unit tests possible for tkinter UI. Verify manually after implementation.

- [ ] **Step 1: Add FreeD fields to _build_env_tab**

In `settings_window.py`, extend `_build_env_tab` after the UDP Port spinbox block. The existing method ends with the `udp_port` row at `r += 1`. Add the following immediately after that spinbox `.grid(...)` call:

```python
        # Output Protocol — dropdown
        r += 1
        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=6)

        r += 1
        ttk.Label(parent, text="Output Protocol", width=20, anchor="e").grid(
            row=r, column=0, padx=6, pady=4)
        proto_var = tk.StringVar(value=cfg.output_protocol)
        self._vars["output_protocol"] = proto_var
        proto_combo = ttk.Combobox(
            parent, textvariable=proto_var,
            values=["freed", "opentrack"],
            state="readonly", width=15,
        )
        proto_combo.grid(row=r, column=1, sticky="w", padx=6)

        # FreeD Host
        r += 1
        ttk.Label(parent, text="FreeD Host", width=20, anchor="e").grid(
            row=r, column=0, padx=6, pady=4)
        freed_host_var = tk.StringVar(value=cfg.freed_host)
        self._vars["freed_host"] = freed_host_var
        freed_host_entry = ttk.Entry(parent, textvariable=freed_host_var, width=18)
        freed_host_entry.grid(row=r, column=1, columnspan=2, sticky="w", padx=6)

        # FreeD Port
        r += 1
        ttk.Label(parent, text="FreeD Port", width=20, anchor="e").grid(
            row=r, column=0, padx=6, pady=4)
        freed_port_var = tk.IntVar(value=cfg.freed_port)
        self._vars["freed_port"] = freed_port_var
        freed_port_spin = ttk.Spinbox(
            parent, from_=1024, to=65535,
            textvariable=freed_port_var, width=7,
        )
        freed_port_spin.grid(row=r, column=1, sticky="w", padx=6)

        def _toggle_freed_fields(*_):
            state = "normal" if proto_var.get() == "freed" else "disabled"
            freed_host_entry.config(state=state)
            freed_port_spin.config(state=state)

        proto_var.trace_add("write", _toggle_freed_fields)
        _toggle_freed_fields()
```

- [ ] **Step 2: Update _collect() to include new fields**

In `_collect`, the `return Config(...)` block currently ends with `z_scale`. Add three new keyword arguments before the closing parenthesis:

```python
        return Config(
            cam_index         = int(v["cam_index"].get()),
            focal_length_px   = float(v["focal_length_px"].get()),
            max_num_faces     = int(v["max_num_faces"].get()),
            lock_snap_dist_px = int(v["lock_snap_dist_px"].get()),
            cam_offset_x_cm   = float(v["cam_offset_x_cm"].get()),
            cam_offset_y_cm   = float(v["cam_offset_y_cm"].get()),
            udp_host          = v["udp_host"].get().strip(),
            udp_port          = int(v["udp_port"].get()),
            output_protocol   = v["output_protocol"].get(),
            freed_host        = v["freed_host"].get().strip(),
            freed_port        = int(v["freed_port"].get()),
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

- [ ] **Step 3: Run the full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Manual UI smoke test**

Run the app and open Settings:

```
python app.py
```

Verify:
- "Output Protocol" dropdown appears in the Environment tab, defaulting to "freed"
- "FreeD Host" and "FreeD Port" fields are enabled when "freed" is selected
- Switching to "opentrack" disables FreeD Host/Port fields
- Apply/Save round-trips correctly (reopen Settings and confirm values persisted)
- Reset Defaults restores "freed" / "127.0.0.1" / 40000

- [ ] **Step 5: Commit**

```bash
git add settings_window.py
git commit -m "feat: add FreeD protocol selector and fields to settings window"
```
