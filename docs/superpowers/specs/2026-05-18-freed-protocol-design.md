# FreeD LiveLink Protocol Support — Design Spec

**Date:** 2026-05-18

## Summary

Add FreeD protocol output alongside the existing OpenTrack UDP output. The user can switch between protocols via a dropdown in the settings window. FreeD is the default. A new `sender.py` module owns all packet-formatting and UDP-send logic, keeping `tracker.py` focused on tracking math.

---

## Architecture

### New file: `sender.py`

Owns all packet construction and UDP dispatch:

- `pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes` — moved from `tracker.py` (48 bytes, 6 little-endian doubles)
- `pack_freed(x_cm, y_cm, z_cm, yaw, pitch, roll) -> bytes` — new (29 bytes, FreeD 0xD1)
- `send(sock, host, port, data: bytes)` — thin wrapper around `sock.sendto`

### Changes to `tracker.py`

- Remove `pack_opentrack` definition (moved to `sender.py`)
- Import `sender`
- In `_tracking_loop`, replace direct `sock.sendto(pack_opentrack(...))` with:
  ```python
  if cfg.output_protocol == "freed":
      data = sender.pack_freed(tx, ty, tz, yaw, pitch, roll)
      sender.send(sock, cfg.freed_host, cfg.freed_port, data)
  else:
      data = sender.pack_opentrack(tx, ty, tz, yaw, pitch, roll)
      sender.send(sock, cfg.udp_host, cfg.udp_port, data)
  ```

### Changes to `config.py`

Three new fields added to the `Config` dataclass:

```python
output_protocol: str = "freed"     # "freed" | "opentrack"
freed_host: str = "127.0.0.1"
freed_port: int = 40000
```

Existing `udp_host` / `udp_port` are unchanged (used only when `output_protocol == "opentrack"`).

### Changes to `settings_window.py`

New "Output Protocol" section in the settings window:

- Dropdown (`ttk.Combobox`): values `["FreeD LiveLink", "OpenTrack"]`, maps to `"freed"` / `"opentrack"`
- FreeD Host entry field
- FreeD Port entry field
- On dropdown change: enable/disable FreeD fields dynamically (`state="normal"` / `"disabled"`)

---

## FreeD Packet Format (0xD1)

Total: **29 bytes**

| Byte(s) | Field | Value / Conversion |
|---------|-------|--------------------|
| 0 | Message Type | `0xD1` |
| 1 | Camera ID | `0x00` (fixed) |
| 2–4 | Pan (Yaw) | `yaw_deg × 32768`, 24-bit signed big-endian |
| 5–7 | Tilt (Pitch) | `pitch_deg × 32768`, 24-bit signed big-endian |
| 8–10 | Roll | `roll_deg × 32768`, 24-bit signed big-endian |
| 11–13 | X position | `x_cm × 10 × 64`, 24-bit signed big-endian |
| 14–16 | Y position | `y_cm × 10 × 64`, 24-bit signed big-endian |
| 17–19 | Z position | `z_cm × 10 × 64`, 24-bit signed big-endian |
| 20–22 | Zoom | `0` |
| 23–25 | Focus | `0` |
| 26–27 | User Defined | `0x00 0x00` |
| 28 | Checksum | `(256 - sum(bytes[0:28]) % 256) % 256` |

Position unit: tracker outputs cm → converted to mm (`×10`) then to FreeD units (`×64`).
Rotation unit: tracker outputs degrees → multiplied by 32768.
24-bit values clamped to `[-8388608, 8388607]`.

---

## Error Handling

- No new error handling needed: `sender.send` uses the same UDP socket as before; UDP is fire-and-forget.
- Invalid `output_protocol` string falls back to OpenTrack silently (defensive `else` branch).

---

## Testing

- Unit test `pack_freed` with known values (verify byte layout and checksum).
- Unit test `pack_opentrack` still passes after move to `sender.py`.
- Manual: run tracker, switch to FreeD in settings, verify UE5 LiveLink FreeD receives data on port 40000.

---

## Out of Scope

- Camera ID configuration (fixed at `0x00`)
- Zoom / Focus values (always `0`)
- Simultaneous dual-protocol output
