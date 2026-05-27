# Unified Host/Port Settings — Design Spec

**Date:** 2026-05-27

## Summary

Replace the four separate network fields (`udp_host`, `udp_port`, `freed_host`, `freed_port`) with a single `host`/`port` pair. When the user switches the output protocol in settings, both fields auto-reset to the protocol's default values (`127.0.0.1:40000` for FreeD, `127.0.0.1:4242` for OpenTrack).

---

## Changes

### config.py

Remove fields: `udp_host`, `udp_port`, `freed_host`, `freed_port`

Add fields:
```python
host: str = "127.0.0.1"
port: int = 40000   # default matches "freed"
```

Add module-level constants (not part of the dataclass):
```python
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS: dict[str, int] = {"freed": 40000, "opentrack": 4242}
```

Existing `config.json` files with old field names will have those keys silently ignored by `load_config` (existing behaviour — unknown keys are already filtered out).

### settings_window.py

**Remove:**
- The `ttk.Separator` between UDP and FreeD sections
- "FreeD Host" entry and its label
- "FreeD Port" spinbox and its label
- `self._vars["freed_host"]`, `self._vars["freed_port"]`
- The `_toggle_freed_fields` callback and `trace_add` wiring

**Rename existing UDP fields:**
- `self._vars["udp_host"]` → `self._vars["host"]`
- `self._vars["udp_port"]` → `self._vars["port"]`
- Update their `ttk.Label` texts to `"Host"` and `"Port"`

**Add protocol-change auto-reset:**
Wire the existing `proto_var` trace to also reset host/port:
```python
def _on_protocol_change(*_):
    proto = proto_var.get()
    host_var.set(DEFAULT_HOST)
    port_var.set(DEFAULT_PORTS[proto])

proto_var.trace_add("write", _on_protocol_change)
```

**Update `_collect()`:**
```python
host = v["host"].get().strip(),
port = int(v["port"].get()),
```
Remove `udp_host`, `udp_port`, `freed_host`, `freed_port` from the `Config(...)` call.

### tracker.py

Replace dual-dispatch send with shared host/port:
```python
if cfg.output_protocol == "freed":
    data = sender.pack_freed(tx, ty, tz, yaw, pitch, roll)
elif cfg.output_protocol == "opentrack":
    data = sender.pack_opentrack(tx, ty, tz, yaw, pitch, roll)
else:
    raise ValueError(f"Unknown output_protocol: {cfg.output_protocol!r}")
sender.send(sock, cfg.host, cfg.port, data)
```

### app.py

Update status bar to use `cfg.host` / `cfg.port` directly. Remove `_endpoint_str` static helper (no longer needed):

```python
# In __init__ and _on_settings_apply:
text=f"Cam #{cfg.cam_index}  ->  {cfg.host}:{cfg.port}  [{cfg.output_protocol}]"
```

---

## Testing

- Update `test_config.py`: replace assertions on old field names with `host`/`port`; add test for `DEFAULT_PORTS` values
- Verify `_collect()` and `_reset()` round-trip correctly through the new field names
- Manual: open settings, switch protocol, confirm host and port auto-reset

---

## Out of Scope

- Per-protocol memory of custom host/port (switching always resets to defaults)
- Adding new protocols
