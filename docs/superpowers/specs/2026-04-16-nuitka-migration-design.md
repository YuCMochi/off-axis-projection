# Nuitka Migration Design

**Date:** 2026-04-16
**Branch:** feat/nuitka-migration

## Goal

Replace PyInstaller with Nuitka as the packaging tool for the Off-Axis Tracker standalone Windows exe. Primary motivations: faster startup time and smaller distribution size.

## Scope

- Replace `off_axis_tracker.spec` + PyInstaller with `pyproject.toml` [tool.nuitka] + Nuitka
- Simplify `build.bat`
- Update `requirements.txt`
- Update `.github/workflows/release.yml` CI pipeline

Python source files (`app.py`, `tracker.py`, `config.py`, `settings_window.py`) are not touched.

## Output Format

`--standalone` onedir: `dist\OffAxisTracker\OffAxisTracker.exe` plus accompanying DLLs and data files. Same structure as the current PyInstaller output — the zip step in CI does not need to change.

## File Changes

### New: `pyproject.toml`

Replaces the role of `off_axis_tracker.spec`. Nuitka reads this automatically when running `python -m nuitka app.py`.

```toml
[tool.nuitka]
standalone = true
enable-plugin = ["numpy", "tk-inter"]
include-package = ["mediapipe", "cv2"]
include-package-data = ["mediapipe", "cv2"]
windows-disable-console = true
output-filename = "OffAxisTracker"
output-dir = "dist"
```

Key decisions:
- `enable-plugin = ["numpy", "tk-inter"]` — Nuitka built-in plugins handle C extension wiring for numpy and tkinter
- `include-package = ["mediapipe", "cv2"]` — forces full package inclusion (mediapipe uses dynamic imports internally)
- `include-package-data = ["mediapipe", "cv2"]` — carries model files bundled inside mediapipe (legacy solutions API) and OpenCV data files; replaces `collect_data_files()` from PyInstaller
- `windows-disable-console = true` — no console window, matches current PyInstaller `console=False`

### Modified: `build.bat`

```bat
@echo off
echo Building Off-Axis Face Tracker...
python -m nuitka app.py
echo.
echo Done. Output: dist\OffAxisTracker\OffAxisTracker.exe
pause
```

### Deleted: `off_axis_tracker.spec`

No longer needed.

### Modified: `requirements.txt`

Add `nuitka` to the dependency list.

### Modified: `.github/workflows/release.yml`

Replace the single PyInstaller build step with one step:

```yaml
- name: Build exe
  run: python -m nuitka app.py
```

`nuitka` is already installed by the existing `pip install -r requirements.txt` step, so no extra install step is needed. All other steps (checkout, Python setup, install dependencies, zip, release) remain unchanged.

## Size and Startup Expectations

Current PyInstaller output: ~304 MB. Nuitka output expected ~200–280 MB — mediapipe model files and C extensions dominate the size and cannot be eliminated regardless of tool. Startup speed improvement comes from Nuitka's AOT compilation eliminating the PyInstaller bootloader unpack phase.

## Non-ASCII Path Issue

The existing workaround for mediapipe's C++ ANSI fopen failure on non-ASCII install paths (fixed in a prior commit) is in Python source code and is not affected by this migration.
