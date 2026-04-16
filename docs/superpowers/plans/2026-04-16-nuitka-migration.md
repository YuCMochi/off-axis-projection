# Nuitka Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PyInstaller with Nuitka as the build tool, producing a faster-starting standalone Windows exe with a smaller footprint.

**Architecture:** Nuitka config lives in `pyproject.toml` under `[tool.nuitka]`, replacing `off_axis_tracker.spec`. `build.bat` calls `python -m nuitka app.py` with no arguments — all options come from `pyproject.toml`. The CI workflow replaces the `pyinstaller` step and updates the zip source path to match Nuitka's output directory naming (`app.dist/` instead of `OffAxisTracker/`).

**Tech Stack:** Nuitka (latest), Python 3.11, MSVC (already installed), GitHub Actions `windows-latest`

---

## File Map

| Action | File |
|---|---|
| Modify | `requirements.txt` |
| Create | `pyproject.toml` |
| Modify | `build.bat` |
| Delete | `off_axis_tracker.spec` |
| Modify | `.github/workflows/release.yml` |

> **Note on output path:** Nuitka names the output folder `{scriptname}.dist/`. Since the entry point is `app.py`, the output lands at `dist/app.dist/OffAxisTracker.exe` — not `dist/OffAxisTracker/`. The CI zip step must be updated accordingly.

---

### Task 1: Swap PyInstaller for Nuitka in requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Edit requirements.txt**

  Replace line:
  ```
  pyinstaller==6.19.0
  ```
  With:
  ```
  nuitka
  ```

- [ ] **Step 2: Verify install works**

  ```bash
  .venv/Scripts/pip install -r requirements.txt
  ```

  Expected: installs Nuitka with no errors. Nuitka will also pull in `ordered-set` and `zstandard` as dependencies automatically.

- [ ] **Step 3: Commit**

  ```bash
  git add requirements.txt
  git commit -m "chore: replace pyinstaller with nuitka in requirements"
  ```

---

### Task 2: Create pyproject.toml with Nuitka config

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

  Create file at project root with this exact content:

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

  Config rationale:
  - `enable-plugin = ["numpy", "tk-inter"]` — Nuitka built-in plugins that handle C extension wiring for numpy and tkinter; without these, DLL loading fails at runtime
  - `include-package = ["mediapipe", "cv2"]` — forces full inclusion; both packages use dynamic imports internally that Nuitka's static analysis cannot detect
  - `include-package-data = ["mediapipe", "cv2"]` — carries model files bundled inside mediapipe's legacy solutions API and OpenCV's haarcascade data; this replaces `collect_data_files()` from the old `.spec`
  - `windows-disable-console = true` — no black console window; mirrors the old `console=False`
  - `output-dir = "dist"` — output goes to `dist/app.dist/` (Nuitka appends `.dist` to the script name)

- [ ] **Step 2: Commit**

  ```bash
  git add pyproject.toml
  git commit -m "chore: add pyproject.toml with Nuitka build config"
  ```

---

### Task 3: Simplify build.bat and delete .spec

**Files:**
- Modify: `build.bat`
- Delete: `off_axis_tracker.spec`

- [ ] **Step 1: Rewrite build.bat**

  Replace the entire contents of `build.bat` with:

  ```bat
  @echo off
  echo Building Off-Axis Face Tracker...
  .venv\Scripts\python -m nuitka app.py
  echo.
  echo Done. Output: dist\app.dist\OffAxisTracker.exe
  pause
  ```

- [ ] **Step 2: Delete off_axis_tracker.spec**

  ```bash
  git rm off_axis_tracker.spec
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add build.bat
  git commit -m "chore: simplify build.bat for Nuitka, remove PyInstaller spec"
  ```

---

### Task 4: Local build verification

This task has no unit tests — it is a manual build smoke test. Run it before touching CI.

- [ ] **Step 1: Run the build**

  Double-click `build.bat`, or in terminal:

  ```bash
  cd "c:/Users/0/Desktop/my projects/off-axis-projection"
  .venv/Scripts/python -m nuitka app.py
  ```

  Expected: Nuitka prints compilation progress. First run takes several minutes (compiling C). Subsequent runs are faster due to caching. No errors at the end.

- [ ] **Step 2: Verify output structure**

  ```bash
  ls dist/app.dist/
  ```

  Expected: `OffAxisTracker.exe` exists, alongside DLL files and a `mediapipe/` subdirectory containing model files.

- [ ] **Step 3: Launch the exe and smoke test**

  Run `dist\app.dist\OffAxisTracker.exe`.

  Verify:
  - GUI window opens (no console window behind it)
  - Clicking Start begins face tracking without crashing
  - UDP data is sent (check with a UDP listener or OpenTrack)
  - Clicking Stop halts tracking cleanly

  If the exe crashes on startup, run with console enabled to see the error: temporarily set `windows-disable-console = false` in `pyproject.toml`, rebuild, read the traceback, then revert.

- [ ] **Step 4: Commit build artifacts note**

  No commit needed here — `dist/` is already in `.gitignore`.

---

### Task 5: Update CI workflow

**Files:**
- Modify: `.github/workflows/release.yml`

- [ ] **Step 1: Edit release.yml**

  Replace the current `Build exe` step:

  ```yaml
  - name: Build exe
    run: pyinstaller off_axis_tracker.spec --clean --noconfirm
  ```

  With:

  ```yaml
  - name: Build exe
    run: python -m nuitka app.py
  ```

  Also update the `Zip output` step path from `dist\OffAxisTracker\*` to `dist\app.dist\*`:

  ```yaml
  - name: Zip output
    run: Compress-Archive -Path dist\app.dist\* -DestinationPath OffAxisTracker.zip
  ```

  The full updated workflow for reference:

  ```yaml
  name: Release

  on:
    push:
      tags:
        - "v*"

  jobs:
    build:
      runs-on: windows-latest
      permissions:
        contents: write

      steps:
        - uses: actions/checkout@v4

        - name: Set up Python
          uses: actions/setup-python@v5
          with:
            python-version: "3.11"

        - name: Install dependencies
          run: pip install -r requirements.txt

        - name: Build exe
          run: python -m nuitka app.py

        - name: Zip output
          run: Compress-Archive -Path dist\app.dist\* -DestinationPath OffAxisTracker.zip

        - name: Create GitHub Release
          uses: softprops/action-gh-release@v2
          with:
            files: OffAxisTracker.zip
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add .github/workflows/release.yml
  git commit -m "ci: replace PyInstaller with Nuitka in release workflow"
  ```

---

### Task 6: Final check and PR

- [ ] **Step 1: Verify branch is clean**

  ```bash
  git status
  git log main..HEAD --oneline
  ```

  Expected: 5 commits ahead of main, working tree clean.

- [ ] **Step 2: Push branch**

  ```bash
  git push -u origin feat/nuitka-migration
  ```
