# Zhang Camera Calibration — Design Spec

**Date:** 2026-06-03  
**Status:** Approved

## Problem

Position tracking (X/Y/Z) is inaccurate because the current camera model assumes:
- `focal_length_px = 320.0` (manually guessed default)
- Zero distortion coefficients
- Principal point exactly at image center

These assumptions cause the depth formula `z = (real_eye_dist_cm × focal) / eye_dist_px` to produce wrong values, which propagates into X and Y as well.

## Goal

Add a Zhang's method (棋盤格) camera calibration wizard to the Settings window that produces a `calibration.json` with accurate intrinsics. The tracker uses these values automatically when the file exists.

---

## Architecture

### New file: `calibrator.py`

`Calibrator` class runs calibration in a daemon thread, mirrors the existing `FaceTracker` pattern.

**Public API:**
```python
Calibrator(cam_index, cam_width, cam_height,
           board_cols, board_rows,  # inner corner counts (not total squares)
           square_mm,
           on_progress, on_done)
calibrator.start()
calibrator.stop()
```

**Thread logic:**
1. Open camera with given index/size settings
2. Each frame: `cv2.findChessboardCorners` → `cv2.cornerSubPix` for sub-pixel accuracy
3. **Stability check:** if all corner displacements vs. previous frame < 2 px for 10 consecutive frames → auto-capture
4. **Cooldown:** 30 frames after each capture (lets user reposition board)
5. After 20 captures: `cv2.calibrateCamera` → write `calibration.json`
6. Progress updates via `on_progress(n: int, total: int)` and `on_done(result: dict | None)`
   — callbacks are called from the worker thread; `SettingsWindow` is responsible for marshalling to the Tk main thread via `root.after(0, ...)`

**Preview window (OpenCV):**
- Shows live camera feed
- Detected corners drawn with `cv2.drawChessboardCorners`
- Green progress bar overlay during stability countdown
- Brief white flash on capture
- "Q / ESC" closes preview without stopping calibration

### `calibration.json` format

Saved next to `config.json` (same `_APP_DIR` logic):

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "rms_error": 0.45,
  "image_size": [640, 480]
}
```

### `tracker.py` changes (minimal)

New module-level function:
```python
def load_calibration() -> dict | None
```
Reads `calibration.json`; returns `None` if file absent or malformed.

Changes inside `_tracking_loop`:
- Call `load_calibration()` once at startup
- If result present: use `np.array(result["camera_matrix"])` as `cam_mtx` and `np.array(result["dist_coeffs"])` as `dist_cfs`
- If absent: fallback to `get_cam_matrix(w, h, focal_length_px)` + `np.zeros((4,1))` (current behaviour)

Changes inside `_solve_pose`:
- Accept `dist` as a parameter instead of hardcoding `np.zeros((4,1))`

Changes inside `_estimate_position`:
- Accept optional `fx` parameter; use it instead of `cfg.focal_length_px` when calibration is loaded

---

## UI — New "相機校正 / Calibrate" Tab

Added as the third tab in `SettingsWindow`.

```
┌─────────────────────────────────────────┐
│  棋盤格內角數 Cols (橫)  [ 9 ] (spinbox) │
│  棋盤格內角數 Rows (縱)  [ 6 ] (spinbox) │
│  方格大小 Square (mm)    [30 ] (spinbox) │
│                                         │
│  [  開始校正 / Start  ]                 │
│                                         │
│  ● 進度：0 / 20 張已擷取                │
│  ● 狀態：等待棋盤格…                    │
│                                         │
│  上次校正結果：                          │
│    Focal X: —   Focal Y: —              │
│    重投影誤差: —  (< 1.0 px = 良好)     │
│  [  套用到 Focal Length 欄位  ]  (disabled until result ready)
└─────────────────────────────────────────┘
```

**"套用" button behaviour:**
- Writes `(fx + fy) / 2` into the `focal_length_px` slider on the Environment tab
- The calibration file is already saved at this point; this just syncs the slider for display

**"開始校正" toggles to "停止校正 / Stop"** while running.

---

## Files Changed

| File | Change |
|------|--------|
| `calibrator.py` | **New** — Calibrator class |
| `tracker.py` | Load calibration, pass dist to solvePnP, use calibrated fx for position |
| `settings_window.py` | Add third tab with calibration UI |
| `config.py` | No changes needed |

---

## Acceptance Criteria

1. Running calibration with a flat checkerboard produces `calibration.json` with RMS < 1.0 px
2. After calibration, Z depth matches real-world distance more accurately than before
3. If `calibration.json` is absent, tracker behaviour is identical to current behaviour
4. Calibration can be stopped mid-way without crashing the app
