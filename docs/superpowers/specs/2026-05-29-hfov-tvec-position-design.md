# Design: HFOV Calibration + tvec-based Position

**Date:** 2026-05-29  
**Branch:** feat/hfov-tvec-position  
**Status:** Approved

## Problem

1. `focal_length_px` default (320 px) is far too low for real webcams (typical: 500–700 px at 640×480), causing Z output to be ~50% of actual distance.
2. `_estimate_position` recomputes position from two eye landmarks, discarding the more accurate `tvec` already produced by `solvePnP`. This method is noisy, drifts when the head rotates, and has no advantage over `tvec`.
3. `real_eye_dist_cm` and `focal_length_px` are unintuitive config fields — users must know camera intrinsics or measure with a tape.

## Solution

Replace the two problematic config fields with `cam_hfov_deg` (horizontal field of view in degrees, typically found on the webcam's spec page), compute focal length at runtime from HFOV + actual captured resolution, and use `tvec` from solvePnP directly for X/Y/Z position.

## Changes

### config.py

- **Remove:** `focal_length_px: float = 320.0`
- **Remove:** `real_eye_dist_cm: float = 9.0`
- **Add:** `cam_hfov_deg: float = 70.0`

`load_config` requires no changes — unrecognised keys are already filtered out, and new fields fall back to their defaults.

### tracker.py

**Focal computation** — in the existing `if cam_mtx is None` block, replace:
```python
cam_mtx = get_cam_matrix(w, h, cfg.focal_length_px)
```
with:
```python
focal_px = (w / 2.0) / math.tan(math.radians(cfg.cam_hfov_deg / 2.0))
cam_mtx = get_cam_matrix(w, h, focal_px)
```

**Position computation** — replace the `_estimate_position` call with:
```python
# tvec: solvePnP translation in camera coords, units = cm (same as FACE_MODEL_3D)
tx = float(tvec[0][0]) + cfg.cam_offset_x_cm   # right = positive
ty = -float(tvec[1][0]) + cfg.cam_offset_y_cm  # up = positive (flip camera Y)
tz = float(tvec[2][0])                          # depth from camera = positive
```

**Remove:** `_estimate_position` static method entirely.

Also update `_solve_pose` signature — it currently calls `get_cam_matrix(w, h, cfg.focal_length_px)` internally. Change the signature to accept a pre-computed `cam_mtx: np.ndarray` parameter and remove the internal `get_cam_matrix` call. The tracking loop already holds `cam_mtx`, so just pass it through.

### settings_window.py

Environment tab rows:
- **Remove** the `focal_length_px` slider row
- **Remove** the `real_eye_dist_cm` slider row
- **Add** `cam_hfov_deg` slider: range 30–120, step 0.5, label `"Camera HFOV (°)"`

`_collect()`:
- Remove `focal_length_px` and `real_eye_dist_cm` fields
- Add `cam_hfov_deg = float(v["cam_hfov_deg"].get())`

### tests/test_tracker_utils.py

Existing tests are unaffected (`get_cam_matrix` signature unchanged).

Add one new test:
```python
def test_hfov_to_focal_90deg():
    import math
    # 90° HFOV on 640-wide image → focal = 320.0 (tan 45° = 1)
    focal = (640 / 2.0) / math.tan(math.radians(90.0 / 2.0))
    assert focal == pytest.approx(320.0, rel=1e-6)
```

## How to Calibrate

1. Look up your webcam model's HFOV in its spec sheet (e.g., Logitech C270 = 60°, C920 = 78°).
2. Set `cam_hfov_deg` in Settings to that value.
3. To verify: stand at a known distance (e.g. arm's length ≈ 60 cm), check Z output.
4. If Z is off, adjust `cam_hfov_deg` proportionally — higher HFOV → smaller focal → smaller Z.

## Files Touched

| File | Change |
|------|--------|
| `config.py` | remove 2 fields, add 1 field |
| `tracker.py` | focal from HFOV, tvec for position, remove `_estimate_position` |
| `settings_window.py` | swap slider rows, update `_collect` |
| `tests/test_tracker_utils.py` | add 1 test |
