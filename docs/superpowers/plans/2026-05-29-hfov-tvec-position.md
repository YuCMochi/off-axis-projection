# HFOV Calibration + tvec-based Position Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inaccurate `focal_length_px`/`real_eye_dist_cm`/`_estimate_position` stack with HFOV-driven focal computation and direct tvec-based position output from solvePnP.

**Architecture:** `cam_hfov_deg` (degrees, user looks it up from webcam spec) replaces `focal_length_px` in Config; focal_px is computed at runtime from HFOV + actual captured resolution; `tvec` from `solvePnP` replaces the noisy 2-eye-landmark `_estimate_position` method.

**Tech Stack:** Python 3, OpenCV (`cv2.solvePnP`, `cv2.projectPoints`), NumPy, tkinter (settings UI), pytest

---

## File Map

| File | Change |
|------|--------|
| `config.py` | Remove `focal_length_px`, remove `real_eye_dist_cm`, add `cam_hfov_deg: float = 70.0` |
| `tracker.py` | `_solve_pose` accepts `cam_mtx` param; `if cam_mtx is None` block uses HFOV; tvec replaces `_estimate_position`; delete `_estimate_position` |
| `settings_window.py` | Remove two slider rows, add `cam_hfov_deg` slider, update `_collect()` |
| `tests/test_tracker_utils.py` | Add `test_config_has_cam_hfov_deg` and `test_hfov_to_focal_90deg` and `test_solve_pose_accepts_cam_mtx` |

---

## Task 1: Update Config — remove old fields, add cam_hfov_deg

**Files:**
- Modify: `config.py:24-45`
- Test: `tests/test_tracker_utils.py`

- [ ] **Step 1: Write two failing tests**

Add to `tests/test_tracker_utils.py`:

```python
def test_config_has_cam_hfov_deg():
    from config import Config
    cfg = Config()
    assert cfg.cam_hfov_deg == pytest.approx(70.0)
    assert not hasattr(cfg, "focal_length_px")
    assert not hasattr(cfg, "real_eye_dist_cm")


def test_hfov_to_focal_90deg():
    import math
    # 90° HFOV on 640-wide image → focal = 320.0 (tan 45° = 1.0)
    focal = (640 / 2.0) / math.tan(math.radians(90.0 / 2.0))
    assert focal == pytest.approx(320.0, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_tracker_utils.py::test_config_has_cam_hfov_deg tests/test_tracker_utils.py::test_hfov_to_focal_90deg -v
```

Expected: `test_config_has_cam_hfov_deg` FAILS (Config still has `focal_length_px`); `test_hfov_to_focal_90deg` PASSES (pure math, no project code).

- [ ] **Step 3: Update config.py**

Replace the `Config` dataclass body (lines 24–45 of `config.py`). The new version removes `focal_length_px` and `real_eye_dist_cm`, and adds `cam_hfov_deg`:

```python
@dataclass
class Config:
    # ── Environment Profile ──────────────────────────────────────────────────
    cam_index: int = 0
    cam_hfov_deg: float = 70.0
    max_num_faces: int = 5
    lock_snap_dist_px: int = 150
    cam_offset_x_cm: float = 0.0
    cam_offset_y_cm: float = 16.2
    # ── Output Protocol ──────────────────────────────────────────────────────
    output_protocol: str = "freed"       # "freed" | "opentrack"
    host: str = "127.0.0.1"
    port: int = 40000
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
```

- [ ] **Step 4: Run tests to verify both pass**

```
pytest tests/test_tracker_utils.py::test_config_has_cam_hfov_deg tests/test_tracker_utils.py::test_hfov_to_focal_90deg -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```
git add config.py tests/test_tracker_utils.py
git commit -m "feat: replace focal_length_px+real_eye_dist_cm with cam_hfov_deg in Config"
```

---

## Task 2: Update tracker.py — HFOV focal, _solve_pose signature, tvec position

**Files:**
- Modify: `tracker.py:259-296`, `tracker.py:374-403`
- Test: `tests/test_tracker_utils.py`

- [ ] **Step 1: Write a failing test for the new _solve_pose signature**

Add to `tests/test_tracker_utils.py`:

```python
def test_solve_pose_accepts_cam_mtx():
    import cv2
    import numpy as np
    from tracker import FaceTracker, get_cam_matrix, FACE_MODEL_3D

    cam = get_cam_matrix(640, 480, 800.0)
    dist = np.zeros((4, 1))
    # Project 3D face model to 2D using a known pose
    rvec0 = np.zeros((3, 1))
    tvec0 = np.array([[0.0], [0.0], [60.0]])
    img_pts, _ = cv2.projectPoints(FACE_MODEL_3D, rvec0, tvec0, cam, dist)
    result = FaceTracker._solve_pose(img_pts.reshape(-1, 2), cam, dist, None, None)
    assert result is not None
    _, tv = result
    assert float(tv[2][0]) == pytest.approx(60.0, rel=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_tracker_utils.py::test_solve_pose_accepts_cam_mtx -v
```

Expected: FAIL — `_solve_pose` currently takes `(image_pts, w, h, cfg, prev_rvec, prev_tvec)`, so the call with `cam` as second arg produces the wrong result or a TypeError.

- [ ] **Step 3: Update _solve_pose signature (tracker.py lines 374–388)**

Replace the entire `_solve_pose` static method:

```python
@staticmethod
def _solve_pose(image_pts, cam_mtx: np.ndarray, dist_cfs: np.ndarray, prev_rvec, prev_tvec):
    if prev_rvec is not None and prev_tvec is not None:
        ok, rv, tv = cv2.solvePnP(
            FACE_MODEL_3D, image_pts, cam_mtx, dist_cfs,
            rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rv, tv = cv2.solvePnP(
            FACE_MODEL_3D, image_pts, cam_mtx, dist_cfs, flags=cv2.SOLVEPNP_SQPNP
        )
    return (rv, tv) if ok else None
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_tracker_utils.py::test_solve_pose_accepts_cam_mtx -v
```

Expected: PASS.

- [ ] **Step 5: Update the cam_mtx initialization block (tracker.py line 259–260)**

Replace:
```python
if cam_mtx is None:
    cam_mtx = get_cam_matrix(w, h, cfg.focal_length_px)
```

With:
```python
if cam_mtx is None:
    focal_px = (w / 2.0) / math.tan(math.radians(cfg.cam_hfov_deg / 2.0))
    cam_mtx = get_cam_matrix(w, h, focal_px)
```

- [ ] **Step 6: Update the _solve_pose call site (tracker.py line 270)**

Replace:
```python
pnp = self._solve_pose(img_pts, w, h, cfg, prev_rvec, prev_tvec)
```

With:
```python
pnp = self._solve_pose(img_pts, cam_mtx, dist_cfs, prev_rvec, prev_tvec)
```

- [ ] **Step 7: Replace _estimate_position call with tvec extraction (tracker.py lines 292–296)**

Replace:
```python
pos = self._estimate_position(lm, w, h, cfg)
if pos:
    tx, ty, tz = pos[0] * cfg.x_scale, pos[1] * cfg.y_scale, pos[2] * cfg.z_scale
else:
    tx, ty, tz = 0.0, 0.0, 0.0
```

With:
```python
tx = (float(tvec[0][0]) + cfg.cam_offset_x_cm) * cfg.x_scale
ty = (-float(tvec[1][0]) + cfg.cam_offset_y_cm) * cfg.y_scale
tz = float(tvec[2][0]) * cfg.z_scale
```

- [ ] **Step 8: Delete the _estimate_position static method (tracker.py lines 390–403)**

Remove the entire method including its decorator:
```python
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
```

Also delete the now-unused constant `MIN_EYE_DIST_PX = 1.0` at line 101 (only referenced by `_estimate_position`).

- [ ] **Step 9: Run full test suite**

```
pytest -v
```

Expected: all tests PASS (including `test_solve_pose_accepts_cam_mtx`).

- [ ] **Step 10: Commit**

```
git add tracker.py tests/test_tracker_utils.py
git commit -m "feat: HFOV-based focal computation and tvec-based position in tracker"
```

---

## Task 3: Update settings_window.py — swap sliders, update _collect()

**Files:**
- Modify: `settings_window.py:44–52`, `settings_window.py:125–147`

*(No automated test for UI widget construction — verify manually by running the app and opening Settings.)*

- [ ] **Step 1: Replace the env tab rows list (settings_window.py lines 44–52)**

Replace:
```python
rows = [
    ("cam_index",        "Camera Index",       "int",   0,   9,    1,    cfg.cam_index),
    ("focal_length_px",  "Focal Length (px)",  "float", 100, 1000, 1,    cfg.focal_length_px),
    ("max_num_faces",    "Max Faces",          "int",   1,   10,   1,    cfg.max_num_faces),
    ("lock_snap_dist_px","Lock Snap Dist (px)","int",   30,  500,  10,   cfg.lock_snap_dist_px),
    ("cam_offset_x_cm", "Cam Offset X (cm)",  "float", -30, 30,   0.5,  cfg.cam_offset_x_cm),
    ("cam_offset_y_cm", "Cam Offset Y (cm)",  "float",  0,  60,   0.5,  cfg.cam_offset_y_cm),
    ("real_eye_dist_cm","Eye Distance (cm)",  "float",  4,  15,   0.5,  cfg.real_eye_dist_cm),
]
```

With:
```python
rows = [
    ("cam_index",        "Camera Index",       "int",   0,   9,    1,    cfg.cam_index),
    ("cam_hfov_deg",     "Camera HFOV (°)",    "float", 30,  120,  0.5,  cfg.cam_hfov_deg),
    ("max_num_faces",    "Max Faces",          "int",   1,   10,   1,    cfg.max_num_faces),
    ("lock_snap_dist_px","Lock Snap Dist (px)","int",   30,  500,  10,   cfg.lock_snap_dist_px),
    ("cam_offset_x_cm", "Cam Offset X (cm)",  "float", -30, 30,   0.5,  cfg.cam_offset_x_cm),
    ("cam_offset_y_cm", "Cam Offset Y (cm)",  "float",  0,  60,   0.5,  cfg.cam_offset_y_cm),
]
```

- [ ] **Step 2: Update _collect() (settings_window.py lines 125–147)**

Replace:
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

With:
```python
def _collect(self) -> Config:
    v = self._vars
    return Config(
        cam_index         = int(v["cam_index"].get()),
        cam_hfov_deg      = float(v["cam_hfov_deg"].get()),
        max_num_faces     = int(v["max_num_faces"].get()),
        lock_snap_dist_px = int(v["lock_snap_dist_px"].get()),
        cam_offset_x_cm   = float(v["cam_offset_x_cm"].get()),
        cam_offset_y_cm   = float(v["cam_offset_y_cm"].get()),
        output_protocol   = v["output_protocol"].get(),
        host              = v["host"].get().strip(),
        port              = int(v["port"].get()),
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

- [ ] **Step 3: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```
git add settings_window.py
git commit -m "feat: replace focal_length_px/real_eye_dist_cm sliders with cam_hfov_deg in Settings UI"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All four files in spec's file table are covered. All spec sections mapped to tasks.
- [x] **Placeholder scan:** No TBD, no "similar to Task N", all code blocks are complete.
- [x] **Type consistency:** `cam_mtx: np.ndarray` and `dist_cfs: np.ndarray` in `_solve_pose` signature match usage at call site (line 270, `dist_cfs` defined at line 247). `cam_hfov_deg` spelling consistent across config.py, tracker.py, settings_window.py, and tests.
- [x] **MIN_EYE_DIST_PX cleanup:** Spec doesn't mention it, but it's only referenced by `_estimate_position`. Included in Task 2 Step 8.
