# Zhang Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Zhang's method (棋盤格) calibration wizard to the Settings window that writes `calibration.json` with accurate camera intrinsics, fixing position tracking accuracy.

**Architecture:** New `calibrator.py` runs calibration in a daemon thread (mirrors `FaceTracker`). `tracker.py` loads `calibration.json` at startup and uses its `camera_matrix` / `dist_coeffs` instead of guessed defaults. `settings_window.py` gets a third "相機校正" tab with spinboxes, start/stop, progress and apply-to-focal-length button.

**Tech Stack:** Python 3.10+, OpenCV (`cv2.findChessboardCorners`, `cv2.calibrateCamera`), NumPy, tkinter/ttk, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `config.py` | Modify | Add `CALIBRATION_PATH` constant |
| `calibrator.py` | Create | `Calibrator` class — camera thread, corner detection, auto-capture, compute + write JSON |
| `tracker.py` | Modify | `load_calibration()`, refactor `_solve_pose` / `_estimate_position` signatures, wire in tracking loop |
| `settings_window.py` | Modify | Add third "相機校正 / Calibrate" tab |
| `tests/test_calibrator.py` | Create | Unit tests for `_is_stable`, `_save_calibration`, `load_calibration` |
| `tests/test_tracker_utils.py` | Modify | Add tests for refactored `_solve_pose` / `_estimate_position` |

---

## Task 1: Add `CALIBRATION_PATH` to `config.py`

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add the constant after `CONFIG_PATH`**

In `config.py`, after line `CONFIG_PATH = _APP_DIR / "config.json"`, add:

```python
CALIBRATION_PATH = _APP_DIR / "calibration.json"
```

- [ ] **Step 2: Verify the existing config tests still pass**

```
pytest tests/test_config.py -v
```

Expected: all green, no failures.

- [ ] **Step 3: Commit**

```
git add config.py
git commit -m "feat: add CALIBRATION_PATH to config"
```

---

## Task 2: `load_calibration()` in `tracker.py`

**Files:**
- Modify: `tracker.py`
- Create: `tests/test_calibrator.py` (first test only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibrator.py`:

```python
import json
import pytest


def test_load_calibration_missing_file(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "calibration.json")
    import tracker
    assert tracker.load_calibration() is None


def test_load_calibration_valid_file(tmp_path, monkeypatch):
    import config
    calib = {
        "camera_matrix": [[612.0, 0, 320.0], [0, 611.0, 240.0], [0, 0, 1]],
        "dist_coeffs": [0.1, -0.2, 0.0, 0.0, 0.05],
        "rms_error": 0.43,
        "image_size": [640, 480],
    }
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(calib), encoding="utf-8")
    monkeypatch.setattr(config, "CALIBRATION_PATH", p)
    import tracker
    result = tracker.load_calibration()
    assert result is not None
    assert result["rms_error"] == pytest.approx(0.43)
    assert result["camera_matrix"][0][0] == pytest.approx(612.0)


def test_load_calibration_malformed_file(tmp_path, monkeypatch):
    import config
    p = tmp_path / "calibration.json"
    p.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(config, "CALIBRATION_PATH", p)
    import tracker
    assert tracker.load_calibration() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_calibrator.py -v
```

Expected: FAIL — `tracker` has no attribute `load_calibration`.

- [ ] **Step 3: Add `load_calibration` to `tracker.py`**

Add this import at the top of `tracker.py`:

```python
from config import Config, CALIBRATION_PATH
import sender
```

(Replace the existing `from config import Config` line.)

Add this function after the `sample_2d` function (before `SmoothFilter`):

```python
def load_calibration() -> "dict | None":
    if not CALIBRATION_PATH.exists():
        return None
    try:
        import json
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        _ = data["camera_matrix"], data["dist_coeffs"], data["image_size"]
        return data
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_calibrator.py -v
```

Expected: all 3 tests green.

- [ ] **Step 5: Commit**

```
git add tracker.py tests/test_calibrator.py
git commit -m "feat: add load_calibration() to tracker"
```

---

## Task 3: Refactor `_solve_pose` and `_estimate_position` signatures

**Files:**
- Modify: `tracker.py`
- Modify: `tests/test_tracker_utils.py`

The current `_solve_pose` builds its own camera matrix internally (ignoring the `cam_mtx` in the loop). This task moves that responsibility to the caller so calibrated intrinsics can be passed in.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tracker_utils.py`:

```python
def test_solve_pose_accepts_cam_and_dist_params():
    from tracker import _solve_pose_with_cam, FACE_MODEL_3D, get_cam_matrix
    import numpy as np
    # Use a dummy set of 6 2D points that roughly correspond to the 3D model
    # viewed head-on; just verify the function signature accepts cam+dist.
    cam = get_cam_matrix(640, 480, 500.0)
    dist = np.zeros((4, 1))
    image_pts = np.array([
        [320.0, 240.0],
        [320.0, 300.0],
        [280.0, 220.0],
        [360.0, 220.0],
        [295.0, 245.0],
        [345.0, 245.0],
    ], dtype=np.float64)
    # Should not raise; ok may be False for degenerate input, that's fine
    result = _solve_pose_with_cam(image_pts, cam, dist, None, None)
    assert result is None or (len(result) == 2)


def test_estimate_position_uses_focal_px_param():
    from tracker import _estimate_position_with_focal
    from config import Config
    import numpy as np

    class _FakeLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    cfg = Config(real_eye_dist_cm=6.5, cam_offset_x_cm=0.0, cam_offset_y_cm=0.0)
    w, h = 640, 480
    # Eyes 100px apart, centred on frame
    landmarks = [None] * 478
    landmarks[33]  = _FakeLM(0.422, 0.5)   # left eye  ~ x=270
    landmarks[263] = _FakeLM(0.578, 0.5)   # right eye ~ x=370  → dist≈100px

    result = _estimate_position_with_focal(landmarks, w, h, cfg, focal_px=500.0)
    assert result is not None
    z_expected = (6.5 * 500.0) / 100.0   # = 32.5 cm
    assert result[2] == pytest.approx(z_expected, rel=0.05)
```

- [ ] **Step 2: Run to verify they fail**

```
pytest tests/test_tracker_utils.py::test_solve_pose_accepts_cam_and_dist_params tests/test_tracker_utils.py::test_estimate_position_uses_focal_px_param -v
```

Expected: FAIL — functions not found.

- [ ] **Step 3: Add `_solve_pose_with_cam` to `tracker.py`**

Add a new **module-level** function (keep the old `_solve_pose` static method untouched for now):

```python
def _solve_pose_with_cam(image_pts, cam: np.ndarray, dist: np.ndarray,
                          prev_rvec, prev_tvec):
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
```

- [ ] **Step 4: Add `_estimate_position_with_focal` to `tracker.py`**

Add another module-level function:

```python
def _estimate_position_with_focal(landmarks, w: int, h: int,
                                   cfg: Config, focal_px: float):
    lx = landmarks[LM_LEFT_EYE].x * w;  ly = landmarks[LM_LEFT_EYE].y * h
    rx = landmarks[LM_RIGHT_EYE].x * w; ry = landmarks[LM_RIGHT_EYE].y * h
    eye_dist_px = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
    if eye_dist_px < MIN_EYE_DIST_PX:
        return None
    z_cm = (cfg.real_eye_dist_cm * focal_px) / eye_dist_px
    cx_px = (landmarks[LM_LEFT_EYE].x + landmarks[LM_RIGHT_EYE].x) / 2.0 * w
    cy_px = (landmarks[LM_LEFT_EYE].y + landmarks[LM_RIGHT_EYE].y) / 2.0 * h
    x_cm = (cx_px - w / 2.0) * z_cm / focal_px + cfg.cam_offset_x_cm
    y_cm = -((cy_px - h / 2.0) * z_cm / focal_px) + cfg.cam_offset_y_cm
    return x_cm, y_cm, z_cm
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_tracker_utils.py -v
```

Expected: all green (including old tests).

- [ ] **Step 6: Commit**

```
git add tracker.py tests/test_tracker_utils.py
git commit -m "feat: add _solve_pose_with_cam and _estimate_position_with_focal"
```

---

## Task 4: Wire calibration into `_tracking_loop`

**Files:**
- Modify: `tracker.py`

- [ ] **Step 1: Replace `_solve_pose` and `_estimate_position` call sites in `_tracking_loop`**

In `_tracking_loop`, **replace** the section starting from `if cam_mtx is None:` through the rest of the pose/position logic. The full diff:

**Remove** (around line 269):
```python
                if cam_mtx is None:
                    cam_mtx = get_cam_matrix(w, h, cfg.focal_length_px)
```

**Replace** with (place this block right after `cap = cv2.VideoCapture(...)` setup, before the `while` loop):

```python
        calib = load_calibration()
        if calib:
            cam_mtx = np.array(calib["camera_matrix"], dtype=np.float64)
            dist_cfs = np.array(calib["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
            focal_px = calib["camera_matrix"][0][0]
            print(f"[tracker] calibration loaded — fx={focal_px:.1f}  rms={calib['rms_error']}")
        else:
            cam_mtx = get_cam_matrix(
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cfg.focal_length_px,
            )
            dist_cfs = np.zeros((4, 1))
            focal_px = cfg.focal_length_px
```

- [ ] **Step 2: Replace `_solve_pose` call**

Find this call in `_tracking_loop`:
```python
                    pnp = self._solve_pose(img_pts, w, h, cfg, prev_rvec, prev_tvec)
```

Replace with:
```python
                    pnp = _solve_pose_with_cam(img_pts, cam_mtx, dist_cfs, prev_rvec, prev_tvec)
```

- [ ] **Step 3: Replace `_estimate_position` call**

Find:
```python
                        pos = self._estimate_position(lm, w, h, cfg)
```

Replace with:
```python
                        pos = _estimate_position_with_focal(lm, w, h, cfg, focal_px)
```

- [ ] **Step 4: Run full test suite**

```
pytest tests/ -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add tracker.py
git commit -m "feat: use calibrated intrinsics in tracking loop"
```

---

## Task 5: Create `calibrator.py`

**Files:**
- Create: `calibrator.py`
- Modify: `tests/test_calibrator.py`

- [ ] **Step 1: Write tests for `_is_stable` and `_save_calibration`**

Append to `tests/test_calibrator.py`:

```python
def test_is_stable_true_when_corners_identical():
    import numpy as np
    from calibrator import Calibrator
    corners = np.random.rand(54, 1, 2).astype(np.float32)
    assert Calibrator._is_stable(corners, corners.copy()) is True


def test_is_stable_false_when_corners_move():
    import numpy as np
    from calibrator import Calibrator
    prev = np.zeros((54, 1, 2), dtype=np.float32)
    curr = prev.copy()
    curr[0, 0, 0] = 5.0  # one corner moved 5px
    assert Calibrator._is_stable(prev, curr) is False


def test_save_calibration_writes_json(tmp_path, monkeypatch):
    import json, numpy as np
    import config
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "calibration.json")
    from calibrator import Calibrator
    cam_mtx = np.array([[612.0, 0, 320.0], [0, 611.0, 240.0], [0, 0, 1.0]])
    dist = np.array([[0.1, -0.2, 0.0, 0.0, 0.05]])
    result = Calibrator._save_calibration(cam_mtx, dist, 0.43, (640, 480))
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["rms_error"] == pytest.approx(0.43)
    assert saved["camera_matrix"][0][0] == pytest.approx(612.0)
    assert len(saved["dist_coeffs"]) == 5
    assert result["image_size"] == [640, 480]
```

- [ ] **Step 2: Run to verify they fail**

```
pytest tests/test_calibrator.py::test_is_stable_true_when_corners_identical tests/test_calibrator.py::test_is_stable_false_when_corners_move tests/test_calibrator.py::test_save_calibration_writes_json -v
```

Expected: FAIL — `calibrator` module not found.

- [ ] **Step 3: Create `calibrator.py`**

```python
"""calibrator.py — Zhang's method camera calibration in a daemon thread."""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

import cv2
import numpy as np

from config import CALIBRATION_PATH

TARGET_FRAMES = 20
STABILITY_FRAMES = 10
STABILITY_MAX_MOVE_PX = 2.0
COOLDOWN_FRAMES = 30


class Calibrator:
    def __init__(
        self,
        cam_index: int,
        cam_width: int,
        cam_height: int,
        board_cols: int,   # inner corner count, horizontal
        board_rows: int,   # inner corner count, vertical
        square_mm: float,
        on_progress: Callable[[int, int], None],
        on_done: Callable[[Optional[dict]], None],
    ):
        self.cam_index = cam_index
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.board_cols = board_cols
        self.board_rows = board_rows
        self.square_mm = square_mm
        self.on_progress = on_progress
        self.on_done = on_done
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None

    @staticmethod
    def _is_stable(prev: np.ndarray, curr: np.ndarray) -> bool:
        return float(np.max(np.linalg.norm(curr - prev, axis=2))) < STABILITY_MAX_MOVE_PX

    @staticmethod
    def _save_calibration(
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rms: float,
        image_size: tuple,
    ) -> dict:
        result = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.flatten().tolist(),
            "rms_error": round(float(rms), 4),
            "image_size": list(image_size),
        }
        CALIBRATION_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _run(self) -> None:
        board_size = (self.board_cols, self.board_rows)
        objp = np.zeros((self.board_cols * self.board_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_cols, 0:self.board_rows].T.reshape(-1, 2)
        objp *= self.square_mm
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        obj_points: list = []
        img_points: list = []

        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.on_done(None)
            return
        if self.cam_width > 0 and self.cam_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        stable_count = 0
        cooldown = 0
        prev_corners: Optional[np.ndarray] = None
        captured = 0
        image_size = (640, 480)
        win = "Camera Calibration  (Q/ESC to close preview)"

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                image_size = (w, h)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, board_size, None)

                if found:
                    corners_ref = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    if cooldown > 0:
                        cooldown -= 1
                        status = f"冷卻中 / Cooldown  ({cooldown})"
                    else:
                        if prev_corners is not None and self._is_stable(prev_corners, corners_ref):
                            stable_count += 1
                        else:
                            stable_count = 0

                        if stable_count >= STABILITY_FRAMES:
                            obj_points.append(objp)
                            img_points.append(corners_ref)
                            captured += 1
                            self.on_progress(captured, TARGET_FRAMES)
                            stable_count = 0
                            cooldown = COOLDOWN_FRAMES
                            prev_corners = None
                            # White flash
                            cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), -1)
                            if captured >= TARGET_FRAMES:
                                cv2.imshow(win, frame)
                                cv2.waitKey(100)
                                break
                            status = f"擷取 {captured}/{TARGET_FRAMES} ✓"
                        else:
                            bar_w = int(w * stable_count / STABILITY_FRAMES)
                            cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 255, 0), -1)
                            status = f"穩定中 {stable_count}/{STABILITY_FRAMES}"
                        prev_corners = corners_ref

                    cv2.drawChessboardCorners(frame, board_size, corners_ref, found)
                else:
                    prev_corners = None
                    stable_count = 0
                    status = "等待棋盤格 / Looking for board"

                cv2.putText(
                    frame, f"{captured}/{TARGET_FRAMES}  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2,
                )
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    cv2.destroyWindow(win)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not self._stop_event.is_set() and captured >= TARGET_FRAMES:
            rms, cam_mtx, dist_cfs, _, _ = cv2.calibrateCamera(
                obj_points, img_points, image_size, None, None
            )
            result = self._save_calibration(cam_mtx, dist_cfs, rms, image_size)
            self.on_done(result)
        else:
            self.on_done(None)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_calibrator.py -v
```

Expected: all green.

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add calibrator.py tests/test_calibrator.py
git commit -m "feat: add Calibrator class with Zhang's method"
```

---

## Task 6: Add calibration tab to `settings_window.py`

**Files:**
- Modify: `settings_window.py`

- [ ] **Step 1: Add the third tab to the notebook in `__init__`**

In `SettingsWindow.__init__`, after the two existing `notebook.add(...)` calls, add:

```python
        calib_frame = ttk.Frame(notebook)
        notebook.add(calib_frame, text="相機校正 / Calibrate")
        self._calibrator = None
        self._calib_result: dict | None = None
        self._build_calib_tab(calib_frame, cfg)
```

- [ ] **Step 2: Add `_build_calib_tab` method**

Add this method to `SettingsWindow` after `_build_tune_tab`:

```python
    def _build_calib_tab(self, parent: ttk.Frame, cfg: Config) -> None:
        ttk.Label(parent, text="棋盤格內角 Cols", width=22, anchor="e").grid(
            row=0, column=0, padx=6, pady=4)
        self._calib_cols = tk.IntVar(value=9)
        ttk.Spinbox(parent, from_=3, to=20, textvariable=self._calib_cols, width=6).grid(
            row=0, column=1, sticky="w", padx=6)

        ttk.Label(parent, text="棋盤格內角 Rows", width=22, anchor="e").grid(
            row=1, column=0, padx=6, pady=4)
        self._calib_rows = tk.IntVar(value=6)
        ttk.Spinbox(parent, from_=3, to=20, textvariable=self._calib_rows, width=6).grid(
            row=1, column=1, sticky="w", padx=6)

        ttk.Label(parent, text="方格大小 Square (mm)", width=22, anchor="e").grid(
            row=2, column=0, padx=6, pady=4)
        self._calib_sq_mm = tk.DoubleVar(value=30.0)
        ttk.Spinbox(parent, from_=5.0, to=200.0, increment=5.0,
                    textvariable=self._calib_sq_mm, width=6).grid(
            row=2, column=1, sticky="w", padx=6)

        ttk.Separator(parent, orient="horizontal").grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=6)

        self._calib_btn = ttk.Button(
            parent, text="開始校正 / Start", command=self._toggle_calibration)
        self._calib_btn.grid(row=4, column=0, columnspan=2, pady=6)

        self._calib_progress_var = tk.StringVar(value="進度：0 / 20 張已擷取")
        ttk.Label(parent, textvariable=self._calib_progress_var).grid(
            row=5, column=0, columnspan=3, padx=6, pady=2)

        self._calib_status_var = tk.StringVar(value="狀態：就緒")
        ttk.Label(parent, textvariable=self._calib_status_var).grid(
            row=6, column=0, columnspan=3, padx=6, pady=2)

        ttk.Separator(parent, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="ew", padx=6, pady=6)

        ttk.Label(parent, text="上次校正結果：", anchor="w").grid(
            row=8, column=0, columnspan=3, sticky="w", padx=6)
        self._calib_result_var = tk.StringVar(
            value="  Focal X: —   Focal Y: —\n  重投影誤差: — px")
        ttk.Label(parent, textvariable=self._calib_result_var, justify="left").grid(
            row=9, column=0, columnspan=3, sticky="w", padx=12, pady=2)

        self._calib_apply_btn = ttk.Button(
            parent, text="套用到 Focal Length 欄位",
            command=self._apply_calibration, state="disabled")
        self._calib_apply_btn.grid(row=10, column=0, columnspan=2, pady=6)

        self._load_existing_calibration()
```

- [ ] **Step 3: Add helper methods for calibration actions**

Add these methods to `SettingsWindow`:

```python
    def _toggle_calibration(self) -> None:
        from calibrator import Calibrator
        if self._calibrator and self._calibrator.running:
            self._calibrator.stop()
            self._calibrator = None
            self._calib_btn.config(text="開始校正 / Start")
            self._calib_status_var.set("狀態：已停止")
            return
        try:
            cfg_snap = self._collect()
        except (ValueError, tk.TclError):
            cfg_snap = None
        cam_index = cfg_snap.cam_index if cfg_snap else 0
        cam_w = cfg_snap.cam_width if cfg_snap else 0
        cam_h = cfg_snap.cam_height if cfg_snap else 0
        self._calibrator = Calibrator(
            cam_index=cam_index,
            cam_width=cam_w,
            cam_height=cam_h,
            board_cols=self._calib_cols.get(),
            board_rows=self._calib_rows.get(),
            square_mm=self._calib_sq_mm.get(),
            on_progress=self._on_calib_progress,
            on_done=self._on_calib_done,
        )
        self._calibrator.start()
        self._calib_btn.config(text="停止校正 / Stop")
        self._calib_progress_var.set("進度：0 / 20 張已擷取")
        self._calib_status_var.set("狀態：校正中…")

    def _on_calib_progress(self, n: int, total: int) -> None:
        self.after(0, lambda: self._calib_progress_var.set(
            f"進度：{n} / {total} 張已擷取"))

    def _on_calib_done(self, result: "dict | None") -> None:
        def _update():
            self._calibrator = None
            self._calib_btn.config(text="開始校正 / Start")
            if result:
                self._calib_result = result
                fx = result["camera_matrix"][0][0]
                fy = result["camera_matrix"][1][1]
                rms = result["rms_error"]
                quality = "✓ 良好" if rms < 1.0 else "⚠ 偏高，建議重校"
                self._calib_result_var.set(
                    f"  Focal X: {fx:.1f} px   Focal Y: {fy:.1f} px\n"
                    f"  重投影誤差: {rms:.3f} px  {quality}"
                )
                self._calib_apply_btn.config(state="normal")
                self._calib_status_var.set("狀態：校正完成 ✓")
            else:
                self._calib_status_var.set("狀態：已取消或失敗")
        self.after(0, _update)

    def _apply_calibration(self) -> None:
        if not self._calib_result:
            return
        fx = self._calib_result["camera_matrix"][0][0]
        fy = self._calib_result["camera_matrix"][1][1]
        self._vars["focal_length_px"].set((fx + fy) / 2.0)

    def _load_existing_calibration(self) -> None:
        from config import CALIBRATION_PATH
        if not CALIBRATION_PATH.exists():
            return
        try:
            import json
            data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            fx = data["camera_matrix"][0][0]
            fy = data["camera_matrix"][1][1]
            rms = data["rms_error"]
            self._calib_result = data
            quality = "✓ 良好" if rms < 1.0 else "⚠ 偏高，建議重校"
            self._calib_result_var.set(
                f"  Focal X: {fx:.1f} px   Focal Y: {fy:.1f} px\n"
                f"  重投影誤差: {rms:.3f} px  {quality}"
            )
            self._calib_apply_btn.config(state="normal")
            self._calib_status_var.set("狀態：已載入上次校正結果")
        except Exception:
            pass
```

- [ ] **Step 4: Run the full test suite**

```
pytest tests/ -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add settings_window.py
git commit -m "feat: add calibration tab to settings window"
```

---

## Task 7: Final integration check

- [ ] **Step 1: Run the app and open Settings**

```
python app.py
```

Open Settings → confirm three tabs appear: 環境設定, 調效參數, 相機校正.

- [ ] **Step 2: Confirm fallback still works**

Delete (or rename) `calibration.json` if it exists. Start tracking — confirm it behaves exactly as before (no crash, uses `focal_length_px` from config).

- [ ] **Step 3: Run calibration with a real checkerboard**

Open 相機校正 tab → set Cols/Rows/mm to match your board → click 開始校正. Hold the board at various angles in front of the camera. Confirm:
- Corners are highlighted in the preview window
- Progress bar counts up automatically
- After 20 frames the preview closes and the result (Focal X/Y + RMS) appears
- RMS is ideally < 1.0 px

- [ ] **Step 4: Apply and verify position improvement**

Click 套用到 Focal Length. Close settings with Save. Start tracking. Confirm Z depth is now more accurate vs. a tape measure.

- [ ] **Step 5: Commit final state**

```
git add -A
git commit -m "feat: Zhang calibration — integration verified"
```
