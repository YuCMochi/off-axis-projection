"""calibrator.py — Zhang's method camera calibration in a daemon thread."""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

import cv2
import numpy as np

import config as _config

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
        _config.CALIBRATION_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
