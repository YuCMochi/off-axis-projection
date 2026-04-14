"""app.py — Entry point + MainWindow control panel."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from config import Config, load_config, save_config
from tracker import FaceTracker
from settings_window import SettingsWindow

POLL_MS = 150   # how often MainWindow refreshes HUD from tracker.live


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Off-Axis Face Tracker")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cfg = load_config()
        self._tracker = FaceTracker(self._cfg)
        self._settings_win: SettingsWindow | None = None

        self._build_ui()
        self._poll()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # ── Status row ──────────────────────────────────────────────────────
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", **pad)

        self._status_dot = tk.Label(status_frame, text="●", fg="#555", font=("Consolas", 16))
        self._status_dot.pack(side="left")
        self._status_lbl = ttk.Label(status_frame, text="待機 / Idle", font=("Consolas", 11))
        self._status_lbl.pack(side="left", padx=6)
        self._cam_lbl = ttk.Label(status_frame,
                                   text=f"Cam #{self._cfg.cam_index}  ->  {self._cfg.udp_host}:{self._cfg.udp_port}",
                                   font=("Consolas", 9), foreground="#888")
        self._cam_lbl.pack(side="right")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)

        # ── HUD labels ───────────────────────────────────────────────────────
        hud_frame = ttk.Frame(self.root)
        hud_frame.pack(fill="x", **pad)

        self._hud_rot = ttk.Label(hud_frame,
                                   text="Yaw:  +0.0   Pitch:  +0.0   Roll:  +0.0 deg",
                                   font=("Consolas", 10), foreground="#4fc")
        self._hud_rot.pack(anchor="w")
        self._hud_pos = ttk.Label(hud_frame,
                                   text="X: +0.0   Y: +0.0   Z: +0.0  cm",
                                   font=("Consolas", 10), foreground="#4cf")
        self._hud_pos.pack(anchor="w")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)

        self._start_btn = ttk.Button(btn_frame, text="Start", width=18,
                                      command=self._toggle_tracker)
        self._start_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Settings", width=18,
                   command=self._open_settings).pack(side="left")

        # ── Error label (hidden by default) ──────────────────────────────────
        self._err_lbl = ttk.Label(self.root, text="", foreground="red",
                                   font=("Consolas", 9), wraplength=300)
        self._err_lbl.pack(padx=10, pady=(0, 4))

    # ── Tracker control ───────────────────────────────────────────────────────

    def _toggle_tracker(self) -> None:
        if self._tracker.running:
            self._tracker.stop()
            self._start_btn.config(text="Start")
        else:
            self._tracker = FaceTracker(self._cfg)
            self._tracker.start(preview=True)
            self._start_btn.config(text="Stop")

    def _open_settings(self) -> None:
        if self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return
        self._settings_win = SettingsWindow(
            self.root, self._cfg, on_apply=self._on_settings_apply
        )

    def _on_settings_apply(self, new_cfg: Config) -> None:
        self._cfg = new_cfg
        # Restart tracker so cam_index / max_num_faces / udp changes take effect
        was_running = self._tracker.running
        if was_running:
            self._tracker.stop()
        self._tracker = FaceTracker(self._cfg)
        if was_running:
            self._tracker.start(preview=True)
        # Update info label
        self._cam_lbl.config(
            text=f"Cam #{self._cfg.cam_index}  ->  {self._cfg.udp_host}:{self._cfg.udp_port}"
        )

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        live = self._tracker.live
        tracking: bool = live["tracking"]
        error: str | None = live["error"]

        if error:
            self._status_dot.config(fg="#f55")
            self._status_lbl.config(text="Error")
            self._err_lbl.config(text=error)
        elif tracking:
            self._status_dot.config(fg="#4f4")
            self._status_lbl.config(text="Tracking")
            self._err_lbl.config(text="")
            self._hud_rot.config(
                text=f"Yaw: {live['yaw']:+6.1f}   Pitch: {live['pitch']:+6.1f}   Roll: {live['roll']:+6.1f} deg"
            )
            self._hud_pos.config(
                text=f"X: {live['x']:+6.1f}   Y: {live['y']:+6.1f}   Z: {live['z']:+6.1f}  cm"
            )
        elif self._tracker.running:
            self._status_dot.config(fg="#fa0")
            self._status_lbl.config(text="Detecting...")
            self._err_lbl.config(text="")
        else:
            self._status_dot.config(fg="#555")
            self._status_lbl.config(text="Idle")
            self._err_lbl.config(text="")

        self.root.after(POLL_MS, self._poll)

    def _on_close(self) -> None:
        if self._tracker.running:
            self._tracker.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
