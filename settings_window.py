"""settings_window.py — Settings Toplevel with two tabs for all Config params."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from config import Config, save_config


class SettingsWindow(tk.Toplevel):
    """Settings editor.  Calls on_apply(cfg) when user clicks Apply or Save."""

    def __init__(self, parent: tk.Misc, cfg: Config, on_apply: Callable[[Config], None]):
        super().__init__(parent)
        self.title("設定 / Settings")
        self.resizable(False, False)
        self.grab_set()   # modal
        self._on_apply = on_apply

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        env_frame = ttk.Frame(notebook)
        tune_frame = ttk.Frame(notebook)
        notebook.add(env_frame,  text="環境設定 / Environment")
        notebook.add(tune_frame, text="調效參數 / Tuning")

        self._vars: dict[str, tk.Variable] = {}
        self._build_env_tab(env_frame, cfg)
        self._build_tune_tab(tune_frame, cfg)

        # Buttons row
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Apply",      command=self._apply).pack(side="left",  padx=4)
        ttk.Button(btn_frame, text="Save",       command=self._save).pack(side="left",   padx=4)
        ttk.Button(btn_frame, text="Cancel",     command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Reset defaults", command=self._reset).pack(side="right", padx=4)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_env_tab(self, parent: ttk.Frame, cfg: Config) -> None:
        rows = [
            ("cam_index",        "Camera Index",       "int",   0,   9,    1,    cfg.cam_index),
            ("focal_length_px",  "Focal Length (px)",  "float", 100, 1000, 1,    cfg.focal_length_px),
            ("max_num_faces",    "Max Faces",          "int",   1,   10,   1,    cfg.max_num_faces),
            ("lock_snap_dist_px","Lock Snap Dist (px)","int",   30,  500,  10,   cfg.lock_snap_dist_px),
            ("cam_offset_x_cm", "Cam Offset X (cm)",  "float", -30, 30,   0.5,  cfg.cam_offset_x_cm),
            ("cam_offset_y_cm", "Cam Offset Y (cm)",  "float",  0,  60,   0.5,  cfg.cam_offset_y_cm),
            ("real_eye_dist_cm","Eye Distance (cm)",  "float",  4,  15,   0.5,  cfg.real_eye_dist_cm),
        ]
        for r, (key, label, kind, lo, hi, res, default) in enumerate(rows):
            self._add_slider_row(parent, r, key, label, kind, lo, hi, res, default)

        # UDP Host — text entry
        r = len(rows)
        ttk.Label(parent, text="UDP Host", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        var = tk.StringVar(value=cfg.udp_host)
        self._vars["udp_host"] = var
        ttk.Entry(parent, textvariable=var, width=18).grid(row=r, column=1, columnspan=2, sticky="w", padx=6)

        # UDP Port — spinbox
        r += 1
        ttk.Label(parent, text="UDP Port", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        var2 = tk.IntVar(value=cfg.udp_port)
        self._vars["udp_port"] = var2
        ttk.Spinbox(parent, from_=1024, to=65535, textvariable=var2, width=7).grid(
            row=r, column=1, sticky="w", padx=6)

    def _build_tune_tab(self, parent: ttk.Frame, cfg: Config) -> None:
        rows = [
            ("smooth_alpha", "Smooth Alpha",     "float", 0.01, 1.0,  0.01, cfg.smooth_alpha),
            ("deadzone_rot", "Deadzone Rot (°)", "float", 0.0,  10.0, 0.1,  cfg.deadzone_rot),
            ("deadzone_pos", "Deadzone Pos (cm)","float", 0.0,  5.0,  0.05, cfg.deadzone_pos),
            ("yaw_scale",   "Yaw Scale",         "float", 0.1,  5.0,  0.1,  cfg.yaw_scale),
            ("pitch_scale", "Pitch Scale",       "float", 0.1,  5.0,  0.1,  cfg.pitch_scale),
            ("roll_scale",  "Roll Scale",        "float", 0.1,  5.0,  0.1,  cfg.roll_scale),
            ("x_scale",     "X Scale",           "float", 0.1,  5.0,  0.1,  cfg.x_scale),
            ("y_scale",     "Y Scale",           "float", 0.1,  5.0,  0.1,  cfg.y_scale),
            ("z_scale",     "Z Scale",           "float", 0.1,  5.0,  0.1,  cfg.z_scale),
        ]
        for r, (key, label, kind, lo, hi, res, default) in enumerate(rows):
            self._add_slider_row(parent, r, key, label, kind, lo, hi, res, default)

    def _add_slider_row(self, parent, row, key, label, kind, lo, hi, res, default):
        ttk.Label(parent, text=label, width=20, anchor="e").grid(row=row, column=0, padx=6, pady=3)
        var = tk.DoubleVar(value=float(default)) if kind == "float" else tk.IntVar(value=int(default))
        self._vars[key] = var
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", length=220)
        scale.grid(row=row, column=1, padx=4, pady=3)
        val_lbl = ttk.Label(parent, text=f"{default}", width=8)
        val_lbl.grid(row=row, column=2, padx=4)

        def _update_label(*_):
            v = var.get()
            val_lbl.config(text=f"{v:.2f}" if kind == "float" else str(int(v)))
        var.trace_add("write", _update_label)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _collect(self) -> Config:
        v = self._vars
        return Config(
            cam_index         = int(v["cam_index"].get()),
            focal_length_px   = float(v["focal_length_px"].get()),
            max_num_faces     = int(v["max_num_faces"].get()),
            lock_snap_dist_px = int(v["lock_snap_dist_px"].get()),
            cam_offset_x_cm   = float(v["cam_offset_x_cm"].get()),
            cam_offset_y_cm   = float(v["cam_offset_y_cm"].get()),
            udp_host          = v["udp_host"].get().strip(),
            udp_port          = int(v["udp_port"].get()),
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

    def _apply(self) -> None:
        cfg = self._collect()
        self._on_apply(cfg)

    def _save(self) -> None:
        cfg = self._collect()
        save_config(cfg)
        self._on_apply(cfg)
        messagebox.showinfo("Saved", "設定已儲存 / Settings saved to config.json", parent=self)
        self.destroy()

    def _reset(self) -> None:
        defaults = Config()
        for key, var in self._vars.items():
            val = getattr(defaults, key)
            var.set(val)
