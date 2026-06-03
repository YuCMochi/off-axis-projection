"""settings_window.py — Settings Toplevel with two tabs for all Config params."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from config import Config, save_config, DEFAULT_HOST, DEFAULT_PORTS


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
        calib_frame = ttk.Frame(notebook)
        notebook.add(calib_frame, text="相機校正 / Calibrate")
        self._calibrator = None
        self._calib_result: dict | None = None

        self._vars: dict[str, tk.Variable] = {}
        self._build_env_tab(env_frame, cfg)
        self._build_tune_tab(tune_frame, cfg)
        self._build_calib_tab(calib_frame, cfg)

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
            ("cam_index",        "Camera Index",        "int",   0,   9,    1,    cfg.cam_index),
            ("cam_width",        "Capture W (px,0=auto)","int",   0,   3840, 160,  cfg.cam_width),
            ("cam_height",       "Capture H (px,0=auto)","int",  0,   2160, 90,   cfg.cam_height),
            ("cam_fps",          "Capture FPS (0=auto)", "int",  0,   240,  1,    cfg.cam_fps),
            ("focal_length_px",  "Focal Length (px)",   "float", 100, 1000, 1,    cfg.focal_length_px),
            ("max_num_faces",    "Max Faces",           "int",   1,   10,   1,    cfg.max_num_faces),
            ("lock_snap_dist_px","Lock Snap Dist (px)", "int",   30,  500,  10,   cfg.lock_snap_dist_px),
            ("cam_offset_x_cm", "Cam Offset X (cm)",   "float", -30, 30,   0.5,  cfg.cam_offset_x_cm),
            ("cam_offset_y_cm", "Cam Offset Y (cm)",   "float",  0,  60,   0.5,  cfg.cam_offset_y_cm),
            ("real_eye_dist_cm","Eye Distance (cm)",   "float",  4,  15,   0.5,  cfg.real_eye_dist_cm),
        ]
        for r, (key, label, kind, lo, hi, res, default) in enumerate(rows):
            self._add_slider_row(parent, r, key, label, kind, lo, hi, res, default)

        # Host — text entry
        r = len(rows)
        ttk.Label(parent, text="Host", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        host_var = tk.StringVar(value=cfg.host)
        self._vars["host"] = host_var
        ttk.Entry(parent, textvariable=host_var, width=18).grid(
            row=r, column=1, columnspan=2, sticky="w", padx=6)

        # Port — spinbox
        r += 1
        ttk.Label(parent, text="Port", width=20, anchor="e").grid(row=r, column=0, padx=6, pady=4)
        port_var = tk.IntVar(value=cfg.port)
        self._vars["port"] = port_var
        ttk.Spinbox(parent, from_=1024, to=65535, textvariable=port_var, width=7).grid(
            row=r, column=1, sticky="w", padx=6)

        # Output Protocol — dropdown
        r += 1
        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=6)

        r += 1
        ttk.Label(parent, text="Output Protocol", width=20, anchor="e").grid(
            row=r, column=0, padx=6, pady=4)
        proto_var = tk.StringVar(value=cfg.output_protocol)
        self._vars["output_protocol"] = proto_var
        ttk.Combobox(
            parent, textvariable=proto_var,
            values=["freed", "opentrack"],
            state="readonly", width=15,
        ).grid(row=r, column=1, sticky="w", padx=6)

        def _on_protocol_change(*_):
            host_var.set(DEFAULT_HOST)
            port_var.set(DEFAULT_PORTS[proto_var.get()])

        proto_var.trace_add("write", _on_protocol_change)

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
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal", length=160)
        scale.grid(row=row, column=1, padx=4, pady=3)

        fmt = (lambda v: f"{v:.2f}") if kind == "float" else (lambda v: str(int(round(v))))
        str_var = tk.StringVar(value=fmt(float(default)))
        entry = ttk.Entry(parent, textvariable=str_var, width=8)
        entry.grid(row=row, column=2, padx=4, pady=3)

        _lock = [False]

        def _slider_to_entry(*_):
            if _lock[0]:
                return
            _lock[0] = True
            str_var.set(fmt(var.get()))
            _lock[0] = False

        def _entry_to_slider(*_):
            if _lock[0]:
                return
            try:
                v = float(str_var.get())
                v = max(float(lo), min(float(hi), v))
                _lock[0] = True
                var.set(int(round(v)) if kind == "int" else v)
                _lock[0] = False
            except ValueError:
                pass

        var.trace_add("write", _slider_to_entry)
        str_var.trace_add("write", _entry_to_slider)
        entry.bind("<FocusOut>", lambda *_: str_var.set(fmt(var.get())))

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

    # ── Actions ───────────────────────────────────────────────────────────────

    def _collect(self) -> Config:
        v = self._vars
        return Config(
            cam_index         = int(v["cam_index"].get()),
            cam_width         = int(v["cam_width"].get()),
            cam_height        = int(v["cam_height"].get()),
            cam_fps           = int(v["cam_fps"].get()),
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

    def _apply(self) -> None:
        try:
            cfg = self._collect()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid input", f"請確認輸入值正確 / Invalid value:\n{exc}", parent=self)
            return
        self._on_apply(cfg)

    def _save(self) -> None:
        try:
            cfg = self._collect()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid input", f"請確認輸入值正確 / Invalid value:\n{exc}", parent=self)
            return
        save_config(cfg)
        self._on_apply(cfg)
        messagebox.showinfo("Saved", "設定已儲存 / Settings saved to config.json", parent=self)
        self.destroy()

    def _reset(self) -> None:
        defaults = Config()
        for key, var in self._vars.items():
            val = getattr(defaults, key)
            var.set(val)
