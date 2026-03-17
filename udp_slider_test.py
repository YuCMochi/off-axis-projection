"""
udp_slider_test.py
==================
六個拉桿 GUI → OpenTrack UDP 模擬輸出工具

用途：逐軸測試 UE5 裡每個軸的方向是否正確。
Usage: Test each axis one at a time in UE5 to check for flipped axes.

使用方式：
  python udp_slider_test.py              # 預設 127.0.0.1:4242
  python udp_slider_test.py --port 4243
  python udp_slider_test.py --host 192.168.1.10
"""

import tkinter as tk
import socket
import struct
import argparse
import traceback

# ── 常數 / Constants ──────────────────────────────────────────────────────────

POS_RANGE  = (-50, 50)     # X/Y 範圍（cm）
Z_RANGE    = (0, 150)      # Z 範圍（cm）
ROT_RANGE  = (-45, 45)     # Yaw/Pitch/Roll 範圍（度）

SEND_INTERVAL_MS = 33      # 發送間隔（毫秒），約 30 Hz（夠用且不會太頻繁）


def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    """打包 OpenTrack UDP 封包（48 bytes, 6 個 little-endian doubles）"""
    return struct.pack('<6d', x, y, z, yaw, pitch, roll)


class SliderApp:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.alive = True       # 視窗是否還活著

        # ── 建立視窗 ──
        self.root = tk.Tk()
        self.root.title(f"UDP Slider Test -> {host}:{port}")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── 用 DoubleVar 綁定拉桿值 ──
        self.vars = {}
        self.value_labels = {}

        slider_defs = [
            ("X",     "cm",  POS_RANGE[0], POS_RANGE[1], 0,  "#FF6B6B"),
            ("Y",     "cm",  POS_RANGE[0], POS_RANGE[1], 0,  "#51CF66"),
            ("Z",     "cm",  Z_RANGE[0],   Z_RANGE[1],   60, "#339AF0"),
            ("Yaw",   "deg", ROT_RANGE[0], ROT_RANGE[1], 0,  "#FF922B"),
            ("Pitch", "deg", ROT_RANGE[0], ROT_RANGE[1], 0,  "#CC5DE8"),
            ("Roll",  "deg", ROT_RANGE[0], ROT_RANGE[1], 0,  "#20C997"),
        ]

        for i, (name, unit, lo, hi, default, color) in enumerate(slider_defs):
            var = tk.DoubleVar(value=default)
            self.vars[name] = var

            # 軸名稱
            tk.Label(self.root, text=name, font=("Consolas", 14, "bold"),
                     fg=color, width=6, anchor="e"
            ).grid(row=i, column=0, padx=(10, 0), pady=6)

            # 拉桿
            tk.Scale(
                self.root, from_=lo, to=hi,
                orient=tk.HORIZONTAL, length=400,
                resolution=0.1, variable=var,
                font=("Consolas", 10), fg=color,
                troughcolor="#2B2B2B", highlightthickness=0,
                command=lambda val, n=name, u=unit: self._update_label(n, val, u)
            ).grid(row=i, column=1, padx=5, pady=6)

            # 數值顯示
            lbl = tk.Label(self.root, text=f"{default:+.1f} {unit}",
                           font=("Consolas", 11), width=12, anchor="w")
            lbl.grid(row=i, column=2, padx=(0, 10), pady=6)
            self.value_labels[name] = (lbl, unit)

        # 全部歸零按鈕
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=len(slider_defs), column=0, columnspan=3, pady=10)
        tk.Button(btn_frame, text="Reset All", font=("Consolas", 11),
                  command=self._reset_all, width=20).pack()

        # 狀態列
        tk.Label(self.root, text=f"-> {host}:{port}  |  ~{1000 // SEND_INTERVAL_MS} Hz",
                 font=("Consolas", 9), fg="gray"
        ).grid(row=len(slider_defs)+1, column=0, columnspan=3, pady=(0, 5))

        # ── 啟動定時發送 ──
        self._send_udp()

    def _update_label(self, name, val, unit):
        lbl, _ = self.value_labels[name]
        lbl.config(text=f"{float(val):+.1f} {unit}")

    def _reset_all(self):
        for name, var in self.vars.items():
            var.set(60.0 if name == "Z" else 0.0)
            lbl, unit = self.value_labels[name]
            lbl.config(text=f"{var.get():+.1f} {unit}")

    def _send_udp(self):
        """在主執行緒定時發送 UDP 封包，用 try/finally 確保 after() 永遠排程"""
        if not self.alive:
            return
        try:
            x     = self.vars["X"].get()
            y     = self.vars["Y"].get()
            z     = self.vars["Z"].get()
            yaw   = self.vars["Yaw"].get()
            pitch = self.vars["Pitch"].get()
            roll  = self.vars["Roll"].get()
            packet = pack_opentrack(x, y, z, yaw, pitch, roll)
            self.sock.sendto(packet, (self.host, self.port))
        except Exception as e:
            print(f"[WARN] UDP send error: {e}")
        finally:
            # 不管有沒有錯，都排程下一次（視窗還活著的話）
            if self.alive:
                self.root.after(SEND_INTERVAL_MS, self._send_udp)

    def _on_close(self):
        """關閉視窗"""
        self.alive = False
        try:
            self.sock.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='OpenTrack UDP Slider Simulator')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=4242, type=int)
    args = parser.parse_args()

    print(f"[INFO] Slider -> {args.host}:{args.port}")
    print("[INFO] Close window to quit")

    SliderApp(args.host, args.port).run()
    print("[INFO] Done")


if __name__ == "__main__":
    main()
