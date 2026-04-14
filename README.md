# Off-Axis Projection — Face Tracker

用臉部追蹤實現 Off-Axis（偏軸）投影效果的工具。
透過 UDP 將臉部座標傳送給其他程式（例如遊戲引擎）來即時調整畫面透視。

[![GitHub release](https://img.shields.io/github/v/release/YuCMochi/off-axis-projection)](https://github.com/YuCMochi/off-axis-projection/releases/latest)
[![Download](https://img.shields.io/badge/Download-OffAxisTracker.zip-blue)](https://github.com/YuCMochi/off-axis-projection/releases/latest/download/OffAxisTracker.zip)

---

## 下載

直接點連結下載最新版：

**[OffAxisTracker.zip — 最新版本](https://github.com/YuCMochi/off-axis-projection/releases/latest/download/OffAxisTracker.zip)**

解壓後執行 `OffAxisTracker.exe`，不需要安裝 Python。

---

## 使用方式

1. 解壓 `OffAxisTracker.zip`
2. 執行 `OffAxisTracker.exe`
3. 點 **Start 開始** 開始追蹤
4. 在 **Settings 設定** 調整攝像頭、UDP 目標、平滑參數等
5. 其他程式（遊戲引擎等）透過 UDP `127.0.0.1:4242` 接收 OpenTrack 格式的座標

---

## UDP 格式（OpenTrack）

48 bytes，6 個 little-endian double：`[X, Y, Z, Yaw, Pitch, Roll]`

---

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `app.py` | 主程式入口，tkinter GUI |
| `tracker.py` | 臉部追蹤核心（mediapipe + OpenCV），daemon thread |
| `config.py` | 設定檔讀寫（config.json） |
| `settings_window.py` | 設定視窗 UI |
| `off_axis_tracker.spec` | PyInstaller 打包設定 |

---

### Python 版本
```
Python 3.11
```

### 安裝步驟

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 執行（開發模式）

```bash
python app.py
```

### 本地打包

```bash
build.bat
# 輸出在 dist\OffAxisTracker\OffAxisTracker.exe
```

---

## 主要套件

- `mediapipe==0.10.9` — 臉部偵測（legacy solutions API）
- `opencv-python` — 攝像頭影像處理
- `numpy` — 數值計算
- `pyinstaller` — 打包成 exe

---

*最後更新：2026-04-14*