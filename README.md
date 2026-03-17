# Off-Axis Projection — Face Tracker

用臉部追蹤實現 Off-Axis（偏軸）投影效果的工具。
透過 UDP 將臉部座標傳送給其他程式（例如遊戲引擎）來即時調整畫面透視。

---

## 📁 檔案說明

| 檔案 | 說明 |
|------|------|
| `face_tracker_udp.py` | 主程式：抓取臉部座標，用 UDP 傳出 |
| `udp_slider_test.py` | 測試工具：用滑桿模擬 UDP 傳輸，方便調試軸向 |
| `face_landmarker.task` | MediaPipe 臉部偵測模型（需自行下載，見下方說明） |

---

## 🐍 環境重建筆記（換電腦必看）

### Python 版本
```
Python 3.11.9
```

### 安裝步驟

**1. 建立虛擬環境**
```bash
python -m venv .venv
```

**2. 啟動虛擬環境**
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

**3. 安裝所有套件（直接複製貼上）**
```bash
pip install mediapipe==0.10.9 opencv-python==4.13.0.92 numpy==2.4.3 matplotlib==3.10.8 sounddevice==0.5.5
```

---

## 📦 完整套件版本清單

> 以下是 `pip freeze` 的完整輸出，供參考。
> 大部分是自動安裝的相依套件，**主要需要手動安裝的只有上面那幾個**。

```
absl-py==2.4.0
attrs==25.4.0
cffi==2.0.0
contourpy==1.3.3
cycler==0.12.1
flatbuffers==25.12.19
fonttools==4.62.1
kiwisolver==1.5.0
matplotlib==3.10.8
mediapipe==0.10.9
numpy==2.4.3
opencv-contrib-python==4.13.0.92
opencv-python==4.13.0.92
packaging==26.0
pillow==12.1.1
protobuf==3.20.3
pycparser==3.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
sounddevice==0.5.5
```

---

## 📥 MediaPipe 模型下載

`face_landmarker.task` 這個檔案沒有上傳到 GitHub（太大了），需要自己下載：

1. 前往 MediaPipe 官方下載頁：
   https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
2. 下載 `face_landmarker.task`
3. 放到專案根目錄（跟 `face_tracker_udp.py` 同一層）

---

## 🚀 執行方式

```bash
# 啟動虛擬環境後執行主程式
python face_tracker_udp.py

# 或者先用測試工具確認 UDP 傳輸正常
python udp_slider_test.py
```

---

*最後更新：2026-03-17*
