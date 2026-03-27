# face_tracker_udp.py 改善筆記

> 記錄日期：2026-03-27

---

## 🐛 已知可修正項目

### 1. Debug 計數器寫法 hacky
- **位置**：Line 474-476
- **問題**：用 `main._fc`（函式屬性）來存幀數計數器，不直覺
- **建議**：改用 `nonlocal` 變數，或把整個 main loop 重構成 class

```python
# 目前（hacky）
if not hasattr(main, '_fc'):
    main._fc = 0
main._fc += 1

# 建議改成（在 main() 開頭宣告）
frame_count = 0
# ...在迴圈中：
frame_count += 1
```

---

### 2. 攝影機斷線時會無限 loop
- **位置**：Line 393-394
- **問題**：`cap.read()` 失敗時只用 `continue`，攝影機拔掉會死循環
- **建議**：加入連續失敗計數器，超過閾值就自動中斷

```python
# 建議加入
MAX_READ_FAILURES = 30  # 連續 30 幀讀取失敗就停止
read_fail_count = 0

# 在迴圈中：
ok, frame = cap.read()
if not ok:
    read_fail_count += 1
    if read_fail_count > MAX_READ_FAILURES:
        print("[ERROR] 攝影機連續讀取失敗，自動停止")
        break
    continue
read_fail_count = 0  # 讀成功就歸零
```

---

### 3. 沒有鏡頭畸變校正
- **位置**：Line 299, 388
- **問題**：`dist = np.zeros((4,1))`，假設鏡頭無畸變
- **影響**：廣角 / 便宜鏡頭的桶形畸變會影響 solvePnP 精度
- **建議**：整合 `calibrate_camera.py` 輸出的畸變係數，或加一個可選的 `--distortion` 參數

---

## 🚀 效能優化提案

### 4. 用卡爾曼濾波 (Kalman Filter) 取代 EMA 平滑

#### 為什麼？

| 比較 | 目前的 EMA | 卡爾曼濾波 |
|---|---|---|
| **延遲** | 一定有延遲（alpha 越小延遲越大） | 可預測性補償，延遲極低 |
| **抖動抑制** | 靠 alpha + 死區 | 根據噪聲模型自動調整 |
| **突然加速** | 反應慢，會被平滑吃掉 | 可以追上突然的動作 |
| **實作難度** | 很簡單 | 中等（但 OpenCV 有現成的） |

#### 原理簡單說

卡爾曼濾波 = **預測 + 修正**：
1. **預測**：根據上一幀的速度，「猜」這一幀的位置（所以幾乎零延遲）
2. **修正**：用實際量測值修正猜測，降低抖動

就像你在開車看 GPS：
- EMA 像「看過去 5 秒的平均位置」→ 穩但延遲
- 卡爾曼像「根據車速預測你現在在哪，再用 GPS 微調」→ 又穩又即時

#### 實作方向

可以用 `cv2.KalmanFilter`，6 個通道（X, Y, Z, Yaw, Pitch, Roll）各自獨立：

```python
# 概念範例（每個軸一個 Kalman Filter）
import cv2
import numpy as np

def create_kalman_1d(process_noise=1e-2, measurement_noise=1e-1):
    """建立 1D 卡爾曼濾波器（狀態=[位置, 速度]）"""
    kf = cv2.KalmanFilter(2, 1)  # 2 states, 1 measurement
    kf.transitionMatrix = np.array([[1, 1],   # 位置 = 位置 + 速度
                                     [0, 1]], dtype=np.float32)  # 速度不變
    kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)
    kf.statePost = np.zeros((2, 1), dtype=np.float32)
    return kf

# 使用方式：
kf_yaw = create_kalman_1d()
# 每幀：
kf_yaw.predict()                          # 預測（根據速度外推）
corrected = kf_yaw.correct(np.array([[raw_yaw]], dtype=np.float32))
smoothed_yaw = corrected[0, 0]            # 修正後的值
```

#### 調參重點
- `process_noise`：越大 → 越信任量測（靈敏但抖），越小 → 越信任預測（穩但慢）
- `measurement_noise`：越大 → 越不信任量測（更平滑），反之更靈敏
- 建議先設 `process_noise=0.01`, `measurement_noise=0.1` 開始試

---

## 📋 優先順序建議

1. ⭐ **卡爾曼濾波** — 最大改善（延遲 + 手感）
2. 🔧 **攝影機斷線保護** — 穩定性必要
3. 🧹 **Debug 計數器重構** — 程式碼品質
4. 📐 **畸變校正** — 精度提升（視鏡頭品質決定優先度）
