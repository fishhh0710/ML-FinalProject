# 資料格式

目前沒有額外的 schema 版本欄或場次 ID。以下契約對應交接版原始碼。

## 共用座標與時間

- 像素原點在影像左上角，x 向右，y 向下。使用輸入原影格座標，與預覽縮放無關。影格尺寸順序為 `[width, height]`。
- `frame` 從 0 開始，是**已處理影格序號**。即時取圖可能跳過舊影格。影片模式以影格序號和 FPS 估算相對時間，缺失或無效 FPS 使用 30。未使用影片 PTS。
- `time_s` 以秒表示、三位小數。即時模式在開局校正完成後開始計時，來源是處理迴圈的單調時鐘。事件時間為確認或記錄時間。
- `slot_id` 是固定桌面格位。自動模式依排、由左到右編為 `S01–S08`。手動模式以點選次序為準。 `card_id` 是實體牌識別碼。同水果的兩張牌有不同 ID。

## 輸入

| 類型 | 格式或條件 |
|---|---|
| 相機或影片 | OpenCV 可讀 BGR 影格。自動定位的輸入首幀或開局預覽需看見兩排白背。 |
| 校正 JSON | 根物件有 `frame_size_px` 和 `slots`。每格含 `slot_id`、順時針 TL/TR/BR/BL 的 `corners_px`、`white_level` 和 `contrast`。亮度以 0–255 灰階衡量，對比是內部亮度中位數減外環第 25 百分位。 |
| 手部模型 | MediaPipe Hand Landmarker `.task`，放置位置見 [SETUP](SETUP.md)。 |

正面 Tag 對照隨程式的 `CARDS` 固定。

| 水果 | 牌 ID 與標記 |
|---|---|
| 香蕉 | `banana_1` (12, 13)，`banana_2` (14, 15) |
| 葡萄 | `grape_1` (4, 5)，`grape_2` (6, 7) |
| 檸檬 | `lemon_1` (0, 1)，`lemon_2` (2, 3) |
| 西瓜 | `watermelon_1` (8, 9)，`watermelon_2` (10, 11) |

## 輸出

**點位 CSV（`--csv`）**：每個偵測到的手一列；無手影格不產生資料列。未指定時，相機為 `pointing.csv`，影片為同目錄 `<stem>_hands.csv`。

- 基本欄：`frame,time_s,hand_idx,handedness,score`。 `hand_idx` 是當幀 MediaPipe 回傳順序，`score` 是 handedness 分數。
- 21 組關鍵點：`x0,y0,z0` 至 `x20,y20,z20`。x/y 為整數像素，z 為 MediaPipe 原始模型相對深度值。
- 指尖：`point_x_px,point_y_px,card_id,fruit,card_pose_source,point_status`。 `point_status` 可為 `matched`、`no_card`、`ambiguous`、`outside_frame`。牌面資訊可空白。
- 啟用格位才附加 `slot_id,slot_observation,slot_last_seen_card_id`。 `slot_observation` 為當幀 `up`、`down` 或 `unknown`。上次看到的牌號可能是歷史資訊，首次翻開前可空白。

**翻牌 CSV（`--flips`）**：只在 `--board` 或 `--auto-board` 產生，預設 `<點位檔名>_flips.csv`（去除 .csv 再加後綴）。欄名依序為 `frame,time_s,event,slot_id,card_id,evidence`。

| event | 代表 |
|---|---|
| `flip_up` | 已確認的白背格，轉成已確認的正面。 |
| `flip_down` | 確認從正面轉回白背，牌號沿用上次看到的牌。 |
| `front_seen` | 未從穩定 `down` 狀態轉入的已確認正面，例如開局已朝上或未經蓋回就換了牌。 |

翻開的 `evidence` 為 `both_tags` 或 `single_tag`，蓋回為 `white_back`。

**時間軸 CSV（`--timeline`）**：需要格位模式，預設 `<點位前綴>_timeline.csv`。欄位依序為 `frame,time_s,event,slot_id,card_id,hand_track_id,handedness,point_started_s,point_last_seen_s,duration_s,evidence`。翻牌事件與上表相同，手部欄留空。停留事件為 `point_start` 和 `point_end`，以場次內 `hand_track_id` 關聯。`point_started_s` 是第一次觀測到同格，`point_last_seen_s` 是最後一次，`duration_s` 是兩者差值。停留證據包括 `dwell_threshold`、`hand_lost`、`slot_lost`、`slot_changed`、`session_end`，牌號可空白。

**Tag 偵測 JSON**：`CardLocator.locate()` 及靜態圖 CLI 輸出 `tag_family`、`frame_size_px`、`detected_marker_ids`、`duplicate_marker_ids`、`tags`、`cards`。每個 `tag` 有 `tag_id,corners_px`，角點按解碼方向排列。每個 `card` 有 `card_id,fruit,marker_ids,visible_marker_ids,pose_source,center_px,corners_px,angle_deg`。牌的四角順序來自參考牌面 TL、TR、BR、BL。角度是投影上緣方向，單位度。

**標註影片**：`--output`／`-o` 可設定輸出路徑，影片輸入預設 `<stem>_hands.mp4`，相機需明示。`mp4v` 編碼，以影格寫入，不含音訊或右側回合/除錯面板。
