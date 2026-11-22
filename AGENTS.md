# 開發代理交接

## 開始工作

1. 讀 [README](README.md) 的啟動方式與 [CURRENT_STATUS](docs/CURRENT_STATUS.md)。用 `git status --short --branch` 確認 checkout。
2. 讀 [ARCHITECTURE](docs/ARCHITECTURE.md)、[DATA_FORMAT](docs/DATA_FORMAT.md) 與任務對應的 [TODO](docs/TODO.md)。
3. 依 [TESTING](docs/TESTING.md) 建立基準，再修改。各任務用獨立分支，先查看使用者已有的變更。

## 模組邊界

- 保留 `hand_tracking.py` 的 CLI 相容性。相機取圖留在 `LiveCapture`，預覽控制留在主執行緒。
- `card_locator.py` 由影像輸出 Tag 和卡牌幾何。辨識與格位狀態分開。
- `board_tracker.py` 處理格位校正與牌面狀態。`unknown` 保留既有穩定狀態。牌背維持全白。
- 實體驗收沿用 [README](README.md) 指定的共用相機及原八張牌。變更 Tag 字典、ID 對照或參考幾何前，先確認對現有實體牌的影響。
- `session_timeline.py` 負責事件與手部停留。保留畫面座標及格式契約。調整時間或狀態時，明示對舊輸出的影響。
- `memory_game.py` 負責兩張一回合與預覽面板。後續語音、評估及離線分析優先做成獨立腳本，沿用事件 CSV。
- 所有檔案路徑應可從乾淨環境提供。測試影片、相機原始資料、執行產物、模型、快取和憑證留在本機。測試用自造資料。

## 交付檢查

- 執行 `python -m unittest discover -s tests -v`，另跑相關 CLI 或影片煙霧測試。
- 變更辨識時，用同一段有人工真值的影片比對 Tag、格位、事件與時間，記錄解析度、參數和差異。
- 寫下執行命令、實際結果和仍需人工驗證的情境。CSV 變更同行更新 [DATA_FORMAT](docs/DATA_FORMAT.md)。
- 工作範圍以指派任務為準。新增必要文件、設定或測試時，保留其餘行為。
