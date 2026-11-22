# 驗證流程

從含 `hand_tracking.py` 的交接包根目錄執行，並使用 [SETUP](SETUP.md) 安裝過依賴的 Python。

## 可重現測試

```powershell
python -m unittest discover -s tests -v
python hand_tracking.py --help
python card_locator.py --help
python -m tests.make_fixture test-output/board.avi
python hand_tracking.py test-output/board.avi --auto-board --csv test-output/points.csv --flips test-output/flips.csv --timeline test-output/timeline.csv -o test-output/annotated.mp4
Import-Csv test-output/flips.csv | Group-Object event
(Import-Csv test-output/timeline.csv).Count
```

`tests.make_fixture` 用現有 AprilTag 字典與幾何生成 1280×720、8 FPS、36 幀的測試影片。順序為白背、八張正面、空桌、同批正面、白背、再翻開。煙霧測試需要模型。期望 `flip_up` **16 筆**、`flip_down` **8 筆**、時間軸 **24 筆**，空桌再恢復正面的區間不應重複計入翻牌。輸出影片 36 幀，點位 CSV 只有標頭，因為合成素材沒有手。單元測試另外用模擬手部位置驗證停留。

無效校正的負向測試：

```powershell
python hand_tracking.py test-output/board.avi --board test-output/not-created.json
$LASTEXITCODE
```

期望錯誤包含 `Board file not found` 且退出碼為 2。

## 2026-11-23 結果

| 項目 | 結果 | 證據 |
|---|---|---|
| 全新虛擬環境 | 通過 | 下載所有 wheels，再以 `--no-index --find-links` 安裝到未繼承系統套件的 Windows x64 Python 3.14.5 虛擬環境。 `pip check` 無套件衝突。 |
| 單元測試 | 5/5 通過 | 16 個 Tag 與八個格位、三幀確認、遮擋與未知、停留事件、回合與面板渲染、預覽開關的狀態處理。 |
| CLI 與錯誤輸入 | 通過 | 兩個 CLI help 能執行。無效校正的退出碼為 2，錯誤提示可操作。 |
| 測試影片與 MediaPipe | 通過 | 全新環境自動下載模型後成功執行。36 幀、16 筆翻開、8 筆蓋回、24 筆時間軸事件。逐筆比對翻牌 CSV 與時間軸的對應欄位一致。 |
| Windows 外接相機 | 此環境開啟失敗 | 以 DirectShow 試開索引 0 回傳關閉，沒有進入實體流程。裝置故障原因未判定。 |
| 真人手、實體牌整局、GUI 鍵盤及校正互動 | 尚未驗證 | 合成驗證覆蓋模組與 CSV，實體遮擋、光照和即時操作須由組員驗收。 |

## 共用設備實跑紀錄

使用者已用 [SETUP](SETUP.md) 的正式指令跑通即時流程。本機核對 `pointing_hd.csv` 有 416 筆手部觀測，`session_hd.csv` 有 44 筆事件（14 次翻開、6 次蓋回，以及各 12 筆停留開始與結束）。這些是程式輸出筆數，未與人工真值比對。實際取圖尺寸需從現場終端機確認，輸出檔不含此資訊。

## 實體驗收

使用 [README](../README.md) 指定的同一台共用相機和同一套八張實體牌，搭配暗色桌面或墊板與 GUI。從 [SETUP](SETUP.md) 的開局指令開始。沿用已跑通的正式指令，不需另外測 640×480。若要比對時間，保存一份與事件可對時的原始影片。交接包只包含合成素材的生成器。

| 手動案例 | 通過標準 |
|---|---|
| 白背自動定位 | 一次找到八格，標號順序及角點包住對應牌。每格先達 `down`。記錄終端機實際取圖尺寸。 |
| 逐張翻開與蓋回 | 八張牌逐張翻開、蓋回並保持到程式確認。人工標註格位與牌號逐筆吻合，維持正面時不重複記錄。 |
| 遮住與移開 | 手遮住正面、牌未翻面，移開不新增翻牌。先看見白背後再翻，才產生新的一次翻開。 |
| 指尖與停留 | 指定格位停超過 400 ms 有成對事件。快掃不計入。牌背可有格位且牌號空白。 |
| 配對與 GUI | 失配後等待蓋回，配對成功計一組。完成顯示 4/4。D 切換事件面板，Q/Esc 或關閉視窗後輸出檔可讀。 |

人工標註至少保留事件類型、格位、牌號和錄影時間。比對時區分首次觀測與程式確認時間。
