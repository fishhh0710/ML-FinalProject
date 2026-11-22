# 安裝和操作

## 環境

交接驗證環境為 Windows x64、Python 3.14.5，根目錄的 `requirements.txt` 對應該環境的四個直接依賴。OpenCV 套件須提供 `cv2.aruco.ArucoDetector` 和 GUI。請避免在同一虛擬環境並裝多個都提供 `cv2` 的 OpenCV 發行套件。其他 Python 版本、macOS 和 Linux 尚未驗證。

在交接包根目錄執行。PowerShell 不需啟用虛擬環境。

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import cv2, numpy, mediapipe, PIL; assert hasattr(cv2.aruco, 'ArucoDetector'); print(cv2.__version__, numpy.__version__, mediapipe.__version__, PIL.__version__)"
```

若環境限制 PyPI，請在可連網的同平台、同 Python 版本環境執行 `python -m pip download -r requirements.txt -d wheelhouse`。複製 wheelhouse 後，可用 `python -m pip install --no-index --find-links wheelhouse -r requirements.txt`。這條安裝路徑已在全新 Windows x64 虛擬環境驗證；離線執行影片仍需預先放好模型。

## 模型與測試素材

- `hand_tracking.py` 初次執行會將 MediaPipe Hand Landmarker 下載到腳本旁的 `hand_landmarker.task`。離線時，先取得官方 `.task` 檔並放在同一目錄。模型不隨交接包分發。單用 `card_locator.py` 或執行單元測試不需要模型。 [官方模型](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- 影片煙霧測試可自行生成無個人影像的素材，見 [TESTING](TESTING.md)。正式實體驗收使用 [README](../README.md) 指定的共用相機、原八張牌與預覽視窗。

下面的 `python` 指你剛安裝依賴的環境。未啟用環境時可改用 `.\.venv\Scripts\python.exe`。

## 執行模式

| 模式 | 指令 |
|---|---|
| 正式即時流程（已於共用設備跑通） | `python -u hand_tracking.py --camera 0 --backend dshow --auto-board --csv pointing_hd.csv --timeline session_hd.csv` |
| 即時原生解析度 | 在上列相機指令加入 `--native-resolution`，僅在取圖逾時或無畫面時用來排查，日常操作與驗收以正式指令為準。 |
| 影片自動找白背 | `python hand_tracking.py input.avi --auto-board --csv pointing_video.csv --flips flips_video.csv --timeline session_video.csv -o annotated.mp4` |
| 影片顯示預覽 | 在影片指令加入 `--show`，開啟遊戲或事件面板。 |
| 手動校正 | `python hand_tracking.py --camera 0 --backend dshow --calibrate-board board.json` |
| 使用既有校正 | `python hand_tracking.py --camera 0 --backend dshow --board board.json --csv pointing_manual.csv` |
| 只檢查 Tag | `python card_locator.py --image front.jpg --output detected.jpg` |

同一台共用相機接到不同電腦時，索引仍可能不同。Windows 可選 `--backend auto`、`dshow` 或 `msmf`。確認 `Camera frame received` 顯示實際取到的尺寸。若自動定位一直等待，讓牌白背朝上、移開手、使用較暗底面及固定相機，等 `Automatically located 8 card slots.` 後再開始。影片首幀即須看清全部白背。

手動校正先按空白鍵定格，再按左上、右上、右下、左下依序點每張牌。按 `U` 撤銷，Enter 儲存。既有校正檔會被拒絕覆寫。重用檔案時，影格尺寸和相機相對格位需一致。

相機模式一律顯示預覽，`D` 切換遊戲及事件面板，`Q` 或 Esc 結束。缺少 Pillow 時使用事件面板。使用不同輸出檔名保存不同場次，現有 CSV 可能被新執行覆寫。
