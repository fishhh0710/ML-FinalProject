# 實體卡牌記憶遊戲

把八張卡排成兩排，用相機看你指哪裡、翻哪張。畫面會即時顯示這回合翻到的兩張牌和配對結果，指尖停留與翻牌也會存成 CSV，方便回看。

## 怎麼跑

用 Windows 和 Python 3.14，在專案資料夾執行：

```powershell
python -m pip install -r requirements.txt
python -u hand_tracking.py --camera 0 --backend dshow --auto-board --csv pointing_hd.csv --timeline session_hd.csv
```

用我們共用的 VCS-AR0234-C 相機和原本那八張牌。開局把牌白背朝上放好，等畫面找到八個格位再翻。第一次執行會下載手部模型。`D` 看事件紀錄，`Q` 或 `Esc` 結束。

安裝或相機有問題看 [SETUP](docs/SETUP.md)，要接手開發看 [AGENTS](AGENTS.md) 和 [TODO](docs/TODO.md)。
