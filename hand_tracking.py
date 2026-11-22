"""
用 MediaPipe Hand Landmarker 追蹤影片中手的位置。

輸出：
  1. 標註後的影片（骨架、外框、手腕軌跡）
  2. CSV：每一幀每隻手 21 個關鍵點的像素座標

用法：
  python hand_tracking.py mediapipe_test_video.mp4
  python hand_tracking.py input.mp4 -o out.mp4 --csv out.csv --num-hands 2 --show
"""
import argparse
import csv
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255)]  # 每隻手一個顏色 (BGR)


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"下載模型到 {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_hand(frame, pts, color, label):
    for c in CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (255, 255, 255), -1)
        cv2.circle(frame, p, 4, color, 1)
    xs, ys = zip(*pts)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color, 2)
    cv2.putText(frame, label, (x1 - 10, y1 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="輸入影片路徑")
    ap.add_argument("-o", "--output", help="輸出影片路徑（預設 <輸入>_hands.mp4）")
    ap.add_argument("--csv", help="關鍵點 CSV 路徑（預設 <輸入>_hands.csv）")
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--trail", type=int, default=30, help="手腕軌跡長度（幀），0 = 不畫")
    ap.add_argument("--show", action="store_true", help="處理時即時顯示畫面（按 q 中止）")
    args = ap.parse_args()

    src = Path(args.video)
    out_path = Path(args.output) if args.output else src.with_name(src.stem + "_hands.mp4")
    csv_path = Path(args.csv) if args.csv else src.with_name(src.stem + "_hands.csv")

    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,  # VIDEO 模式會跨幀做追蹤，比逐張偵測穩定
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_conf,
        min_hand_presence_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf,
    )

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trails = {}  # handedness label -> deque of wrist points
    header = ["frame", "time_s", "hand_idx", "handedness", "score"]
    header += [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]

    with vision.HandLandmarker.create_from_options(options) as landmarker, \
            open(csv_path, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(header)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = int(frame_idx * 1000 / fps)  # 時間戳必須單調遞增
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, ts_ms)

            seen = set()
            for i, (lms, hd) in enumerate(zip(result.hand_landmarks, result.handedness)):
                label, score = hd[0].category_name, hd[0].score
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                color = COLORS[i % len(COLORS)]
                draw_hand(frame, pts, color, f"{label} {score:.2f}")

                row = [frame_idx, f"{ts_ms / 1000:.3f}", i, label, f"{score:.3f}"]
                for lm, (px, py) in zip(lms, pts):
                    row += [px, py, f"{lm.z:.4f}"]
                cw.writerow(row)

                if args.trail > 0:
                    trails.setdefault(label, deque(maxlen=args.trail)).append(pts[0])
                    seen.add(label)

            # 畫手腕軌跡；沒偵測到的手讓軌跡逐漸消失
            for label, tr in trails.items():
                if label not in seen and tr:
                    tr.popleft()
                for a, b in zip(tr, list(tr)[1:]):
                    cv2.line(frame, a, b, (0, 255, 255), 2)

            cv2.putText(frame, f"frame {frame_idx}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(frame)

            if args.show:
                cv2.imshow("hand tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"{frame_idx}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"完成：{frame_idx} 幀\n影片 -> {out_path}\nCSV  -> {csv_path}")


if __name__ == "__main__":
    main()
