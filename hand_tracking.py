"""Track hands and match index fingertips to AprilTag cards on the same frame.

Video:  python hand_tracking.py session.mp4 --csv pointing.csv --show
Camera: python hand_tracking.py --camera 0 --backend dshow --csv pointing.csv

CSV coordinates are pixels from the image's upper-left corner.
"""

import argparse
import csv
import json
import math
import time
import urllib.request
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2
import numpy as np

from card_locator import CardLocator, annotate
from board_tracker import BoardTracker, EVENT_COLUMNS, calibrate_board, detect_white_board
from session_timeline import EventTimeline, HandSample


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255)]
INDEX_FINGERTIP = 8


def ensure_model():
    if not MODEL_PATH.is_file():
        print(f"Downloading the hand model to {MODEL_PATH} ...")
        partial_path = MODEL_PATH.with_suffix(".task.part")
        urllib.request.urlretrieve(MODEL_URL, partial_path)
        partial_path.replace(MODEL_PATH)


def draw_hand(frame, pts, color, label, connections):
    for c in connections:
        cv2.line(frame, pts[c.start], pts[c.end], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (255, 255, 255), -1)
        cv2.circle(frame, p, 4, color, 1)
    xs, ys = zip(*pts)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color, 2)
    cv2.putText(frame, label, (x1 - 10, y1 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def card_at_point(point, frame_size, cards):
    """Test a fingertip against visible card polygons, including their edges."""
    x, y = point
    width, height = frame_size
    if not (0 <= x < width and 0 <= y < height):
        return None, "outside_frame"

    hits = [
        card for card in cards
        if cv2.pointPolygonTest(np.float32(card["corners_px"]), point, False) >= 0
    ]
    if len(hits) == 1:
        return hits[0], "matched"
    return None, "ambiguous" if hits else "no_card"


class PreviewWindow:
    def __init__(self, name):
        self.name = name
        self.was_visible = False
        self.hidden_since = None

    def show(self, image):
        if self.hidden_since is None:
            cv2.imshow(self.name, image)

    def poll(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return key, "Q / Esc pressed"
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if visible:
            self.was_visible = True
            self.hidden_since = None
        elif self.was_visible:
            now = time.monotonic()
            if self.hidden_since is None:
                self.hidden_since = now
            elif now - self.hidden_since >= 0.3:
                return key, "preview window closed"
        return key, ""


class LiveCapture:
    """Keep DirectShow calls off the OpenCV window thread."""

    def __init__(self, index, backend, native_resolution):
        self.frames = Queue(maxsize=1)
        self.stop = Event()
        self.stage = "Opening camera"
        self.error = ""
        self.backend_name = ""
        self.fps = 0.0
        self.worker = Thread(target=self._capture, args=(index, backend, native_resolution), daemon=True)
        self.worker.start()

    def _capture(self, index, backend, native_resolution):
        try:
            for use_hd in ((False,) if native_resolution else (True, False)):
                cap = cv2.VideoCapture(index, backend)
                try:
                    if not cap.isOpened():
                        self.error = "Camera could not be opened. Close other apps using it."
                        return
                    if use_hd:
                        self.stage = "Requesting 1920x1080"
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    self.stage = "Waiting for first camera frame" + ("" if use_hd else " (native resolution)")
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    self.backend_name = cap.getBackendName()
                    self.fps = cap.get(cv2.CAP_PROP_FPS)
                    while not self.stop.is_set():
                        try:
                            self.frames.put_nowait(frame)
                        except Full:
                            try:
                                self.frames.get_nowait()
                            except Empty:
                                pass
                            self.frames.put_nowait(frame)
                        self.stage = "Waiting for next camera frame"
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            self.error = "Camera stopped returning frames."
                            return
                    return
                finally:
                    cap.release()
                if self.stop.is_set():
                    return
            self.error = "Camera could not return its first frame."
        except cv2.error as exc:
            self.error = f"{self.stage}: {exc}"

    def read(self, window, allow_d=False):
        deadline = time.monotonic() + 8
        toggle_debug = False
        while True:
            try:
                frame = self.frames.get(timeout=0.025)
                return frame, toggle_debug
            except Empty:
                if self.error:
                    raise SystemExit(self.error)
                if time.monotonic() >= deadline:
                    raise SystemExit(f"Camera timed out: {self.stage}. Try --native-resolution or close other camera apps.")
                key, reason = window.poll()
                if reason:
                    raise SystemExit(f"Camera capture cancelled: {reason}.")
                if allow_d and key in (ord("d"), ord("D")):
                    toggle_debug = not toggle_debug

    def release(self):
        self.stop.set()
        self.worker.join(timeout=0.3)


def read_frame(cap, window, allow_d=False):
    if isinstance(cap, LiveCapture):
        frame, toggle_debug = cap.read(window, allow_d)
        return True, frame, toggle_debug
    ok, frame = cap.read()
    return ok, frame, False


def open_capture(parser, args):
    backend = {
        None: cv2.CAP_ANY, "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
    }[args.backend]
    if args.camera is not None:
        window = PreviewWindow("Starting camera")
        print(f"Opening camera {args.camera} ({args.backend or 'auto'}) ...", flush=True)
        canvas = np.full((150, 660, 3), 28, dtype=np.uint8)
        cv2.putText(canvas, f"Opening camera {args.camera}...  Q / Esc to cancel",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
        window.show(canvas)
        cap = LiveCapture(args.camera, backend, args.native_resolution)
        try:
            frame, _ = cap.read(window)
        except BaseException:
            cap.release()
            raise
        finally:
            cv2.destroyAllWindows()
        print(f"Camera frame received: {frame.shape[1]}x{frame.shape[0]} ({cap.backend_name}).", flush=True)
        return cap, frame

    source = str(args.video)
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        parser.error(f"Cannot open input: {source} (backend: {args.backend or 'auto'})")
    ok, frame = cap.read()
    if not ok:
        cap.release()
        parser.error(f"Cannot read the first frame from input: {source}")
    return cap, frame


def detect_board_at_start(parser, cap, frame, locator, count, live):
    window = PreviewWindow("Finding white-backed cards")
    previous = None
    confirmations = 0
    next_check = 0.0
    next_hint = time.monotonic() + 8
    preview = None
    print(f"Place {count} cards white-side up in two rows and clear your hands.", flush=True)
    try:
        while True:
            now = time.monotonic()
            if live and now >= next_hint:
                print("Camera is running. Still looking for all white-backed cards in two rows.", flush=True)
                next_hint = now + 8
            if now >= next_check:
                next_check = now + 0.15
                calibration = detect_white_board(frame, count)
                preview = BoardTracker(calibration, frame) if calibration else None
                if preview:
                    tags = locator.locate(frame)["tags"]
                    if any(preview.at(tuple(np.mean(tag["corners_px"], axis=0))) for tag in tags):
                        preview = None
                current = np.array([slot.corners for slot in preview.slots]) if preview else None
                if current is None:
                    confirmations = 0
                elif previous is not None and np.max(np.linalg.norm(current - previous, axis=2)) <= 8:
                    confirmations += 1
                else:
                    confirmations = 1
                previous = current
                if preview and (not live or confirmations >= 3):
                    print(f"Automatically located {count} card slots.")
                    return preview, frame
                if not live:
                    parser.error("No clear white-back grid was found in the video's first frame.")
            image = frame.copy()
            if preview:
                preview.draw(image)
            height, width = image.shape[:2]
            scale = min(1.0, 1200 / width, 750 / height)
            image = cv2.resize(image, (round(width * scale), round(height * scale)))
            cv2.putText(image, f"Finding {count} white backs in two rows. Q: quit", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 190, 255), 2, cv2.LINE_AA)
            window.show(image)
            _, reason = window.poll()
            if reason:
                raise SystemExit(f"Automatic board detection cancelled: {reason}.")
            ok, frame, _ = read_frame(cap, window)
            if not ok:
                parser.error("Camera stopped while locating the cards.")
    finally:
        if live:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("video", type=Path, nargs="?", help="Input video")
    source.add_argument("--camera", type=int, help="Camera index")
    ap.add_argument("--backend", choices=("auto", "dshow", "msmf"), help="Camera capture backend")
    ap.add_argument("--native-resolution", action="store_true", help="Keep the camera's default resolution")
    ap.add_argument("-o", "--output", type=Path, help="Annotated video path")
    ap.add_argument("--csv", type=Path, help="Landmarks and pointing CSV path")
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--trail", type=int, default=30, help="Wrist trail in frames, 0 to disable")
    ap.add_argument("--show", action="store_true", help="Preview video processing (always on for camera)")
    board_mode = ap.add_mutually_exclusive_group()
    board_mode.add_argument("--calibrate-board", type=Path, help="Save fixed slots with all cards face down")
    board_mode.add_argument("--board", type=Path, help="Load the fixed slots for flip tracking")
    board_mode.add_argument("--auto-board", action="store_true", help="Locate white-backed cards automatically")
    ap.add_argument("--slots", type=int, default=8, help="Number of cards to mark during calibration")
    ap.add_argument("--flips", type=Path, help="Flip event CSV path (requires board tracking)")
    ap.add_argument("--timeline", type=Path, help="Combined pointing and flip CSV path")
    ap.add_argument("--dwell-ms", type=int, default=400, help="Minimum fingertip dwell to log a point")
    args = ap.parse_args()
    live = args.camera is not None
    if not live and (args.backend or args.native_resolution):
        ap.error("--backend and --native-resolution require --camera")
    if args.trail < 0 or args.num_hands < 1 or not 0 <= args.min_conf <= 1:
        ap.error("Use --trail >= 0, --num-hands >= 1 and --min-conf between 0 and 1.")
    if args.slots < 1:
        ap.error("--slots must be at least 1.")
    if (args.flips or args.timeline) and not (args.board or args.auto_board):
        ap.error("--flips and --timeline require --auto-board or --board.")
    if args.dwell_ms < 1:
        ap.error("--dwell-ms must be positive.")
    if args.board and not args.board.is_file():
        ap.error(f"Board file not found: {args.board}. Use --auto-board to start without a board file.")

    if args.calibrate_board:
        if args.calibrate_board.exists():
            ap.error("Calibration already exists. Choose a new filename.")
        cap, frame = open_capture(ap, args)
        try:
            if live:
                print("Place the cards white-side up and clear your hands. Press Space to freeze the frame.")
                window = PreviewWindow("Calibration preview")
                while True:
                    height, width = frame.shape[:2]
                    scale = min(1.0, 1200 / width, 750 / height)
                    window.show(cv2.resize(frame, (round(width * scale), round(height * scale))))
                    key, reason = window.poll()
                    if key == 32:
                        break
                    if reason:
                        return
                    ok, frame, _ = read_frame(cap, window)
                    if not ok:
                        ap.error("Camera stopped during calibration.")
                cv2.destroyWindow(window.name)
            calibrate_board(frame, args.calibrate_board, args.slots)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    out_path = args.output
    if out_path is None and not live:
        out_path = args.video.with_name(args.video.stem + "_hands.mp4")
    csv_path = args.csv
    if csv_path is None:
        csv_path = Path("pointing.csv") if live else args.video.with_name(args.video.stem + "_hands.csv")
    flips_path, timeline_path = None, None
    if args.board or args.auto_board:
        flips_path = args.flips or csv_path.with_name(csv_path.stem + "_flips.csv")
        timeline_path = args.timeline or csv_path.with_name(csv_path.stem + "_timeline.csv")
    paths = [p.resolve() for p in (args.video, out_path, csv_path, args.board, flips_path, timeline_path)
             if p is not None]
    if len(paths) != len(set(paths)):
        ap.error("Input files and output files must all use different paths.")

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as exc:
        raise SystemExit(f"MediaPipe Tasks is required ({exc}). Install it in this Python environment: python -m pip install mediapipe") from exc

    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_conf,
        min_hand_presence_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf,
    )
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    locator = CardLocator()
    trails = {}
    header = ["frame", "time_s", "hand_idx", "handedness", "score"]
    header += [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
    header += ["point_x_px", "point_y_px", "card_id", "fruit", "card_pose_source", "point_status"]

    with ExitStack() as stack:
        cap, frame = open_capture(ap, args)
        stack.callback(cap.release)
        print("Starting MediaPipe hand tracking ...", flush=True)
        landmarker = stack.enter_context(vision.HandLandmarker.create_from_options(options))
        tracker, flip_writer, flip_file = None, None, None
        if args.auto_board:
            tracker, frame = detect_board_at_start(ap, cap, frame, locator, args.slots, live)
        elif args.board is not None:
            try:
                tracker = BoardTracker(json.loads(args.board.read_text(encoding="utf-8")), frame)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                ap.error(f"Cannot use board calibration: {exc}")
        if tracker is not None:
            header += ["slot_id", "slot_observation", "slot_last_seen_card_id"]
            flip_file = stack.enter_context(flips_path.open("w", newline="", encoding="utf-8"))
            flip_writer = csv.writer(flip_file)
            flip_writer.writerow(EVENT_COLUMNS)
        show = live or args.show
        window = PreviewWindow("Hand and card tracking")
        game, game_panel = None, None
        show_debug = False
        if show:
            stack.callback(cv2.destroyAllWindows)
            if tracker is not None and len(tracker.slots) >= 2:
                try:
                    from memory_game import GamePanel, MemoryGame
                except ImportError as exc:
                    if exc.name != "PIL":
                        raise
                    print("Install Pillow for the game view: python -m pip install Pillow")
                else:
                    game = MemoryGame(len(tracker.slots))
                    game_panel = GamePanel(game)
        fps = cap.fps if live else cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        h, w = frame.shape[:2]
        if live:
            print(f"Camera {args.camera}: {cap.backend_name}, {w}x{h}. Press Q or Esc to stop.", flush=True)
        writer = None
        if out_path:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            stack.callback(writer.release)
            if not writer.isOpened():
                ap.error(f"Cannot write output video: {out_path}")
        csv_file = stack.enter_context(csv_path.open("w", newline="", encoding="utf-8"))
        cw = csv.writer(csv_file)
        cw.writerow(header)
        frame_idx = 0
        last_ts_ms = -1
        last_frame_idx = 0
        timeline = None
        if tracker is not None:
            timeline_file = stack.enter_context(timeline_path.open("w", newline="", encoding="utf-8"))
            timeline = EventTimeline(csv.writer(timeline_file), (w, h), dwell_ms=args.dwell_ms)
            stack.callback(lambda: timeline.finish(last_frame_idx, max(0, last_ts_ms)))
        start_ns = time.monotonic_ns()
        while True:
            raw_ms = (time.monotonic_ns() - start_ns) // 1_000_000 if live else int(frame_idx * 1000 / fps)
            ts_ms = max(last_ts_ms + 1, raw_ms)
            last_ts_ms = ts_ms
            last_frame_idx = frame_idx
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, ts_ms)
            card_result = locator.locate(frame)
            frame_size = card_result["frame_size_px"]
            drawing = annotate(cv2, frame, card_result)
            hands_px = [[(int(lm.x * frame_size[0]), int(lm.y * frame_size[1])) for lm in lms]
                        for lms in result.hand_landmarks]
            if tracker:
                events = tracker.update(frame, card_result, hands_px)
                if game is not None:
                    game.update(events, ts_ms, frame, card_result["cards"])
                for event in events:
                    flip_writer.writerow([frame_idx, f"{ts_ms / 1000:.3f}", *event])
                if events:
                    flip_file.flush()
                tracker.draw(drawing)

            seen = set()
            hand_samples = []
            for i, lms in enumerate(result.hand_landmarks):
                categories = result.handedness[i] if i < len(result.handedness) else []
                label = categories[0].category_name if categories else "Unknown"
                score = f"{categories[0].score:.3f}" if categories else ""
                pts = hands_px[i]
                tip = lms[INDEX_FINGERTIP]
                point = (tip.x * frame_size[0], tip.y * frame_size[1])
                card, status = card_at_point(point, frame_size, card_result["cards"])
                slot = tracker.at(point) if tracker else None
                if timeline:
                    card_id = slot.card_id if slot else ""
                    if slot and card and tracker.at(tuple(card["center_px"])) is slot:
                        card_id = card["card_id"]
                    hand_samples.append(HandSample(pts[0], label, slot.name if slot else "", card_id))

                color = COLORS[i % len(COLORS)]
                draw_hand(drawing, pts, color, f"{label} {score}", connections)
                if status != "outside_frame":
                    tip_px = pts[INDEX_FINGERTIP]
                    cv2.circle(drawing, tip_px, 8, (0, 255, 255), 2)
                    card_label = card["card_id"] if card else slot.name if slot else status
                    if card and card["pose_source"] == "single_tag":
                        card_label += " (1 tag)"
                    cv2.putText(drawing, f"{card_label} {tip_px}", (tip_px[0] + 10, max(20, tip_px[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                row = [frame_idx, f"{ts_ms / 1000:.3f}", i, label, score]
                for lm, (px, py) in zip(lms, pts):
                    row += [px, py, f"{lm.z:.4f}"]
                row += [f"{point[0]:.2f}", f"{point[1]:.2f}"]
                row += [card[key] if card else "" for key in ("card_id", "fruit", "pose_source")]
                row.append(status)
                if tracker:
                    row += [slot.name, slot.observation, slot.card_id] if slot else ["", "", ""]
                cw.writerow(row)

                if args.trail:
                    trails.setdefault(label, deque(maxlen=args.trail)).append(pts[0])
                    seen.add(label)

            if timeline:
                previous_count = timeline.event_count
                timeline.update(frame_idx, ts_ms, hand_samples, events)
                if timeline.event_count != previous_count:
                    timeline_file.flush()

            for label, trail in trails.items():
                if label not in seen and trail:
                    trail.popleft()
                for a, b in zip(trail, list(trail)[1:]):
                    cv2.line(drawing, a, b, (0, 255, 255), 2)

            cv2.putText(drawing, f"frame {frame_idx}  hands {len(result.hand_landmarks)}  cards {len(card_result['cards'])}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if writer:
                writer.write(drawing)
            frame_idx += 1
            if show:
                if game_panel is not None and not show_debug:
                    preview = game_panel.render(drawing)
                else:
                    preview = timeline.preview(drawing, ts_ms) if timeline else drawing
                window.show(preview)
                key, reason = window.poll()
                if key in (ord("d"), ord("D")):
                    show_debug = not show_debug
                if reason:
                    print(f"Stopped: {reason}.", flush=True)
                    break
            if frame_idx % 100 == 0:
                csv_file.flush()
                print(f"{frame_idx} frames")
            ok, frame, toggle_debug = read_frame(cap, window, allow_d=True)
            if toggle_debug:
                show_debug = not show_debug
            if not ok:
                if live:
                    raise SystemExit(f"Camera stopped returning frames. CSV saved to {csv_path}")
                break

    print(f"Processed {frame_idx} frames. CSV: {csv_path}")
    if flips_path:
        print(f"Flip events: {flips_path}")
    if timeline_path:
        print(f"Session timeline: {timeline_path}")
    if out_path:
        print(f"Video: {out_path}")


if __name__ == "__main__":
    main()
