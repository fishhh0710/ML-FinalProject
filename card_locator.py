"""Detect the fruit cards' AprilTag 36h11 markers in a photo or live video.

Install: python -m pip install "opencv-contrib-python>=4.10"
Image:   python card_locator.py --image photo.jpg --output detected.jpg
Camera:  python card_locator.py --camera 0 --backend dshow
Video:   python card_locator.py --video recording.mp4

For MediaPipe integration, instantiate CardLocator once and call locate(frame)
with the same uncropped, unmirrored frame used for hand landmarks. Coordinates
are (x, y) pixels, measured from the top-left corner of that input frame.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


TAG_FAMILY = "tag36h11"

# For each card: ID, outline (TL, TR, BR, BL), then the two AprilTag IDs and
# their corners in OpenCV's decoded tag orientation, all in the reference photo.
CARDS = (
    ("banana_1", ((532, 321), (652, 326), (623, 553), (502, 541)), {
        12: ((607.56, 360.42), (611.05, 334), (637.46, 334.92), (633.29, 361.46)),
        13: ((513.69, 530.75), (515.51, 501.9), (542.42, 503.85), (538.36, 533.27)),
    }),
    ("watermelon_1", ((667, 325), (800, 331), (776, 567), (640, 554)), {
        8: ((751.73, 368.43), (755.77, 338), (785.07, 341.01), (781.91, 369.66)),
        9: ((648.39, 543.36), (651.86, 512.48), (681, 516), (677.9, 545.98)),
    }),
    ("grape_1", ((814, 333), (953, 344), (933, 580), (790, 570)), {
        4: ((902.67, 379.21), (905.78, 348.59), (935.8, 351.3), (933.31, 381.84)),
        5: ((803, 557), (807.06, 525.96), (837.95, 529.33), (834.46, 560.78)),
    }),
    ("lemon_1", ((972, 345), (1116, 358), (1095, 594), (952, 581)), {
        0: ((1069.55, 391.23), (1072.05, 363.06), (1100.15, 365.84), (1098.77, 393.87)),
        1: ((962.99, 572.8), (966.04, 541.91), (996.07, 544.58), (993.98, 575.21)),
    }),
    ("banana_2", ((496, 557), (617, 570), (603, 798), (483, 778)), {
        14: ((577.1, 609.78), (580.1, 580.02), (607.95, 583.91), (604.54, 613.93)),
        15: ((496.99, 770.95), (497.05, 743.63), (523.19, 746.82), (522.87, 774.88)),
    }),
    ("watermelon_2", ((635, 571), (770, 585), (752, 820), (620, 800)), {
        10: ((724.76, 627.27), (727.7, 595.44), (758.2, 598.98), (755.21, 630.13)),
        11: ((630.73, 793.21), (632.01, 763.98), (660.47, 767.71), (659.17, 797.38)),
    }),
    ("grape_2", ((783, 585), (927, 598), (907, 836), (765, 822)), {
        6: ((878.64, 642.09), (882.41, 610.06), (913.09, 613.02), (910.24, 645.11)),
        7: ((782.1, 813.36), (785.03, 783.95), (816.03, 787.99), (813.61, 817.44)),
    }),
    ("lemon_2", ((947, 600), (1090, 611), (1068, 849), (926, 837)), {
        2: ((1044.22, 655.08), (1048.65, 624.22), (1078.98, 626.99), (1076.33, 658.31)),
        3: ((935.07, 821.98), (938.46, 792.25), (969.35, 795.09), (966.11, 825.6)),
    }),
)


def load_opencv():
    try:
        import cv2
    except ImportError:
        raise SystemExit('Install OpenCV: python -m pip install "opencv-contrib-python>=4.10"')
    if not hasattr(getattr(cv2, "aruco", None), "ArucoDetector"):
        raise SystemExit("OpenCV 4.10 or newer with the aruco module is required.")
    return cv2


def card_positions(cv, markers, duplicate_ids):
    """Recover visible card polygons using current and reference tag corners."""
    cards = []
    for label, outline, templates in CARDS:
        if duplicate_ids.intersection(templates):
            continue
        visible_ids = [tag_id for tag_id in templates if tag_id in markers]
        if not visible_ids:
            continue

        src = np.float32([templates[tag_id] for tag_id in visible_ids]).reshape(-1, 2)
        dst = np.concatenate([markers[tag_id] for tag_id in visible_ids])
        if len(visible_ids) == 1:
            transform = cv.getPerspectiveTransform(src, dst)
        else:
            transform, _ = cv.findHomography(src, dst)
            if transform is None:
                continue
            predicted = cv.perspectiveTransform(src[None], transform)[0]
            marker_quads = dst.reshape(-1, 4, 2)
            edges = marker_quads - np.roll(marker_quads, 1, axis=1)
            fit_error = np.max(np.linalg.norm(predicted - dst, axis=1))
            if fit_error > max(2.0, 0.15 * np.median(np.linalg.norm(edges, axis=2))):
                continue

        if not np.isfinite(transform).all():
            continue
        # A homography with a horizon inside the card cannot give a finite card outline.
        denominators = np.column_stack((outline, np.ones(4))) @ transform[2]
        if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
            continue
        corners = cv.perspectiveTransform(np.float32([outline]), transform)[0]
        if not np.isfinite(corners).all() or not cv.isContourConvex(corners):
            continue
        if cv.contourArea(corners, oriented=True) < 1:
            continue

        tl, tr, br, bl = np.column_stack((corners, np.ones(4)))
        center = np.cross(np.cross(tl, br), np.cross(tr, bl))
        center_xy = center[:2] / center[2]
        edge = corners[1] - corners[0]
        cards.append({
            "card_id": label,
            "fruit": label.rsplit("_", 1)[0],
            "marker_ids": list(templates),
            "visible_marker_ids": visible_ids,
            "pose_source": "both_tags" if len(visible_ids) == 2 else "single_tag",
            "center_px": center_xy.round(2).tolist(),
            "corners_px": corners.astype(float).round(2).tolist(),
            "angle_deg": round(math.degrees(math.atan2(edge[1], edge[0])), 2),
        })
    return cards


class CardLocator:
    """Find raw AprilTags and known fruit cards in one frame."""

    def __init__(self):
        self.cv = load_opencv()
        aruco = self.cv.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(dictionary, params)

    def locate(self, frame):
        """Accept a uint8 BGR or grayscale image. Return a JSON-ready result."""
        gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        detections = [] if ids is None else sorted(
            ((int(tag_id), np.asarray(quad, dtype=np.float32).reshape(4, 2))
             for tag_id, quad in zip(ids.ravel(), corners)),
            key=lambda detection: detection[0],
        )
        markers = {}
        duplicates = set()
        for tag_id, quad in detections:
            if tag_id in markers:
                duplicates.add(tag_id)
            markers[tag_id] = quad

        height, width = gray.shape
        return {
            "tag_family": TAG_FAMILY,
            "frame_size_px": [width, height],
            "detected_marker_ids": list(markers),
            "duplicate_marker_ids": sorted(duplicates),
            "tags": [
                {"tag_id": tag_id, "corners_px": quad.astype(float).round(2).tolist()}
                for tag_id, quad in detections
            ],
            "cards": card_positions(self.cv, markers, duplicates),
        }


def annotate(cv, frame, result):
    image = frame.copy()
    for tag in result["tags"]:
        tag_quad = np.rint(tag["corners_px"]).astype(np.int32)
        cv.polylines(image, [tag_quad], True, (250, 200, 60), 1, cv.LINE_AA)
        x, y = map(int, tag_quad[0])
        cv.putText(image, str(tag["tag_id"]), (x, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.45, (250, 200, 60), 1, cv.LINE_AA)
    for card in result["cards"]:
        color = (45, 224, 85) if card["pose_source"] == "both_tags" else (45, 190, 245)
        corners = np.rint(card["corners_px"]).astype(np.int32)
        center = tuple(np.rint(card["center_px"]).astype(int))
        cv.polylines(image, [corners], True, color, 2, cv.LINE_AA)
        cv.circle(image, center, 5, color, -1, cv.LINE_AA)
        text_point = (center[0] + 8, center[1] - 8)
        cv.putText(image, card["card_id"], text_point, cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, card["card_id"], text_point, cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to a photo")
    source.add_argument("--camera", type=int, help="Webcam index, usually 0")
    source.add_argument("--video", type=Path, help="Path to a video")
    parser.add_argument("--output", type=Path, help="Save the annotated photo")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), help="Camera capture backend")
    parser.add_argument("--native-resolution", action="store_true", help="Keep the camera's default resolution")
    args = parser.parse_args()
    if args.image is None and args.output is not None:
        parser.error("--output requires --image")
    if args.camera is None and (args.backend is not None or args.native_resolution):
        parser.error("--backend and --native-resolution require --camera")
    locator = CardLocator()
    cv = locator.cv
    if args.image is not None:
        frame = cv.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv.IMREAD_COLOR)
        if frame is None:
            parser.error(f"Cannot decode image: {args.image}")
        result = locator.locate(frame)
        if args.output is not None:
            ok, encoded = cv.imencode(args.output.suffix, annotate(cv, frame, result))
            if not ok:
                parser.error(f"Cannot encode output: {args.output}")
            args.output.write_bytes(encoded.tobytes())
        print(json.dumps(result, indent=2))
        return

    input_source = args.camera if args.video is None else str(args.video)
    backends = {"auto": cv.CAP_ANY, "dshow": cv.CAP_DSHOW, "msmf": cv.CAP_MSMF}
    backend = backends[args.backend or "auto"]
    capture = cv.VideoCapture(input_source, backend)
    try:
        if not capture.isOpened():
            parser.error(f"Cannot open input {input_source} (backend: {args.backend or 'auto'})")
        if args.camera is not None and not args.native_resolution:
            capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, frame = capture.read()
        if not ok and args.camera is not None and not args.native_resolution:
            capture.release()
            capture = cv.VideoCapture(input_source, backend)
            ok, frame = capture.read()
        if not ok:
            parser.error(f"Cannot read the first frame from input {input_source}")
        if args.camera is not None:
            height, width = frame.shape[:2]
            print(f"Camera {args.camera}: {capture.getBackendName()}, {width}x{height}")
        print("Press Q or Esc to quit. Press P to print the current positions.")
        while True:
            result = locator.locate(frame)
            cv.imshow("Fruit card locator", annotate(cv, frame, result))
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                print(json.dumps(result, indent=2), flush=True)
            ok, frame = capture.read()
            if not ok:
                if args.video is not None:
                    break
                raise SystemExit("Camera stopped returning frames.")
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
