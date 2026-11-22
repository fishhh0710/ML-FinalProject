"""Generate a non-personal, 36-frame video with known flip events."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from card_locator import CARDS


def fixture_images():
    back = np.full((720, 1280, 3), 58, dtype=np.uint8)
    front = back.copy()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for i, (_, outline, tags) in enumerate(CARDS):
        x, y = 170 + 245 * (i % 4), 80 + 295 * (i // 4)
        quad = np.float32([[x, y], [x + 174, y], [x + 174, y + 236], [x, y + 236]])
        for image in (front, back):
            cv2.fillConvexPoly(image, quad.astype(np.int32), (240, 240, 240))
        cv2.circle(front, (x + 87, y + 118), 38, (50, 140, 210), -1)
        transform = cv2.getPerspectiveTransform(np.float32(outline), quad)
        for tag_id, corners in tags.items():
            target = cv2.perspectiveTransform(np.float32([corners]), transform)[0]
            dst = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [95, 0], [95, 95], [0, 95]]), target
            )
            marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 96)
            patch = cv2.warpPerspective(marker, dst, (1280, 720), flags=cv2.INTER_NEAREST,
                                        borderValue=255)
            mask = np.zeros((720, 1280), np.uint8)
            cv2.fillConvexPoly(mask, np.rint(target).astype(np.int32), 1)
            front[mask > 0] = patch[mask > 0, None]
    return back, front


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output .avi path")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    back, front = fixture_images()
    empty = np.full_like(back, 58)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"MJPG"), 8, (1280, 720))
    try:
        if not writer.isOpened():
            raise SystemExit("Could not open MJPG/AVI writer.")
        for image in (back, front, empty, front, back, front):
            for _ in range(6):
                writer.write(image)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
