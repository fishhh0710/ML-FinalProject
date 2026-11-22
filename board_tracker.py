"""Track card flips in fixed slots using front AprilTags and plain white backs."""

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


PATCH_SIZE = (48, 64)
CARD_PATCH = np.float32([[7, 7], [40, 7], [40, 56], [7, 56]])
# Side strips outside the card. Skip corners and the boundary itself.
RING = np.zeros((64, 48), dtype=bool)
RING[15:49, 3:6] = RING[15:49, 42:45] = True
RING[3:6, 15:33] = RING[58:61, 15:33] = True
EVENT_COLUMNS = ("frame", "time_s", "event", "slot_id", "card_id", "evidence")


@dataclass
class Slot:
    name: str
    corners: np.ndarray
    transform: np.ndarray
    contrast: float
    white_level: float
    face: str = "unknown"
    card_id: str = ""
    observation: str = "unknown"
    pending: tuple = ()
    count: int = 0


def slot_patch(gray, transform):
    patch = cv2.warpPerspective(gray, transform, PATCH_SIZE)
    return patch[12:52, 12:36], patch[RING]


class BoardTracker:
    def __init__(self, calibration, frame, confirmations=3):
        height, width = frame.shape[:2]
        if not isinstance(calibration, dict) or calibration.get("frame_size_px") != [width, height]:
            raise ValueError("Calibration must use the current input resolution.")
        self.size = (width, height)
        self.confirmations = confirmations
        self.slots = []
        entries = calibration.get("slots", [])
        if not entries:
            raise ValueError("Calibration has no slots.")
        for entry in entries:
            corners = np.float32(entry["corners_px"])
            if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                    or not cv2.isContourConvex(corners) or cv2.contourArea(corners, oriented=True) < 100
                    or np.any(corners < 0) or np.any(corners >= [width, height])):
                raise ValueError(f"Invalid corners for {entry['slot_id']}.")
            if any(cv2.intersectConvexConvex(corners, slot.corners)[0] > 1 for slot in self.slots):
                raise ValueError("Calibration slots must not overlap.")
            contrast, white = float(entry["contrast"]), float(entry["white_level"])
            if not (25 <= contrast <= 255 and 100 <= white <= 255):
                raise ValueError(f"{entry['slot_id']} needs a white card back on a darker surface.")
            self.slots.append(Slot(entry["slot_id"], corners,
                                   cv2.getPerspectiveTransform(corners, CARD_PATCH), contrast, white))

    def at(self, point):
        hits = [slot for slot in self.slots
                if cv2.pointPolygonTest(slot.corners, point, False) >= 0]
        return hits[0] if len(hits) == 1 else None

    def update(self, frame, card_result, hands_px):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("Input resolution changed after calibration.")
        fronts = {slot.name: [] for slot in self.slots}
        for card in card_result["cards"]:
            slot = self.at(tuple(card["center_px"]))
            if slot:
                fronts[slot.name].append(card)
        tags_by_slot = {slot.name: set() for slot in self.slots}
        for tag in card_result["tags"]:
            slot = self.at(tuple(np.mean(tag["corners_px"], axis=0)))
            if slot:
                tags_by_slot[slot.name].add(tag["tag_id"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hand_boxes = [cv2.boundingRect(np.int32(hand)) for hand in hands_px if len(hand)]
        events = []
        for slot in self.slots:
            candidates = fronts[slot.name]
            if len(candidates) == 1 and tags_by_slot[slot.name].issubset(candidates[0]["marker_ids"]):
                card = candidates[0]
                observation = ("up", card["card_id"])
                evidence = card["pose_source"]
            elif candidates or tags_by_slot[slot.name] or self._hand_overlaps(slot, hand_boxes):
                observation, evidence = (), ""
            elif self._white_back(gray, slot):
                observation, evidence = ("down", ""), "white_back"
            else:
                observation, evidence = (), ""

            slot.observation = observation[0] if observation else "unknown"
            if observation != slot.pending:
                slot.pending, slot.count = observation, 0
            if not observation:
                continue
            slot.count += 1
            if slot.count != self.confirmations:
                continue

            face, card_id = observation
            if face == "up" and (slot.face != "up" or slot.card_id != card_id):
                slot.card_id = card_id
                events.append(("flip_up" if slot.face == "down" else "front_seen",
                               slot.name, card_id, evidence))
            if face == "down" and slot.face == "up":
                events.append(("flip_down", slot.name, slot.card_id, evidence))
            slot.face = face
        return events

    @staticmethod
    def _hand_overlaps(slot, boxes):
        x, y, w, h = cv2.boundingRect(slot.corners)
        pad = max(6, round(min(w, h) * 0.10))
        return any(x - pad < a + c and a < x + w + pad and y - pad < b + d and b < y + h + pad
                   for a, b, c, d in boxes)

    @staticmethod
    def _white_back(gray, slot):
        inside, outside = slot_patch(gray, slot.transform)
        white = float(np.median(inside))
        contrast = white - float(np.percentile(outside, 25))
        return (white >= max(100, slot.white_level - 40)
                and contrast >= max(20, slot.contrast * 0.50)
                and np.mean(np.abs(inside.astype(float) - white) <= 24) >= 0.94)

    def draw(self, frame):
        for slot in self.slots:
            color = (240, 195, 40) if slot.observation != "unknown" else (140, 140, 140)
            cv2.polylines(frame, [np.rint(slot.corners).astype(np.int32)], True, color, 1, cv2.LINE_AA)
            x, y = np.rint(slot.corners[0]).astype(int)
            cv2.putText(frame, f"{slot.name} {slot.observation}", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def calibrate_board(frame, path: Path, count):
    """Click four corners per face-down card on a frozen calibration image."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1200 / width, 750 / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
    sx, sy = width / preview.shape[1], height / preview.shape[0]
    points = []
    window = "Board setup"
    print("Click TL, TR, BR, BL for each white-back card. U undoes a click. Enter saves.")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count * 4 and y >= 42:
            points.append((x, y - 42))

    cv2.setMouseCallback(window, on_click)
    config = None
    try:
        while True:
            drawing = cv2.copyMakeBorder(preview, 42, 0, 0, 0, cv2.BORDER_CONSTANT)
            for i, (x, y) in enumerate(points):
                cv2.circle(drawing, (x, y + 42), 4, (0, 255, 255), -1)
                if i % 4 == 3:
                    quad = np.int32(points[i - 3:i + 1]) + [0, 42]
                    cv2.polylines(drawing, [quad.astype(np.int32)], True, (0, 255, 255), 2)
            n = len(points)
            instruction = (f"Slot {n // 4 + 1}/{count}: click {('TL', 'TR', 'BR', 'BL')[n % 4]}"
                           if n < count * 4 else "Enter: save")
            cv2.putText(drawing, instruction + "   U: undo   Esc: cancel", (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, drawing)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("Board calibration cancelled.")
            if key == ord("u") and points:
                points.pop()
            if key in (10, 13) and n == count * 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                entries = []
                for i in range(count):
                    corners = np.float32(points[4 * i:4 * i + 4]) * np.float32([sx, sy])
                    transform = cv2.getPerspectiveTransform(corners, CARD_PATCH)
                    inside, outside = slot_patch(gray, transform)
                    white = float(np.median(inside))
                    entries.append({"slot_id": f"S{i + 1:02}", "corners_px": corners.tolist(),
                                    "contrast": white - float(np.percentile(outside, 25)), "white_level": white})
                candidate = {"frame_size_px": [width, height], "slots": entries}
                try:
                    BoardTracker(candidate, frame)
                except ValueError as exc:
                    print(exc)
                    print("Press U to revise corners, or Esc to restart with a darker surface.")
                    continue
                config = candidate
                break
    finally:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window)
    with path.open("x", encoding="utf-8") as out:
        json.dump(config, out, indent=2)
    print(f"Board calibration saved to {path}.")

def detect_white_board(frame, count=8):
    """Find a two-row group of white-backed cards and build slot calibration."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1000 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    sx, sy = width / small.shape[1], height / small.shape[0]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    area = gray.size
    kernel = np.ones((3, 3), dtype=np.uint8)

    for cutoff in (195, 215, 175, 230):
        mask = np.uint8((gray >= cutoff) & (hsv[:, :, 1] <= 75)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if not 0.002 * area <= contour_area <= 0.060 * area:
                continue
            perimeter = cv2.arcLength(contour, True)
            quad = cv2.approxPolyDP(contour, 0.023 * perimeter, True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            a, b = cv2.minAreaRect(quad)[1]
            if min(a, b) <= 0 or not 1.12 <= max(a, b) / min(a, b) <= 3.1:
                continue
            if contour_area < 0.74 * a * b:
                continue
            corners = quad.reshape(4, 2).astype(np.float32)
            if np.any(corners < 3) or np.any(corners >= [gray.shape[1] - 3, gray.shape[0] - 3]):
                continue
            if cv2.contourArea(corners, oriented=True) < 0:
                corners = corners[::-1]
            corners = np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0)
            corners *= np.float32([sx, sy])
            transform = cv2.getPerspectiveTransform(corners, CARD_PATCH)
            inside, outside = slot_patch(full_gray, transform)
            white = float(np.median(inside))
            contrast = white - float(np.percentile(outside, 25))
            if contrast < 25 or np.mean(np.abs(inside.astype(float) - white) <= 24) < 0.94:
                continue
            candidates.append((corners, white, contrast))

        group = select_card_grid(candidates, count)
        if group is None:
            continue
        entries = [
            {"slot_id": f"S{i:02}", "corners_px": corners.tolist(),
             "contrast": contrast, "white_level": white}
            for i, (corners, white, contrast) in enumerate(group, 1)
        ]
        calibration = {"frame_size_px": [width, height], "slots": entries}
        try:
            BoardTracker(calibration, frame)
        except ValueError:
            continue
        return calibration
    return None


def select_card_grid(candidates, count):
    """Prefer similarly sized cards laid out in two aligned rows."""
    if len(candidates) < count or len(candidates) > count + 5:
        return None
    best = None
    next_score = float("inf")
    best_score = float("inf")
    centers = [np.mean(corners, axis=0) for corners, _, _ in candidates]
    areas = [cv2.contourArea(corners) for corners, _, _ in candidates]
    for indices in combinations(range(len(candidates)), count):
        sizes = np.array([areas[i] for i in indices])
        if sizes.max() > 2.5 * sizes.min():
            continue
        step = float(np.sqrt(np.median(sizes)))
        ordered = sorted(indices, key=lambda i: centers[i][1])
        row_size = (count + 1) // 2
        rows = [ordered[:row_size], ordered[row_size:]]
        if count <= 2:
            rows = [ordered]
        rows = [sorted(row, key=lambda i: centers[i][0]) for row in rows]
        positions = [np.array([centers[i] for i in row]) for row in rows]
        score = float(np.std(np.log(sizes)))
        valid = True
        for points in positions:
            if len(points) <= 1:
                continue
            gaps = np.diff(points[:, 0])
            if gaps.min() < 0.54 * step or gaps.max() > 2.5 * gaps.min():
                valid = False
                break
            slope = (points[-1, 1] - points[0, 1]) / (points[-1, 0] - points[0, 0])
            residuals = points[:, 1] - slope * points[:, 0]
            if abs(slope) > 0.45 or np.ptp(residuals) > 0.48 * step:
                valid = False
                break
            score += float(np.std(gaps) / np.mean(gaps) + np.std(residuals) / step)
        if not valid:
            continue
        if len(positions) == 2:
            upper, lower = positions
            separation = np.mean(lower[:, 1]) - np.mean(upper[:, 1])
            if not 0.60 * step <= separation <= 2.85 * step:
                continue
            offset = abs(np.mean(lower[:, 0]) - np.mean(upper[:, 0]))
            if offset > 0.8 * step:
                continue
            score += offset / step
        if score < best_score:
            best, next_score, best_score = rows, best_score, score
        elif score < next_score:
            next_score = score
    if best is None or next_score - best_score < 0.10:
        return None
    return [candidates[i] for row in best for i in row]
