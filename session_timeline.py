"""Convert per-frame fingertip observations into card-board events."""

from collections import deque
from dataclasses import dataclass
from math import dist

import cv2
import numpy as np


COLUMNS = (
    "frame", "time_s", "event", "slot_id", "card_id", "hand_track_id",
    "handedness", "point_started_s", "point_last_seen_s", "duration_s", "evidence",
)


@dataclass
class HandSample:
    wrist: tuple
    handedness: str
    slot_id: str
    card_id: str


@dataclass
class HandTrack:
    sample: HandSample
    seen_ms: int
    slot_id: str = ""
    card_id: str = ""
    start_ms: int = 0
    point_seen_ms: int = 0
    confirmed: bool = False


def seconds(ms):
    return f"{ms / 1000:.3f}"


class EventTimeline:
    def __init__(self, writer, frame_size, dwell_ms=400, gap_ms=250):
        self.writer = writer
        writer.writerow(COLUMNS)
        self.match_distance = 0.16 * dist((0, 0), frame_size)
        self.dwell_ms = dwell_ms
        self.gap_ms = gap_ms
        self.tracks = {}
        self.next_id = 1
        self.recent = deque(maxlen=18)
        self.event_count = 0

    def emit(self, frame, ms, event, slot, card, evidence, hand_id="", handedness="",
             start="", last_seen="", duration=""):
        self.writer.writerow((frame, seconds(ms), event, slot, card, hand_id, handedness,
                              start, last_seen, duration, evidence))
        self.recent.appendleft((ms, event, slot, card, hand_id, duration))
        self.event_count += 1

    def point_event(self, frame, ms, event, hand_id, track, evidence):
        self.emit(frame, ms, event, track.slot_id, track.card_id, evidence, hand_id,
                  track.sample.handedness, seconds(track.start_ms), seconds(track.point_seen_ms),
                  seconds(track.point_seen_ms - track.start_ms))

    def close_point(self, frame, ms, hand_id, track, evidence):
        if track.confirmed:
            self.point_event(frame, ms, "point_end", hand_id, track, evidence)
        track.slot_id = ""
        track.card_id = ""
        track.confirmed = False

    def update(self, frame, ms, hands, flips):
        for event, slot, card, evidence in flips:
            self.emit(frame, ms, event, slot, card, evidence)
        for hand_id, track in list(self.tracks.items()):
            if ms - track.seen_ms > self.gap_ms:
                self.close_point(frame, ms, hand_id, track, "hand_lost")
                del self.tracks[hand_id]

        matches = []
        for hand_id, track in self.tracks.items():
            for i, sample in enumerate(hands):
                distance = dist(track.sample.wrist, sample.wrist)
                if distance <= self.match_distance:
                    if track.sample.handedness != sample.handedness:
                        distance += self.match_distance * 0.25
                    matches.append((distance, hand_id, i))
        used_tracks, assigned = set(), {}
        for _, hand_id, i in sorted(matches):
            if hand_id not in used_tracks and i not in assigned:
                used_tracks.add(hand_id)
                assigned[i] = hand_id

        for i, sample in enumerate(hands):
            if i not in assigned:
                assigned[i] = self.next_id
                self.tracks[self.next_id] = HandTrack(sample, ms)
                self.next_id += 1
            hand_id = assigned[i]
            track = self.tracks[hand_id]
            track.sample, track.seen_ms = sample, ms
            if track.slot_id and ms - track.point_seen_ms > self.gap_ms:
                self.close_point(frame, ms, hand_id, track, "slot_lost")
            if not sample.slot_id:
                continue
            if track.slot_id != sample.slot_id:
                self.close_point(frame, ms, hand_id, track, "slot_changed")
                track.slot_id, track.card_id, track.start_ms = sample.slot_id, sample.card_id, ms
            if sample.card_id and not track.card_id:
                track.card_id = sample.card_id
            track.point_seen_ms = ms
            if not track.confirmed and ms - track.start_ms >= self.dwell_ms:
                track.confirmed = True
                self.point_event(frame, ms, "point_start", hand_id, track, "dwell_threshold")

    def finish(self, frame, ms):
        for hand_id, track in self.tracks.items():
            self.close_point(frame, ms, hand_id, track, "session_end")

    def preview(self, frame, ms):
        """Add a compact event panel to the live preview only."""
        height, width = frame.shape[:2]
        scale = min(1.0, 960 / width, 700 / height)
        view = cv2.resize(frame, (round(width * scale), round(height * scale)))
        panel = np.full((max(390, view.shape[0]), 374, 3), (30, 29, 27), dtype=np.uint8)

        def label(text, y, color=(224, 224, 224), scale=0.49):
            cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        label("Session events", 29, (245, 245, 245), 0.73)
        label(f"{seconds(ms)} s   |   {self.event_count} events", 52, (174, 174, 174))
        label("Fingertip dwell", 85, (240, 215, 100), 0.57)
        y = 108
        active = [(i, track) for i, track in self.tracks.items() if track.slot_id]
        for hand_id, track in active[:4]:
            elapsed = seconds(track.point_seen_ms - track.start_ms)
            state = "active" if track.confirmed else "pending"
            card = track.card_id or "unrevealed"
            label(f"H{hand_id} {track.sample.handedness}  {track.slot_id}  {card}  {elapsed} s", y)
            label(state, y + 16, (145, 189, 142) if track.confirmed else (160, 160, 160), 0.43)
            y += 37
        if not active:
            label("No fingertip dwell in a slot", y, (160, 160, 160))
            y += 20
        y += 12
        cv2.line(panel, (14, y), (360, y), (75, 75, 75), 1)
        y += 24
        label("Recent events (newest first)", y, (240, 215, 100), 0.55)
        for event_ms, event, slot, card, hand_id, elapsed in self.recent:
            y += 22
            if y >= panel.shape[0] - 27:
                break
            details = f"H{hand_id} {card or '--'}" if hand_id else card
            if event == "point_end":
                details += f" {elapsed}s"
            label(f"{event_ms / 1000:6.2f}  {event:<11} {slot}  {details}", y,
                  (198, 225, 157) if event.startswith("flip") else (218, 218, 218), 0.44)
        label("Q / Esc: stop and save", panel.shape[0] - 12, (163, 163, 163), 0.45)
        if view.shape[0] < panel.shape[0]:
            view = cv2.copyMakeBorder(view, 0, panel.shape[0] - view.shape[0], 0, 0,
                                      cv2.BORDER_CONSTANT, value=(24, 24, 24))
        return np.hstack((view, panel))
