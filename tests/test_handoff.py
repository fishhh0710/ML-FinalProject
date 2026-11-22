import csv
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from board_tracker import BoardTracker, detect_white_board
from card_locator import CardLocator
from hand_tracking import PreviewWindow
from memory_game import GamePanel, MemoryGame
from session_timeline import EventTimeline, HandSample
from tests.make_fixture import fixture_images


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.back, cls.front = fixture_images()
        cls.locator = CardLocator()
        cls.front_result = cls.locator.locate(cls.front)
        cls.empty_result = {"cards": [], "tags": []}

    def test_apriltags_and_board(self):
        self.assertEqual(self.front_result["detected_marker_ids"], list(range(16)))
        self.assertEqual(len(self.front_result["cards"]), 8)
        self.assertIsNone(detect_white_board(np.full_like(self.back, 58)))
        config = detect_white_board(self.back)
        self.assertEqual([s["slot_id"] for s in config["slots"]], [f"S{i:02}" for i in range(1, 9)])

    def test_flip_debounce_and_missing_evidence(self):
        board = BoardTracker(detect_white_board(self.back), self.back)

        def step(image, result, hands=(), frames=3):
            return [event for _ in range(frames) for event in board.update(image, result, hands)]

        self.assertFalse(step(self.back, self.empty_result))
        self.assertFalse(step(self.front, self.front_result, frames=2))
        self.assertEqual(len(step(self.front, self.front_result, frames=1)), 8)
        self.assertFalse(step(np.full_like(self.back, 58), self.empty_result))
        self.assertFalse(step(self.front, self.front_result))
        events = step(self.back, self.empty_result, [[(200, 100), (260, 210)]])
        self.assertEqual(len(events), 7)
        self.assertEqual(step(self.back, self.empty_result), [("flip_down", "S01", "banana_1", "white_back")])
        self.assertEqual(len(step(self.front, self.front_result)), 8)

    def test_dwell_threshold_and_game_debug_preview(self):
        out = io.StringIO()
        timeline = EventTimeline(csv.writer(out), (1280, 720))
        for frame, ms in enumerate((0, 130, 260, 420, 550)):
            slot = "S01" if ms < 550 else "S02"
            timeline.update(frame, ms, [HandSample((100, 100), "Right", slot, "")], [])
        timeline.finish(4, 550)
        rows = list(csv.DictReader(io.StringIO(out.getvalue())))
        self.assertEqual([r["event"] for r in rows], ["point_start", "point_end"])
        self.assertEqual(rows[1]["duration_s"], "0.420")
        self.assertGreater(timeline.preview(self.back, 550).shape[1], 960)

    def test_pair_rules_and_panel(self):
        game = MemoryGame(8)

        def update(events):
            game.update(events, 1000, self.front, self.front_result["cards"])

        first = ("flip_up", "S01", "banana_1", "both_tags")
        second = ("flip_up", "S02", "grape_1", "both_tags")
        update([first, second])
        self.assertEqual(game.turns, 1)
        self.assertEqual(game.pairs, 0)
        update([("flip_down", "S01", "banana_1", "white_back"), first])
        update([("flip_down", "S02", "grape_1", "white_back")])
        self.assertTrue(game.awaiting_face_down)
        update([("flip_down", "S01", "banana_1", "white_back")])
        update([first, ("flip_up", "S05", "banana_2", "both_tags")])
        self.assertEqual(game.pairs, 1)
        self.assertEqual(game.turns, 2)
        self.assertEqual(GamePanel(game).render(self.front).shape[0], 682)

    def test_preview_ignores_initial_invisible_state_then_detects_close(self):
        preview = PreviewWindow("test")
        with patch.object(cv2, "waitKey", return_value=-1), \
                patch.object(cv2, "getWindowProperty", side_effect=[-1, 0, 1, -1, -1]), \
                patch("hand_tracking.time.monotonic", side_effect=[1, 1.4]):
            for _ in range(4):
                self.assertEqual(preview.poll()[1], "")
            self.assertEqual(preview.poll()[1], "preview window closed")
        with patch.object(cv2, "waitKey", return_value=ord("Q")):
            self.assertEqual(preview.poll()[1], "Q / Esc pressed")


if __name__ == "__main__":
    unittest.main()
