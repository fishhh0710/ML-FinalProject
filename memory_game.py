"""Real-time two-card memory game scoreboard for the physical board."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FRUIT_LABELS = {
    "banana": ("香蕉", "Banana", "#f5cf70"),
    "grape": ("葡萄", "Grape", "#c0a5ed"),
    "lemon": ("檸檬", "Lemon", "#e8e480"),
    "watermelon": ("西瓜", "Watermelon", "#85d5ae"),
}
PANEL_WIDTH, PANEL_HEIGHT = 494, 682


@dataclass
class CardFace:
    slot: str
    card_id: str
    thumbnail: Image.Image | None

    @property
    def fruit(self):
        return self.card_id.rsplit("_", 1)[0]


@dataclass
class Turn:
    number: int
    first: CardFace
    second: CardFace
    matched: bool
    time_ms: int


class MemoryGame:
    def __init__(self, slots):
        self.goal = slots // 2
        self.matched_slots = set()
        self.awaiting_face_down = set()
        self.selection = []
        self.completed = deque(maxlen=4)
        self.turns = 0
        self.revision = 0

    @property
    def pairs(self):
        return len(self.matched_slots) // 2

    @property
    def finished(self):
        return self.pairs >= self.goal

    def update(self, events, ms, frame, visible_cards):
        for event, slot, card_id, _ in events:
            if self.finished:
                break
            if event == "flip_down":
                if slot in self.awaiting_face_down:
                    self.awaiting_face_down.remove(slot)
                    if not self.awaiting_face_down:
                        self.selection.clear()
                    self.revision += 1
                elif len(self.selection) == 1 and self.selection[0].slot == slot:
                    self.selection.clear()
                    self.revision += 1
                continue
            if event != "flip_up" or slot in self.matched_slots:
                continue
            if self.awaiting_face_down:
                self.awaiting_face_down.add(slot)
                continue
            if any(face.slot == slot for face in self.selection):
                continue
            if len(self.selection) == 2:
                self.selection.clear()
            face = CardFace(slot, card_id, card_thumbnail(frame, card_id, visible_cards))
            self.selection.append(face)
            self.revision += 1
            if len(self.selection) == 1:
                continue
            first, second = self.selection
            self.turns += 1
            matched = first.fruit == second.fruit
            self.completed.appendleft(Turn(self.turns, first, second, matched, ms))
            if matched:
                self.matched_slots.update((first.slot, second.slot))
            else:
                self.awaiting_face_down.update((first.slot, second.slot))


def card_thumbnail(frame, card_id, visible_cards):
    card = next((item for item in visible_cards if item["card_id"] == card_id), None)
    if card is None:
        return None
    corners = np.float32(card["corners_px"])
    if (not np.isfinite(corners).all() or np.any(corners < 0)
            or np.any(corners >= [frame.shape[1], frame.shape[0]])):
        return None
    target = np.float32([[0, 0], [131, 0], [131, 181], [0, 181]])
    transform = cv2.getPerspectiveTransform(corners, target)
    crop = cv2.warpPerspective(frame, transform, (132, 182))
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class GamePanel:
    def __init__(self, game):
        self.game = game
        font_root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        self.regular = font_root / "msjh.ttc"
        self.bold = font_root / "msjhbd.ttc"
        self.zh = self.regular.is_file()
        self.cached_revision = -1
        self.panel = None

    def text(self, zh, en):
        return zh if self.zh else en

    @lru_cache(maxsize=32)
    def font(self, size, strong):
        if self.zh:
            path = self.bold if strong and self.bold.is_file() else self.regular
        else:
            path = "DejaVuSans-Bold.ttf" if strong else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            return ImageFont.load_default(size=size)

    def render(self, frame):
        if self.cached_revision != self.game.revision:
            self.panel = self.build_panel()
            self.cached_revision = self.game.revision
        height, width = frame.shape[:2]
        ratio = min(1.0, 860 / width, PANEL_HEIGHT / height)
        camera = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
        canvas = np.full((PANEL_HEIGHT, camera.shape[1] + PANEL_WIDTH, 3), (23, 22, 22), dtype=np.uint8)
        top = (PANEL_HEIGHT - camera.shape[0]) // 2
        canvas[top:top + camera.shape[0], :camera.shape[1]] = camera
        canvas[:, camera.shape[1]:] = self.panel
        return canvas

    def build_panel(self):
        game = self.game
        panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#121b2d")
        draw = ImageDraw.Draw(panel)

        def label(x, y, text, size=16, color="#e9effa", strong=False):
            draw.text((x, y), text, font=self.font(size, strong), fill=color)

        def fruit_name(face):
            zh, en, color = FRUIT_LABELS.get(face.fruit, (face.fruit, face.fruit, "#d7e0f3"))
            return self.text(zh, en), color

        draw.rounded_rectangle((24, 22, 98, 48), radius=12, fill="#214239")
        draw.ellipse((34, 32, 41, 39), fill="#79dfaa")
        label(48, 25, "LIVE", 14, "#a4efc5", True)
        label(24, 55, self.text("記憶配對", "Memory Match"), 32, strong=True)
        label(351, 59, f"{game.pairs} / {game.goal}", 25, "#9be8bb", True)
        label(352, 86, self.text("成功配對", "PAIRS"), 13, "#a5b7d1")
        draw.rounded_rectangle((25, 108, 469, 114), radius=3, fill="#273751")
        if game.pairs:
            draw.rounded_rectangle((25, 108, 25 + round(444 * game.pairs / game.goal), 114),
                                   radius=3, fill="#83dbac")

        if game.finished:
            status = self.text("全部配對完成", "All pairs matched")
            round_number = game.turns
        elif game.awaiting_face_down:
            status = self.text("配對未成功，請蓋回翻開的牌", "No match. Turn the open cards back.")
            round_number = game.turns
        elif len(game.selection) == 2:
            status = self.text("配對成功，可開始下一回合", "Match! Ready for the next turn.")
            round_number = game.turns
        elif game.selection:
            status = self.text("等待第二張牌", "Waiting for the second card")
            round_number = game.turns + 1
        else:
            status = self.text("等待第一張牌", "Waiting for the first card")
            round_number = game.turns + 1
        label(24, 130, self.text(f"第 {round_number:02} 回合", f"TURN {round_number:02}"), 22, strong=True)
        label(24, 165, status, 16, "#aebed6")

        for i, left in enumerate((24, 252)):
            right = left + 216
            face = game.selection[i] if i < len(game.selection) else None
            draw.rounded_rectangle((left, 196, right, 402), radius=16, fill="#1d2a42",
                                   outline="#344762", width=1)
            label(left + 14, 209, self.text(f"第 {i + 1} 張", f"CARD {i + 1}"), 13, "#adbdd6")
            if face:
                name, accent = fruit_name(face)
                if face.thumbnail:
                    thumb = ImageOps.contain(face.thumbnail, (105, 130))
                    panel.paste(thumb, (left + (216 - thumb.width) // 2, 232))
                else:
                    draw.rounded_rectangle((left + 73, 255, left + 143, 325), radius=16, fill=accent)
                label(left + 15, 370, name, 20, accent, True)
                label(right - 57, 374, face.slot, 14, "#b5c5dc")
            else:
                draw.rounded_rectangle((left + 69, 240, left + 147, 349), radius=12,
                                       fill="#283953", outline="#3e5675")
                label(left + 99, 274, "?", 29, "#9cb0d0", True)
                label(left + 15, 371, self.text("待翻開", "Face down"), 17, "#a4b5cf")

        label(24, 429, self.text("回合紀錄", "Recent turns"), 22, strong=True)
        if not game.completed:
            label(26, 479, self.text("翻開兩張牌後會顯示結果", "Results appear after two flips."), 15, "#9cacc6")
        for i, turn in enumerate(game.completed):
            y = 467 + i * 43
            a, _ = fruit_name(turn.first)
            b, _ = fruit_name(turn.second)
            draw.rounded_rectangle((24, y, 469, y + 37), radius=9, fill="#1d2a42")
            label(34, y + 7, f"{turn.number:02}", 15, "#a9bddd", True)
            label(70, y + 7, f"{a} {turn.first.slot}  /  {b} {turn.second.slot}", 15)
            verdict = self.text("成功", "Match") if turn.matched else self.text("未配對", "No match")
            label(354, y + 7, verdict, 14, "#9be8bb" if turn.matched else "#f1aa91", True)
            label(418, y + 9, f"{turn.time_ms / 1000:.1f}s", 13, "#a3b5ce")
        label(24, 653, self.text("D 切換事件除錯　 Q / Esc 結束", "D  Event debug   /   Q or Esc  Quit"), 14, "#99acc8")
        return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
