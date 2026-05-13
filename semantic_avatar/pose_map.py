"""Lightweight server-side semantic pose map generation."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .schema import SemanticPacket


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(_clamp(x) * (width - 1)), int(_clamp(y) * (height - 1))


class PoseMapRenderer:
    """Render compact semantic controls into RGB conditioning maps."""

    def __init__(self, width: int, height: int, *, line_width: int | None = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.line_width = line_width or max(2, round(min(self.width, self.height) / 160))

    def render(self, packet: SemanticPacket) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(image)

        head = _point(packet.head_x, packet.head_y, self.width, self.height)
        shoulder = _point(packet.shoulder_x, packet.shoulder_y, self.width, self.height)

        confidence = _clamp(packet.confidence)
        dim = int(64 * (1.0 - confidence))
        torso_color = (64, 220 - dim, 255 - dim)
        head_color = (255 - dim, 236 - dim, 96)
        axis_color = (255, 96, 96)
        landmark_color = (128, 255, 160)

        shoulder_span = int(self.width * 0.28)
        neck_y = int(head[1] + (shoulder[1] - head[1]) * 0.55)
        left_shoulder = (shoulder[0] - shoulder_span // 2, shoulder[1])
        right_shoulder = (shoulder[0] + shoulder_span // 2, shoulder[1])
        neck = (head[0], neck_y)

        draw.line([left_shoulder, neck, right_shoulder], fill=torso_color, width=self.line_width)
        draw.line([head, neck], fill=torso_color, width=self.line_width)

        radius = max(8, int(min(self.width, self.height) * 0.055))
        draw.ellipse(
            [head[0] - radius, head[1] - radius, head[0] + radius, head[1] + radius],
            outline=head_color,
            width=self.line_width,
        )

        yaw_offset = math.sin(math.radians(packet.yaw)) * radius * 0.75
        pitch_offset = math.sin(math.radians(packet.pitch)) * radius * 0.55
        roll = math.radians(packet.roll)
        axis_length = radius * 1.35
        axis_dx = math.sin(roll) * axis_length
        axis_dy = math.cos(roll) * axis_length
        nose = (int(head[0] + yaw_offset), int(head[1] + pitch_offset))
        draw.line([head, nose], fill=axis_color, width=self.line_width)
        draw.line(
            [
                (int(nose[0] - axis_dx * 0.35), int(nose[1] + axis_dy * 0.35)),
                (int(nose[0] + axis_dx * 0.35), int(nose[1] - axis_dy * 0.35)),
            ],
            fill=(96, 160, 255),
            width=max(1, self.line_width - 1),
        )

        for x, y in packet.pose_landmarks:
            px, py = _point(x, y, self.width, self.height)
            dot = max(1, self.line_width)
            draw.ellipse([px - dot, py - dot, px + dot, py + dot], fill=landmark_color)

        return image

    def render_array(self, packet: SemanticPacket, *, normalize: bool = True) -> np.ndarray:
        image = self.render(packet)
        array = np.asarray(image, dtype=np.float32)
        if normalize:
            array = array / 127.5 - 1.0
        return array
