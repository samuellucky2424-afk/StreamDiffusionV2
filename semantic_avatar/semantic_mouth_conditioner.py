"""Temporal mouth conditioning for realtime semantic avatar driving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Deque

from .semantic_pose import SemanticPacket

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291
FACE_TOP = 10
FACE_BOTTOM = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _point(points: list[tuple[float, float]], index: int) -> tuple[float, float] | None:
    if 0 <= index < len(points):
        return points[index]
    return None


@dataclass(slots=True)
class MouthFeatures:
    mouth_open: float = 0.0
    jaw_open: float = 0.0
    lip_separation: float = 0.0
    lip_width: float = 0.0
    corner_stretch: float = 0.0
    smile: float = 0.0
    compression: float = 0.0
    roundness: float = 0.0
    velocity: float = 0.0
    viseme: str = "closed"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mouth_open": round(self.mouth_open, 4),
            "jaw_open": round(self.jaw_open, 4),
            "lip_separation": round(self.lip_separation, 4),
            "lip_width": round(self.lip_width, 4),
            "corner_stretch": round(self.corner_stretch, 4),
            "smile": round(self.smile, 4),
            "compression": round(self.compression, 4),
            "roundness": round(self.roundness, 4),
            "velocity": round(self.velocity, 4),
            "viseme": self.viseme,
        }


class SemanticMouthConditioner:
    """Smooth mouth controls while preserving fast speech-like changes."""

    def __init__(self, history_size: int = 16) -> None:
        self.history: Deque[MouthFeatures] = deque(maxlen=max(4, int(history_size)))
        self.state: MouthFeatures | None = None
        self.last_t = perf_counter()

    def reset(self) -> None:
        self.history.clear()
        self.state = None
        self.last_t = perf_counter()

    def update(self, packet: SemanticPacket) -> MouthFeatures:
        now = perf_counter()
        dt = max(1 / 120, min(0.25, now - self.last_t))
        self.last_t = now

        target = self._raw_features(packet)
        previous = self.state
        if previous is None:
            self.state = target
            self.history.append(target)
            return target

        velocity = abs(target.jaw_open - previous.jaw_open) / dt
        open_alpha = _clamp(0.42 + velocity * 0.035, 0.42, 0.82)
        close_alpha = _clamp(0.30 + velocity * 0.025, 0.30, 0.66)

        jaw_open = self._smooth(previous.jaw_open, target.jaw_open, open_alpha, close_alpha)
        lip_separation = self._smooth(previous.lip_separation, target.lip_separation, open_alpha, close_alpha)
        mouth_open = max(jaw_open * 0.78, lip_separation, self._smooth(previous.mouth_open, target.mouth_open, open_alpha, close_alpha))
        smile = self._smooth(previous.smile, target.smile, 0.38, 0.30)
        lip_width = self._smooth(previous.lip_width, target.lip_width, 0.34 + smile * 0.12, 0.28)
        corner_stretch = self._smooth(previous.corner_stretch, target.corner_stretch, 0.46, 0.34)
        compression = self._smooth(previous.compression, target.compression, 0.34, 0.38)
        roundness = self._smooth(previous.roundness, target.roundness, open_alpha, close_alpha)

        state = MouthFeatures(
            mouth_open=_clamp(mouth_open),
            jaw_open=_clamp(jaw_open),
            lip_separation=_clamp(lip_separation),
            lip_width=_clamp(lip_width),
            corner_stretch=_clamp(corner_stretch),
            smile=_clamp(smile),
            compression=_clamp(compression),
            roundness=_clamp(roundness),
            velocity=round(velocity, 4),
            viseme=self._viseme(jaw_open, lip_width, roundness, smile),
        )
        self.state = state
        self.history.append(state)
        return state

    def _raw_features(self, packet: SemanticPacket) -> MouthFeatures:
        landmark_gap, landmark_width = self._landmark_mouth(packet.face_landmarks)
        jaw_open = max(packet.jaw_open, packet.mouth_open * 0.92, landmark_gap)
        lip_separation = max(packet.mouth_upper_lower, landmark_gap, packet.mouth_open * 0.82)
        lip_width = packet.lip_width or landmark_width or _clamp(0.36 + packet.smile * 0.34)
        corner_stretch = max(packet.lip_corner_stretch, packet.smile * 0.72, _clamp((lip_width - 0.43) * 1.55))
        roundness = _clamp(jaw_open * (1.0 - corner_stretch * 0.45))
        compression = _clamp((1.0 - lip_separation) * max(0.0, 0.32 - jaw_open) + corner_stretch * 0.08)
        mouth_open = max(packet.mouth_open, jaw_open * 0.88, lip_separation)

        return MouthFeatures(
            mouth_open=_clamp(mouth_open),
            jaw_open=_clamp(jaw_open),
            lip_separation=_clamp(lip_separation),
            lip_width=_clamp(lip_width),
            corner_stretch=_clamp(corner_stretch),
            smile=_clamp(packet.smile),
            compression=_clamp(compression),
            roundness=roundness,
            velocity=0.0,
            viseme=self._viseme(jaw_open, lip_width, roundness, packet.smile),
        )

    def _landmark_mouth(self, points: list[tuple[float, float]]) -> tuple[float, float]:
        if len(points) <= max(FACE_BOTTOM, RIGHT_CORNER):
            return (0.0, 0.0)

        vertical_scale = max(_distance(_point(points, FACE_TOP), _point(points, FACE_BOTTOM)), 0.001)
        horizontal_scale = max(
            _distance(_point(points, LEFT_EYE_OUTER), _point(points, RIGHT_EYE_OUTER)),
            vertical_scale * 0.55,
            0.001,
        )
        lip_gap = _distance(_point(points, UPPER_LIP), _point(points, LOWER_LIP))
        lip_width = _distance(_point(points, LEFT_CORNER), _point(points, RIGHT_CORNER))
        return (_clamp((lip_gap / vertical_scale) * 5.4), _clamp((lip_width / horizontal_scale) * 1.45))

    @staticmethod
    def _smooth(previous: float, target: float, rising_alpha: float, falling_alpha: float, deadzone: float = 0.006) -> float:
        if abs(target - previous) < deadzone:
            return previous
        alpha = rising_alpha if target > previous else falling_alpha
        return previous + (target - previous) * _clamp(alpha, 0.01, 1.0)

    @staticmethod
    def _viseme(jaw_open: float, lip_width: float, roundness: float, smile: float) -> str:
        if jaw_open > 0.62:
            return "wide_open" if lip_width > 0.62 else "open"
        if roundness > 0.42:
            return "round"
        if smile > 0.52 or lip_width > 0.72:
            return "wide"
        if jaw_open > 0.18:
            return "mid"
        return "closed"


__all__ = ["MouthFeatures", "SemanticMouthConditioner"]
