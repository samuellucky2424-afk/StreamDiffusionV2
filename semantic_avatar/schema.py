"""Semantic packet schema used by the first pose-conditioning bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_landmarks(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []

    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            x = _float_or_none(item.get("x"))
            y = _float_or_none(item.get("y"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = _float_or_none(item[0])
            y = _float_or_none(item[1])
        else:
            continue
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


@dataclass
class SemanticPacket:
    """Compact semantic motion packet from the existing MediaPipe frontend."""

    t: float | None = None
    frame_id: int | None = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    head_x: float = 0.5
    head_y: float = 0.34
    shoulder_x: float = 0.5
    shoulder_y: float = 0.62
    confidence: float = 1.0
    pose_landmarks: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SemanticPacket":
        if data is None:
            return cls(confidence=0.0)

        def pick(*keys: str, default: float | None = None) -> float | None:
            for key in keys:
                value = _float_or_none(data.get(key))
                if value is not None:
                    return value
            return default

        frame_id = data.get("frameId", data.get("frame_id"))
        try:
            frame_id = int(frame_id) if frame_id is not None else None
        except (TypeError, ValueError):
            frame_id = None

        landmarks = (
            data.get("poseLandmarks")
            or data.get("pose_landmarks")
            or data.get("landmarks")
            or []
        )

        return cls(
            t=pick("t", "timestamp"),
            frame_id=frame_id,
            yaw=pick("yaw", default=0.0) or 0.0,
            pitch=pick("pitch", default=0.0) or 0.0,
            roll=pick("roll", default=0.0) or 0.0,
            head_x=pick("headX", "head_x", default=0.5) or 0.5,
            head_y=pick("headY", "head_y", default=0.34) or 0.34,
            shoulder_x=pick("shoulderX", "shoulder_x", default=0.5) or 0.5,
            shoulder_y=pick("shoulderY", "shoulder_y", default=0.62) or 0.62,
            confidence=pick("confidence", default=1.0) or 0.0,
            pose_landmarks=_normalize_landmarks(landmarks),
        )

    def to_debug_dict(self) -> dict[str, float | int | None]:
        return {
            "t": self.t,
            "frame_id": self.frame_id,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "head_x": self.head_x,
            "head_y": self.head_y,
            "shoulder_x": self.shoulder_x,
            "shoulder_y": self.shoulder_y,
            "confidence": self.confidence,
            "pose_landmark_count": len(self.pose_landmarks),
        }
