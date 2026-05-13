"""Semantic packet parsing and lightweight pose utilities.

This module is intentionally transport/model agnostic. It normalizes the
browser's MediaPipe-derived JSON into one shape that the adapter can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _pick_float(data: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = _float_or_none(data.get(key))
        if value is not None:
            return value
    return default


def _pick_nested_float(data: Mapping[str, Any], path: Sequence[str], default: float = 0.0) -> float:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return _float_or_none(current) if _float_or_none(current) is not None else default


def _normalize_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        x = _float_or_none(value.get("x"))
        y = _float_or_none(value.get("y"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _float_or_none(value[0])
        y = _float_or_none(value[1])
    else:
        return None
    if x is None or y is None:
        return None
    return (_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0))


def _normalize_landmarks(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        point = _normalize_point(item)
        if point is not None:
            points.append(point)
    return points


def _normalize_landmark_groups(value: Any) -> list[list[tuple[float, float]]]:
    if not isinstance(value, list):
        return []
    groups: list[list[tuple[float, float]]] = []
    for item in value:
        source = item.get("landmarks") if isinstance(item, Mapping) else item
        points = _normalize_landmarks(source)
        if points:
            groups.append(points)
    return groups


@dataclass(slots=True)
class SemanticPacket:
    """Normalized semantic motion packet sent by the browser.

    Angles are degrees. Unit controls are normalized to ``0..1``. Points are
    normalized image coordinates where ``(0, 0)`` is top-left.
    """

    timestamp: float | None = None
    frame_id: int | None = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    blink: float = 0.0
    blink_left: float = 0.0
    blink_right: float = 0.0
    mouth_open: float = 0.0
    jaw_open: float = 0.0
    mouth_upper_lower: float = 0.0
    lip_width: float = 0.0
    lip_corner_stretch: float = 0.0
    brow: float = 0.0
    smile: float = 0.0
    cheek: float = 0.0
    eye_x: float = 0.0
    eye_y: float = 0.0
    head_x: float = 0.5
    head_y: float = 0.35
    shoulder_x: float = 0.5
    shoulder_y: float = 0.62
    shoulder_rotation: float = 0.0
    torso_rotation: float = 0.0
    neck_rotation: float = 0.0
    confidence: float = 1.0
    face_landmarks: list[tuple[float, float]] = field(default_factory=list)
    pose_landmarks: list[tuple[float, float]] = field(default_factory=list)
    hand_landmarks: list[list[tuple[float, float]]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SemanticPacket":
        if not data:
            return cls(confidence=0.0)

        payload = dict(data)
        timestamp = _pick_float(payload, "timestamp", "wallTime", "t", default=0.0)
        if timestamp <= 0:
            timestamp = None

        frame_id_raw = payload.get("frameId", payload.get("frame_id"))
        try:
            frame_id = int(frame_id_raw) if frame_id_raw is not None else None
        except (TypeError, ValueError):
            frame_id = None

        left_shoulder = _normalize_point(payload.get("shoulderLeft"))
        right_shoulder = _normalize_point(payload.get("shoulderRight"))
        shoulders = payload.get("shoulders")
        if isinstance(shoulders, Mapping):
            left_shoulder = left_shoulder or _normalize_point(shoulders.get("left"))
            right_shoulder = right_shoulder or _normalize_point(shoulders.get("right"))

        compact_left = _normalize_point(payload.get("sl"))
        compact_right = _normalize_point(payload.get("sr"))
        left_shoulder = left_shoulder or compact_left
        right_shoulder = right_shoulder or compact_right

        if left_shoulder and right_shoulder:
            shoulder_x = (left_shoulder[0] + right_shoulder[0]) * 0.5
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) * 0.5
        else:
            shoulder_x = _pick_float(payload, "shoulderX", "shoulder_x", default=0.5)
            shoulder_y = _pick_float(payload, "shoulderY", "shoulder_y", default=0.62)

        blink_left = _clamp(_pick_float(payload, "blinkLeft", "blink_left", "bl", default=0.0), 0.0, 1.0)
        blink_right = _clamp(_pick_float(payload, "blinkRight", "blink_right", "br", default=blink_left), 0.0, 1.0)
        blink = _clamp(_pick_float(payload, "blink", default=(blink_left + blink_right) * 0.5), 0.0, 1.0)

        eye_direction = payload.get("eyeDirection")
        eye_x_default = _pick_nested_float(eye_direction, ("x",), default=0.0) if isinstance(eye_direction, Mapping) else 0.0
        eye_y_default = _pick_nested_float(eye_direction, ("y",), default=0.0) if isinstance(eye_direction, Mapping) else 0.0

        face_landmarks = (
            payload.get("faceLandmarks")
            or payload.get("face_landmarks")
            or payload.get("landmarks")
            or []
        )
        pose_landmarks = payload.get("poseLandmarks") or payload.get("pose_landmarks") or []
        hand_landmarks = payload.get("handLandmarks") or payload.get("hand_landmarks") or payload.get("hands") or []

        head = _normalize_point(payload.get("headPosition"))
        head_x = head[0] if head else _pick_float(payload, "headX", "head_x", default=0.5)
        head_y = head[1] if head else _pick_float(payload, "headY", "head_y", default=0.35)
        jaw_open = _clamp(_pick_float(payload, "jawOpen", "jaw_open", "jaw", default=0.0), 0.0, 1.0)
        mouth_open = _clamp(
            _pick_float(payload, "mouthOpen", "mouth_open", "mo", default=jaw_open),
            0.0,
            1.0,
        )
        neck_rotation = _clamp(
            _pick_float(payload, "neckRotation", "neck_rotation", "nr", default=0.0),
            -90.0,
            90.0,
        )

        return cls(
            timestamp=timestamp,
            frame_id=frame_id,
            yaw=_clamp(_pick_float(payload, "yaw", "y", default=0.0), -90.0, 90.0),
            pitch=_clamp(_pick_float(payload, "pitch", "p", default=0.0), -60.0, 60.0),
            roll=_clamp(_pick_float(payload, "roll", "r", default=0.0), -90.0, 90.0),
            blink=blink,
            blink_left=blink_left,
            blink_right=blink_right,
            mouth_open=mouth_open,
            jaw_open=jaw_open,
            mouth_upper_lower=_clamp(
                _pick_float(payload, "mouthUpperLower", "lipSeparation", "lipGap", "ul", default=mouth_open),
                0.0,
                1.0,
            ),
            lip_width=_clamp(_pick_float(payload, "lipWidth", "mouthWidth", "lw", default=0.0), 0.0, 1.0),
            lip_corner_stretch=_clamp(
                _pick_float(payload, "lipCornerStretch", "cornerStretch", "lcs", default=0.0),
                0.0,
                1.0,
            ),
            brow=_clamp(_pick_float(payload, "brow", "browRaise", "brow_raise", default=0.0), 0.0, 1.0),
            smile=_clamp(_pick_float(payload, "smile", "sm", default=0.0), 0.0, 1.0),
            cheek=_clamp(_pick_float(payload, "cheek", "cheekMovement", "cheekRaise", "ck", default=0.0), 0.0, 1.0),
            eye_x=_clamp(_pick_float(payload, "pupilX", "eyeDirectionX", "ex", default=eye_x_default), -1.0, 1.0),
            eye_y=_clamp(_pick_float(payload, "pupilY", "eyeDirectionY", "ey", default=eye_y_default), -1.0, 1.0),
            head_x=_clamp(head_x, 0.0, 1.0),
            head_y=_clamp(head_y, 0.0, 1.0),
            shoulder_x=_clamp(shoulder_x, 0.0, 1.0),
            shoulder_y=_clamp(shoulder_y, 0.0, 1.0),
            shoulder_rotation=_clamp(_pick_float(payload, "shoulderRotation", "shoulder_rotation", "torsoAngle", "ta", default=0.0), -90.0, 90.0),
            torso_rotation=_clamp(_pick_float(payload, "torsoRotation", "torso_rotation", "ta", default=0.0), -90.0, 90.0),
            neck_rotation=neck_rotation,
            confidence=_clamp(_pick_float(payload, "confidence", "poseConfidence", "c", default=1.0), 0.0, 1.0),
            face_landmarks=_normalize_landmarks(face_landmarks),
            pose_landmarks=_normalize_landmarks(pose_landmarks),
            hand_landmarks=_normalize_landmark_groups(hand_landmarks),
            raw=payload,
        )

    def to_debug_dict(self) -> dict[str, float | int | None]:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "yaw": round(self.yaw, 3),
            "pitch": round(self.pitch, 3),
            "roll": round(self.roll, 3),
            "blink": round(self.blink, 3),
            "mouth_open": round(self.mouth_open, 3),
            "jaw_open": round(self.jaw_open, 3),
            "mouth_upper_lower": round(self.mouth_upper_lower, 3),
            "lip_width": round(self.lip_width, 3),
            "lip_corner_stretch": round(self.lip_corner_stretch, 3),
            "brow": round(self.brow, 3),
            "smile": round(self.smile, 3),
            "cheek": round(self.cheek, 3),
            "head_x": round(self.head_x, 3),
            "head_y": round(self.head_y, 3),
            "shoulder_rotation": round(self.shoulder_rotation, 3),
            "neck_rotation": round(self.neck_rotation, 3),
            "confidence": round(self.confidence, 3),
            "face_landmarks": len(self.face_landmarks),
            "pose_landmarks": len(self.pose_landmarks),
            "hand_landmarks": sum(len(hand) for hand in self.hand_landmarks),
        }


__all__ = ["SemanticPacket"]
