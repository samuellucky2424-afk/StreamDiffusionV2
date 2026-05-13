"""Dense facial semantic encoder for realtime avatar conditioning.

The encoder stays intentionally lightweight: it converts semantic packets into
compact controls plus raster guidance maps that can be folded into the existing
image-conditioning path before StreamDiffusionV2 denoising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from .semantic_mouth_conditioner import MouthFeatures, SemanticMouthConditioner
from .semantic_pose import SemanticPacket

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [263, 387, 385, 362, 380, 373, 263]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _lerp(previous: float, target: float, alpha: float) -> float:
    return previous + (target - previous) * alpha


def _xy(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return (point[0] * width, point[1] * height)


def _path(points: list[tuple[float, float]], indices: list[int], width: int, height: int) -> list[tuple[float, float]]:
    if not points or max(indices) >= len(points):
        return []
    return [_xy(points[index], width, height) for index in indices]


@dataclass(slots=True)
class FacialConditioning:
    control: dict[str, Any]
    conditioning_image: Image.Image
    face_map: Image.Image
    mouth_mask: Image.Image
    eye_mask: Image.Image
    body_map: Image.Image
    shape: tuple[int, int, int]
    metrics: dict[str, Any] = field(default_factory=dict)


class SemanticFaceEncoder:
    """Encode dense face/body semantics into compact realtime conditioning."""

    def __init__(self) -> None:
        self.mouth = SemanticMouthConditioner()
        self._state: dict[str, float] = {}
        self._blink_state = {"left": 0.0, "right": 0.0}
        self.frames = 0
        self.last_metrics: dict[str, Any] = {}

    def reset(self) -> None:
        self.mouth.reset()
        self._state.clear()
        self._blink_state = {"left": 0.0, "right": 0.0}
        self.frames = 0
        self.last_metrics = {}

    def encode(self, packet: SemanticPacket, size: tuple[int, int]) -> FacialConditioning:
        started = perf_counter()
        width, height = size
        mouth = self.mouth.update(packet)
        control = self._controls(packet, mouth)

        map_started = perf_counter()
        face_map = self._draw_face_map(packet, control, size)
        mouth_mask = self._draw_mouth_mask(packet, control, mouth, size)
        eye_mask = self._draw_eye_mask(packet, control, size)
        body_map = self._draw_body_map(packet, control, size)
        conditioning_image = self._combine_maps(face_map, mouth_mask, eye_mask, body_map)
        map_ms = (perf_counter() - map_started) * 1000.0
        total_ms = (perf_counter() - started) * 1000.0

        self.frames += 1
        self.last_metrics = {
            "semantic_face_encoder_frames": self.frames,
            "semantic_face_encoder_ms": round(total_ms, 3),
            "semantic_controlnet_latency_ms": round(map_ms, 3),
            "conditioning_tensor_shape": f"{height}x{width}x3",
            "conditioning_tensor_width": width,
            "conditioning_tensor_height": height,
            "yaw": round(control["yaw"], 3),
            "pitch": round(control["pitch"], 3),
            "roll": round(control["roll"], 3),
            "mouth_open": round(control["mouth_open"], 3),
            "jaw_open": round(control["jaw_open"], 3),
            "blink_left": round(control["blink_left"], 3),
            "blink_right": round(control["blink_right"], 3),
            "blink": round((control["blink_left"] + control["blink_right"]) * 0.5, 3),
            "smile": round(control["smile"], 3),
            "brow": round(control["brow"], 3),
            "semantic_face_landmarks": len(packet.face_landmarks),
            "semantic_pose_landmarks": len(packet.pose_landmarks),
            "semantic_hand_landmarks": sum(len(hand) for hand in packet.hand_landmarks),
            "semantic_mouth_viseme": mouth.viseme,
        }
        return FacialConditioning(
            control=control,
            conditioning_image=conditioning_image,
            face_map=face_map,
            mouth_mask=mouth_mask,
            eye_mask=eye_mask,
            body_map=body_map,
            shape=(height, width, 3),
            metrics=self.last_metrics,
        )

    def _controls(self, packet: SemanticPacket, mouth: MouthFeatures) -> dict[str, Any]:
        yaw = self._smooth("yaw", _clamp(packet.yaw * 1.68, -64.0, 64.0), 0.55, 0.72, 0.12)
        pitch = self._smooth("pitch", _clamp(packet.pitch * 1.42, -48.0, 48.0), 0.50, 0.66, 0.12)
        roll = self._smooth(
            "roll",
            _clamp(packet.roll * 1.30 + packet.shoulder_rotation * 0.16, -54.0, 54.0),
            0.52,
            0.70,
            0.12,
        )
        shoulder_rotation = self._smooth("shoulder_rotation", _clamp(packet.shoulder_rotation * 1.28, -62.0, 62.0), 0.42, 0.55, 0.18)
        neck_rotation = self._smooth("neck_rotation", _clamp(packet.neck_rotation * 1.25 + packet.yaw * 0.18, -55.0, 55.0), 0.44, 0.58, 0.18)
        torso_rotation = self._smooth("torso_rotation", _clamp(packet.torso_rotation * 1.18, -55.0, 55.0), 0.36, 0.50, 0.18)

        blink_left = self._smooth_blink("left", packet.blink_left or packet.blink)
        blink_right = self._smooth_blink("right", packet.blink_right or packet.blink)
        mouth_open = self._smooth("mouth_open", mouth.mouth_open, 0.58, 0.78, 0.005)
        jaw_open = self._smooth("jaw_open", mouth.jaw_open, 0.60, 0.82, 0.005)
        smile = self._smooth("smile", max(packet.smile, mouth.corner_stretch * 0.55), 0.42, 0.55, 0.006)
        brow = self._smooth("brow", packet.brow, 0.45, 0.56, 0.006)
        cheek = self._smooth("cheek", max(packet.cheek, smile * 0.45), 0.38, 0.50, 0.006)
        eye_x = self._smooth("eye_x", _clamp(packet.eye_x, -1.0, 1.0), 0.42, 0.56, 0.01)
        eye_y = self._smooth("eye_y", _clamp(packet.eye_y, -1.0, 1.0), 0.42, 0.56, 0.01)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "blink_left": blink_left,
            "blink_right": blink_right,
            "mouth_open": mouth_open,
            "jaw_open": jaw_open,
            "lip_separation": mouth.lip_separation,
            "lip_width": mouth.lip_width,
            "lip_corner_stretch": mouth.corner_stretch,
            "mouth_roundness": mouth.roundness,
            "smile": smile,
            "brow": brow,
            "cheek": cheek,
            "eye_x": eye_x,
            "eye_y": eye_y,
            "head_x": packet.head_x,
            "head_y": packet.head_y,
            "shoulder_x": packet.shoulder_x,
            "shoulder_y": packet.shoulder_y,
            "shoulder_rotation": shoulder_rotation,
            "torso_rotation": torso_rotation,
            "neck_rotation": neck_rotation,
            "confidence": packet.confidence,
            "mouth": mouth.to_dict(),
        }

    def _smooth(self, name: str, target: float, alpha: float, fast_alpha: float, deadzone: float) -> float:
        previous = self._state.get(name)
        if previous is None:
            self._state[name] = target
            return target
        delta = abs(target - previous)
        if delta < deadzone:
            return previous
        effective_alpha = min(fast_alpha, alpha + delta * 0.012)
        value = _lerp(previous, target, effective_alpha)
        self._state[name] = value
        return value

    def _smooth_blink(self, side: str, target: float) -> float:
        target = _clamp(target, 0.0, 1.0)
        previous = self._blink_state[side]
        if target > 0.54:
            alpha = 0.88
        elif previous > 0.42 and target > 0.28:
            alpha = 0.68
        else:
            alpha = 0.34
        value = _lerp(previous, target, alpha)
        if value < 0.04:
            value = 0.0
        self._blink_state[side] = value
        return value

    def _draw_face_map(self, packet: SemanticPacket, control: dict[str, Any], size: tuple[int, int]) -> Image.Image:
        width, height = size
        image = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")
        points = packet.face_landmarks
        line_w = max(1, width // 240)
        drew_feature_path = False

        for indices, color in (
            (FACE_OVAL, (72, 255, 218, 132)),
            (LEFT_EYE, (80, 170, 255, 156)),
            (RIGHT_EYE, (80, 170, 255, 156)),
            (LEFT_BROW, (255, 214, 108, 132)),
            (RIGHT_BROW, (255, 214, 108, 132)),
            (MOUTH_OUTER, (255, 96, 132, 172)),
            (MOUTH_INNER, (255, 72, 108, 190)),
        ):
            path = _path(points, indices, width, height)
            if len(path) > 1:
                draw.line(path, fill=color, width=line_w)
                drew_feature_path = True

        if points:
            dot = max(1, width // 520)
            step = max(1, len(points) // 156)
            for lx, ly in points[::step]:
                px, py = lx * width, ly * height
                draw.ellipse([px - dot, py - dot, px + dot, py + dot], fill=(90, 240, 255, 72))
        if not drew_feature_path:
            self._draw_synthetic_face(draw, control, size)

        return image.filter(ImageFilter.GaussianBlur(radius=0.35))

    def _draw_mouth_mask(self, packet: SemanticPacket, control: dict[str, Any], mouth: MouthFeatures, size: tuple[int, int]) -> Image.Image:
        width, height = size
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        path = _path(packet.face_landmarks, MOUTH_OUTER, width, height)
        if len(path) > 2:
            draw.polygon(path, fill=185)
            inner = _path(packet.face_landmarks, MOUTH_INNER, width, height)
            if len(inner) > 2:
                draw.polygon(inner, fill=255)
        else:
            cx = control["head_x"] * width
            cy = (control["head_y"] + 0.12) * height
            mouth_w = width * (0.052 + mouth.lip_width * 0.062 + control["smile"] * 0.018)
            mouth_h = height * (0.006 + mouth.jaw_open * 0.055 + mouth.lip_separation * 0.035)
            draw.ellipse([cx - mouth_w, cy - mouth_h, cx + mouth_w, cy + mouth_h], fill=245)
        return mask.filter(ImageFilter.GaussianBlur(radius=max(1, width // 260)))

    def _draw_eye_mask(self, packet: SemanticPacket, control: dict[str, Any], size: tuple[int, int]) -> Image.Image:
        width, height = size
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for indices in (LEFT_EYE, RIGHT_EYE):
            path = _path(packet.face_landmarks, indices, width, height)
            if len(path) > 2:
                draw.polygon(path, fill=235)

        if len(packet.face_landmarks) <= max(max(LEFT_EYE), max(RIGHT_EYE)):
            cx = control["head_x"] * width
            cy = control["head_y"] * height - height * 0.045
            span = width * 0.065
            eye_w = width * 0.026
            for side, blink in ((-1, control["blink_left"]), (1, control["blink_right"])):
                eh = height * max(0.002, 0.011 * (1.0 - blink))
                ex = cx + side * span + control["eye_x"] * width * 0.006
                ey = cy + control["eye_y"] * height * 0.006
                draw.ellipse([ex - eye_w, ey - eh, ex + eye_w, ey + eh], fill=235)
        return mask.filter(ImageFilter.GaussianBlur(radius=max(1, width // 300)))

    def _draw_body_map(self, packet: SemanticPacket, control: dict[str, Any], size: tuple[int, int]) -> Image.Image:
        width, height = size
        image = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")
        line_w = max(2, width // 170)
        pose = packet.pose_landmarks

        if pose and len(pose) > 16:
            for a, b in POSE_CONNECTIONS:
                if a < len(pose) and b < len(pose):
                    draw.line([_xy(pose[a], width, height), _xy(pose[b], width, height)], fill=(70, 255, 240, 142), width=line_w)
            dot = max(2, width // 280)
            for index in (11, 12, 13, 14, 15, 16, 23, 24):
                if index < len(pose):
                    px, py = _xy(pose[index], width, height)
                    draw.ellipse([px - dot, py - dot, px + dot, py + dot], fill=(110, 255, 230, 132))
        else:
            sx = control["shoulder_x"] * width
            sy = control["shoulder_y"] * height
            angle = math.radians(control["shoulder_rotation"])
            span = width * 0.34
            dx = math.cos(angle) * span * 0.5
            dy = math.sin(angle) * span * 0.5
            draw.line([(sx - dx, sy - dy), (sx + dx, sy + dy)], fill=(70, 255, 240, 142), width=line_w)

        for hand in packet.hand_landmarks:
            if len(hand) < 5:
                continue
            for a, b in HAND_CONNECTIONS:
                if a < len(hand) and b < len(hand):
                    draw.line([_xy(hand[a], width, height), _xy(hand[b], width, height)], fill=(255, 224, 116, 140), width=max(1, width // 260))

        return image.filter(ImageFilter.GaussianBlur(radius=0.45))

    def _draw_synthetic_face(self, draw: ImageDraw.ImageDraw, control: dict[str, Any], size: tuple[int, int]) -> None:
        width, height = size
        cx = control["head_x"] * width
        cy = control["head_y"] * height
        face_w = width * 0.24
        face_h = height * 0.31
        roll = math.radians(control["roll"])
        draw.ellipse([cx - face_w, cy - face_h, cx + face_w, cy + face_h], outline=(72, 255, 218, 90), width=max(1, width // 260))
        for side in (-1, 1):
            ex = cx + side * face_w * 0.42 + math.sin(roll) * height * 0.012
            ey = cy - face_h * 0.20
            draw.line([(ex - face_w * 0.14, ey), (ex + face_w * 0.14, ey)], fill=(80, 170, 255, 130), width=max(1, width // 240))

    def _combine_maps(self, face_map: Image.Image, mouth_mask: Image.Image, eye_mask: Image.Image, body_map: Image.Image) -> Image.Image:
        combined = Image.blend(face_map, body_map, 0.48)
        combined = Image.composite(Image.new("RGB", combined.size, (160, 22, 54)), combined, mouth_mask)
        combined = Image.composite(Image.new("RGB", combined.size, (32, 88, 210)), combined, eye_mask)
        return combined


__all__ = ["FacialConditioning", "SemanticFaceEncoder"]
