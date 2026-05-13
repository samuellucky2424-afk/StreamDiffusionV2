"""Semantic-to-conditioning adapter for realtime avatar driving.

The websocket layer is allowed to know about semantic JSON. StreamDiffusionV2 is
not. This adapter is the bridge: it turns semantic packets plus a cached
portrait into synthetic driving frames that the existing v2v pipeline can read
as normal image conditioning.
"""

from __future__ import annotations

import io
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .semantic_pose import SemanticPacket

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(
        ImageOps.exif_transpose(image).convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.34),
    )


@dataclass(slots=True)
class PortraitSession:
    avatar_id: str
    filename: str
    image: Image.Image
    created_at: float = field(default_factory=time)
    width: int = 512
    height: int = 512
    frames_rendered: int = 0
    last_packet: SemanticPacket | None = None

    def metrics(self) -> dict[str, float | int | str | None]:
        return {
            "avatar_id": self.avatar_id,
            "filename": self.filename,
            "created_at": self.created_at,
            "width": self.width,
            "height": self.height,
            "frames_rendered": self.frames_rendered,
            "last_packet": self.last_packet.to_debug_dict() if self.last_packet else None,
        }


class SemanticAvatarAdapter:
    """Convert semantic packets into synthetic RGB conditioning frames."""

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        *,
        debug_dir: str | os.PathLike[str] | None = None,
        debug_every_n: int = 0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.sessions: dict[str, PortraitSession] = {}
        self.latest_avatar_id: str | None = None
        self.lock = threading.RLock()
        self.frames_rendered = 0
        self.last_render_ms = 0.0
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_every_n = max(0, int(debug_every_n))
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, width: int = 512, height: int = 512) -> "SemanticAvatarAdapter":
        return cls(
            width=width,
            height=height,
            debug_dir=os.getenv("SEMANTIC_AVATAR_DEBUG_DIR"),
            debug_every_n=int(os.getenv("SEMANTIC_AVATAR_DEBUG_EVERY_N", "0") or "0"),
        )

    def upload_portrait(self, filename: str, payload: bytes) -> dict[str, Any]:
        if not payload:
            raise ValueError("Empty avatar upload.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("Avatar image must be 10 MB or smaller.")

        try:
            source = Image.open(io.BytesIO(payload))
            image = _fit_cover(source, (self.width, self.height))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unable to decode avatar image: {exc}") from exc

        avatar_id = uuid.uuid4().hex
        session = PortraitSession(
            avatar_id=avatar_id,
            filename=filename or "avatar.png",
            image=image,
            width=self.width,
            height=self.height,
        )
        with self.lock:
            self.sessions[avatar_id] = session
            self.latest_avatar_id = avatar_id

        return {
            "avatar_id": avatar_id,
            "width": self.width,
            "height": self.height,
            "filename": session.filename,
        }

    def get_session(self, avatar_id: str | None = None) -> PortraitSession:
        with self.lock:
            resolved_id = avatar_id or self.latest_avatar_id
            if not resolved_id or resolved_id not in self.sessions:
                raise KeyError("Upload a portrait before opening the semantic avatar stream.")
            return self.sessions[resolved_id]

    def packet_to_image(
        self,
        packet_data: Mapping[str, Any] | SemanticPacket,
        *,
        avatar_id: str | None = None,
    ) -> Image.Image:
        packet = packet_data if isinstance(packet_data, SemanticPacket) else SemanticPacket.from_mapping(packet_data)
        session = self.get_session(avatar_id)
        started = perf_counter()
        image = self._render_synthetic_driving_frame(session, packet)
        self.last_render_ms = (perf_counter() - started) * 1000.0
        self.frames_rendered += 1
        session.frames_rendered += 1
        session.last_packet = packet
        self._maybe_write_debug_frame(image, session, packet)
        return image

    def packet_to_array(
        self,
        packet_data: Mapping[str, Any] | SemanticPacket,
        *,
        avatar_id: str | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        image = self.packet_to_image(packet_data, avatar_id=avatar_id)
        array = np.asarray(image, dtype=np.float32)
        if normalize:
            array = array / 127.5 - 1.0
        return array

    def _render_synthetic_driving_frame(self, session: PortraitSession, packet: SemanticPacket) -> Image.Image:
        base = session.image
        background = base.filter(ImageFilter.GaussianBlur(radius=14))
        background = Image.blend(background, Image.new("RGB", base.size, (20, 22, 28)), 0.34)

        yaw = _clamp(packet.yaw, -45.0, 45.0)
        pitch = _clamp(packet.pitch, -35.0, 35.0)
        roll = _clamp(packet.roll + packet.shoulder_rotation * 0.18, -45.0, 45.0)
        confidence_alpha = 0.55 + packet.confidence * 0.45

        layer = base.convert("RGBA")
        yaw_squash = 1.0 - abs(yaw) / 260.0
        pitch_scale = 1.0 + abs(pitch) / 380.0
        scaled_w = max(8, int(self.width * yaw_squash))
        scaled_h = max(8, int(self.height * pitch_scale))
        layer = layer.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        layer = layer.rotate(roll, resample=Image.Resampling.BICUBIC, expand=True)

        x = int((self.width - layer.width) * 0.5 + yaw * 0.85 + (packet.head_x - 0.5) * 46)
        y = int((self.height - layer.height) * 0.5 + pitch * 0.42 + (packet.head_y - 0.35) * 54)
        alpha = layer.getchannel("A").point(lambda value: int(value * confidence_alpha))
        layer.putalpha(alpha)

        frame = background.convert("RGBA")
        frame.alpha_composite(layer, (x, y))
        self._draw_expression_overlay(frame, packet)
        return frame.convert("RGB")

    def _draw_expression_overlay(self, frame: Image.Image, packet: SemanticPacket) -> None:
        draw = ImageDraw.Draw(frame, "RGBA")
        width, height = frame.size
        cx = packet.head_x * width
        cy = packet.head_y * height
        face_w = width * 0.34
        face_h = height * 0.42
        eye_y = cy - face_h * 0.12 + packet.pitch * 0.35
        mouth_y = cy + face_h * 0.22 + packet.mouth_open * height * 0.018
        left_eye_x = cx - face_w * 0.18 + packet.eye_x * width * 0.01
        right_eye_x = cx + face_w * 0.18 + packet.eye_x * width * 0.01
        blink_left = _clamp(packet.blink_left or packet.blink, 0.0, 1.0)
        blink_right = _clamp(packet.blink_right or packet.blink, 0.0, 1.0)
        brow_lift = packet.brow * height * 0.025

        guide_alpha = int(52 + packet.confidence * 52)
        dark = (32, 20, 26, guide_alpha + 40)
        warm = (255, 196, 138, guide_alpha)
        cyan = (70, 230, 255, max(28, guide_alpha - 20))

        def eye(x: float, blink: float) -> None:
            ew = face_w * 0.105
            eh = max(1.0, face_h * 0.025 * (1.0 - blink))
            if blink > 0.72:
                draw.line([(x - ew, eye_y), (x + ew, eye_y)], fill=dark, width=max(2, width // 170))
            else:
                draw.ellipse([x - ew, eye_y - eh, x + ew, eye_y + eh], outline=dark, width=max(1, width // 210))
                pupil_x = x + packet.eye_x * ew * 0.45
                pupil_y = eye_y + packet.eye_y * eh * 0.65
                draw.ellipse([pupil_x - 1.5, pupil_y - 1.5, pupil_x + 1.5, pupil_y + 1.5], fill=dark)

        eye(left_eye_x, blink_left)
        eye(right_eye_x, blink_right)

        brow_y = eye_y - face_h * 0.085 - brow_lift
        brow_slant = packet.roll * 0.06
        draw.line(
            [(left_eye_x - face_w * 0.09, brow_y + brow_slant), (left_eye_x + face_w * 0.11, brow_y - brow_slant)],
            fill=dark,
            width=max(2, width // 180),
        )
        draw.line(
            [(right_eye_x - face_w * 0.11, brow_y - brow_slant), (right_eye_x + face_w * 0.09, brow_y + brow_slant)],
            fill=dark,
            width=max(2, width // 180),
        )

        mouth_w = face_w * (0.14 + packet.smile * 0.11)
        mouth_h = face_h * (0.018 + packet.mouth_open * 0.11)
        if packet.mouth_open > 0.08:
            draw.ellipse(
                [cx - mouth_w, mouth_y - mouth_h, cx + mouth_w, mouth_y + mouth_h],
                fill=(78, 26, 42, guide_alpha + 58),
                outline=warm,
                width=max(1, width // 240),
            )
        else:
            curve = [
                (cx - mouth_w, mouth_y),
                (cx - mouth_w * 0.35, mouth_y + packet.smile * face_h * 0.05),
                (cx + mouth_w * 0.35, mouth_y + packet.smile * face_h * 0.05),
                (cx + mouth_w, mouth_y),
            ]
            draw.line(curve, fill=dark, width=max(2, width // 190), joint="curve")

        shoulder_y = packet.shoulder_y * height
        shoulder_span = width * (0.24 + packet.confidence * 0.08)
        shoulder_angle = np.deg2rad(packet.shoulder_rotation + packet.roll * 0.2)
        dx = np.cos(shoulder_angle) * shoulder_span * 0.5
        dy = np.sin(shoulder_angle) * shoulder_span * 0.5
        sx = packet.shoulder_x * width
        draw.line([(sx - dx, shoulder_y - dy), (sx + dx, shoulder_y + dy)], fill=cyan, width=max(2, width // 160))

        if packet.face_landmarks:
            dot = max(1, width // 420)
            for lx, ly in packet.face_landmarks[:: max(1, len(packet.face_landmarks) // 80)]:
                px, py = lx * width, ly * height
                draw.ellipse([px - dot, py - dot, px + dot, py + dot], fill=(80, 230, 255, 46))

    def _maybe_write_debug_frame(self, image: Image.Image, session: PortraitSession, packet: SemanticPacket) -> None:
        if not self.debug_dir or self.debug_every_n <= 0:
            return
        if self.frames_rendered % self.debug_every_n != 0:
            return
        frame_id = packet.frame_id if packet.frame_id is not None else self.frames_rendered
        image.save(self.debug_dir / f"{session.avatar_id}_{frame_id:06d}.jpg", quality=90)

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            active_session = self.sessions.get(self.latest_avatar_id or "")
            session_count = len(self.sessions)
        return {
            "semantic_adapter_frames": self.frames_rendered,
            "semantic_adapter_last_render_ms": round(self.last_render_ms, 3),
            "semantic_adapter_width": self.width,
            "semantic_adapter_height": self.height,
            "avatar_session_count": session_count,
            "active_avatar": active_session.metrics() if active_session else None,
        }


__all__ = [
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_CONTENT_TYPES",
    "PortraitSession",
    "SemanticAvatarAdapter",
]
