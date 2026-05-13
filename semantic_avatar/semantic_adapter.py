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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .semantic_face_encoder import FacialConditioning, SemanticFaceEncoder
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
        debug_semantic_overlay: bool = False,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.sessions: dict[str, PortraitSession] = {}
        self.latest_avatar_id: str | None = None
        self.lock = threading.RLock()
        self.frames_rendered = 0
        self.last_render_ms = 0.0
        self.face_encoder = SemanticFaceEncoder()
        self.last_conditioning_metrics: dict[str, Any] = {}
        self.debug_semantic_overlay = bool(debug_semantic_overlay)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_every_n = max(0, int(debug_every_n))
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        width: int = 512,
        height: int = 512,
        *,
        debug_semantic_overlay: bool | None = None,
    ) -> "SemanticAvatarAdapter":
        if debug_semantic_overlay is None:
            debug_semantic_overlay = os.getenv("SEMANTIC_AVATAR_DEBUG_OVERLAY", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        return cls(
            width=width,
            height=height,
            debug_dir=os.getenv("SEMANTIC_AVATAR_DEBUG_DIR"),
            debug_every_n=int(os.getenv("SEMANTIC_AVATAR_DEBUG_EVERY_N", "0") or "0"),
            debug_semantic_overlay=debug_semantic_overlay,
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
            self.face_encoder.reset()

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
        conditioning = self.face_encoder.encode(packet, (self.width, self.height))
        self.last_conditioning_metrics = dict(conditioning.metrics)
        control = conditioning.control

        base = self._deform_portrait(session.image, packet, control)
        background = base.filter(ImageFilter.GaussianBlur(radius=14))
        background = Image.blend(background, Image.new("RGB", base.size, (20, 22, 28)), 0.34)

        yaw = _clamp(control["yaw"], -58.0, 58.0)
        pitch = _clamp(control["pitch"], -46.0, 46.0)
        roll = _clamp(control["roll"] + control["neck_rotation"] * 0.10, -54.0, 54.0)
        confidence_alpha = 0.55 + packet.confidence * 0.45

        layer = base.convert("RGBA")
        yaw_squash = 1.0 - abs(yaw) / 210.0
        pitch_scale = 1.0 + abs(pitch) / 310.0
        scaled_w = max(8, int(self.width * yaw_squash))
        scaled_h = max(8, int(self.height * pitch_scale))
        layer = layer.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        layer = layer.rotate(roll, resample=Image.Resampling.BICUBIC, expand=True)

        x = int((self.width - layer.width) * 0.5 + yaw * 1.12 + (control["head_x"] - 0.5) * 58)
        y = int((self.height - layer.height) * 0.5 + pitch * 0.58 + (control["head_y"] - 0.35) * 68)
        alpha = layer.getchannel("A").point(lambda value: int(value * confidence_alpha))
        layer.putalpha(alpha)

        frame = background.convert("RGBA")
        frame.alpha_composite(layer, (x, y))
        if self.debug_semantic_overlay:
            self._composite_conditioning_maps(frame, conditioning, control)
            self._draw_expression_overlay(frame, packet, control)
        return frame.convert("RGB")

    def _deform_portrait(self, image: Image.Image, packet: SemanticPacket, control: dict[str, Any]) -> Image.Image:
        """Apply natural-looking portrait deformations before VAE encoding.

        This is the hidden semantic-to-image-conditioning stage. It avoids
        painting landmark/mask graphics while still giving StreamDiffusionV2 a
        changing visual condition to denoise from.
        """

        frame = image.convert("RGBA")
        width, height = frame.size
        cx = width * 0.5 + (control["head_x"] - 0.5) * width * 0.035
        face_cy = height * 0.42 + (control["head_y"] - 0.35) * height * 0.035
        face_w = width * 0.38
        face_h = height * 0.46

        self._shift_brow_regions(frame, cx, face_cy, face_w, face_h, control)
        self._deform_eye_region(frame, cx - face_w * 0.22, face_cy - face_h * 0.17, face_w, face_h, control["blink_left"], control)
        self._deform_eye_region(frame, cx + face_w * 0.22, face_cy - face_h * 0.17, face_w, face_h, control["blink_right"], control)
        self._deform_mouth_region(frame, cx, face_cy + face_h * 0.22, face_w, face_h, packet, control)
        return frame.convert("RGB")

    def _deform_mouth_region(
        self,
        frame: Image.Image,
        cx: float,
        cy: float,
        face_w: float,
        face_h: float,
        packet: SemanticPacket,
        control: dict[str, Any],
    ) -> None:
        mouth_open = _clamp(control["mouth_open"], 0.0, 1.0)
        jaw_open = _clamp(control["jaw_open"], 0.0, 1.0)
        smile = _clamp(control["smile"], 0.0, 1.0)
        lip_width = _clamp(control["lip_width"], 0.0, 1.0)
        width, height = frame.size
        region_w = max(12, int(face_w * (0.34 + lip_width * 0.22 + smile * 0.16)))
        region_h = max(10, int(face_h * (0.105 + jaw_open * 0.10)))
        bbox = self._safe_box(cx - region_w * 0.5, cy - region_h * 0.5, cx + region_w * 0.5, cy + region_h * 0.5, width, height)
        if bbox is None:
            return

        crop = frame.crop(bbox)
        crop_w, crop_h = crop.size
        scale_x = 1.0 + smile * 0.16 + lip_width * 0.08
        scale_y = 0.92 + mouth_open * 0.70
        resized = crop.resize(
            (max(1, int(crop_w * scale_x)), max(1, int(crop_h * scale_y))),
            Image.Resampling.BICUBIC,
        )
        paste_x = int((bbox[0] + bbox[2] - resized.width) * 0.5)
        paste_y = int((bbox[1] + bbox[3] - resized.height) * 0.5 + jaw_open * height * 0.008)
        mask = Image.new("L", resized.size, 0)
        ImageDraw.Draw(mask).ellipse([0, 0, resized.width, resized.height], fill=205)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, crop_w // 18)))
        resized.putalpha(mask)
        self._alpha_composite_clipped(frame, resized, paste_x, paste_y)

        if mouth_open > 0.08:
            # Darken the existing mouth region instead of drawing a semantic graphic.
            aperture = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(aperture, "RGBA")
            hole_w = face_w * (0.075 + lip_width * 0.035 + smile * 0.025)
            hole_h = face_h * (0.010 + mouth_open * 0.060 + jaw_open * 0.035)
            mouth_color = self._sample_dark_color(frame, bbox)
            draw.ellipse(
                [cx - hole_w, cy - hole_h, cx + hole_w, cy + hole_h],
                fill=(*mouth_color, int(42 + mouth_open * 86)),
            )
            aperture = aperture.filter(ImageFilter.GaussianBlur(radius=max(1.0, width / 360.0)))
            frame.alpha_composite(aperture)

    def _deform_eye_region(
        self,
        frame: Image.Image,
        cx: float,
        cy: float,
        face_w: float,
        face_h: float,
        blink: float,
        control: dict[str, Any],
    ) -> None:
        blink = _clamp(blink, 0.0, 1.0)
        width, height = frame.size
        region_w = max(10, int(face_w * 0.20))
        region_h = max(8, int(face_h * 0.075))
        gaze_x = control["eye_x"] * width * 0.006
        gaze_y = control["eye_y"] * height * 0.004
        bbox = self._safe_box(
            cx - region_w * 0.5 + gaze_x,
            cy - region_h * 0.5 + gaze_y,
            cx + region_w * 0.5 + gaze_x,
            cy + region_h * 0.5 + gaze_y,
            width,
            height,
        )
        if bbox is None:
            return

        crop = frame.crop(bbox)
        crop_w, crop_h = crop.size
        target_h = max(1, int(crop_h * (1.0 - blink * 0.72)))
        compressed = crop.resize((crop_w, target_h), Image.Resampling.BICUBIC)
        eyelid = crop.filter(ImageFilter.GaussianBlur(radius=max(1.0, crop_h / 7.0)))
        frame.alpha_composite(eyelid, (bbox[0], bbox[1]))
        frame.alpha_composite(compressed, (bbox[0], int((bbox[1] + bbox[3] - target_h) * 0.5)))

    def _shift_brow_regions(
        self,
        frame: Image.Image,
        cx: float,
        face_cy: float,
        face_w: float,
        face_h: float,
        control: dict[str, Any],
    ) -> None:
        brow = _clamp(control["brow"], 0.0, 1.0)
        if brow < 0.03:
            return
        width, height = frame.size
        shift = int(-brow * height * 0.018)
        for side in (-1, 1):
            bx = cx + side * face_w * 0.22
            by = face_cy - face_h * 0.30
            region_w = max(12, int(face_w * 0.22))
            region_h = max(8, int(face_h * 0.07))
            bbox = self._safe_box(bx - region_w * 0.5, by - region_h * 0.5, bx + region_w * 0.5, by + region_h * 0.5, width, height)
            if bbox is None:
                continue
            crop = frame.crop(bbox)
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, crop.width, crop.height], radius=max(2, crop.height // 2), fill=170)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, crop.height // 4)))
            frame.paste(crop, (bbox[0], bbox[1] + shift), mask)

    @staticmethod
    def _safe_box(left: float, top: float, right: float, bottom: float, width: int, height: int) -> tuple[int, int, int, int] | None:
        box = (
            max(0, int(left)),
            max(0, int(top)),
            min(width, int(right)),
            min(height, int(bottom)),
        )
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return None
        return box

    @staticmethod
    def _alpha_composite_clipped(frame: Image.Image, overlay: Image.Image, x: int, y: int) -> None:
        left = max(0, -x)
        top = max(0, -y)
        right = min(overlay.width, frame.width - x)
        bottom = min(overlay.height, frame.height - y)
        if right <= left or bottom <= top:
            return
        clipped = overlay.crop((left, top, right, bottom))
        frame.alpha_composite(clipped, (max(0, x), max(0, y)))

    @staticmethod
    def _sample_dark_color(frame: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
        crop = frame.crop(bbox).convert("RGB").resize((1, 1), Image.Resampling.BILINEAR)
        r, g, b = crop.getpixel((0, 0))
        enhancer = ImageEnhance.Brightness(Image.new("RGB", (1, 1), (r, g, b)))
        dark = enhancer.enhance(0.36).getpixel((0, 0))
        return (int(dark[0]), int(dark[1]), int(dark[2]))

    def _composite_conditioning_maps(
        self,
        frame: Image.Image,
        conditioning: FacialConditioning,
        control: dict[str, Any],
    ) -> None:
        strength = _clamp(0.20 + control["confidence"] * 0.18, 0.16, 0.38)
        guide = conditioning.conditioning_image.convert("RGBA")
        alpha = guide.convert("L").point(lambda value: int(value * strength))
        guide.putalpha(alpha)
        frame.alpha_composite(guide)

    def _draw_expression_overlay(self, frame: Image.Image, packet: SemanticPacket, control: dict[str, Any]) -> None:
        draw = ImageDraw.Draw(frame, "RGBA")
        width, height = frame.size
        cx = control["head_x"] * width
        cy = control["head_y"] * height
        face_w = width * 0.34
        face_h = height * 0.42
        eye_y = cy - face_h * 0.12 + control["pitch"] * 0.32
        mouth_y = cy + face_h * 0.22 + control["mouth_open"] * height * 0.030
        left_eye_x = cx - face_w * 0.18 + control["eye_x"] * width * 0.012
        right_eye_x = cx + face_w * 0.18 + control["eye_x"] * width * 0.012
        blink_left = _clamp(control["blink_left"], 0.0, 1.0)
        blink_right = _clamp(control["blink_right"], 0.0, 1.0)
        brow_lift = control["brow"] * height * 0.034

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
                pupil_x = x + control["eye_x"] * ew * 0.60
                pupil_y = eye_y + control["eye_y"] * eh * 0.85
                draw.ellipse([pupil_x - 1.5, pupil_y - 1.5, pupil_x + 1.5, pupil_y + 1.5], fill=dark)

        eye(left_eye_x, blink_left)
        eye(right_eye_x, blink_right)

        brow_y = eye_y - face_h * 0.085 - brow_lift
        brow_slant = control["roll"] * 0.06
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

        lip_width = _clamp(control["lip_width"], 0.0, 1.0)
        mouth_w = face_w * (0.13 + lip_width * 0.09 + control["smile"] * 0.10)
        mouth_h = face_h * (0.016 + control["jaw_open"] * 0.18 + control["lip_separation"] * 0.08)
        if control["mouth_open"] > 0.055:
            draw.ellipse(
                [cx - mouth_w, mouth_y - mouth_h, cx + mouth_w, mouth_y + mouth_h],
                fill=(78, 20, 36, guide_alpha + 76),
                outline=warm,
                width=max(1, width // 240),
            )
        else:
            curve = [
                (cx - mouth_w, mouth_y),
                (cx - mouth_w * 0.35, mouth_y + control["smile"] * face_h * 0.06),
                (cx + mouth_w * 0.35, mouth_y + control["smile"] * face_h * 0.06),
                (cx + mouth_w, mouth_y),
            ]
            draw.line(curve, fill=dark, width=max(2, width // 190), joint="curve")

        shoulder_y = control["shoulder_y"] * height
        shoulder_span = width * (0.24 + packet.confidence * 0.08)
        shoulder_angle = np.deg2rad(control["shoulder_rotation"] + control["roll"] * 0.2)
        dx = np.cos(shoulder_angle) * shoulder_span * 0.5
        dy = np.sin(shoulder_angle) * shoulder_span * 0.5
        sx = control["shoulder_x"] * width
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
            "semantic_debug_overlay": self.debug_semantic_overlay,
            "semantic_conditioning_mode": "debug_overlay" if self.debug_semantic_overlay else "hidden_pre_denoise",
            "avatar_session_count": session_count,
            "active_avatar": active_session.metrics() if active_session else None,
            **self.last_conditioning_metrics,
        }


__all__ = [
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_CONTENT_TYPES",
    "PortraitSession",
    "SemanticAvatarAdapter",
]
