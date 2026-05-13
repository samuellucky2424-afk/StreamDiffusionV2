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

from .face_expression_encoder import FaceExpressionEncoder
from .identity_lock import IdentityLock
from .semantic_face_encoder import (
    FACE_OVAL,
    LEFT_BROW,
    LEFT_EYE,
    MOUTH_OUTER,
    RIGHT_BROW,
    RIGHT_EYE,
    FacialConditioning,
    SemanticFaceEncoder,
)
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
    identity_cached: bool = False

    def metrics(self) -> dict[str, float | int | str | None]:
        return {
            "avatar_id": self.avatar_id,
            "filename": self.filename,
            "created_at": self.created_at,
            "width": self.width,
            "height": self.height,
            "frames_rendered": self.frames_rendered,
            "identity_cached": self.identity_cached,
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
        debug_face_mask: bool = False,
        debug_identity: bool = False,
        debug_conditioning: bool = False,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.sessions: dict[str, PortraitSession] = {}
        self.latest_avatar_id: str | None = None
        self.lock = threading.RLock()
        self.frames_rendered = 0
        self.last_render_ms = 0.0
        self.face_encoder = SemanticFaceEncoder()
        self.expression_encoder = FaceExpressionEncoder()
        self.identity_lock = IdentityLock()
        self.last_conditioning_metrics: dict[str, Any] = {}
        self.debug_semantic_overlay = bool(debug_semantic_overlay)
        self.debug_face_mask = bool(debug_face_mask)
        self.debug_identity = bool(debug_identity)
        self.debug_conditioning = bool(debug_conditioning)
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
        debug_face_mask: bool | None = None,
        debug_identity: bool | None = None,
        debug_conditioning: bool | None = None,
    ) -> "SemanticAvatarAdapter":
        if debug_semantic_overlay is None:
            debug_semantic_overlay = os.getenv("SEMANTIC_AVATAR_DEBUG_OVERLAY", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if debug_face_mask is None:
            debug_face_mask = os.getenv("SEMANTIC_AVATAR_DEBUG_FACE_MASK", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if debug_identity is None:
            debug_identity = os.getenv("SEMANTIC_AVATAR_DEBUG_IDENTITY", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if debug_conditioning is None:
            debug_conditioning = os.getenv("SEMANTIC_AVATAR_DEBUG_CONDITIONING", "").lower() in {
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
            debug_face_mask=debug_face_mask,
            debug_identity=debug_identity,
            debug_conditioning=debug_conditioning,
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
        identity_embedding = self.identity_lock.cache_portrait(avatar_id, image)
        session.identity_cached = True
        with self.lock:
            self.sessions[avatar_id] = session
            self.latest_avatar_id = avatar_id
            self.face_encoder.reset()
            self.expression_encoder.reset()

        return {
            "avatar_id": avatar_id,
            "width": self.width,
            "height": self.height,
            "filename": session.filename,
            "identity_lock": True,
            "identity_embedding_dim": int(identity_embedding.vector.shape[0]),
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
        expression = self.expression_encoder.encode(packet, control, conditioning_shape=conditioning.shape)
        self.last_conditioning_metrics.update(expression.metrics)

        frame = self._deform_portrait(session.image, packet, control).convert("RGBA")
        identity_result = self.identity_lock.condition_frame(
            session.avatar_id,
            frame.convert("RGB"),
            expression_vector=expression.vector,
        )
        frame = identity_result.image.convert("RGBA")
        self.last_conditioning_metrics.update(identity_result.metrics)
        self.last_conditioning_metrics.update(
            {
                "identity_payload_shape": f"1x{identity_result.metrics.get('identity_embedding_dim')}",
                "identity_embedding_shape": f"1x{identity_result.metrics.get('identity_embedding_dim')}",
                "identity_conditioning_payload_active": bool(identity_result.metrics.get("identity_lock_active")),
                "debug_identity": self.debug_identity,
                "debug_conditioning": self.debug_conditioning,
            }
        )
        if self.debug_semantic_overlay:
            self._composite_conditioning_maps(frame, conditioning, control)
            self._draw_expression_overlay(frame, packet, control)
        if self.debug_face_mask:
            self._draw_face_mask_debug(frame, conditioning, control)
        if self.debug_identity or self.debug_conditioning:
            self._draw_conditioning_debug(frame)
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

        face_points = self._portrait_face_points(packet, width, height)
        masks = self._build_region_masks(packet, control, face_points, (width, height), cx, face_cy, face_w, face_h)
        self.last_conditioning_metrics.update(
            {
                "local_face_deform_active": True,
                "global_transform_disabled": True,
                "portrait_anchor_locked": True,
                "latent_conditioning_active": True,
                "controlnet_conditioning_active": True,
                "controlnet_model_active": False,
                "controlnet_maps_active": True,
                "human_controlnet_conditioning_active": True,
                "openpose_upper_body_conditioning_active": True,
                "face_mesh_conditioning_active": bool(packet.face_landmarks),
                "hand_conditioning_active": bool(packet.hand_landmarks),
                "face_mask_resolution": f"{height}x{width}",
                "face_mask_regions": ",".join(sorted(masks)),
            }
        )

        self._apply_head_structure_deform(frame, masks, control)
        self._shift_brow_regions(frame, cx, face_cy, face_w, face_h, control, masks=masks)
        self._deform_eye_region(
            frame,
            cx - face_w * 0.22,
            face_cy - face_h * 0.17,
            face_w,
            face_h,
            control["blink_left"],
            control,
            mask=masks.get("left_eye"),
        )
        self._deform_eye_region(
            frame,
            cx + face_w * 0.22,
            face_cy - face_h * 0.17,
            face_w,
            face_h,
            control["blink_right"],
            control,
            mask=masks.get("right_eye"),
        )
        self._deform_mouth_region(frame, cx, face_cy + face_h * 0.22, face_w, face_h, packet, control, mask=masks.get("mouth"))
        self._deform_body_regions(frame, masks, control)
        return frame.convert("RGB")

    def _portrait_face_points(
        self,
        packet: SemanticPacket,
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        if not packet.face_landmarks or len(packet.face_landmarks) <= max(FACE_OVAL):
            return []

        source_points = [(x * width, y * height) for x, y in packet.face_landmarks]
        face_path = [source_points[index] for index in FACE_OVAL if index < len(source_points)]
        if not face_path:
            return []

        left = min(point[0] for point in face_path)
        right = max(point[0] for point in face_path)
        top = min(point[1] for point in face_path)
        bottom = max(point[1] for point in face_path)
        source_w = max(1.0, right - left)
        source_h = max(1.0, bottom - top)
        source_cx = (left + right) * 0.5
        source_cy = (top + bottom) * 0.5

        target_cx = width * 0.5
        target_cy = height * 0.43
        target_w = width * 0.42
        target_h = height * 0.54
        scale = min(target_w / source_w, target_h / source_h)

        return [
            (
                target_cx + (point[0] - source_cx) * scale,
                target_cy + (point[1] - source_cy) * scale,
            )
            for point in source_points
        ]

    def _build_region_masks(
        self,
        packet: SemanticPacket,
        control: dict[str, Any],
        face_points: list[tuple[float, float]],
        size: tuple[int, int],
        cx: float,
        face_cy: float,
        face_w: float,
        face_h: float,
    ) -> dict[str, Image.Image]:
        width, height = size
        masks: dict[str, Image.Image] = {}

        def polygon_mask(indices: list[int], blur: float = 3.0, inflate: int = 0) -> Image.Image | None:
            if not face_points or max(indices) >= len(face_points):
                return None
            mask = Image.new("L", size, 0)
            points = [face_points[index] for index in indices]
            if inflate > 0:
                local_cx = sum(point[0] for point in points) / len(points)
                local_cy = sum(point[1] for point in points) / len(points)
                points = [
                    (
                        local_cx + (point[0] - local_cx) * (1.0 + inflate / 100.0),
                        local_cy + (point[1] - local_cy) * (1.0 + inflate / 100.0),
                    )
                    for point in points
                ]
            draw = ImageDraw.Draw(mask)
            draw.polygon(points, fill=255)
            return mask.filter(ImageFilter.GaussianBlur(radius=blur))

        def ellipse_mask(name: str, box: tuple[float, float, float, float], blur: float = 4.0) -> None:
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).ellipse(box, fill=255)
            masks[name] = mask.filter(ImageFilter.GaussianBlur(radius=blur))

        mouth = polygon_mask(MOUTH_OUTER, blur=4.0, inflate=18)
        left_eye = polygon_mask(LEFT_EYE, blur=3.0, inflate=35)
        right_eye = polygon_mask(RIGHT_EYE, blur=3.0, inflate=35)
        left_brow = polygon_mask(LEFT_BROW, blur=5.0, inflate=80)
        right_brow = polygon_mask(RIGHT_BROW, blur=5.0, inflate=80)

        if mouth:
            masks["mouth"] = mouth
        else:
            ellipse_mask("mouth", (cx - face_w * 0.18, face_cy + face_h * 0.15, cx + face_w * 0.18, face_cy + face_h * 0.32))
        if left_eye:
            masks["left_eye"] = left_eye
        else:
            ellipse_mask("left_eye", (cx - face_w * 0.34, face_cy - face_h * 0.24, cx - face_w * 0.08, face_cy - face_h * 0.10))
        if right_eye:
            masks["right_eye"] = right_eye
        else:
            ellipse_mask("right_eye", (cx + face_w * 0.08, face_cy - face_h * 0.24, cx + face_w * 0.34, face_cy - face_h * 0.10))
        if left_brow:
            masks["left_brow"] = left_brow
        else:
            ellipse_mask("left_brow", (cx - face_w * 0.36, face_cy - face_h * 0.36, cx - face_w * 0.06, face_cy - face_h * 0.23))
        if right_brow:
            masks["right_brow"] = right_brow
        else:
            ellipse_mask("right_brow", (cx + face_w * 0.06, face_cy - face_h * 0.36, cx + face_w * 0.36, face_cy - face_h * 0.23))

        if face_points and max(FACE_OVAL) < len(face_points):
            lower_face = [face_points[index] for index in FACE_OVAL if index < len(face_points) and face_points[index][1] > face_cy - face_h * 0.02]
            if len(lower_face) >= 3:
                jaw_mask = Image.new("L", size, 0)
                ImageDraw.Draw(jaw_mask).polygon(lower_face, fill=190)
                masks["jaw"] = jaw_mask.filter(ImageFilter.GaussianBlur(radius=8))
        if "jaw" not in masks:
            ellipse_mask("jaw", (cx - face_w * 0.34, face_cy + face_h * 0.02, cx + face_w * 0.34, face_cy + face_h * 0.47), blur=8)

        ellipse_mask("left_cheek", (cx - face_w * 0.38, face_cy - face_h * 0.02, cx - face_w * 0.02, face_cy + face_h * 0.28), blur=8)
        ellipse_mask("right_cheek", (cx + face_w * 0.02, face_cy - face_h * 0.02, cx + face_w * 0.38, face_cy + face_h * 0.28), blur=8)
        ellipse_mask("neck", (cx - face_w * 0.20, face_cy + face_h * 0.34, cx + face_w * 0.20, face_cy + face_h * 0.67), blur=8)

        shoulder_y = control["shoulder_y"] * height
        shoulder_x = control["shoulder_x"] * width
        upper_body = Image.new("L", size, 0)
        body_draw = ImageDraw.Draw(upper_body)
        body_draw.rounded_rectangle(
            [
                shoulder_x - width * 0.30,
                max(face_cy + face_h * 0.44, shoulder_y - height * 0.08),
                shoulder_x + width * 0.30,
                min(height, shoulder_y + height * 0.22),
            ],
            radius=max(8, width // 18),
            fill=150,
        )
        masks["upper_body"] = upper_body.filter(ImageFilter.GaussianBlur(radius=10))
        return masks

    def _apply_head_structure_deform(
        self,
        frame: Image.Image,
        masks: dict[str, Image.Image],
        control: dict[str, Any],
    ) -> None:
        width, height = frame.size
        yaw = _clamp(control["yaw"] / 64.0, -1.0, 1.0)
        pitch = _clamp(control["pitch"] / 48.0, -1.0, 1.0)
        roll = _clamp(control["roll"] / 54.0, -1.0, 1.0)
        cheek_shift = width * 0.014 * yaw
        vertical_shift = height * 0.008 * pitch

        self._apply_masked_transform(frame, masks.get("left_cheek"), shift=(-cheek_shift, vertical_shift), scale=(1.0 - yaw * 0.020, 1.0 + pitch * 0.020))
        self._apply_masked_transform(frame, masks.get("right_cheek"), shift=(-cheek_shift, vertical_shift), scale=(1.0 - yaw * 0.020, 1.0 + pitch * 0.020))
        self._apply_masked_transform(frame, masks.get("jaw"), shift=(-cheek_shift * 0.45, vertical_shift * 1.35), scale=(1.0 + abs(yaw) * 0.020, 1.0 + pitch * 0.030))
        self._apply_masked_transform(frame, masks.get("neck"), shift=(-cheek_shift * 0.25 + roll * width * 0.006, vertical_shift * 0.5), scale=(1.0, 1.0 + abs(pitch) * 0.018))
        self._apply_masked_transform(frame, masks.get("mouth"), shift=(-cheek_shift * 0.30, vertical_shift * 0.65), scale=(1.0 + abs(yaw) * 0.012, 1.0))
        self._apply_masked_transform(frame, masks.get("left_eye"), shift=(-cheek_shift * 0.18 - roll * width * 0.002, vertical_shift * 0.35), scale=(1.0 - yaw * 0.012, 1.0))
        self._apply_masked_transform(frame, masks.get("right_eye"), shift=(-cheek_shift * 0.18 + roll * width * 0.002, vertical_shift * 0.35), scale=(1.0 - yaw * 0.012, 1.0))

    def _deform_mouth_region(
        self,
        frame: Image.Image,
        cx: float,
        cy: float,
        face_w: float,
        face_h: float,
        packet: SemanticPacket,
        control: dict[str, Any],
        mask: Image.Image | None = None,
    ) -> None:
        mouth_open = _clamp(control["mouth_open"], 0.0, 1.0)
        jaw_open = _clamp(control["jaw_open"], 0.0, 1.0)
        smile = _clamp(control["smile"], 0.0, 1.0)
        lip_width = _clamp(control["lip_width"], 0.0, 1.0)
        width, height = frame.size
        region_w = max(12, int(face_w * (0.34 + lip_width * 0.22 + smile * 0.16)))
        region_h = max(10, int(face_h * (0.105 + jaw_open * 0.10)))
        if mask is not None and mask.getbbox() is not None:
            bbox = mask.getbbox()
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
        else:
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
        alpha = Image.new("L", resized.size, 0)
        ImageDraw.Draw(alpha).ellipse([0, 0, resized.width, resized.height], fill=205)
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=max(2, crop_w // 18)))
        resized.putalpha(alpha)
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
        mask: Image.Image | None = None,
    ) -> None:
        blink = _clamp(blink, 0.0, 1.0)
        width, height = frame.size
        region_w = max(10, int(face_w * 0.20))
        region_h = max(8, int(face_h * 0.075))
        gaze_x = control["eye_x"] * width * 0.006
        gaze_y = control["eye_y"] * height * 0.004
        if mask is not None and mask.getbbox() is not None:
            bbox = mask.getbbox()
        else:
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
        masks: dict[str, Image.Image] | None = None,
    ) -> None:
        brow = _clamp(control["brow"], 0.0, 1.0)
        if brow < 0.03:
            return
        width, height = frame.size
        shift = int(-brow * height * 0.018)
        for side, mask_name in ((-1, "left_brow"), (1, "right_brow")):
            mask = masks.get(mask_name) if masks else None
            if mask is not None and mask.getbbox() is not None:
                bbox = mask.getbbox()
                crop = frame.crop(bbox)
                frame.paste(crop, (bbox[0], bbox[1] + shift), mask.crop(bbox))
                continue
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

    def _deform_body_regions(
        self,
        frame: Image.Image,
        masks: dict[str, Image.Image],
        control: dict[str, Any],
    ) -> None:
        shoulder = _clamp(control["shoulder_rotation"] / 62.0, -1.0, 1.0)
        neck = _clamp(control["neck_rotation"] / 55.0, -1.0, 1.0)
        width, height = frame.size
        self._apply_masked_transform(
            frame,
            masks.get("upper_body"),
            shift=(shoulder * width * 0.010, abs(shoulder) * height * 0.004),
            scale=(1.0 + abs(shoulder) * 0.015, 1.0),
        )
        self._apply_masked_transform(
            frame,
            masks.get("neck"),
            shift=(neck * width * 0.010, 0.0),
            scale=(1.0 + abs(neck) * 0.012, 1.0),
        )

    def _apply_masked_transform(
        self,
        frame: Image.Image,
        mask: Image.Image | None,
        *,
        shift: tuple[float, float] = (0.0, 0.0),
        scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if mask is None:
            return
        bbox = mask.getbbox()
        if bbox is None:
            return
        crop = frame.crop(bbox)
        crop_mask = mask.crop(bbox)
        scaled_w = max(1, int(crop.width * max(0.85, min(1.18, scale[0]))))
        scaled_h = max(1, int(crop.height * max(0.85, min(1.20, scale[1]))))
        transformed = crop.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        transformed_mask = crop_mask.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        transformed.putalpha(transformed_mask.point(lambda value: int(value * 0.58)))
        paste_x = int((bbox[0] + bbox[2] - scaled_w) * 0.5 + shift[0])
        paste_y = int((bbox[1] + bbox[3] - scaled_h) * 0.5 + shift[1])
        self._alpha_composite_clipped(frame, transformed, paste_x, paste_y)

    def _draw_face_mask_debug(
        self,
        frame: Image.Image,
        conditioning: FacialConditioning,
        control: dict[str, Any],
    ) -> None:
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        for mask, color in (
            (conditioning.mouth_mask, (255, 64, 96, 86)),
            (conditioning.eye_mask, (64, 128, 255, 72)),
            (conditioning.face_map.convert("L"), (64, 255, 210, 52)),
            (conditioning.body_map.convert("L"), (255, 220, 80, 48)),
        ):
            layer = Image.new("RGBA", frame.size, color)
            layer.putalpha(mask.point(lambda value: int(value * (color[3] / 255.0))))
            overlay.alpha_composite(layer)
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.rectangle([8, 8, 258, 76], fill=(0, 0, 0, 150))
        draw.text((16, 16), "LOCAL_FACE_DEFORM active", fill=(210, 255, 238, 255))
        draw.text((16, 34), "GLOBAL_TRANSFORM disabled", fill=(210, 255, 238, 255))
        draw.text((16, 52), f"mask {frame.height}x{frame.width} yaw {control['yaw']:.1f}", fill=(210, 255, 238, 255))
        frame.alpha_composite(overlay)

    def _draw_conditioning_debug(self, frame: Image.Image) -> None:
        metrics = self.last_conditioning_metrics
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        width = frame.width
        draw.rounded_rectangle([8, 84, min(width - 8, 292), 178], radius=8, fill=(0, 0, 0, 156))
        lines = [
            f"IDENTITY_LOCK {str(metrics.get('identity_lock_active')).lower()}",
            f"drift {metrics.get('identity_drift_score')} cache {metrics.get('temporal_cache_usage')}",
            f"expr {metrics.get('face_expression_tensor_shape')} motion {metrics.get('face_expression_motion_energy')}",
            f"latent {metrics.get('latent_identity_conditioning_active')} controlnet {metrics.get('controlnet_maps_active')}",
            f"cross-attn {metrics.get('cross_attention_identity_conditioning_active')}",
        ]
        for index, line in enumerate(lines):
            draw.text((16, 94 + index * 16), line, fill=(214, 255, 236, 255))
        frame.alpha_composite(overlay)

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
            "semantic_debug_face_mask": self.debug_face_mask,
            "semantic_debug_identity": self.debug_identity,
            "semantic_debug_conditioning": self.debug_conditioning,
            "semantic_conditioning_mode": "debug_overlay" if self.debug_semantic_overlay else "hidden_pre_denoise",
            "avatar_session_count": session_count,
            "active_avatar": active_session.metrics() if active_session else None,
            **self.identity_lock.metrics(),
            **self.last_conditioning_metrics,
        }


__all__ = [
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_CONTENT_TYPES",
    "PortraitSession",
    "SemanticAvatarAdapter",
]
