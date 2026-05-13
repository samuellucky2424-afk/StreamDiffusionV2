"""Lightweight portrait identity locking for realtime semantic avatars.

This module is deliberately upstream-compatible with StreamDiffusionV2. The
Wan v2v demo path does not expose PuLID or InstantID attention adapters, so the
identity lock works at the existing image-conditioning boundary: it caches a
portrait embedding and folds color, low-frequency structure, and temporal
reference information into the hidden conditioning frame before denoising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter, time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _safe_std(array: np.ndarray) -> np.ndarray:
    return np.maximum(array, np.asarray([0.035, 0.035, 0.035], dtype=np.float32))


@dataclass(slots=True)
class IdentityEmbedding:
    avatar_id: str
    vector: np.ndarray
    reference_image: Image.Image
    reference_low: Image.Image
    color_mean: tuple[float, float, float]
    color_std: tuple[float, float, float]
    created_at: float = field(default_factory=time)
    frames_conditioned: int = 0
    last_drift_score: float | None = None
    last_condition_ms: float = 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "identity_lock_active": True,
            "identity_embedding_cached": True,
            "identity_embedding_dim": int(self.vector.shape[0]),
            "identity_cache_frames": self.frames_conditioned,
            "identity_created_at": self.created_at,
            "identity_drift_score": None if self.last_drift_score is None else round(self.last_drift_score, 4),
            "identity_condition_ms": round(self.last_condition_ms, 3),
        }


@dataclass(slots=True)
class IdentityConditioningResult:
    image: Image.Image
    metrics: dict[str, Any]


class IdentityLock:
    """Cache portrait identity and stabilize generated conditioning frames."""

    def __init__(
        self,
        *,
        base_strength: float = 0.20,
        global_strength: float = 0.055,
        temporal_strength: float = 0.075,
    ) -> None:
        self.base_strength = _clamp(base_strength, 0.0, 0.6)
        self.global_strength = _clamp(global_strength, 0.0, 0.18)
        self.temporal_strength = _clamp(temporal_strength, 0.0, 0.24)
        self._cache: dict[str, IdentityEmbedding] = {}
        self._last_frames: dict[str, Image.Image] = {}

    def cache_portrait(self, avatar_id: str, image: Image.Image) -> IdentityEmbedding:
        reference = image.convert("RGB").copy()
        vector = self._embed_image(reference)
        arr = np.asarray(reference, dtype=np.float32) / 255.0
        color_mean = tuple(float(v) for v in arr.reshape(-1, 3).mean(axis=0))
        color_std = tuple(float(v) for v in _safe_std(arr.reshape(-1, 3).std(axis=0)))
        embedding = IdentityEmbedding(
            avatar_id=avatar_id,
            vector=vector,
            reference_image=reference,
            reference_low=reference.filter(ImageFilter.GaussianBlur(radius=2.4)),
            color_mean=color_mean,
            color_std=color_std,
        )
        self._cache[avatar_id] = embedding
        self._last_frames.pop(avatar_id, None)
        return embedding

    def has_identity(self, avatar_id: str | None) -> bool:
        return bool(avatar_id and avatar_id in self._cache)

    def reset_temporal(self, avatar_id: str | None = None) -> None:
        if avatar_id is None:
            self._last_frames.clear()
            return
        self._last_frames.pop(avatar_id, None)

    def condition_frame(
        self,
        avatar_id: str,
        frame: Image.Image,
        *,
        expression_vector: np.ndarray | None = None,
        region_mask: Image.Image | None = None,
    ) -> IdentityConditioningResult:
        started = perf_counter()
        embedding = self._cache.get(avatar_id)
        if embedding is None:
            return IdentityConditioningResult(
                image=frame.convert("RGB"),
                metrics={
                    "identity_lock_active": False,
                    "identity_embedding_cached": False,
                    "latent_identity_conditioning_active": False,
                    "generator_identity_conditioning_active": False,
                    "cross_attention_identity_conditioning_active": False,
                },
            )

        source = frame.convert("RGB")
        reference = embedding.reference_image.resize(source.size, Image.Resampling.BICUBIC)
        reference_low = embedding.reference_low.resize(source.size, Image.Resampling.BICUBIC)
        mask = region_mask.convert("L").resize(source.size) if region_mask is not None else self._default_identity_mask(source.size)

        motion = self._expression_motion(expression_vector)
        local_strength = self.base_strength * (1.0 - min(0.45, motion * 0.28))
        global_strength = self.global_strength * (1.0 - min(0.35, motion * 0.22))
        temporal_strength = self.temporal_strength * (1.0 - min(0.55, motion * 0.45))

        matched = self._match_color(source, reference, strength=0.48)
        globally_matched = Image.blend(source, matched, global_strength)
        identity_reference = Image.blend(matched, reference_low, local_strength)
        identity_mask = mask.point(lambda value: int(value * 0.78))
        conditioned = Image.composite(identity_reference, globally_matched, identity_mask)

        previous = self._last_frames.get(avatar_id)
        temporal_cache_used = previous is not None
        if previous is not None and previous.size == conditioned.size:
            conditioned = Image.blend(conditioned, previous, temporal_strength)

        drift = self.identity_drift_score(avatar_id, conditioned)
        embedding.frames_conditioned += 1
        embedding.last_drift_score = drift
        embedding.last_condition_ms = (perf_counter() - started) * 1000.0
        self._last_frames[avatar_id] = conditioned.copy()

        metrics = {
            **embedding.metrics(),
            "latent_identity_conditioning_active": True,
            "generator_identity_conditioning_active": True,
            "cross_attention_identity_conditioning_active": False,
            "cross_attention_identity_conditioning_mode": "unavailable_in_upstream_wan_demo",
            "identity_conditioning_mode": "cached_portrait_pre_denoise",
            "identity_preserve_strength": round(local_strength, 4),
            "identity_global_strength": round(global_strength, 4),
            "identity_temporal_strength": round(temporal_strength, 4),
            "identity_expression_motion": round(motion, 4),
            "temporal_cache_usage": "hit" if temporal_cache_used else "miss",
            "temporal_identity_persistence": True,
        }
        return IdentityConditioningResult(image=conditioned, metrics=metrics)

    def identity_drift_score(self, avatar_id: str, image: Image.Image) -> float | None:
        embedding = self._cache.get(avatar_id)
        if embedding is None:
            return None
        vector = self._embed_image(image)
        similarity = float(np.dot(embedding.vector, vector))
        return _clamp(1.0 - similarity, 0.0, 2.0)

    def build_conditioning_payload(self, avatar_id: str) -> dict[str, Any]:
        embedding = self._cache.get(avatar_id)
        if embedding is None:
            return {}
        return {
            "identity_embedding": embedding.vector.astype(np.float32),
            "identity_embedding_shape": tuple(int(v) for v in embedding.vector.shape),
            "identity_reference_size": embedding.reference_image.size,
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "identity_session_count": len(self._cache),
            "identity_temporal_cache_entries": len(self._last_frames),
        }

    def _default_identity_mask(self, size: tuple[int, int]) -> Image.Image:
        width, height = size
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            [
                width * 0.22,
                height * 0.075,
                width * 0.78,
                height * 0.79,
            ],
            fill=255,
        )
        draw.rounded_rectangle(
            [
                width * 0.18,
                height * 0.53,
                width * 0.82,
                height * 0.98,
            ],
            radius=max(8, width // 16),
            fill=92,
        )
        return mask.filter(ImageFilter.GaussianBlur(radius=max(3, width // 64)))

    @staticmethod
    def _match_color(frame: Image.Image, reference: Image.Image, *, strength: float) -> Image.Image:
        strength = _clamp(strength, 0.0, 1.0)
        arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        ref = np.asarray(reference.convert("RGB").resize(frame.size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        ref_flat = ref.reshape(-1, 3)
        mean = flat.mean(axis=0)
        std = _safe_std(flat.std(axis=0))
        ref_mean = ref_flat.mean(axis=0)
        ref_std = _safe_std(ref_flat.std(axis=0))
        adjusted = (arr - mean) * (ref_std / std) + ref_mean
        adjusted = np.clip(adjusted, 0.0, 1.0)
        blended = arr * (1.0 - strength) + adjusted * strength
        return Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _expression_motion(expression_vector: np.ndarray | None) -> float:
        if expression_vector is None:
            return 0.0
        values = np.asarray(expression_vector, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return 0.0
        return _clamp(float(np.mean(np.abs(values))), 0.0, 1.0)

    @staticmethod
    def _embed_image(image: Image.Image) -> np.ndarray:
        crop = ImageOps.fit(image.convert("RGB"), (128, 128), method=Image.Resampling.BICUBIC, centering=(0.5, 0.38))
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        gray = np.asarray(crop.convert("L").resize((16, 16), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0

        histograms: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(arr[..., channel], bins=16, range=(0.0, 1.0))
            histograms.append(hist.astype(np.float32) / max(1.0, float(hist.sum())))

        gy, gx = np.gradient(gray)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / max(float(edge.max()), 1e-6)
        edge_small = np.asarray(Image.fromarray((edge * 255.0).astype(np.uint8), "L").resize((8, 8), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0

        mean = arr.reshape(-1, 3).mean(axis=0)
        std = _safe_std(arr.reshape(-1, 3).std(axis=0))
        vector = np.concatenate(
            [
                gray.reshape(-1) - float(gray.mean()),
                edge_small.reshape(-1),
                *histograms,
                mean,
                std,
            ]
        ).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return vector
        return vector / norm


__all__ = ["IdentityConditioningResult", "IdentityEmbedding", "IdentityLock"]
