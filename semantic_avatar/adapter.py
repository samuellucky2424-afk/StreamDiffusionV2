"""Adapter from semantic packets to stream-compatible conditioning frames."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .pose_map import PoseMapRenderer
from .schema import SemanticPacket

LOGGER = logging.getLogger(__name__)


class SemanticPoseConditioningAdapter:
    """Convert semantic packets into normalized RGB pose-map arrays."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        debug_dir: str | os.PathLike[str] | None = None,
        debug_every_n: int = 0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.renderer = PoseMapRenderer(width=self.width, height=self.height)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_every_n = max(0, int(debug_every_n))
        self.frames_rendered = 0
        self.last_render_ms = 0.0
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, width: int, height: int) -> "SemanticPoseConditioningAdapter":
        debug_dir = os.getenv("SEMANTIC_POSE_DEBUG_DIR")
        debug_every_n = int(os.getenv("SEMANTIC_POSE_DEBUG_EVERY_N", "0") or "0")
        return cls(width=width, height=height, debug_dir=debug_dir, debug_every_n=debug_every_n)

    def packet_to_array(self, packet_data: Mapping[str, Any] | SemanticPacket) -> np.ndarray:
        packet = packet_data if isinstance(packet_data, SemanticPacket) else SemanticPacket.from_mapping(packet_data)
        started = perf_counter()
        array = self.renderer.render_array(packet, normalize=True)
        self.last_render_ms = (perf_counter() - started) * 1000.0
        self.frames_rendered += 1
        self._maybe_write_debug_map(packet)
        return array

    def _maybe_write_debug_map(self, packet: SemanticPacket) -> None:
        if not self.debug_dir or self.debug_every_n <= 0:
            return
        if self.frames_rendered % self.debug_every_n != 0:
            return

        image = self.renderer.render(packet)
        frame_id = packet.frame_id if packet.frame_id is not None else self.frames_rendered
        image.save(self.debug_dir / f"pose_{frame_id:06d}.png")
        LOGGER.info(
            "semantic pose map debug frame=%s render_ms=%.3f packet=%s",
            frame_id,
            self.last_render_ms,
            packet.to_debug_dict(),
        )

    def metrics(self) -> dict[str, float | int]:
        return {
            "semantic_pose_frames": self.frames_rendered,
            "semantic_pose_last_render_ms": self.last_render_ms,
            "semantic_pose_width": self.width,
            "semantic_pose_height": self.height,
        }
