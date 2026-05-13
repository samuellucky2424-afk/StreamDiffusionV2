"""Compact facial expression tensor encoder for semantic avatar conditioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .semantic_pose import SemanticPacket


FEATURE_NAMES = (
    "yaw",
    "pitch",
    "roll",
    "mouth_open",
    "jaw_open",
    "lip_separation",
    "lip_width",
    "lip_corner_stretch",
    "mouth_roundness",
    "smile",
    "blink_left",
    "blink_right",
    "brow",
    "cheek",
    "eye_x",
    "eye_y",
    "neck_rotation",
    "shoulder_rotation",
    "torso_rotation",
    "head_x",
    "head_y",
    "shoulder_x",
    "shoulder_y",
    "confidence",
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class ExpressionConditioning:
    vector: np.ndarray
    velocity: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def tensor_shape(self) -> tuple[int, int]:
        return (1, int(self.vector.shape[0]))

    def payload(self) -> dict[str, Any]:
        return {
            "expression_tensor": self.vector.astype(np.float32),
            "expression_velocity": self.velocity.astype(np.float32),
            "expression_tensor_shape": self.tensor_shape,
            "expression_feature_names": self.feature_names,
        }


class FaceExpressionEncoder:
    """Convert dense MediaPipe semantics into a smoothed expression tensor."""

    def __init__(self) -> None:
        self._state: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self.frames = 0
        self.last_metrics: dict[str, Any] = {}

    def reset(self) -> None:
        self._state = None
        self._velocity = None
        self.frames = 0
        self.last_metrics = {}

    def encode(
        self,
        packet: SemanticPacket,
        control: Mapping[str, Any],
        *,
        conditioning_shape: tuple[int, int, int] | None = None,
    ) -> ExpressionConditioning:
        started = perf_counter()
        raw = self._raw_vector(packet, control)
        if self._state is None:
            smoothed = raw
            velocity = np.zeros_like(raw)
        else:
            delta = raw - self._state
            velocity = self._smooth_velocity(delta)
            alpha = self._alpha(raw, delta)
            smoothed = self._state + delta * alpha

        self._state = smoothed.astype(np.float32)
        self._velocity = velocity.astype(np.float32)
        self.frames += 1

        motion_energy = float(np.mean(np.abs(velocity)))
        mouth_energy = float(np.mean(np.abs(smoothed[3:9])))
        eye_energy = float(np.mean(np.abs(smoothed[10:13])))
        pose_energy = float(np.mean(np.abs(smoothed[[0, 1, 2, 16, 17, 18]])))
        shape = (1, int(smoothed.shape[0]))
        self.last_metrics = {
            "face_expression_encoder_frames": self.frames,
            "face_expression_encoder_ms": round((perf_counter() - started) * 1000.0, 3),
            "face_expression_tensor_shape": f"{shape[0]}x{shape[1]}",
            "face_expression_features": len(FEATURE_NAMES),
            "face_expression_motion_energy": round(motion_energy, 4),
            "face_expression_mouth_energy": round(mouth_energy, 4),
            "face_expression_eye_energy": round(eye_energy, 4),
            "face_expression_pose_energy": round(pose_energy, 4),
            "face_expression_conditioning_active": True,
            "face_expression_conditioning_stage": "pre_denoise_hidden_conditioning",
            "expression_smoothing_active": True,
            "motion_damping_active": True,
            "blink_hysteresis_active": True,
            "mouth_smoothing_active": True,
        }
        if conditioning_shape is not None:
            self.last_metrics["conditioning_source_shape"] = "x".join(str(int(value)) for value in conditioning_shape)

        return ExpressionConditioning(
            vector=smoothed.astype(np.float32),
            velocity=velocity.astype(np.float32),
            metrics=self.last_metrics,
        )

    def metrics(self) -> dict[str, Any]:
        return dict(self.last_metrics)

    def _raw_vector(self, packet: SemanticPacket, control: Mapping[str, Any]) -> np.ndarray:
        mouth = control.get("mouth")
        mouth_roundness = 0.0
        if isinstance(mouth, Mapping):
            mouth_roundness = float(mouth.get("roundness") or 0.0)

        values = [
            _clamp(float(control.get("yaw", packet.yaw)) / 64.0, -1.0, 1.0),
            _clamp(float(control.get("pitch", packet.pitch)) / 48.0, -1.0, 1.0),
            _clamp(float(control.get("roll", packet.roll)) / 54.0, -1.0, 1.0),
            _clamp(float(control.get("mouth_open", packet.mouth_open)), 0.0, 1.0),
            _clamp(float(control.get("jaw_open", packet.jaw_open)), 0.0, 1.0),
            _clamp(float(control.get("lip_separation", packet.mouth_upper_lower)), 0.0, 1.0),
            _clamp(float(control.get("lip_width", packet.lip_width)), 0.0, 1.0),
            _clamp(float(control.get("lip_corner_stretch", packet.lip_corner_stretch)), 0.0, 1.0),
            _clamp(mouth_roundness, 0.0, 1.0),
            _clamp(float(control.get("smile", packet.smile)), 0.0, 1.0),
            _clamp(float(control.get("blink_left", packet.blink_left)), 0.0, 1.0),
            _clamp(float(control.get("blink_right", packet.blink_right)), 0.0, 1.0),
            _clamp(float(control.get("brow", packet.brow)), 0.0, 1.0),
            _clamp(float(control.get("cheek", packet.cheek)), 0.0, 1.0),
            _clamp(float(control.get("eye_x", packet.eye_x)), -1.0, 1.0),
            _clamp(float(control.get("eye_y", packet.eye_y)), -1.0, 1.0),
            _clamp(float(control.get("neck_rotation", packet.neck_rotation)) / 55.0, -1.0, 1.0),
            _clamp(float(control.get("shoulder_rotation", packet.shoulder_rotation)) / 62.0, -1.0, 1.0),
            _clamp(float(control.get("torso_rotation", packet.torso_rotation)) / 55.0, -1.0, 1.0),
            _clamp((float(control.get("head_x", packet.head_x)) - 0.5) * 2.0, -1.0, 1.0),
            _clamp((float(control.get("head_y", packet.head_y)) - 0.35) * 2.0, -1.0, 1.0),
            _clamp((float(control.get("shoulder_x", packet.shoulder_x)) - 0.5) * 2.0, -1.0, 1.0),
            _clamp((float(control.get("shoulder_y", packet.shoulder_y)) - 0.62) * 2.0, -1.0, 1.0),
            _clamp(float(control.get("confidence", packet.confidence)), 0.0, 1.0),
        ]
        return np.asarray(values, dtype=np.float32)

    def _smooth_velocity(self, delta: np.ndarray) -> np.ndarray:
        if self._velocity is None:
            return delta
        return self._velocity * 0.35 + delta * 0.65

    @staticmethod
    def _alpha(raw: np.ndarray, delta: np.ndarray) -> np.ndarray:
        alpha = np.full_like(raw, 0.42, dtype=np.float32)
        fast_indices = np.asarray([3, 4, 5, 10, 11], dtype=np.int64)
        alpha[fast_indices] = 0.72
        pose_indices = np.asarray([0, 1, 2, 16, 17, 18], dtype=np.int64)
        alpha[pose_indices] = 0.50
        alpha = np.minimum(0.88, alpha + np.abs(delta) * 0.22)
        deadzone = np.full_like(raw, 0.006, dtype=np.float32)
        deadzone[fast_indices] = 0.002
        return np.where(np.abs(delta) < deadzone, 0.0, alpha)


__all__ = ["ExpressionConditioning", "FaceExpressionEncoder", "FEATURE_NAMES"]
