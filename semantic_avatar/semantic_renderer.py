"""Renderer bridge between semantic conditioning and StreamDiffusionV2 queues."""

from __future__ import annotations

import io
import queue
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

from PIL import Image

from .semantic_adapter import SemanticAvatarAdapter
from .semantic_metrics import SemanticAvatarMetrics


@dataclass(slots=True)
class EncodedFrame:
    payload: bytes
    media_type: str
    encode_ms: float
    decode_ms: float
    stream_latency_ms: float | None


def _qsize(q: Any) -> int:
    try:
        return int(q.qsize())
    except (AttributeError, NotImplementedError):
        return 0


def _drain_queue(q: Any, max_items: int | None = None) -> int:
    drained = 0
    while max_items is None or drained < max_items:
        try:
            q.get_nowait()
        except AttributeError:
            try:
                q.get(False)
            except Exception:  # noqa: BLE001
                break
        except queue.Empty:
            break
        except Exception:  # noqa: BLE001
            break
        drained += 1
    return drained


class PipelineQueueSemanticRenderer:
    """Feed synthetic semantic frames into an existing demo Pipeline instance."""

    def __init__(
        self,
        pipeline: Any,
        adapter: SemanticAvatarAdapter,
        metrics: SemanticAvatarMetrics,
        *,
        prompt: str | None = None,
        max_input_queue_frames: int = 16,
        max_output_frames_per_tick: int = 4,
        image_format: str = "JPEG",
        image_quality: int = 68,
    ) -> None:
        self.pipeline = pipeline
        self.adapter = adapter
        self.metrics = metrics
        self.prompt = prompt
        self.max_input_queue_frames = max(1, int(max_input_queue_frames))
        self.max_output_frames_per_tick = max(1, int(max_output_frames_per_tick))
        self.image_format = image_format.upper()
        self.image_quality = max(1, min(95, int(image_quality)))
        self._submitted_at: list[float] = []

    def submit_packet(
        self,
        packet: Mapping[str, Any],
        *,
        avatar_id: str | None,
        received_perf_t: float | None = None,
        prompt: str | None = None,
    ) -> dict[str, float | int | None]:
        started = perf_counter()
        if received_perf_t is not None:
            self.metrics.queue_delay_ms.add((started - received_perf_t) * 1000.0)

        image_array = self.adapter.packet_to_array(packet, avatar_id=avatar_id, normalize=True)
        adapter_ms = (perf_counter() - started) * 1000.0
        self.metrics.adapter_latency_ms.add(adapter_ms)

        self._maybe_update_prompt(prompt)

        dropped = 0
        input_queue = getattr(self.pipeline, "input_queue", None)
        if input_queue is None:
            raise RuntimeError("Pipeline does not expose an input_queue.")

        input_qsize = _qsize(input_queue)
        if input_qsize >= self.max_input_queue_frames:
            dropped = _drain_queue(input_queue)
            self.metrics.packet_dropped(dropped, stale=True)

        input_queue.put(image_array)
        self._submitted_at.append(perf_counter())
        while len(self._submitted_at) > self.max_input_queue_frames * 2:
            self._submitted_at.pop(0)

        self.metrics.packet_submitted()
        self.metrics.set_queue_sizes(input_size=_qsize(input_queue))
        return {
            "adapter_ms": adapter_ms,
            "dropped_input_frames": dropped,
            "input_queue_size": _qsize(input_queue),
        }

    def collect_encoded_frames(self) -> list[EncodedFrame]:
        decode_started = perf_counter()
        images = self._produce_images()
        decode_ms = (perf_counter() - decode_started) * 1000.0
        if not images:
            self.metrics.set_queue_sizes(
                input_size=_qsize(getattr(self.pipeline, "input_queue", None)),
                output_size=_qsize(getattr(self.pipeline, "output_queue", None)),
            )
            return []

        frames: list[EncodedFrame] = []
        for image in images[-self.max_output_frames_per_tick :]:
            stream_latency_ms = self._pop_stream_latency()
            encode_started = perf_counter()
            payload = self._encode_image(image)
            encode_ms = (perf_counter() - encode_started) * 1000.0
            self.metrics.encode_latency_ms.add(encode_ms)
            self.metrics.decode_latency_ms.add(decode_ms / max(1, len(images)))
            if stream_latency_ms is not None:
                self.metrics.stream_latency_ms.add(stream_latency_ms)
                denoise_ms = max(0.0, stream_latency_ms - encode_ms - (decode_ms / max(1, len(images))))
                self.metrics.denoise_latency_ms.add(denoise_ms)
            frames.append(
                EncodedFrame(
                    payload=payload,
                    media_type="image/webp" if self.image_format == "WEBP" else "image/jpeg",
                    encode_ms=encode_ms,
                    decode_ms=decode_ms / max(1, len(images)),
                    stream_latency_ms=stream_latency_ms,
                )
            )

        self.metrics.frame_output(len(frames))
        self.metrics.set_queue_sizes(
            input_size=_qsize(getattr(self.pipeline, "input_queue", None)),
            output_size=_qsize(getattr(self.pipeline, "output_queue", None)),
        )
        return frames

    def _produce_images(self) -> list[Image.Image]:
        if hasattr(self.pipeline, "produce_outputs"):
            return list(self.pipeline.produce_outputs())

        output_queue = getattr(self.pipeline, "output_queue", None)
        if output_queue is None:
            return []

        images: list[Image.Image] = []
        while _qsize(output_queue) > 0:
            item = output_queue.get()
            if isinstance(item, Image.Image):
                images.append(item)
        return images

    def _encode_image(self, image: Image.Image) -> bytes:
        frame = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        if self.image_format in {"JPEG", "JPG", "WEBP"}:
            save_kwargs["quality"] = self.image_quality
        if self.image_format in {"JPEG", "JPG"}:
            image = image.convert("RGB")
            fmt = "JPEG"
        else:
            fmt = "WEBP"
        image.save(frame, format=fmt, **save_kwargs)
        return frame.getvalue()

    def _maybe_update_prompt(self, prompt: str | None) -> None:
        requested = prompt or self.prompt
        if not requested:
            return
        runtime_state = getattr(self.pipeline, "runtime_state", None)
        if runtime_state is not None and runtime_state.get("prompt") != requested:
            runtime_state["prompt"] = requested
        if getattr(self.pipeline, "prompt", None) != requested:
            try:
                self.pipeline.prompt = requested
            except Exception:  # noqa: BLE001
                pass

    def _pop_stream_latency(self) -> float | None:
        if not self._submitted_at:
            return None
        submitted_at = self._submitted_at.pop(0)
        return (perf_counter() - submitted_at) * 1000.0

    def metrics_snapshot(self) -> dict[str, Any]:
        payload = self.metrics.snapshot()
        payload.update(self.adapter.metrics())
        return payload


def latest_frame_payload(frames: Iterable[EncodedFrame]) -> EncodedFrame | None:
    latest = None
    for latest in frames:
        pass
    return latest


__all__ = ["EncodedFrame", "PipelineQueueSemanticRenderer", "latest_frame_payload"]
