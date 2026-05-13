"""Realtime RGB frame websocket bridge for StreamDiffusionV2 video-to-video."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image, ImageOps

from .semantic_adapter import SemanticAvatarAdapter
from .semantic_metrics import SemanticAvatarMetrics
from .semantic_renderer import EncodedFrame, latest_frame_payload

LOGGER = logging.getLogger(__name__)


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
        except Exception:  # noqa: BLE001
            break
        drained += 1
    return drained


@dataclass(slots=True)
class VideoAvatarRouteConfig:
    target_fps: int = 8
    jpeg_quality: int = 68
    max_input_queue_frames: int = 8
    max_output_frames_per_tick: int = 4
    image_format: str = "JPEG"
    metrics_interval_s: float = 1.0
    default_prompt: str | None = None


class VideoFrameStreamRenderer:
    """Feed binary webcam frames into the existing StreamDiffusionV2 queue."""

    def __init__(
        self,
        *,
        pipeline: Any,
        width: int,
        height: int,
        metrics: SemanticAvatarMetrics,
        adapter: SemanticAvatarAdapter | None = None,
        prompt: str | None = None,
        max_input_queue_frames: int = 8,
        max_output_frames_per_tick: int = 4,
        image_format: str = "JPEG",
        image_quality: int = 68,
    ) -> None:
        self.pipeline = pipeline
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.metrics = metrics
        self.adapter = adapter
        self.prompt = prompt
        self.max_input_queue_frames = max(1, int(max_input_queue_frames))
        self.max_output_frames_per_tick = max(1, int(max_output_frames_per_tick))
        self.image_format = image_format.upper()
        self.image_quality = max(1, min(95, int(image_quality)))
        self._submitted_at: list[float] = []
        self.frames_decoded = 0
        self.last_frame_bytes = 0
        self.last_decode_ms = 0.0
        self.last_input_shape = f"{self.height}x{self.width}x3"
        self.last_identity_metrics: dict[str, Any] = {}

    def submit_frame(
        self,
        payload: bytes,
        *,
        avatar_id: str | None = None,
        received_perf_t: float | None = None,
        prompt: str | None = None,
    ) -> dict[str, float | int | str | None]:
        started = perf_counter()
        if received_perf_t is not None:
            self.metrics.queue_delay_ms.add((started - received_perf_t) * 1000.0)

        image = self._decode_payload(payload)
        decode_ms = (perf_counter() - started) * 1000.0
        self.last_decode_ms = decode_ms
        self.metrics.adapter_latency_ms.add(decode_ms)
        self.frames_decoded += 1
        self.last_frame_bytes = len(payload)
        image = self._apply_identity_conditioning(image, avatar_id)

        self._maybe_update_prompt(prompt)
        input_queue = getattr(self.pipeline, "input_queue", None)
        if input_queue is None:
            raise RuntimeError("Pipeline does not expose an input_queue.")

        dropped = 0
        if _qsize(input_queue) >= self.max_input_queue_frames:
            dropped = _drain_queue(input_queue)
            self.metrics.packet_dropped(dropped, stale=True)

        input_queue.put(self._to_array(image))
        self._submitted_at.append(perf_counter())
        while len(self._submitted_at) > self.max_input_queue_frames * 2:
            self._submitted_at.pop(0)

        self.metrics.packet_submitted()
        self.metrics.set_queue_sizes(input_size=_qsize(input_queue))
        return {
            "decode_ms": round(decode_ms, 3),
            "dropped_input_frames": dropped,
            "input_queue_size": _qsize(input_queue),
            "input_shape": self.last_input_shape,
        }

    def collect_encoded_frames(self) -> list[EncodedFrame]:
        decode_started = perf_counter()
        images = self._produce_images()
        output_decode_ms = (perf_counter() - decode_started) * 1000.0
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
            per_frame_decode_ms = output_decode_ms / max(1, len(images))
            self.metrics.encode_latency_ms.add(encode_ms)
            self.metrics.decode_latency_ms.add(per_frame_decode_ms)
            if stream_latency_ms is not None:
                self.metrics.stream_latency_ms.add(stream_latency_ms)
                self.metrics.denoise_latency_ms.add(max(0.0, stream_latency_ms - encode_ms - per_frame_decode_ms))
            frames.append(
                EncodedFrame(
                    payload=payload,
                    media_type="image/webp" if self.image_format == "WEBP" else "image/jpeg",
                    encode_ms=encode_ms,
                    decode_ms=per_frame_decode_ms,
                    stream_latency_ms=stream_latency_ms,
                )
            )

        self.metrics.frame_output(len(frames))
        self.metrics.set_queue_sizes(
            input_size=_qsize(getattr(self.pipeline, "input_queue", None)),
            output_size=_qsize(getattr(self.pipeline, "output_queue", None)),
        )
        return frames

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            **self.metrics.snapshot(),
            "video_avatar_active": True,
            "video_avatar_frames_decoded": self.frames_decoded,
            "video_avatar_last_frame_bytes": self.last_frame_bytes,
            "video_avatar_decode_ms": round(self.last_decode_ms, 3),
            "video_avatar_input_shape": self.last_input_shape,
            "video_avatar_width": self.width,
            "video_avatar_height": self.height,
            "video_avatar_conditioning_mode": "hybrid_rgb_identity_pre_denoise"
            if self.last_identity_metrics.get("identity_lock_active")
            else "rgb_frame_pre_denoise",
            "conditioning_tensor_shape": self.last_input_shape,
            **self.last_identity_metrics,
        }

    def _decode_payload(self, payload: bytes) -> Image.Image:
        with Image.open(io.BytesIO(payload)) as source:
            return ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (self.width, self.height),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )

    def _apply_identity_conditioning(self, image: Image.Image, avatar_id: str | None) -> Image.Image:
        if not avatar_id or self.adapter is None or not self.adapter.identity_lock.has_identity(avatar_id):
            self.last_identity_metrics = {
                "video_avatar_identity_lock_active": False,
                "video_avatar_avatar_id": avatar_id,
            }
            return image
        result = self.adapter.identity_lock.condition_frame(avatar_id, image)
        self.last_identity_metrics = {
            **result.metrics,
            "video_avatar_identity_lock_active": True,
            "video_avatar_avatar_id": avatar_id,
        }
        return result.image

    @staticmethod
    def _to_array(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.float32) / 127.5 - 1.0

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


def attach_video_avatar_routes(
    app: Any,
    *,
    pipeline: Any,
    adapter: SemanticAvatarAdapter | None = None,
    width: int = 512,
    height: int = 512,
    route_config: VideoAvatarRouteConfig | None = None,
) -> VideoFrameStreamRenderer:
    config = route_config or VideoAvatarRouteConfig()
    metrics = SemanticAvatarMetrics()
    renderer = VideoFrameStreamRenderer(
        pipeline=pipeline,
        width=width,
        height=height,
        metrics=metrics,
        adapter=adapter,
        prompt=config.default_prompt,
        max_input_queue_frames=config.max_input_queue_frames,
        max_output_frames_per_tick=config.max_output_frames_per_tick,
        image_format=config.image_format,
        image_quality=config.jpeg_quality,
    )

    @app.get("/video-avatar/health")
    async def video_avatar_health() -> dict[str, Any]:
        return {"ok": True, **renderer.metrics_snapshot()}

    async def video_avatar_ws(websocket: WebSocket) -> None:
        await _run_video_avatar_ws(websocket, renderer, config)

    app.add_api_websocket_route("/ws/video-avatar", video_avatar_ws)
    LOGGER.info("Registered video avatar routes: WS /ws/video-avatar, GET /video-avatar/health")
    return renderer


async def _run_video_avatar_ws(
    websocket: WebSocket,
    renderer: VideoFrameStreamRenderer,
    config: VideoAvatarRouteConfig,
) -> None:
    prompt = websocket.query_params.get("prompt") or config.default_prompt
    avatar_id = websocket.query_params.get("avatar_id")
    await websocket.accept()

    latest: bytes | None = None
    latest_received_t = 0.0
    latest_seq = 0
    submitted_seq = 0
    session_drops = 0
    stop = asyncio.Event()
    send_lock = asyncio.Lock()
    min_interval = 1.0 / max(1, int(config.target_fps))
    last_metrics_t = perf_counter()

    async def receiver() -> None:
        nonlocal latest, latest_received_t, latest_seq, session_drops, prompt
        try:
            while not stop.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect

                text = message.get("text")
                raw_bytes = message.get("bytes")
                if text:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        renderer.metrics.packet_dropped()
                        continue
                    if payload.get("type") == "ping":
                        sent_t = payload.get("t")
                        async with send_lock:
                            await websocket.send_text(json.dumps({"type": "pong", "t": sent_t}, separators=(",", ":")))
                    elif payload.get("type") == "config":
                        prompt = payload.get("prompt") or prompt
                    continue

                if not raw_bytes:
                    continue
                renderer.metrics.packet_received()
                if latest is not None:
                    session_drops += 1
                    renderer.metrics.packet_dropped(stale=True)
                latest = raw_bytes
                latest_received_t = perf_counter()
                latest_seq += 1
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.exception("video avatar websocket receiver failed")
        finally:
            stop.set()

    async def render_sender() -> None:
        nonlocal latest, submitted_seq, last_metrics_t
        last_tick_t = 0.0
        try:
            while not stop.is_set():
                now = perf_counter()
                wait = min_interval - (now - last_tick_t)
                if wait > 0:
                    await asyncio.sleep(wait)
                last_tick_t = perf_counter()

                frame = latest
                frame_received_t = latest_received_t
                seq = latest_seq
                latest = None

                if frame is not None and seq != submitted_seq:
                    try:
                        await asyncio.to_thread(
                            renderer.submit_frame,
                            frame,
                            avatar_id=avatar_id,
                            received_perf_t=frame_received_t,
                            prompt=prompt,
                        )
                        submitted_seq = seq
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("video avatar frame submit failed")
                        async with send_lock:
                            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}, separators=(",", ":")))

                frames = await asyncio.to_thread(renderer.collect_encoded_frames)
                latest_frame = latest_frame_payload(frames)
                if latest_frame is not None:
                    async with send_lock:
                        await websocket.send_bytes(latest_frame.payload)

                if perf_counter() - last_metrics_t >= config.metrics_interval_s:
                    snapshot = renderer.metrics_snapshot()
                    async with send_lock:
                        await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))
                    LOGGER.info(
                        "video-avatar fps in=%.2f out=%.2f queue=%s dropped=%s frame=%sB input=%s identity=%s drift=%s decode=%sms encode=%sms denoise=%sms stream=%sms",
                        snapshot.get("semantic_fps") or 0.0,
                        snapshot.get("output_fps") or 0.0,
                        snapshot.get("queue_size") or 0,
                        snapshot.get("dropped_packets") or 0,
                        snapshot.get("video_avatar_last_frame_bytes") or 0,
                        snapshot.get("video_avatar_input_shape"),
                        snapshot.get("video_avatar_identity_lock_active"),
                        snapshot.get("identity_drift_score"),
                        snapshot.get("video_avatar_decode_ms"),
                        snapshot.get("encode_latency_ms"),
                        snapshot.get("denoise_latency_ms"),
                        snapshot.get("stream_latency_ms"),
                    )
                    last_metrics_t = perf_counter()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.exception("video avatar websocket sender failed")
        finally:
            stop.set()

    try:
        await asyncio.gather(receiver(), render_sender())
    finally:
        LOGGER.info("video avatar websocket closed session_drops=%d", session_drops)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["VideoAvatarRouteConfig", "VideoFrameStreamRenderer", "attach_video_avatar_routes"]
