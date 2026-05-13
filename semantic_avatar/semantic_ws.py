"""FastAPI routes for semantic-driven StreamDiffusionV2 avatar streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO
from uuid import uuid4

from fastapi import File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .semantic_adapter import MAX_UPLOAD_BYTES, SUPPORTED_CONTENT_TYPES, SemanticAvatarAdapter
from .semantic_metrics import SemanticAvatarMetrics
from .semantic_renderer import PipelineQueueSemanticRenderer, latest_frame_payload

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SemanticAvatarRouteConfig:
    target_fps: int = 8
    jpeg_quality: int = 68
    max_input_queue_frames: int = 16
    max_output_frames_per_tick: int = 4
    image_format: str = "JPEG"
    metrics_interval_s: float = 1.0
    default_prompt: str | None = None
    debug_semantic_overlay: bool = False
    debug_face_mask: bool = False


def attach_semantic_avatar_routes(
    app: Any,
    *,
    pipeline: Any,
    width: int = 512,
    height: int = 512,
    route_config: SemanticAvatarRouteConfig | None = None,
) -> PipelineQueueSemanticRenderer:
    """Register upload and websocket routes on an existing FastAPI app."""

    config = route_config or SemanticAvatarRouteConfig()
    metrics = SemanticAvatarMetrics()
    adapter = SemanticAvatarAdapter.from_env(
        width=width,
        height=height,
        debug_semantic_overlay=config.debug_semantic_overlay,
        debug_face_mask=config.debug_face_mask,
    )
    renderer = PipelineQueueSemanticRenderer(
        pipeline=pipeline,
        adapter=adapter,
        metrics=metrics,
        prompt=config.default_prompt,
        max_input_queue_frames=config.max_input_queue_frames,
        max_output_frames_per_tick=config.max_output_frames_per_tick,
        image_format=config.image_format,
        image_quality=config.jpeg_quality,
    )

    @app.get("/semantic-avatar/health")
    async def semantic_avatar_health() -> dict[str, Any]:
        return {"ok": True, **renderer.metrics_snapshot()}

    @app.post("/avatar/upload")
    async def upload_avatar(image: UploadFile = File(...)) -> dict[str, Any]:
        if image.content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP portrait.")
        payload = await image.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Avatar image must be 10 MB or smaller.")
        started = perf_counter()
        try:
            result = await asyncio.to_thread(renderer.adapter.upload_portrait, image.filename or "avatar.png", payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["preprocess_ms"] = round((perf_counter() - started) * 1000.0, 2)
        LOGGER.info(
            "semantic avatar uploaded avatar_id=%s size=%sx%s preprocess_ms=%.2f",
            result["avatar_id"],
            result["width"],
            result["height"],
            result["preprocess_ms"],
        )
        return result

    async def semantic_avatar_ws(websocket: WebSocket) -> None:
        await _run_semantic_avatar_ws(websocket, renderer, config)

    app.add_api_websocket_route("/ws/semantic-avatar", semantic_avatar_ws)
    LOGGER.info(
        "Registered semantic avatar routes: POST /avatar/upload, WS /ws/semantic-avatar, GET /semantic-avatar/health"
    )
    return renderer


async def _run_semantic_avatar_ws(
    websocket: WebSocket,
    renderer: PipelineQueueSemanticRenderer,
    config: SemanticAvatarRouteConfig,
) -> None:
    avatar_id = websocket.query_params.get("avatar_id")
    prompt = websocket.query_params.get("prompt") or config.default_prompt
    await websocket.accept()

    latest: dict[str, Any] | None = None
    latest_received_t = 0.0
    latest_seq = 0
    submitted_seq = 0
    session_drops = 0
    stop = asyncio.Event()
    send_lock = asyncio.Lock()
    min_interval = 1.0 / max(1, int(config.target_fps))
    last_metrics_t = perf_counter()
    record_file, record_path = _open_recording_file(avatar_id)

    async def receiver() -> None:
        nonlocal latest, latest_received_t, latest_seq, session_drops
        try:
            while not stop.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect
                text = message.get("text")
                raw_bytes = message.get("bytes")
                if text is None and raw_bytes is not None:
                    text = raw_bytes.decode("utf-8", errors="ignore")
                if not text:
                    continue

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    renderer.metrics.packet_dropped()
                    continue

                if payload.get("type") == "ping":
                    sent_t = payload.get("t")
                    async with send_lock:
                        await websocket.send_text(json.dumps({"type": "pong", "t": sent_t}, separators=(",", ":")))
                    continue

                if payload.get("type") == "pong":
                    continue

                renderer.metrics.packet_received()
                if record_file is not None:
                    _record_packet(record_file, payload)
                packet_t = payload.get("timestamp", payload.get("wallTime", payload.get("t")))
                if isinstance(packet_t, (int, float)) and packet_t > 10_000_000:
                    renderer.metrics.websocket_rtt_ms.add(max(0.0, time.time() * 1000.0 - float(packet_t)))
                if latest is not None:
                    session_drops += 1
                    renderer.metrics.packet_dropped(stale=True)
                latest = payload
                latest_received_t = perf_counter()
                latest_seq += 1
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.exception("semantic avatar websocket receiver failed")
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

                packet = latest
                packet_received_t = latest_received_t
                seq = latest_seq
                latest = None

                if packet is not None and seq != submitted_seq:
                    try:
                        await asyncio.to_thread(
                            renderer.submit_packet,
                            packet,
                            avatar_id=avatar_id,
                            received_perf_t=packet_received_t,
                            prompt=prompt,
                        )
                        submitted_seq = seq
                    except KeyError as exc:
                        async with send_lock:
                            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}, separators=(",", ":")))
                        await asyncio.sleep(0.25)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("semantic avatar packet submit failed")
                        async with send_lock:
                            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}, separators=(",", ":")))

                frames = await asyncio.to_thread(renderer.collect_encoded_frames)
                latest_frame = latest_frame_payload(frames)
                if latest_frame is not None:
                    async with send_lock:
                        await websocket.send_bytes(latest_frame.payload)

                if perf_counter() - last_metrics_t >= config.metrics_interval_s:
                    snapshot = renderer.metrics_snapshot()
                    if record_path is not None:
                        snapshot["semantic_recording_path"] = str(record_path)
                    async with send_lock:
                        await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))
                    LOGGER.info(
                        "semantic-avatar fps in=%.2f out=%.2f queue=%s dropped=%s mouth=%s blink=%s ypr=%s/%s/%s controlnet=%sms q_delay=%sms encode=%sms denoise=%sms decode=%sms stream=%sms",
                        snapshot.get("semantic_fps") or 0.0,
                        snapshot.get("output_fps") or 0.0,
                        snapshot.get("queue_size") or 0,
                        snapshot.get("dropped_packets") or 0,
                        snapshot.get("mouth_open"),
                        snapshot.get("blink"),
                        snapshot.get("yaw"),
                        snapshot.get("pitch"),
                        snapshot.get("roll"),
                        snapshot.get("semantic_controlnet_latency_ms"),
                        snapshot.get("queue_delay_ms"),
                        snapshot.get("encode_latency_ms"),
                        snapshot.get("denoise_latency_ms"),
                        snapshot.get("decode_latency_ms"),
                        snapshot.get("stream_latency_ms"),
                    )
                    last_metrics_t = perf_counter()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.exception("semantic avatar websocket sender failed")
        finally:
            stop.set()

    try:
        await asyncio.gather(receiver(), render_sender())
    finally:
        LOGGER.info("semantic avatar websocket closed avatar_id=%s session_drops=%d", avatar_id, session_drops)
        if record_file is not None:
            record_file.close()
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _open_recording_file(avatar_id: str | None) -> tuple[TextIO | None, Path | None]:
    record_dir = os.getenv("SEMANTIC_AVATAR_RECORD_DIR")
    if not record_dir:
        return None, None
    try:
        root = Path(record_dir)
        root.mkdir(parents=True, exist_ok=True)
        safe_avatar = "".join(ch for ch in (avatar_id or "no-avatar") if ch.isalnum() or ch in {"-", "_"})[:40]
        path = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_avatar}_{uuid4().hex[:8]}.jsonl"
        return path.open("a", encoding="utf-8", buffering=1), path
    except Exception:  # noqa: BLE001
        LOGGER.exception("semantic avatar recording could not be opened")
        return None, None


def _record_packet(record_file: TextIO, payload: dict[str, Any]) -> None:
    try:
        record_file.write(
            json.dumps(
                {
                    "received_ms": int(time.time() * 1000.0),
                    "packet": payload,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    except Exception:  # noqa: BLE001
        LOGGER.exception("semantic avatar recording write failed")


__all__ = ["SemanticAvatarRouteConfig", "attach_semantic_avatar_routes"]
