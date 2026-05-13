"""Low-latency semantic avatar metric helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Deque


@dataclass(slots=True)
class RollingAverage:
    maxlen: int = 120
    values: Deque[float] = field(default_factory=deque)

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self.values.append(float(value))
        while len(self.values) > self.maxlen:
            self.values.popleft()

    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    @property
    def avg(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None


@dataclass(slots=True)
class SemanticAvatarMetrics:
    """Shared metrics object for route, adapter, and renderer."""

    packets_received: int = 0
    packets_submitted: int = 0
    dropped_packets: int = 0
    stale_packets: int = 0
    output_frames: int = 0
    websocket_rtt_ms: RollingAverage = field(default_factory=RollingAverage)
    queue_delay_ms: RollingAverage = field(default_factory=RollingAverage)
    adapter_latency_ms: RollingAverage = field(default_factory=RollingAverage)
    encode_latency_ms: RollingAverage = field(default_factory=RollingAverage)
    denoise_latency_ms: RollingAverage = field(default_factory=RollingAverage)
    decode_latency_ms: RollingAverage = field(default_factory=RollingAverage)
    stream_latency_ms: RollingAverage = field(default_factory=RollingAverage)
    _rate_started: float = field(default_factory=perf_counter)
    _last_rate_t: float = field(default_factory=perf_counter)
    _packets_since: int = 0
    _frames_since: int = 0
    _input_queue_size: int = 0
    _output_queue_size: int = 0
    input_fps: float = 0.0
    output_fps: float = 0.0

    def packet_received(self) -> None:
        self.packets_received += 1
        self._packets_since += 1

    def packet_submitted(self) -> None:
        self.packets_submitted += 1

    def packet_dropped(self, count: int = 1, *, stale: bool = False) -> None:
        self.dropped_packets += int(count)
        if stale:
            self.stale_packets += int(count)

    def frame_output(self, count: int = 1) -> None:
        self.output_frames += int(count)
        self._frames_since += int(count)

    def set_queue_sizes(self, input_size: int | None = None, output_size: int | None = None) -> None:
        if input_size is not None:
            self._input_queue_size = max(0, int(input_size))
        if output_size is not None:
            self._output_queue_size = max(0, int(output_size))

    def refresh_rates(self) -> None:
        now = perf_counter()
        elapsed = now - self._last_rate_t
        if elapsed < 1.0:
            return
        self.input_fps = self._packets_since / elapsed
        self.output_fps = self._frames_since / elapsed
        self._packets_since = 0
        self._frames_since = 0
        self._last_rate_t = now

    def snapshot(self) -> dict[str, float | int | None | str]:
        self.refresh_rates()

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 2)

        return {
            "type": "metrics",
            "packets_received": self.packets_received,
            "packets_submitted": self.packets_submitted,
            "dropped_packets": self.dropped_packets,
            "stale_packets": self.stale_packets,
            "output_frames": self.output_frames,
            "semantic_fps": round(self.input_fps, 2),
            "render_fps": round(self.output_fps, 2),
            "output_fps": round(self.output_fps, 2),
            "queue_size": self._input_queue_size,
            "output_queue_size": self._output_queue_size,
            "websocket_rtt_ms": rounded(self.websocket_rtt_ms.last),
            "queue_delay_ms": rounded(self.queue_delay_ms.last),
            "adapter_latency_ms": rounded(self.adapter_latency_ms.last),
            "encode_latency_ms": rounded(self.encode_latency_ms.last),
            "denoise_latency_ms": rounded(self.denoise_latency_ms.last),
            "decode_latency_ms": rounded(self.decode_latency_ms.last),
            "stream_latency_ms": rounded(self.stream_latency_ms.last),
            "avg_queue_delay_ms": rounded(self.queue_delay_ms.avg),
            "avg_encode_latency_ms": rounded(self.encode_latency_ms.avg),
            "avg_denoise_latency_ms": rounded(self.denoise_latency_ms.avg),
            "avg_decode_latency_ms": rounded(self.decode_latency_ms.avg),
        }


__all__ = ["RollingAverage", "SemanticAvatarMetrics"]
