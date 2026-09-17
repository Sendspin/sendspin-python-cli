"""Tests for the source streamer command handling and framing."""

from __future__ import annotations

import asyncio

from aiosendspin.models.source import ClientStreamStartSource, SourceCommandPayload
from aiosendspin.models.types import AudioCodec, SourceCommand, SourceSignal

from sendspin.source_stream import SourceStreamConfig, SourceStreamer


class _FakeClient:
    """Records the source-related calls a SourceStreamer makes."""

    def __init__(self) -> None:
        self.stream_starts: list[ClientStreamStartSource] = []
        self.stream_ends = 0
        self.chunks: list[tuple[bytes, int]] = []
        self.signals: list[SourceSignal | None] = []
        self._clock = 1_000_000

    def now_us(self) -> int:
        self._clock += 1000
        return self._clock

    async def send_client_stream_start(self, source: ClientStreamStartSource) -> None:
        self.stream_starts.append(source)

    async def send_client_stream_end(self) -> None:
        self.stream_ends += 1

    async def send_source_audio_chunk(self, data: bytes, *, capture_timestamp_us: int) -> bool:
        self.chunks.append((data, capture_timestamp_us))
        return True

    async def send_source_state(self, *, signal: SourceSignal | None = None) -> None:
        self.signals.append(signal)


def _config(*, codec: AudioCodec = AudioCodec.PCM, line_sense: bool = False) -> SourceStreamConfig:
    return SourceStreamConfig(
        codec=codec,
        input_kind="sine",
        device=None,
        sample_rate=48000,
        channels=2,
        frame_ms=20,
        sine_hz=440.0,
        signal_threshold_db=-50.0,
        line_sense=line_sense,
    )


def _make() -> tuple[SourceStreamer, _FakeClient]:
    client = _FakeClient()
    return SourceStreamer(client, _config()), client  # type: ignore[arg-type]


async def test_begin_stream_announces_format_and_starts() -> None:
    """Beginning a stream sends client_stream/start and marks streaming active."""
    streamer, client = _make()
    await streamer._begin_stream()  # noqa: SLF001
    assert len(client.stream_starts) == 1
    assert client.stream_starts[0].codec == AudioCodec.PCM
    assert streamer._streaming.is_set()  # noqa: SLF001


async def test_end_stream_sends_end_and_stops() -> None:
    """Ending a stream sends client_stream/end and clears streaming."""
    streamer, client = _make()
    await streamer._begin_stream()  # noqa: SLF001
    await streamer._end_stream()  # noqa: SLF001
    assert client.stream_ends == 1
    assert not streamer._streaming.is_set()  # noqa: SLF001


async def test_send_frame_streams_only_when_active() -> None:
    """Frames are streamed only after the stream has begun."""
    streamer, client = _make()
    pcm = b"\x01\x02\x03\x04" * 16

    await streamer._send_frame(pcm)  # noqa: SLF001  (not started yet)
    assert client.chunks == []

    await streamer._begin_stream()  # noqa: SLF001
    await streamer._send_frame(pcm)  # noqa: SLF001
    assert len(client.chunks) == 1
    assert client.chunks[0][0] == pcm  # PCM passthrough


async def test_line_sense_reports_signal_changes() -> None:
    """With line sensing enabled, signal presence changes are reported once."""
    client = _FakeClient()
    streamer = SourceStreamer(client, _config(line_sense=True))  # type: ignore[arg-type]

    loud = b"\x00\x40" * 64  # non-trivial amplitude
    silence = b"\x00\x00" * 64

    streamer._maybe_report_signal(loud)  # noqa: SLF001
    streamer._maybe_report_signal(loud)  # no change -> not re-reported
    streamer._maybe_report_signal(silence)  # noqa: SLF001
    await asyncio.sleep(0.05)  # let the scheduled send_source_state tasks run

    assert client.signals == [SourceSignal.PRESENT, SourceSignal.ABSENT]


async def test_handle_source_command_dispatches_start_stop() -> None:
    """A server start command begins streaming; a stop command ends it."""
    streamer, client = _make()

    streamer.handle_source_command(SourceCommandPayload(command=SourceCommand.START))
    await asyncio.sleep(0.05)
    assert streamer._streaming.is_set()  # noqa: SLF001
    assert len(client.stream_starts) == 1

    streamer.handle_source_command(SourceCommandPayload(command=SourceCommand.STOP))
    await asyncio.sleep(0.05)
    assert not streamer._streaming.is_set()  # noqa: SLF001
    assert client.stream_ends == 1
