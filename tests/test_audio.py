from __future__ import annotations

from types import SimpleNamespace

from sendspin.audio import AudioPlayer, PlaybackState, _QueuedChunk


class _NoCallbackStatus:
    input_underflow = False
    output_underflow = False

    def __bool__(self) -> bool:
        return False


def test_drop_correction_discards_one_frame_without_repeating_previous() -> None:
    now_us = 0

    def now() -> int:
        return now_us

    player = AudioPlayer(lambda ts: ts, lambda ts: ts, now_us=now)
    player._format = SimpleNamespace(  # noqa: SLF001
        sample_rate=48_000,
        channels=1,
        bit_depth=16,
        frame_size=2,
    )
    player._playback_state = PlaybackState.PLAYING  # noqa: SLF001
    player._drop_every_n_frames = 1  # noqa: SLF001
    player._frames_until_next_drop = 1  # noqa: SLF001
    player._queue.put(  # noqa: SLF001
        _QueuedChunk(
            server_timestamp_us=0,
            audio_data=b"\x01\x00\x02\x00\x03\x00",
        )
    )

    out = bytearray(4)
    player._audio_callback(  # noqa: SLF001
        memoryview(out),
        frames=2,
        time=SimpleNamespace(outputBufferDacTime=0.0),
        status=_NoCallbackStatus(),
    )

    assert bytes(out) == b"\x01\x00\x03\x00"


def test_sync_correction_waits_for_startup_grace_period() -> None:
    now_us = 1_000_000

    def now() -> int:
        return now_us

    player = AudioPlayer(lambda ts: ts, lambda ts: ts, now_us=now)
    player._format = SimpleNamespace(sample_rate=48_000)  # noqa: SLF001
    player._playback_started_loop_time_us = now_us  # noqa: SLF001

    player._update_correction_schedule(50_000)  # noqa: SLF001

    assert player._drop_every_n_frames == 0  # noqa: SLF001
    assert player._insert_every_n_frames == 0  # noqa: SLF001


class _RecordingStream:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def test_clear_keeps_stream_open_so_device_never_closes() -> None:
    """clear() must drain the queue without stopping the stream.

    The callback renders silence while starved, so keeping the stream
    running holds the audio device open and DACs with lock-loss muting
    relays (e.g. XMOS USB) stay silent across pause/resume/seek.
    """
    player = AudioPlayer(lambda ts: ts, lambda ts: ts, now_us=lambda: 0)
    stream = _RecordingStream()
    player._stream = stream  # noqa: SLF001
    player._stream_started = True  # noqa: SLF001
    player._queue.put(  # noqa: SLF001
        _QueuedChunk(server_timestamp_us=0, audio_data=b"\x01\x00\x02\x00")
    )

    player.clear()

    assert player._queue.empty()  # noqa: SLF001
    assert player._stream_started  # noqa: SLF001
    assert stream.stops == 0


class _RecordingStreamFull:
    def __init__(self) -> None:
        self.stops = 0
        self.starts = 0
        self.closes = 0

    def stop(self) -> None:
        self.stops += 1

    def start(self) -> None:
        self.starts += 1

    def close(self) -> None:
        self.closes += 1


def _make_player() -> AudioPlayer:
    return AudioPlayer(lambda ts: ts, lambda ts: ts, now_us=lambda: 0)


def test_close_stream_keeps_device_open_on_silence() -> None:
    """End-of-stream must not release the audio device.

    The server sends stream/end on pause, so fully closing here is what
    drops DAC lock and clicks the relay. Silence keeps it held.
    """
    player = _make_player()
    stream = _RecordingStreamFull()
    player._stream = stream  # noqa: SLF001
    player._stream_started = True  # noqa: SLF001
    player._queue.put(  # noqa: SLF001
        _QueuedChunk(server_timestamp_us=0, audio_data=b"\x01\x00\x02\x00")
    )

    player.close_stream()

    assert player._queue.empty()  # noqa: SLF001
    assert player._stream_started  # noqa: SLF001
    assert player._stream is stream  # noqa: SLF001
    assert (stream.stops, stream.closes) == (0, 0)


def test_set_format_same_format_keeps_existing_stream() -> None:
    """Resume after pause re-sends the identical format: no recreate."""
    from aiosendspin.client import PCMFormat

    player = _make_player()
    stream = _RecordingStreamFull()
    fmt = PCMFormat(sample_rate=48000, channels=2, bit_depth=16)
    player._format = fmt  # noqa: SLF001
    player._stream = stream  # noqa: SLF001

    player.set_format(SimpleNamespace(pcm_format=fmt), device=None)

    assert player._stream is stream  # noqa: SLF001
    assert (stream.stops, stream.closes) == (0, 0)


def test_set_format_change_stops_old_stream_synchronously() -> None:
    """A format switch must stop the old stream before swapping formats.

    Otherwise an in-flight old-stream callback computes against the new
    format and crashes (ValueError: memoryview assignment, seen live on
    resume-after-announce). stop() is safe here: set_format runs on the
    worker thread, never inside a callback.
    """
    import concurrent.futures
    from unittest.mock import patch

    from aiosendspin.client import PCMFormat

    player = _make_player()
    old = _RecordingStreamFull()
    player._stream = old  # noqa: SLF001
    player._format = PCMFormat(sample_rate=48000, channels=2, bit_depth=16)  # noqa: SLF001
    player._stream_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # noqa: SLF001
    new = _RecordingStreamFull()
    new.latency = 0.05
    try:
        with patch("sendspin.audio.sounddevice.RawOutputStream", return_value=new):
            player.set_format(
                SimpleNamespace(
                    pcm_format=PCMFormat(sample_rate=44100, channels=2, bit_depth=24),
                    codec=SimpleNamespace(value="flac"),
                ),
                device=SimpleNamespace(device_id=6),
            )
    finally:
        # Drain the async close: deterministic counts below.
        player._stream_executor.shutdown(wait=True)  # noqa: SLF001
    # One synchronous stop (before the swap) + one async stop/close pair.
    assert old.stops == 2
    assert old.closes == 1
    assert player._format.sample_rate == 44100  # noqa: SLF001
    assert player._stream is new  # noqa: SLF001
    assert player._last_output_frame == b""  # noqa: SLF001


def test_stale_callback_capped_to_buffer_instead_of_crash() -> None:
    """Replay the live crash: old-stream buffer, new-format frame size.

    A 2048-frame callback on a buffer sized for 2-byte frames while the
    format claims 6-byte frames must degrade to silence, never raise.
    """
    now_us = 0

    def now() -> int:
        return now_us

    player = AudioPlayer(lambda ts: ts, lambda ts: ts, now_us=now)
    player._format = SimpleNamespace(  # noqa: SLF001
        sample_rate=48000,
        channels=2,
        bit_depth=24,
        frame_size=6,
    )
    player._playback_state = PlaybackState.PLAYING  # noqa: SLF001
    player._queue.put(  # noqa: SLF001
        _QueuedChunk(server_timestamp_us=0, audio_data=b"\x01" * 600)
    )

    out = bytearray(2048 * 2)  # old stream buffer: 2-byte frames
    player._audio_callback(  # noqa: SLF001
        memoryview(out),
        frames=2048,
        time=SimpleNamespace(outputBufferDacTime=0.0),
        status=_NoCallbackStatus(),
    )

    assert bytes(out[-2:]) == b"\x00\x00"
