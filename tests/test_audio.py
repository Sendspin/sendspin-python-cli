from __future__ import annotations

from types import SimpleNamespace

from sendspin.audio import AudioPlayer, ClockSource, PlaybackState, _QueuedChunk


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


def _player_past_grace_period(clock_source: ClockSource) -> AudioPlayer:
    now_us = 10_000_000
    player = AudioPlayer(
        lambda ts: ts, lambda ts: ts, now_us=lambda: now_us, clock_source=clock_source
    )
    player._format = SimpleNamespace(sample_rate=48_000)  # noqa: SLF001
    player._playback_state = PlaybackState.PLAYING  # noqa: SLF001
    player._playback_started_loop_time_us = 1_000_000  # noqa: SLF001
    player._force_reanchor = False  # noqa: SLF001
    return player


def test_server_clock_schedules_corrections_for_sustained_error() -> None:
    player = _player_past_grace_period(ClockSource.SERVER)

    for _ in range(20):
        player._update_correction_schedule(50_000)  # noqa: SLF001

    assert player._drop_every_n_frames > 0  # noqa: SLF001


def test_dac_clock_never_modifies_sample_stream() -> None:
    player = _player_past_grace_period(ClockSource.DAC)

    for error_us in (50_000, -50_000, 2_000_000, -2_000_000):
        for _ in range(20):
            player._update_correction_schedule(error_us)  # noqa: SLF001
        assert player._drop_every_n_frames == 0  # noqa: SLF001
        assert player._insert_every_n_frames == 0  # noqa: SLF001
        assert player._force_reanchor is False  # noqa: SLF001


def _player_with_queued_audio(clock_source: ClockSource, queued_us: int) -> AudioPlayer:
    player = AudioPlayer(lambda ts: ts, lambda ts: ts, clock_source=clock_source)
    player._stream_started = True  # noqa: SLF001
    player._queue.put(_QueuedChunk(server_timestamp_us=0, audio_data=b"\x00\x00"))  # noqa: SLF001
    player._queued_duration_us = queued_us  # noqa: SLF001
    return player


def test_server_clock_discards_remainder_on_stream_end() -> None:
    player = _player_with_queued_audio(ClockSource.SERVER, 20_000)

    assert player.end_of_stream_playout_us() == 0


def test_dac_clock_plays_out_drift_remainder_on_stream_end() -> None:
    player = _player_with_queued_audio(ClockSource.DAC, 20_000)

    assert player.end_of_stream_playout_us() == 20_000


def test_dac_clock_discards_large_remainder_on_stream_end() -> None:
    player = _player_with_queued_audio(ClockSource.DAC, 30_000_000)

    assert player.end_of_stream_playout_us() == 0


def _late_start_player(clock_source: ClockSource, late_us: int) -> AudioPlayer:
    player = AudioPlayer(lambda ts: ts, lambda ts: ts, clock_source=clock_source)
    player._format = SimpleNamespace(  # noqa: SLF001
        sample_rate=48_000, channels=1, bit_depth=16, frame_size=2
    )
    player._playback_state = PlaybackState.WAITING_FOR_START  # noqa: SLF001
    player._scheduled_start_dac_time_us = 10_000_000  # noqa: SLF001
    player._queue.put(  # noqa: SLF001
        _QueuedChunk(server_timestamp_us=0, audio_data=b"\x01\x00" * 48_000)
    )
    player._handle_start_gating(  # noqa: SLF001
        memoryview(bytearray(8)),
        0,
        4,
        SimpleNamespace(outputBufferDacTime=(10_000_000 + late_us) / 1_000_000),
    )
    return player


def test_dac_clock_keeps_first_samples_on_slightly_late_start() -> None:
    player = _late_start_player(ClockSource.DAC, 2_500)

    assert player._playback_state == PlaybackState.PLAYING  # noqa: SLF001
    assert player._queue.qsize() == 1  # noqa: SLF001
    assert player._current_chunk is None  # noqa: SLF001


def test_server_clock_skips_frames_on_late_start() -> None:
    player = _late_start_player(ClockSource.SERVER, 2_500)

    assert player._playback_state == PlaybackState.PLAYING  # noqa: SLF001
    assert player._current_chunk is not None or player._queue.qsize() == 0  # noqa: SLF001
    assert player._current_chunk_offset > 0  # noqa: SLF001
