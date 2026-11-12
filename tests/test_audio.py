from __future__ import annotations

from types import SimpleNamespace

from sendspin.audio import AudioPlayer, PlaybackState, _QueuedChunk, find_dop_frames


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


def _dop_buffer(frames: int, *, bytes_per_sample: int = 3, channels: int = 2) -> bytearray:
    """Build packed little-endian DoP frames (marker in the top byte of each sample)."""
    out = bytearray()
    for frame in range(frames):
        marker = 0x05 if frame % 2 == 0 else 0xFA
        for _ in range(channels):
            out += bytes(bytes_per_sample - 3) + bytes([0x69, 0x96, marker])
    return out


def _volume_player(volume: int, *, bit_depth: int = 24) -> AudioPlayer:
    player = AudioPlayer(lambda ts: ts, lambda ts: ts)
    player._format = SimpleNamespace(  # noqa: SLF001
        sample_rate=176_400,
        channels=2,
        bit_depth=bit_depth,
        frame_size=2 * bit_depth // 8,
    )
    player._volume = volume  # noqa: SLF001
    player._muted = False  # noqa: SLF001
    return player


def _pcm_buffer(frames: int) -> bytearray:
    """Build packed 24-bit stereo PCM frames at a steady quarter-scale level."""
    return bytearray(b"\x00\x00\x20" * 2 * frames)


def test_find_dop_frames_detects_markers() -> None:
    mask, last = find_dop_frames(bytes(_dop_buffer(16)), 3, 2)
    assert mask.all()
    assert last == 0xFA
    assert find_dop_frames(bytes(_dop_buffer(16, bytes_per_sample=4)), 4, 2)[0].all()
    assert not find_dop_frames(bytes(_dop_buffer(16)), 2, 3)[0].any()
    assert not find_dop_frames(bytes(6 * 16), 3, 2)[0].any()


def test_find_dop_frames_rejects_short_runs_inside_pcm() -> None:
    buffer = _pcm_buffer(8) + _dop_buffer(4) + _pcm_buffer(8)
    mask, last = find_dop_frames(bytes(buffer), 3, 2)
    assert not mask.any()
    assert last is None


def test_find_dop_frames_keeps_a_short_tail_that_continues_dop() -> None:
    # Three DoP frames left at the end of a stream, followed by zero padding.
    buffer = _dop_buffer(3) + bytearray(6 * 61)
    mask, last = find_dop_frames(bytes(buffer), 3, 2, previous_marker=0xFA)
    assert mask[:3].all()
    assert not mask[3:].any()
    assert last is None
    # The same tail does not count as DoP if the previous buffer was not DoP.
    assert not find_dop_frames(bytes(buffer), 3, 2)[0].any()


def test_volume_leaves_a_padded_dop_tail_untouched() -> None:
    player = _volume_player(40)
    first = bytearray(_dop_buffer(64))
    player._apply_volume(memoryview(first))  # noqa: SLF001
    assert first == _dop_buffer(64)

    # The next callback gets two DoP frames, then zero padding after an underrun.
    tail = _dop_buffer(2) + bytearray(6 * 62)
    buffer = bytearray(tail)
    player._apply_volume(memoryview(buffer))  # noqa: SLF001
    assert buffer == tail


def test_volume_scales_only_the_pcm_around_dop() -> None:
    # Silence while waiting for the start, then DoP, then a transition back to PCM.
    dop = _dop_buffer(32)
    buffer = bytearray(6 * 8) + dop + _pcm_buffer(24)
    player = _volume_player(40)
    player._apply_volume(memoryview(buffer))  # noqa: SLF001
    assert buffer[: 6 * 8] == bytearray(6 * 8)
    assert buffer[6 * 8 : 6 * 40] == dop
    assert buffer[6 * 40 :] != _pcm_buffer(24)


def test_clear_forgets_the_dop_state() -> None:
    player = _volume_player(40)
    player._apply_volume(memoryview(bytearray(_dop_buffer(64))))  # noqa: SLF001
    assert player._dop_last_marker is not None  # noqa: SLF001
    player.clear()
    assert player._dop_last_marker is None  # noqa: SLF001


def test_volume_leaves_dop_untouched() -> None:
    dop = _dop_buffer(64)
    player = _volume_player(40)
    buffer = bytearray(dop)
    player._apply_volume(memoryview(buffer))  # noqa: SLF001
    assert buffer == dop


def test_volume_still_scales_plain_pcm() -> None:
    pcm = bytearray(b"\x00\x00\x40" * 2 * 64)
    player = _volume_player(40)
    buffer = bytearray(pcm)
    player._apply_volume(memoryview(buffer))  # noqa: SLF001
    assert buffer != pcm
