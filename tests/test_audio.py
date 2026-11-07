from __future__ import annotations

from types import SimpleNamespace

from sendspin.audio import AudioPlayer, PlaybackState, _QueuedChunk, is_dop_buffer


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


def test_is_dop_buffer_detects_markers() -> None:
    assert is_dop_buffer(bytes(_dop_buffer(16)), 3, 2)
    assert is_dop_buffer(bytes(_dop_buffer(16, bytes_per_sample=4)), 4, 2)
    assert not is_dop_buffer(bytes(_dop_buffer(16)), 2, 3)
    broken = _dop_buffer(16)
    broken[3 * 2 * 3 + 2] = 0x05  # fourth frame repeats the marker of the third
    assert not is_dop_buffer(bytes(broken), 3, 2)
    assert not is_dop_buffer(bytes(6 * 16), 3, 2)


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
