"""Static delay regressions using the player's declared aiosendspin dependency."""

from __future__ import annotations

# ruff: noqa: SLF001

from types import SimpleNamespace

import pytest
from aiosendspin.client import PCMFormat, SendspinClient
from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.time_sync import TimeElement
from aiosendspin.models.core import ServerCommandPayload
from aiosendspin.models.player import PlayerCommandPayload
from aiosendspin.models.types import PlayerCommand
from aiosendspin.noise import Identity, InMemoryClientPairingStore

from sendspin.audio import AudioPlayer, PlaybackState, _QueuedChunk

NOW_US = 10_000_000


def _connection(
    delay_ms: float, offset_us: int, drift: float
) -> tuple[SendspinClient, SendspinConnection]:
    client = SendspinClient(
        identity=Identity.generate(),
        client_name="Timing test",
        roles=[],
        pairing_store=InMemoryClientPairingStore(),
        static_delay_ms=delay_ms,
    )
    connection = SendspinConnection(client)
    client._admitted_connection = connection
    for timestamp_us in (NOW_US - 2, NOW_US - 1):
        connection._time_filter.update(offset_us, 1, timestamp_us)
    connection._time_filter._current_time_element = TimeElement(
        last_update=NOW_US,
        offset=float(offset_us),
        drift=drift,
        use_drift=drift != 0,
    )
    return client, connection


def _player(connection: SendspinConnection) -> AudioPlayer:
    player = AudioPlayer(
        connection.compute_play_time,
        connection.compute_server_time,
        now_us=lambda: NOW_US,
    )
    player._format = PCMFormat(sample_rate=48_000, channels=1, bit_depth=16)
    return player


@pytest.mark.parametrize(
    ("initial_delay_ms", "new_delay_ms"), [(225, 250), (250, 225), (0, 250), (250, 0)]
)
@pytest.mark.parametrize(
    ("offset_us", "drift"), [(0, 0.0), (150_000, 0.0), (150_000, 0.0001), (-150_000, -0.0001)]
)
@pytest.mark.parametrize("change_from_server", [False, True])
async def test_live_delay_is_applied_once(
    monkeypatch: pytest.MonkeyPatch,
    initial_delay_ms: float,
    new_delay_ms: float,
    offset_us: int,
    drift: float,
    *,
    change_from_server: bool,
) -> None:
    """The SDK changes the timing target while source frames retain their timestamps."""
    client, connection = _connection(initial_delay_ms, offset_us, drift)
    player = _player(connection)
    cursor_us = connection.compute_server_time(NOW_US)
    player._server_ts_cursor_us = cursor_us
    player._server_ts_cursor_remainder = 17
    player._playback_state = PlaybackState.PLAYING
    player._scheduled_start_loop_time_us = NOW_US
    current = _QueuedChunk(cursor_us, b"\x01\x00\x02\x00")
    queued = _QueuedChunk(cursor_us + 42, b"\x03\x00\x04\x00")
    player._current_chunk = current
    player._current_chunk_offset = 2
    player._queue.put(queued)
    player._queued_duration_us = 42
    errors: list[int] = []
    monkeypatch.setattr(player, "_update_correction_schedule", errors.append)

    if change_from_server:
        connection._handle_server_command(
            ServerCommandPayload(
                player=PlayerCommandPayload(
                    command=PlayerCommand.SET_STATIC_DELAY,
                    static_delay_ms=int(new_delay_ms),
                )
            )
        )
    else:
        client.set_static_delay_ms(new_delay_ms)
    delta_us = round((new_delay_ms - initial_delay_ms) * 1_000)

    # Queued, duplicate and stale notifications all leave source position intact.
    for notification_us in (delta_us, delta_us, -delta_us):
        player.apply_delay_change(notification_us)
        assert player._server_ts_cursor_us == cursor_us
        assert player._server_ts_cursor_remainder == 17
        assert player._current_chunk is current
        assert player._current_chunk_offset == 2
        assert player._queued_duration_us == 42
    assert player._queue.get_nowait() is queued
    assert player._playback_state is PlaybackState.PLAYING

    player._update_playback_position_from_dac(SimpleNamespace(outputBufferDacTime=10.0))
    player.submit(cursor_us + 84, b"\x05\x00\x06\x00")
    expected_error_us = connection.compute_server_time(NOW_US) - cursor_us
    assert errors == [expected_error_us]
    assert expected_error_us == pytest.approx(delta_us * (1 + drift), abs=1)

    # A fresh connection at the saved delay uses the same playback mapping.
    _, reconnected = _connection(new_delay_ms, offset_us, drift)
    assert connection.compute_play_time(cursor_us) == reconnected.compute_play_time(cursor_us)


@pytest.mark.parametrize("new_delay_ms", [0, 250])
async def test_delay_change_before_start_uses_current_sdk_mapping(new_delay_ms: float) -> None:
    """Initial buffering schedules the first chunk with the current static delay."""
    client, connection = _connection(225, 150_000, 0.0001)
    player = _player(connection)
    first_timestamp_us = connection.compute_server_time(NOW_US) + 1_000_000
    client.set_static_delay_ms(new_delay_ms)
    player.apply_delay_change(round((new_delay_ms - 225) * 1_000))
    player.submit(first_timestamp_us, b"\x01\x00\x02\x00")
    assert player._server_ts_cursor_us == 0
    assert player._scheduled_start_loop_time_us == connection.compute_play_time(first_timestamp_us)
    assert player._playback_state is PlaybackState.WAITING_FOR_START


@pytest.mark.parametrize("new_delay_ms", [0, 250])
async def test_delay_change_while_waiting_reschedules_first_chunk(new_delay_ms: float) -> None:
    """Buffered startup follows the new mapping on the next submitted chunk."""
    client, connection = _connection(225, 150_000, 0.0001)
    player = _player(connection)
    first_timestamp_us = connection.compute_server_time(NOW_US) + 1_000_000
    player.submit(first_timestamp_us, b"\x01\x00\x02\x00")
    initial_start_us = player._scheduled_start_loop_time_us

    client.set_static_delay_ms(new_delay_ms)
    player.apply_delay_change(round((new_delay_ms - 225) * 1_000))
    player.submit(first_timestamp_us + 41, b"\x03\x00\x04\x00")

    assert player._server_ts_cursor_us == 0
    assert player._scheduled_start_loop_time_us != initial_start_us
    assert player._scheduled_start_loop_time_us == connection.compute_play_time(first_timestamp_us)
    assert player._playback_state is PlaybackState.WAITING_FOR_START
