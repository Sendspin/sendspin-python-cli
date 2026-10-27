"""Tests for ServeWorker."""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from aiosendspin.clock import ManualClock

from sendspin.serve.ipc import AudioChunk, Shutdown, WorkerListening
from sendspin.serve.worker import ServeWorker


@pytest.fixture
def mp_context() -> mp.context.SpawnContext:
    return mp.get_context("spawn")


@pytest.fixture
def audio_queue(mp_context: mp.context.SpawnContext) -> mp.Queue:
    return mp_context.Queue()


@pytest.fixture
def status_queue(mp_context: mp.context.SpawnContext) -> mp.Queue:
    return mp_context.Queue()


@pytest.fixture
def total_listeners(mp_context: mp.context.SpawnContext) -> mp.Value:
    return mp_context.Value("i", 0)


def test_worker_init(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )
    assert worker.worker_id == 0
    assert worker.port == 8928


@pytest.mark.asyncio
async def test_worker_signals_listening_on_start(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    """Worker should put WorkerListening on the status queue after starting."""
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )

    # Put a Shutdown immediately so the worker exits after starting
    audio_queue.put(Shutdown())

    with patch.object(worker, "_start_server", new_callable=AsyncMock):
        await worker.run()

    msg = status_queue.get(timeout=1)
    assert isinstance(msg, WorkerListening)
    assert msg.worker_id == 0
    assert msg.port == 8928


@pytest.mark.asyncio
async def test_worker_processes_audio_chunk(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    """Worker should call prepare_audio/commit_audio for each AudioChunk."""
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )

    chunk = AudioChunk(
        pcm_bytes=b"\x00" * 100,
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        play_start_us=1_000_000,
    )
    audio_queue.put(chunk)
    audio_queue.put(Shutdown())

    mock_stream = MagicMock()
    mock_stream.prepare_audio = MagicMock()
    mock_stream.commit_audio = AsyncMock(return_value=1_000_000)

    with (
        patch.object(worker, "_start_server", new_callable=AsyncMock),
        patch.object(worker, "_get_stream", return_value=mock_stream),
    ):
        await worker.run()

    mock_stream.prepare_audio.assert_called_once()
    mock_stream.commit_audio.assert_called_once_with(play_start_us=1_000_000)


def _chunk(play_start_us: int) -> AudioChunk:
    # 4800 frames of 48 kHz stereo s16 is 100 ms.
    return AudioChunk(
        pcm_bytes=b"\x00" * 4800 * 4,
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        play_start_us=play_start_us,
    )


async def _run_until_stream_appears(
    worker: ServeWorker, audio_queue: mp.Queue, chunks: list[AudioChunk]
) -> MagicMock:
    """Feed chunks to a worker whose stream exists only from the last chunk on."""
    for chunk in chunks:
        audio_queue.put(chunk)
    audio_queue.put(Shutdown())

    mock_stream = MagicMock()
    mock_stream.commit_audio = AsyncMock()
    streams = [None] * (len(chunks) - 1) + [mock_stream]

    with (
        patch.object(worker, "_start_server", new_callable=AsyncMock),
        patch.object(worker, "_get_stream", side_effect=streams),
    ):
        await worker.run()
    return mock_stream


@pytest.mark.asyncio
async def test_worker_replays_recent_chunks_to_new_stream(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    """A first listener gets the unplayed chunks received before its stream."""
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )
    start_us = 1_000_000
    worker._clock = ManualClock(start_us)
    # The first chunk has finished playing.
    chunks = [_chunk(start_us + i * 100_000) for i in range(-1, 3)]

    stream = await _run_until_stream_appears(worker, audio_queue, chunks)

    assert stream.mock_calls == [
        call.prepare_historical_audio(chunks[1].pcm_bytes, ANY, start_time_us=start_us),
        call.prepare_historical_audio(chunks[2].pcm_bytes, ANY, start_time_us=start_us),
        call.prepare_audio(chunks[3].pcm_bytes, ANY),
        call.commit_audio(play_start_us=chunks[3].play_start_us),
    ]


@pytest.mark.asyncio
async def test_worker_skips_chunks_from_previous_timeline(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    """Chunks the new timeline doesn't continue from are not replayed behind it."""
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )
    start_us = 1_000_000
    worker._clock = ManualClock(start_us)
    chunks = [_chunk(start_us + 3_000_000), _chunk(start_us), _chunk(start_us + 100_000)]

    stream = await _run_until_stream_appears(worker, audio_queue, chunks)

    stream.prepare_historical_audio.assert_called_once_with(
        chunks[1].pcm_bytes, ANY, start_time_us=start_us
    )


def test_worker_clears_group_when_last_client_disconnects(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    """When the last client disconnects, active_group and stream should be cleared."""
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )

    # Set up a mock group with one client that will be "removed"
    mock_stream = MagicMock()
    mock_stream.stop = MagicMock()
    mock_group = MagicMock()
    mock_group.clients = []  # Empty after removal

    worker._active_group = mock_group
    worker._stream = mock_stream

    # Create a mock server with no connected clients
    mock_server = MagicMock()
    mock_server.connected_clients = {}
    worker._server = mock_server

    # Simulate ClientRemovedEvent
    from aiosendspin.server import ClientRemovedEvent

    event = ClientRemovedEvent(client_id="test-client")
    worker._on_server_event(mock_server, event)

    # Group and stream should be cleared
    assert worker._active_group is None
    assert worker._stream is None
    mock_stream.stop.assert_called_once()


@pytest.mark.asyncio
async def test_worker_passes_allow_unencrypted_to_server(
    audio_queue: mp.Queue,
    status_queue: mp.Queue,
    total_listeners: mp.Value,
) -> None:
    worker = ServeWorker(
        worker_id=0,
        port=8928,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
        allow_unencrypted=True,
    )

    with patch("sendspin.serve.worker.SendspinPlayerServer.start_server", new_callable=AsyncMock):
        await worker._start_server()

    assert worker._server is not None
    assert worker._server.allow_unencrypted
    await worker._shutdown_server()
