from __future__ import annotations

import asyncio
from types import SimpleNamespace

import sendspin.audio_connector as audio_connector
from sendspin.audio_connector import AudioStreamHandler
from sendspin.settings import ClientSettings


class _FakeWorker:
    instances: list[_FakeWorker] = []

    def __init__(
        self,
        *,
        audio_device: object,
        use_software_volume: bool,
        volume: int,
        muted: bool,
    ) -> None:
        self.audio_device = audio_device
        self.use_software_volume = use_software_volume
        self.volume = volume
        self.muted = muted
        self.running = False
        self.cleared = False
        self.stream_closed = False
        self.submitted: list[tuple[int, bytes | bytearray, object]] = []
        _FakeWorker.instances.append(self)

    def start(
        self,
        compute_play_time: object,
        compute_server_time: object,
        now_us: object | None = None,
        is_clock_synced: object | None = None,
    ) -> None:
        self.running = True
        self.compute_play_time = compute_play_time
        self.compute_server_time = compute_server_time
        self.now_us = now_us
        self.is_clock_synced = is_clock_synced

    def is_running(self) -> bool:
        return self.running

    def submit_chunk(
        self, server_timestamp_us: int, audio_data: bytes | bytearray, fmt: object
    ) -> None:
        self.submitted.append((server_timestamp_us, audio_data, fmt))

    def clear(self) -> None:
        self.cleared = True

    def close_stream(self) -> None:
        self.stream_closed = True

    def set_volume(self, volume: int, *, muted: bool) -> None:
        self.volume = volume
        self.muted = muted

    async def stop(self) -> None:
        self.running = False


class _FakeClient:
    def __init__(self) -> None:
        self.connected = True
        self.audio_chunk_listeners: list[object] = []
        self.stream_start_listeners: list[object] = []
        self.stream_end_listeners: list[object] = []
        self.stream_clear_listeners: list[object] = []

    def compute_play_time(self, timestamp_us: int) -> int:
        return timestamp_us

    def compute_server_time(self, timestamp_us: int) -> int:
        return timestamp_us

    def now_us(self) -> int:
        return 0

    def is_time_synchronized(self) -> bool:
        return True

    async def send_player_state(self, **_: object) -> None:
        return

    def add_audio_chunk_listener(self, callback: object):
        return self._add_listener(self.audio_chunk_listeners, callback)

    def add_stream_start_listener(self, callback: object):
        return self._add_listener(self.stream_start_listeners, callback)

    def add_stream_end_listener(self, callback: object):
        return self._add_listener(self.stream_end_listeners, callback)

    def add_stream_clear_listener(self, callback: object):
        return self._add_listener(self.stream_clear_listeners, callback)

    @staticmethod
    def _add_listener(callbacks: list[object], callback: object):
        callbacks.append(callback)

        def unsubscribe() -> None:
            callbacks.remove(callback)

        return unsubscribe


class _FakeHookController:
    def __init__(self, settings: ClientSettings) -> None:
        self.settings = settings
        self.calls: list[tuple[int, bool]] = []

    async def set_state(self, volume: int, *, muted: bool) -> None:
        self.calls.append((volume, muted))

    async def get_state(self) -> tuple[int, bool]:
        return self.settings.player_volume, self.settings.player_muted

    async def start_monitoring(self, callback: object) -> None:
        self.callback = callback

    async def stop_monitoring(self) -> None:
        return


def _make_format() -> SimpleNamespace:
    return SimpleNamespace(
        codec=SimpleNamespace(value="pcm"),
        pcm_format=SimpleNamespace(sample_rate=48_000, bit_depth=16, channels=2),
    )


def test_audio_worker_restarts_on_stream_start_after_disconnect(monkeypatch) -> None:
    monkeypatch.setattr(audio_connector, "_AudioSyncWorker", _FakeWorker)
    _FakeWorker.instances.clear()

    async def exercise() -> None:
        handler = AudioStreamHandler(
            audio_device=SimpleNamespace(index=0, name="Fake Device"),
            volume=10,
            muted=False,
        )
        client = _FakeClient()
        handler.attach_client(client)
        handler.set_volume(37, muted=True)
        await asyncio.sleep(0)

        await handler.handle_disconnect()
        assert len(_FakeWorker.instances) == 1
        assert not _FakeWorker.instances[0].running

        fmt = _make_format()
        # Simulate a player stream/start message (payload.player must be set)
        stream_start = SimpleNamespace(
            payload=SimpleNamespace(player=SimpleNamespace(), visualizer=None)
        )
        handler._on_stream_start(stream_start)

        assert len(_FakeWorker.instances) == 2
        restarted_worker = _FakeWorker.instances[1]
        assert restarted_worker.running
        assert restarted_worker.volume == 37
        assert restarted_worker.muted is True

        handler._on_audio_chunk(123_456, b"payload", fmt, 0)

        assert restarted_worker.submitted == [(123_456, b"payload", fmt)]

    asyncio.run(exercise())


def test_visualizer_stream_start_does_not_clear_audio_worker(monkeypatch) -> None:
    """A visualizer-only stream/start must not touch the audio worker."""
    monkeypatch.setattr(audio_connector, "_AudioSyncWorker", _FakeWorker)
    _FakeWorker.instances.clear()

    async def exercise() -> None:
        handler = AudioStreamHandler(
            audio_device=SimpleNamespace(index=0, name="Fake Device"),
            volume=10,
            muted=False,
        )
        client = _FakeClient()
        handler.attach_client(client)
        await asyncio.sleep(0)

        assert len(_FakeWorker.instances) == 1
        worker = _FakeWorker.instances[0]
        assert worker.running

        # Send a visualizer-only stream/start (no player payload)
        vis_stream_start = SimpleNamespace(
            payload=SimpleNamespace(player=None, visualizer=SimpleNamespace())
        )
        handler._on_stream_start(vis_stream_start)

        # Worker should be untouched — still the same one, still running
        assert len(_FakeWorker.instances) == 1
        assert worker.running

    asyncio.run(exercise())


def test_attach_client_replaces_previous_client_listeners(monkeypatch) -> None:
    monkeypatch.setattr(audio_connector, "_AudioSyncWorker", _FakeWorker)
    _FakeWorker.instances.clear()

    handler = AudioStreamHandler(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        volume=10,
        muted=False,
    )
    first_client = _FakeClient()
    second_client = _FakeClient()

    handler.attach_client(first_client)
    assert len(first_client.audio_chunk_listeners) == 1
    assert len(first_client.stream_start_listeners) == 1
    assert len(first_client.stream_end_listeners) == 1
    assert len(first_client.stream_clear_listeners) == 1

    handler.attach_client(second_client)

    assert first_client.audio_chunk_listeners == []
    assert first_client.stream_start_listeners == []
    assert first_client.stream_end_listeners == []
    assert first_client.stream_clear_listeners == []
    assert len(second_client.audio_chunk_listeners) == 1
    assert len(second_client.stream_start_listeners) == 1
    assert len(second_client.stream_end_listeners) == 1
    assert len(second_client.stream_clear_listeners) == 1


def test_external_volume_controller_updates_logical_volume(tmp_path) -> None:
    async def exercise() -> None:
        settings = ClientSettings(
            _settings_file=tmp_path / "settings.json",
            player_volume=22,
            player_muted=True,
        )
        changes: list[tuple[int, bool]] = []
        controller = _FakeHookController(settings)
        handler = AudioStreamHandler(
            audio_device=SimpleNamespace(index=0, name="Fake Device"),
            volume=10,
            muted=False,
            on_volume_change=lambda volume, muted: changes.append((volume, muted)),
            volume_controller=controller,
        )

        await handler.read_initial_volume()
        assert handler.volume == 22
        assert handler.muted is True
        assert handler.uses_external_volume_controller is True

        handler.set_volume(41, muted=False)
        await asyncio.sleep(0)

        assert controller.calls == [(41, False)]
        assert handler.volume == 41
        assert handler.muted is False
        assert changes == [(41, False)]

    asyncio.run(exercise())


def test_stream_end_closes_stream_not_just_clears(monkeypatch) -> None:
    """stream_end must fully close the stream (release the device), not just clear."""
    monkeypatch.setattr(audio_connector, "_AudioSyncWorker", _FakeWorker)
    _FakeWorker.instances.clear()

    handler = AudioStreamHandler(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        volume=10,
        muted=False,
    )
    client = _FakeClient()
    handler.attach_client(client)

    worker = _FakeWorker.instances[0]
    assert not worker.stream_closed

    handler._on_stream_end(None)

    assert worker.stream_closed, "_on_stream_end must call close_stream(), not just clear()"
    assert not worker.cleared, "_on_stream_end must not call clear() separately"
