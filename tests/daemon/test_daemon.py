from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiosendspin.models.types import PlayerCommand
from aiosendspin.noise import Identity, InMemoryClientPairingStore

from sendspin.daemon.daemon import DaemonArgs, SendspinDaemon
from sendspin.settings import ClientSettings


class _FakeAudioHandler:
    def __init__(self, *, volume: int, muted: bool) -> None:
        self.volume = volume
        self.muted = muted
        self.calls: list[tuple[int, bool]] = []
        self.delay_changes: list[int] = []
        self.disconnects = 0

    def set_volume(self, volume: int, *, muted: bool) -> None:
        self.calls.append((volume, muted))
        self.volume = volume
        self.muted = muted

    def notify_delay_change(self, delta_us: int) -> None:
        self.delay_changes.append(delta_us)

    async def handle_disconnect(self) -> None:
        self.disconnects += 1


def _make_daemon(tmp_path: Path, *, settings_volume: int, settings_muted: bool) -> SendspinDaemon:
    settings = ClientSettings(
        _settings_file=tmp_path / "settings.json",
        player_volume=settings_volume,
        player_muted=settings_muted,
    )
    args = DaemonArgs(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test Client",
        settings=settings,
        use_mpris=False,
    )
    return SendspinDaemon(args)


def test_volume_command_uses_audio_handler_muted_state_for_external_volume(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=True)
    daemon._audio_handler = _FakeAudioHandler(volume=41, muted=False)

    payload = SimpleNamespace(
        player=SimpleNamespace(command=PlayerCommand.VOLUME, volume=67, mute=None)
    )

    daemon._handle_server_command(payload)

    assert daemon._audio_handler.calls == [(67, False)]


def test_mute_command_uses_audio_handler_volume_state_for_external_volume(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path, settings_volume=12, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=53, muted=False)

    payload = SimpleNamespace(
        player=SimpleNamespace(command=PlayerCommand.MUTE, volume=None, mute=True)
    )

    daemon._handle_server_command(payload)

    assert daemon._audio_handler.calls == [(53, True)]


def test_set_output_delay_uses_applied_tracker_for_delta(tmp_path: Path) -> None:
    """Sync delta is computed from `_output_delay_ms`, not stale settings.

    Reproduces the CLI-override case: settings stays at 0 while the client was
    initialized to 500 from `--output-delay-ms`. A server-initiated delay change
    to 200 must produce delta = -300ms (200 - 500), not -200ms (200 - 0).
    """
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=25, muted=False)
    daemon._output_delay_ms = 500.0
    # aiosendspin auto-applies before the callback fires, so the client already
    # reports the new value.
    daemon._client = SimpleNamespace(output_delay_ms=200.0)  # type: ignore[assignment]

    payload = SimpleNamespace(
        player=SimpleNamespace(
            command=PlayerCommand.SET_OUTPUT_DELAY,
            volume=None,
            mute=None,
            output_delay_ms=200,
        )
    )

    # `settings.update` schedules a debounced save via asyncio; wrap in a loop.
    async def run() -> None:
        daemon._handle_server_command(payload)

    asyncio.run(run())

    assert daemon._audio_handler.delay_changes == [-300_000]
    assert daemon._output_delay_ms == 200.0
    assert daemon._settings.output_delay_ms == 200.0


class _FakeAdmissionClient:
    """Stands in for SendspinClient's admission: a new server_info object means admitted."""

    def __init__(self, *, current: SimpleNamespace | None, admit: bool) -> None:
        self.server_info = current
        self.activities: list[SimpleNamespace] = []
        self._admit = admit
        self.close = asyncio.Event()

    async def attach_websocket(self, ws: object) -> None:
        if not self._admit:
            return  # rejected under the admission rules, or bring-up failed
        self.server_info = SimpleNamespace(name="New Server")
        self.activities = [SimpleNamespace(value="pairing")]
        await self.close.wait()


async def _connect(daemon: SendspinDaemon, client: _FakeAdmissionClient) -> None:
    """Run one incoming connection until it is admitted (then close it) or turned away."""
    task = asyncio.create_task(daemon._handle_server_connection(None))  # type: ignore[arg-type]
    for _ in range(100):
        if (
            task.done()
            or client.server_info is not None
            and client.server_info.name == "New Server"
        ):
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    client.close.set()
    await task


def test_displacing_connection_drops_old_server_audio(tmp_path: Path) -> None:
    """aiosendspin fires no event on displacement, so the daemon resets audio itself."""
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=25, muted=False)  # type: ignore[assignment]
    client = _FakeAdmissionClient(current=SimpleNamespace(name="Old Server"), admit=True)
    daemon._client = client  # type: ignore[assignment]

    asyncio.run(_connect(daemon, client))

    assert daemon._audio_handler.disconnects == 1


def test_first_admitted_connection_keeps_audio(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=25, muted=False)  # type: ignore[assignment]
    client = _FakeAdmissionClient(current=None, admit=True)
    daemon._client = client  # type: ignore[assignment]

    asyncio.run(_connect(daemon, client))

    assert daemon._audio_handler.disconnects == 0


def test_rejected_connection_logs_once_at_info_and_keeps_audio(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=25, muted=False)  # type: ignore[assignment]
    client = _FakeAdmissionClient(current=SimpleNamespace(name="Old Server"), admit=False)
    daemon._client = client  # type: ignore[assignment]

    with caplog.at_level(logging.DEBUG, logger="sendspin.daemon.daemon"):
        asyncio.run(_connect(daemon, client))
        asyncio.run(_connect(daemon, client))

    assert daemon._audio_handler.disconnects == 0
    levels = [r.levelno for r in caplog.records if "was not admitted" in r.getMessage()]
    assert levels == [logging.INFO, logging.DEBUG]


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(None, "legacy-server"), ("current-server", "current-server")],
)
def test_last_playback_server_is_seeded_from_settings(
    tmp_path: Path, stored: str | None, expected: str
) -> None:
    """The settings value moves into the pairing store, without overwriting a newer one."""
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=False)
    daemon._settings.last_played_server_id = "legacy-server"
    store = daemon._args.pairing_store

    async def run() -> str | None:
        if stored is not None:
            await store.set_last_playback_server_id(stored)
        await daemon._seed_last_playback_server()
        return await store.get_last_playback_server_id()

    assert asyncio.run(run()) == expected
