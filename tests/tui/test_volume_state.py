from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from aiosendspin.models.types import PlayerCommand
from aiosendspin.noise import Identity, InMemoryClientPairingStore

from sendspin.settings import ClientSettings
from sendspin.tui.app import AppArgs, AppState, SendspinApp
from sendspin.tui.keyboard import CommandHandler


class _FakeAudioHandler:
    def __init__(self, *, volume: int, muted: bool) -> None:
        self.volume = volume
        self.muted = muted
        self.calls: list[tuple[int, bool]] = []
        self.delay_changes: list[int] = []
        self.send_player_volume_calls = 0

    def set_volume(self, volume: int, *, muted: bool) -> None:
        self.calls.append((volume, muted))
        self.volume = volume
        self.muted = muted

    def notify_delay_change(self, delta_us: int) -> None:
        self.delay_changes.append(delta_us)

    def send_player_volume(self) -> None:
        self.send_player_volume_calls += 1


class _FakeUI:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.delays: list[float] = []

    def add_event(self, event: str) -> None:
        self.events.append(event)

    def set_delay(self, delay_ms: float) -> None:
        self.delays.append(delay_ms)


def _make_settings(tmp_path: Path) -> ClientSettings:
    return ClientSettings(_settings_file=tmp_path / "settings.json")


def _make_app(tmp_path: Path) -> SendspinApp:
    args = AppArgs(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test Client",
        settings=_make_settings(tmp_path),
        use_mpris=False,
    )
    return SendspinApp(args)


def test_tui_volume_command_uses_audio_handler_muted_state(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app._audio_handler = _FakeAudioHandler(volume=41, muted=False)
    app._ui = _FakeUI()
    app._state.player_muted = True

    payload = SimpleNamespace(
        player=SimpleNamespace(command=PlayerCommand.VOLUME, volume=67, mute=None)
    )

    app._handle_server_command(payload)

    assert app._audio_handler.calls == [(67, False)]


def test_tui_mute_command_uses_audio_handler_volume_state(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app._audio_handler = _FakeAudioHandler(volume=53, muted=False)
    app._ui = _FakeUI()
    app._state.player_volume = 12

    payload = SimpleNamespace(
        player=SimpleNamespace(command=PlayerCommand.MUTE, volume=None, mute=True)
    )

    app._handle_server_command(payload)

    assert app._audio_handler.calls == [(53, True)]


def test_tui_set_static_delay_uses_applied_tracker_for_delta(tmp_path: Path) -> None:
    """Sync delta is computed from `_applied_delay_ms`, not stale settings.

    Reproduces the CLI-override case: settings stays at 0 while the client was
    initialized to 500 from `--static-delay-ms`. A server-initiated delay change
    to 200 must produce delta = -300ms (200 - 500), not -200ms (200 - 0).
    """
    app = _make_app(tmp_path)
    app._audio_handler = _FakeAudioHandler(volume=25, muted=False)
    app._ui = _FakeUI()
    app._applied_delay_ms = 500.0
    # aiosendspin auto-applies before the callback fires, so the client already
    # reports the new value.
    app._client = SimpleNamespace(static_delay_ms=200.0)

    payload = SimpleNamespace(
        player=SimpleNamespace(
            command=PlayerCommand.SET_STATIC_DELAY,
            volume=None,
            mute=None,
            static_delay_ms=200,
        )
    )

    # `settings.update` schedules a debounced save via asyncio; wrap in a loop.
    async def run() -> None:
        app._handle_server_command(payload)

    asyncio.run(run())

    assert app._audio_handler.delay_changes == [-300_000]
    assert app._applied_delay_ms == 200.0
    assert app._ui.delays == [200.0]
    assert app._settings.static_delay_ms == 200.0


def test_keyboard_volume_change_uses_audio_handler_state(tmp_path: Path) -> None:
    state = AppState(player_volume=10, player_muted=True)
    audio_handler = _FakeAudioHandler(volume=41, muted=False)
    ui = _FakeUI()
    handler = CommandHandler(
        get_client=lambda: SimpleNamespace(),
        state=state,
        audio_handler=audio_handler,
        ui=ui,
        settings=_make_settings(tmp_path),
    )

    handler.change_player_volume(5)

    assert audio_handler.calls == [(46, False)]


def test_keyboard_toggle_mute_uses_audio_handler_state(tmp_path: Path) -> None:
    state = AppState(player_volume=10, player_muted=True)
    audio_handler = _FakeAudioHandler(volume=41, muted=False)
    ui = _FakeUI()
    handler = CommandHandler(
        get_client=lambda: SimpleNamespace(),
        state=state,
        audio_handler=audio_handler,
        ui=ui,
        settings=_make_settings(tmp_path),
    )

    handler.toggle_player_mute()

    assert audio_handler.calls == [(41, True)]


def test_keyboard_adjust_delay_notifies_app_tracker(tmp_path: Path) -> None:
    """Local `,`/`.` adjustments propagate to the app's applied-delay tracker.

    Without this, a later visualizer toggle would rebuild the client from the
    pre-adjust value, losing the user's tweak.
    """
    audio_handler = _FakeAudioHandler(volume=50, muted=False)
    ui = _FakeUI()
    captured: list[float] = []
    client = SimpleNamespace(static_delay_ms=0.0)

    def set_static_delay_ms(value: float) -> None:
        client.static_delay_ms = max(0.0, min(5000.0, value))

    client.set_static_delay_ms = set_static_delay_ms

    handler = CommandHandler(
        get_client=lambda: client,
        state=AppState(),
        audio_handler=audio_handler,
        ui=ui,
        settings=_make_settings(tmp_path),
        on_delay_changed=captured.append,
    )

    asyncio.run(handler.adjust_delay(50))

    assert client.static_delay_ms == 50.0
    assert audio_handler.delay_changes == [50_000]
    assert captured == [50.0]
