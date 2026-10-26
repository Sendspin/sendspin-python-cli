"""Tests for the TUI client's visualizer configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiosendspin.client import SendspinClient
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.models.visualizer import VisualizerStatePayload
from aiosendspin.noise import Identity, InMemoryClientPairingStore

from sendspin.settings import ClientSettings
from sendspin.tui.app import AppArgs, SendspinApp


def _make_app(tmp_path: Path, *, visualizer: bool) -> SendspinApp:
    args = AppArgs(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        identity=Identity.generate(),
        pairing_store=InMemoryClientPairingStore(),
        client_name="Test Client",
        settings=ClientSettings(_settings_file=tmp_path / "settings.json"),
        use_mpris=False,
    )
    app = SendspinApp(args)
    app._audio_handler = SimpleNamespace(volume=50, muted=False)
    app._supported_formats = [
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)
    ]
    app._supports_artwork = False
    app._visualizer_enabled = visualizer
    return app


@pytest.mark.parametrize("visualizer", [False, True])
def test_create_client_requests_visualizer_types_in_state(tmp_path: Path, visualizer: bool) -> None:
    """The visualizer role is always declared; the setting only picks the requested types."""
    app = _make_app(tmp_path, visualizer=visualizer)

    async def create() -> SendspinClient:
        return app._create_client()

    client = asyncio.run(create())

    assert Roles.VISUALIZER in client.roles
    # The stream configuration is requested in client/state, not the hello support object.
    assert client.visualizer_support is not None
    assert not client.visualizer_support.has_stream_config
    assert client.visualizer_state is not None
    if visualizer:
        assert "spectrum" in client.visualizer_state.types
        assert client.visualizer_state.spectrum is not None
    else:
        assert client.visualizer_state.types == []
        assert client.visualizer_state.spectrum is None


class _FakeClient:
    def __init__(self) -> None:
        self.states: list[VisualizerStatePayload] = []
        self.disconnects = 0

    async def set_visualizer_state(self, state: VisualizerStatePayload) -> None:
        self.states.append(state)

    async def disconnect(self) -> None:
        self.disconnects += 1


class _FakeUI:
    def __init__(self) -> None:
        self.enabled: list[bool] = []

    def set_visualizer_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def clear_beats(self) -> None:
        pass

    def clear_peaks(self) -> None:
        pass


def test_toggle_visualizer_updates_state_without_reconnecting(tmp_path: Path) -> None:
    """Toggling must keep the connection, so playback continues."""
    app = _make_app(tmp_path, visualizer=False)
    client = _FakeClient()
    ui = _FakeUI()
    app._client = client  # type: ignore[assignment]
    app._ui = ui  # type: ignore[assignment]

    async def toggle_twice() -> None:
        await app._toggle_visualizer()
        await app._toggle_visualizer()

    asyncio.run(toggle_twice())

    assert client.disconnects == 0
    assert app._client is client
    assert ui.enabled == [True, False]
    assert "spectrum" in client.states[0].types
    assert client.states[1].types == []
