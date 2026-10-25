"""Tests for the roles the TUI client is built with."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiosendspin.client import SendspinClient
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, Roles
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
def test_create_client_accepts_visualizer_toggle(tmp_path: Path, visualizer: bool) -> None:
    """Toggling the visualizer rebuilds the client; aiosendspin must accept both role sets."""
    app = _make_app(tmp_path, visualizer=visualizer)

    async def create() -> SendspinClient:
        return app._create_client()

    client = asyncio.run(create())

    assert (Roles.VISUALIZER in client.roles) is visualizer
    if visualizer:
        # The stream configuration is requested in client/state, not the hello support object.
        assert client.visualizer_support is not None
        assert not client.visualizer_support.has_stream_config
        assert client.visualizer_state is not None
        assert "spectrum" in client.visualizer_state.types
        assert client.visualizer_state.spectrum is not None
    else:
        assert client.visualizer_state is None
