"""Tests for the TUI metadata-clear handling."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import MediaCommand, RepeatMode

from sendspin.tui.app import AppState, SendspinApp


def _fake_ui() -> SimpleNamespace:
    return SimpleNamespace(
        batch_update=contextlib.nullcontext,
        set_metadata=lambda **kwargs: None,
        set_progress=lambda *args: None,
        set_repeat_shuffle=lambda *args: None,
        add_event=lambda *args: None,
    )


def _app_with(state: AppState) -> Any:
    return SimpleNamespace(_ui=_fake_ui(), _state=state)


def test_app_state_clear_metadata_resets_now_playing() -> None:
    state = AppState(
        title="Track",
        artist="Artist",
        album="Album",
        track_progress=1000,
        track_duration=5000,
    )

    assert state.clear_metadata() is True
    assert state.title is None
    assert state.artist is None
    assert state.album is None
    assert state.track_progress is None
    assert state.track_duration is None
    # Clearing an already-clear state reports no change.
    assert state.clear_metadata() is False


def test_handle_metadata_update_clears_on_removed_role() -> None:
    """A removed metadata role must not leave stale now-playing text behind."""
    state = AppState(title="Track", artist="Artist", album="Album")

    SendspinApp._handle_metadata_update(_app_with(state), None)

    assert state.title is None
    assert state.artist is None
    assert state.album is None


def test_update_metadata_clears_omitted_fields() -> None:
    """Metadata is full state: a field the server leaves out is cleared."""
    state = AppState(
        title="Old",
        artist="Artist",
        album="Album",
        track_progress=1000,
        track_duration=5000,
    )

    assert state.update_metadata(SessionUpdateMetadata(timestamp=0, title="New")) is True
    assert state.title == "New"
    assert state.artist is None
    assert state.album is None
    assert state.track_progress is None
    assert state.track_duration is None
    # Re-applying the same state reports no change.
    assert state.update_metadata(SessionUpdateMetadata(timestamp=1, title="New")) is False


def test_update_metadata_sets_progress() -> None:
    state = AppState()
    metadata = SessionUpdateMetadata(
        timestamp=0,
        title="Track",
        progress=Progress(track_progress=1000, track_duration=5000, playback_speed=1000),
    )

    assert state.update_metadata(metadata) is True
    assert state.track_progress == 1000
    assert state.track_duration == 5000


def test_handle_server_state_takes_repeat_and_shuffle_from_controller() -> None:
    """Repeat and shuffle moved from the metadata object to the controller object."""
    state = AppState()
    shown: list[tuple[RepeatMode | None, bool | None]] = []
    app = _app_with(state)
    app._ui.set_volume = lambda *args, **kwargs: None
    app._ui.set_repeat_shuffle = lambda repeat, shuffle: shown.append((repeat, shuffle))
    controller = ControllerStatePayload(
        supported_commands=[MediaCommand.PLAY],
        volume=50,
        muted=False,
        repeat=RepeatMode.ALL,
        shuffle=True,
    )

    SendspinApp._handle_server_state(app, ServerStatePayload(controller=controller))

    assert state.repeat_mode == RepeatMode.ALL
    assert state.shuffle is True
    assert shown == [(RepeatMode.ALL, True)]
