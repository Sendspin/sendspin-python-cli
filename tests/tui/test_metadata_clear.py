"""Tests for the TUI metadata-clear handling."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

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


def test_handle_metadata_update_clears_on_null_payload() -> None:
    """A spec-legal metadata clear must not leave stale now-playing text behind."""
    state = AppState(title="Track", artist="Artist", album="Album")

    SendspinApp._handle_metadata_update(_app_with(state), SimpleNamespace(metadata=None))

    assert state.title is None
    assert state.artist is None
    assert state.album is None
