"""Tests for settings persistence."""

from __future__ import annotations

from pathlib import Path

from sendspin.settings import _load_identity


def test_load_identity_creates_key_file_without_group_or_other_access(tmp_path: Path) -> None:
    """A generated private key must never be readable by group or other."""
    path = tmp_path / "identity-tui.json"
    settings_path = tmp_path / "settings-tui.json"

    identity = _load_identity(path, settings_path)

    assert path.exists()
    assert path.stat().st_mode & 0o077 == 0

    # The persisted key must still round-trip to the same client identity.
    reloaded = _load_identity(path, settings_path)
    assert reloaded.peer_id == identity.peer_id
