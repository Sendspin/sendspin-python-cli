from __future__ import annotations

import asyncio
import json

import pytest

import sendspin.hook_volume as hook_volume
from sendspin.hook_volume import HookVolumeController
from sendspin.settings import ClientSettings


class _FakeProcess:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def test_get_state_uses_persisted_settings(tmp_path) -> None:
    async def exercise() -> None:
        settings = ClientSettings(
            _settings_file=tmp_path / "settings.json",
            player_volume=64,
            player_muted=True,
        )
        controller = HookVolumeController("/usr/bin/set-volume", settings)

        assert await controller.get_state() == (64, True)

    asyncio.run(exercise())


def test_client_settings_load_hook_set_volume(tmp_path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "player_volume": 33,
                "player_muted": False,
                "hook_set_volume": "/usr/bin/set-volume",
            }
        )
        + "\n"
    )

    async def exercise() -> None:
        settings = ClientSettings(_settings_file=settings_file)
        await settings.load()

        assert settings.player_volume == 33
        assert settings.player_muted is False
        assert settings.hook_set_volume == "/usr/bin/set-volume"

    asyncio.run(exercise())


def test_set_state_runs_hook_and_persists_logical_volume(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, stdout: int, stderr: int) -> _FakeProcess:
        calls.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def exercise() -> None:
        settings = ClientSettings(_settings_file=tmp_path / "settings.json")
        controller = HookVolumeController("/usr/bin/set-volume --zone main", settings)

        await controller.set_state(42, muted=True)
        await settings.flush()

        assert calls == [("/usr/bin/set-volume", "--zone", "main", "0")]
        assert await controller.get_state() == (42, True)

        data = json.loads((tmp_path / "settings.json").read_text())
        assert data["player_volume"] == 42
        assert data["player_muted"] is True

    asyncio.run(exercise())


def test_set_state_kills_hook_that_exceeds_the_timeout(monkeypatch, tmp_path) -> None:
    """A hung volume hook is killed and reported instead of awaited forever."""

    class _HangingProcess:
        returncode: int | None = None

        def __init__(self) -> None:
            self.killed = False
            self.waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(30)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.waited = True
            return -9

    proc = _HangingProcess()

    async def fake_exec(*argv: str, stdout: int, stderr: int) -> _HangingProcess:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(hook_volume, "_HOOK_TIMEOUT_SECONDS", 0.01)

    async def exercise() -> None:
        settings = ClientSettings(
            _settings_file=tmp_path / "settings.json",
            player_volume=100,
            player_muted=False,
        )
        controller = HookVolumeController("/usr/bin/set-volume", settings)

        with pytest.raises(RuntimeError, match="timed out"):
            await controller.set_state(50, muted=False)

        assert proc.killed is True
        assert proc.waited is True
        # A hook that failed must not persist a volume the device never accepted.
        assert await controller.get_state() == (100, False)

    asyncio.run(exercise())
