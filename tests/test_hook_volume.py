from __future__ import annotations

import asyncio
import json

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


def test_set_state_runs_hook_and_persists_hook_state(monkeypatch, tmp_path) -> None:
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
        assert data["hook_volume_state"] == {"volume": 42, "muted": True}
        assert data["player_volume"] == 25
        assert data["player_muted"] is False

        reloaded = ClientSettings(_settings_file=tmp_path / "settings.json")
        await reloaded.load()
        assert reloaded.hook_volume_state == {"volume": 42, "muted": True}

    asyncio.run(exercise())
