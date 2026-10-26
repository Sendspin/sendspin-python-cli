"""Script-backed volume control backend."""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import TYPE_CHECKING

from sendspin.volume_controller import VolumeChangeCallback

if TYPE_CHECKING:
    from sendspin.settings import ClientSettings

logger = logging.getLogger(__name__)


class HookVolumeController:
    """Controls external volume via a user-provided script.

    The hook script receives the effective output volume as the last argument
    in the range 0-100. When muted, the hook receives ``0`` while the logical
    volume and mute state remain persisted separately in settings.
    """

    def __init__(self, command: str, settings: ClientSettings) -> None:
        """Initialize the controller."""
        argv = shlex.split(command)
        if not argv:
            raise ValueError("Hook volume command must not be empty")

        self._argv = argv
        self._settings = settings

    async def set_state(self, volume: int, *, muted: bool) -> None:
        """Set external volume and persist the logical state."""
        if not 0 <= volume <= 100:
            raise ValueError(f"Volume must be 0-100, got {volume}")

        effective_volume = 0 if muted else volume
        logger.debug(
            "Running volume hook %s with effective volume %d",
            self._argv[0],
            effective_volume,
        )

        proc = await asyncio.create_subprocess_exec(
            *self._argv,
            str(effective_volume),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Volume hook timed out: {' '.join(self._argv)}") from None
        if proc.returncode != 0:
            raise RuntimeError(
                "Volume hook failed "
                f"(exit {proc.returncode}): {' '.join(self._argv)}\n"
                f"stderr: {stderr.decode().strip() if stderr else '(empty)'}"
            )

        if stdout or stderr:
            logger.debug(
                "Volume hook output: stdout=%s stderr=%s",
                stdout.decode().strip() if stdout else "(empty)",
                stderr.decode().strip() if stderr else "(empty)",
            )

        self._settings.update(player_volume=volume, player_muted=muted)

    async def get_state(self) -> tuple[int, bool]:
        """Return the persisted logical volume state."""
        volume = max(0, min(100, self._settings.player_volume))
        muted = bool(self._settings.player_muted)
        return volume, muted

    async def start_monitoring(self, _callback: VolumeChangeCallback) -> None:
        """Hook-based volume control does not support external monitoring."""

    async def stop_monitoring(self) -> None:
        """Hook-based volume control does not support external monitoring."""
