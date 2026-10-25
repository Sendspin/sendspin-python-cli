"""Keyboard input handling for the Sendspin CLI."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import readchar
from aiosendspin.models.types import MediaCommand, PlaybackStateType, RepeatMode

if TYPE_CHECKING:
    from aiosendspin.client import SendspinClient

    from sendspin.audio_connector import AudioStreamHandler
    from sendspin.settings import ClientSettings
    from sendspin.tui.app import AppState
    from sendspin.tui.ui import SendspinUI

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles keyboard commands."""

    def __init__(
        self,
        get_client: Callable[[], SendspinClient | None],
        state: AppState,
        audio_handler: AudioStreamHandler,
        ui: SendspinUI,
        settings: ClientSettings,
        on_delay_changed: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the command handler."""
        self._get_client = get_client
        self._state = state
        self._audio_handler = audio_handler
        self._ui = ui
        self._settings = settings
        self._on_delay_changed = on_delay_changed

    async def send_media_command(self, command: MediaCommand) -> None:
        """Send a media command with validation."""
        if command not in self._state.supported_commands:
            self._ui.add_event(f"Server does not support {command.value}")
            return
        client = self._get_client()
        if client is not None:
            await client.send_group_command(command)

    async def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        if self._state.playback_state == PlaybackStateType.PLAYING:
            await self.send_media_command(MediaCommand.PAUSE)
        else:
            await self.send_media_command(MediaCommand.PLAY)

    def change_player_volume(self, delta: int) -> None:
        """Adjust player (local) volume by delta."""
        target = max(0, min(100, self._audio_handler.volume + delta))
        self._audio_handler.set_volume(target, muted=self._audio_handler.muted)
        self._ui.add_event(f"Player volume: {target}%")

    def toggle_player_mute(self) -> None:
        """Toggle player (local) mute state."""
        muted = not self._audio_handler.muted
        self._audio_handler.set_volume(self._audio_handler.volume, muted=muted)
        self._ui.add_event("Player muted" if muted else "Player unmuted")

    async def change_group_volume(self, delta: int) -> None:
        """Adjust group volume by delta."""
        if MediaCommand.VOLUME not in self._state.supported_commands:
            self._ui.add_event("Server does not support volume control")
            return
        current = self._state.volume or 0
        target = max(0, min(100, current + delta))
        client = self._get_client()
        if client is not None:
            await client.send_group_command(MediaCommand.VOLUME, volume=target)

    async def toggle_group_mute(self) -> None:
        """Toggle group mute state."""
        if MediaCommand.MUTE not in self._state.supported_commands:
            self._ui.add_event("Server does not support mute control")
            return
        muted = not self._state.muted
        client = self._get_client()
        if client is not None:
            await client.send_group_command(MediaCommand.MUTE, mute=muted)

    async def cycle_repeat(self) -> None:
        """Cycle repeat mode: OFF -> ALL -> ONE -> OFF."""
        _REPEAT_CYCLE: dict[RepeatMode | None, MediaCommand] = {
            None: MediaCommand.REPEAT_ALL,
            RepeatMode.OFF: MediaCommand.REPEAT_ALL,
            RepeatMode.ALL: MediaCommand.REPEAT_ONE,
            RepeatMode.ONE: MediaCommand.REPEAT_OFF,
        }
        command = _REPEAT_CYCLE.get(self._state.repeat_mode, MediaCommand.REPEAT_ALL)
        await self.send_media_command(command)

    async def toggle_shuffle(self) -> None:
        """Toggle shuffle on/off."""
        if self._state.shuffle:
            await self.send_media_command(MediaCommand.UNSHUFFLE)
        else:
            await self.send_media_command(MediaCommand.SHUFFLE)

    async def adjust_delay(self, delta: float) -> None:
        """Adjust output delay by delta milliseconds."""
        client = self._get_client()
        if client is None:
            return
        old_delay = client.output_delay_ms
        new_delay = max(0, min(5000, old_delay + delta))
        client.set_output_delay_ms(new_delay)
        actual_delta_us = int((client.output_delay_ms - old_delay) * 1000)
        if actual_delta_us != 0:
            self._audio_handler.notify_delay_change(actual_delta_us)
        self._ui.set_delay(client.output_delay_ms)
        self._settings.update(output_delay_ms=client.output_delay_ms)
        if self._on_delay_changed is not None:
            self._on_delay_changed(client.output_delay_ms)
        self._audio_handler.send_player_volume()

    def close_server_selector(self) -> None:
        """Close the server selector panel."""
        self._ui.hide_server_selector()


async def keyboard_loop(
    get_client: Callable[[], SendspinClient | None],
    state: AppState,
    audio_handler: AudioStreamHandler,
    ui: SendspinUI,
    settings: ClientSettings,
    show_server_selector: Callable[[], None],
    on_server_selected: Callable[[], Awaitable[None]],
    request_shutdown: Callable[[], None],
    on_toggle_visualizer: Callable[[], Awaitable[None]],
    on_delay_changed: Callable[[float], None] | None = None,
) -> None:
    """Run the keyboard input loop.

    Args:
        get_client: Callable returning the current Sendspin client instance.
        state: Application state.
        audio_handler: Audio stream handler.
        ui: UI instance.
        settings: Settings manager for persisting player settings.
        show_server_selector: Function to show the server selector UI.
        on_server_selected: Async callback when a server is selected.
        request_shutdown: Callback to request application shutdown.
        on_toggle_visualizer: Async callback to toggle the visualizer.
        on_delay_changed: Optional callback invoked with the new output delay
            (milliseconds) after a local adjust, so the app can keep its
            applied-delay tracker in sync.
    """
    handler = CommandHandler(
        get_client,
        state,
        audio_handler,
        ui,
        settings,
        on_delay_changed=on_delay_changed,
    )

    # Key dispatch table: key -> (highlight_name | None, action)
    # Actions can be sync or async. For keys that need case-insensitive matching, use lowercase.
    shortcuts: dict[str, tuple[str | None, Callable[[], Awaitable[None] | None]]] = {
        # Letter keys
        " ": ("space", handler.toggle_play_pause),
        "m": ("mute", handler.toggle_player_mute),
        "g": ("switch", lambda: handler.send_media_command(MediaCommand.SWITCH)),
        "r": ("repeat", handler.cycle_repeat),
        "x": ("shuffle", handler.toggle_shuffle),
        "v": ("visualizer", on_toggle_visualizer),
        "t": ("theme", ui.cycle_color_mode),
        # Delay adjustment
        ",": ("delay-", lambda: handler.adjust_delay(-10)),
        ".": ("delay+", lambda: handler.adjust_delay(10)),
        # Group volume/mute (uppercase M matched before lowercase fallback)
        "M": ("group-mute", handler.toggle_group_mute),
        # Arrow keys
        readchar.key.LEFT: (
            "prev",
            lambda: handler.send_media_command(MediaCommand.PREVIOUS),
        ),
        readchar.key.RIGHT: (
            "next",
            lambda: handler.send_media_command(MediaCommand.NEXT),
        ),
        readchar.key.UP: ("up", lambda: handler.change_player_volume(5)),
        readchar.key.DOWN: ("down", lambda: handler.change_player_volume(-5)),
        # Group volume
        "]": ("group-up", lambda: handler.change_group_volume(5)),
        "[": ("group-down", lambda: handler.change_group_volume(-5)),
    }

    loop = asyncio.get_running_loop()
    key_queue: asyncio.Queue[str] = asyncio.Queue()
    stop_reader = threading.Event()

    def read_keys() -> None:
        """Read keys on a daemon thread and forward them to the event loop."""
        while not stop_reader.is_set():
            try:
                key = readchar.readkey()
            except KeyboardInterrupt:
                key = "\x03"
            except Exception:  # noqa: BLE001
                logger.exception("Keyboard input failed")
                try:
                    loop.call_soon_threadsafe(request_shutdown)
                    loop.call_soon_threadsafe(key_queue.put_nowait, "\x03")
                except RuntimeError:
                    pass
                return

            if stop_reader.is_set():
                return

            try:
                loop.call_soon_threadsafe(key_queue.put_nowait, key)
            except RuntimeError:
                return

    threading.Thread(target=read_keys, name="sendspin-keyboard", daemon=True).start()

    try:
        while True:
            try:
                key = await key_queue.get()
            except (asyncio.CancelledError, KeyboardInterrupt):
                request_shutdown()
                break

            if key == "\x03":
                request_shutdown()
                break

            # Handle server selector mode
            if ui.is_server_selector_visible():
                if key in "rR":
                    show_server_selector()
                    continue
                if key == readchar.key.UP:
                    ui.highlight_shortcut("selector-up")
                    ui.move_server_selection(-1)
                    continue
                if key == readchar.key.DOWN:
                    ui.highlight_shortcut("selector-down")
                    ui.move_server_selection(1)
                    continue
                if key in ("\r", "\n", readchar.key.ENTER):
                    ui.highlight_shortcut("selector-enter")
                    await on_server_selected()
                    continue
                if key in "qQ":
                    ui.hide_server_selector()
                    continue
                # Ignore other keys when selector is open
                continue

            # Handle quit
            if key in "qQ":
                ui.highlight_shortcut("quit")
                request_shutdown()
                break

            # Handle 's' to open server selector
            if key in "sS":
                ui.highlight_shortcut("server")
                show_server_selector()
                continue

            # Handle shortcuts via dispatch table (case-insensitive for letter keys)
            action = shortcuts.get(key) or shortcuts.get(key.lower())
            if action:
                highlight_name, action_handler = action
                if highlight_name and ui:
                    ui.highlight_shortcut(highlight_name)
                result = action_handler()
                if result is not None:
                    await result
                continue

            # Ignore unhandled escape sequences
            if key.startswith("\x1b"):
                continue
    finally:
        stop_reader.set()
