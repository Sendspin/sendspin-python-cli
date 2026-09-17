"""Core application logic for the Sendspin client."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiosendspin.models.metadata import SessionUpdateMetadata
    from aiosendspin.noise import ClientPairingStore, Identity

    from sendspin.volume_controller import VolumeController

from aiohttp import ClientError
from aiosendspin.client import PairingSupport, SendspinClient
from aiosendspin.models.artwork import ArtworkChannel, ClientHelloArtworkSupport
from aiosendspin.models.core import (
    GroupUpdateServerPayload,
    ServerCommandPayload,
    ServerStatePayload,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    SupportedAudioFormat,
)
from aiosendspin.models.types import (
    ArtworkSource,
    MediaCommand,
    PairAbortReason,
    PictureFormat,
    PlaybackStateType,
    PlayerCommand,
    RepeatMode,
    Roles,
    UndefinedField,
)
from aiosendspin.models.visualizer import (
    BeatTiming,
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    VisualizerFrame,
)
from aiosendspin_mpris import MPRIS_AVAILABLE, SendspinMpris
from PIL.Image import Image as PILImage

from sendspin.artwork_connector import ArtworkHandler
from sendspin.audio_connector import AudioStreamHandler
from sendspin.audio_devices import AudioDevice, detect_supported_audio_formats
from sendspin.discovery import DiscoveredServer, ServiceDiscovery
from sendspin.hooks import run_hook
from sendspin.settings import ClientSettings
from sendspin.tui.artwork import detect_support as detect_artwork_support
from sendspin.tui.keyboard import keyboard_loop
from sendspin.tui.ui import ColorMode, SendspinUI
from sendspin.tui.visualizer import (
    SPECTRUM_F_MAX,
    SPECTRUM_F_MIN,
    SPECTRUM_N_BINS,
    SPECTRUM_SCALE,
    PeakEvent,
)
from sendspin.utils import create_task, get_device_info
from sendspin.visualizer_connector import BeatHandler, PeakHandler, VisualizerHandler

logger = logging.getLogger(__name__)


class ServerSwitchRequested(Exception):
    """Raised when a connection attempt is cancelled due to server switch."""


@dataclass
class AppState:
    """Holds state mirrored from the server for CLI presentation."""

    selected_server: DiscoveredServer | None = None
    playback_state: PlaybackStateType | None = None
    supported_commands: set[MediaCommand] = field(default_factory=set)
    volume: int | None = None
    muted: bool | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    track_progress: int | None = None
    track_duration: int | None = None
    player_volume: int = 100
    player_muted: bool = False
    group_id: str | None = None
    repeat_mode: RepeatMode | None = None
    shuffle: bool | None = None

    def update_metadata(self, metadata: SessionUpdateMetadata) -> bool:
        """Merge new metadata into the state and report if anything changed."""
        changed = False

        # Update simple metadata fields
        for attr in ("title", "artist", "album"):
            value = getattr(metadata, attr)
            if not isinstance(value, UndefinedField) and getattr(self, attr) != value:
                setattr(self, attr, value)
                changed = True

        # Update repeat and shuffle
        for attr in ("repeat_mode", "shuffle"):
            # metadata uses "repeat" for the field name, state uses "repeat_mode"
            meta_attr = "repeat" if attr == "repeat_mode" else attr
            value = getattr(metadata, meta_attr)
            if not isinstance(value, UndefinedField) and getattr(self, attr) != value:
                setattr(self, attr, value)
                changed = True

        # Update progress fields from nested progress object
        if isinstance(metadata.progress, UndefinedField):
            return changed

        if metadata.progress is None:
            # Clear progress fields
            if self.track_progress is not None or self.track_duration is not None:
                self.track_progress = None
                self.track_duration = None
                changed = True
        else:
            # Update from nested progress object
            if self.track_progress != metadata.progress.track_progress:
                self.track_progress = metadata.progress.track_progress
                changed = True
            if self.track_duration != metadata.progress.track_duration:
                self.track_duration = metadata.progress.track_duration
                changed = True

        return changed

    def describe(self) -> str:
        """Return a human-friendly description of the current state."""
        lines: list[str] = []
        if self.title:
            lines.append(f"Now playing: {self.title}")
        if self.artist:
            lines.append(f"Artist: {self.artist}")
        if self.album:
            lines.append(f"Album: {self.album}")
        if self.track_duration:
            progress_s = (self.track_progress or 0) / 1000
            duration_s = self.track_duration / 1000
            lines.append(f"Progress: {progress_s:>5.1f} / {duration_s:>5.1f} s")
        if self.volume is not None:
            vol_line = f"Volume: {self.volume}%"
            if self.muted:
                vol_line += " (muted)"
            lines.append(vol_line)
        if self.playback_state is not None:
            lines.append(f"State: {self.playback_state.value}")
        return "\n".join(lines)


class ConnectionManager:
    """Manages connection state and reconnection logic with exponential backoff."""

    def __init__(
        self,
        discovery: ServiceDiscovery,
        max_backoff: float = 300.0,
    ) -> None:
        """Initialize the connection manager."""
        self._discovery = discovery
        self._error_backoff = 1.0
        self._max_backoff = max_backoff
        self._last_attempted_url = ""
        self._pending_server: DiscoveredServer | None = None  # URL set by user for server switch

    def set_pending_server(self, server: DiscoveredServer) -> None:
        """Set a pending server for server switch."""
        self._pending_server = server

    def consume_pending_server(self) -> DiscoveredServer | None:
        """Get and clear the pending server if set."""
        server = self._pending_server
        self._pending_server = None
        return server

    def set_last_attempted_url(self, url: str) -> None:
        """Record the URL that was last attempted."""
        self._last_attempted_url = url

    def reset_backoff(self) -> None:
        """Reset backoff to initial value after successful connection."""
        self._error_backoff = 1.0

    def should_reset_backoff(self, current_url: str | None) -> bool:
        """Check if URL changed, indicating server came back online."""
        return bool(current_url and current_url != self._last_attempted_url)

    def update_backoff_and_url(self, current_url: str | None) -> tuple[str | None, float]:
        """Update URL and backoff based on discovery.

        Returns (new_url, new_backoff).
        """
        if self.should_reset_backoff(current_url):
            logger.info("Server URL changed to %s, reconnecting immediately", current_url)
            assert current_url is not None
            self._last_attempted_url = current_url
            self._error_backoff = 1.0
            return current_url, 1.0
        self._error_backoff = min(self._error_backoff * 2, self._max_backoff)
        return None, self._error_backoff

    def get_error_backoff(self) -> float:
        """Get the current error backoff duration."""
        return self._error_backoff

    def increase_backoff(self) -> None:
        """Increase the backoff duration for the next retry."""
        self._error_backoff = min(self._error_backoff * 2, self._max_backoff)

    async def handle_error_backoff(self, ui: SendspinUI) -> None:
        """Sleep for error backoff duration."""
        ui.add_event(f"Connection error, retrying in {self._error_backoff:.0f}s...")
        await asyncio.sleep(self._error_backoff)

    async def discover_server(self) -> DiscoveredServer:
        """Wait for server to reappear on the network."""
        return await self._discovery.wait_for_server()


@dataclass
class AppArgs:
    """Configuration for the Sendspin application."""

    audio_device: AudioDevice
    identity: Identity
    pairing_store: ClientPairingStore
    client_name: str
    settings: ClientSettings
    url: str | None = None
    url_from_settings: bool = False
    static_delay_ms: float | None = None
    use_mpris: bool = True
    preferred_format: SupportedAudioFormat | None = None
    volume_controller: VolumeController | None = None
    hook_start: str | None = None
    hook_stop: str | None = None
    manufacturer: str | None = None
    product_name: str | None = None
    interface: str | None = None


class SendspinApp:
    """Main Sendspin application."""

    def __init__(self, args: AppArgs) -> None:
        """Initialize the application."""
        self._args = args
        self._ui: SendspinUI | None = None

        server: DiscoveredServer | None = None
        if args.url:
            label = "Last used" if args.url_from_settings else "Command-line argument"
            server = DiscoveredServer.from_url(label, args.url)

        self._state = AppState(selected_server=server)

        self._client: SendspinClient | None = None
        self._audio_handler: AudioStreamHandler | None = None
        self._visualizer_handler: VisualizerHandler | None = None
        self._beat_handler: BeatHandler | None = None
        self._peak_handler: PeakHandler | None = None
        self._artwork_handler: ArtworkHandler | None = None
        self._settings = args.settings
        self._visualizer_enabled: bool = args.settings.visualizer
        # Probe terminal graphics support before Rich Live takes the tty.
        self._supports_artwork: bool = detect_artwork_support()
        # Currently-applied static delay in milliseconds, mirroring
        # `SendspinClient.static_delay_ms`. Tracked separately from settings
        # because CLI overrides aren't persisted to settings, so
        # `settings.static_delay_ms` can lag the value actually given to the client.
        self._applied_delay_ms: float = 0.0
        interfaces = [args.interface] if args.interface else None
        self._discovery = ServiceDiscovery(interfaces=interfaces)
        self._connection_manager = ConnectionManager(self._discovery)
        self._connect_task: asyncio.Task[None] | None = None
        self._mpris: SendspinMpris | None = None
        self._listener_unsubscribes: list[Callable[[], None]] = []

    @staticmethod
    def _build_artwork_support() -> ClientHelloArtworkSupport:
        """Build artwork support payload for client/hello (artwork@v1)."""
        return ClientHelloArtworkSupport(
            channels=[
                ArtworkChannel(
                    source=ArtworkSource.ALBUM,
                    format=PictureFormat.PNG,
                    media_width=128,
                    media_height=128,
                ),
            ],
        )

    @staticmethod
    def _build_visualizer_support() -> ClientHelloVisualizerSupport:
        """Build visualizer support payload for client/hello (visualizer@v1)."""
        return ClientHelloVisualizerSupport(
            buffer_capacity=65536,
            types=["loudness", "spectrum", "beat", "peak", "f_peak", "pitch"],
            rate_max=30,
            spectrum=ClientHelloVisualizerSpectrum(
                n_disp_bins=SPECTRUM_N_BINS,
                scale=SPECTRUM_SCALE,
                f_min=SPECTRUM_F_MIN,
                f_max=SPECTRUM_F_MAX,
            ),
        )

    def _create_client(self) -> SendspinClient:
        """Create a new SendspinClient with roles based on current visualizer state."""
        args = self._args
        roles = [Roles.CONTROLLER, Roles.PLAYER, Roles.METADATA, Roles.COLOR]
        visualizer_support = None
        if self._visualizer_enabled:
            visualizer_support = self._build_visualizer_support()
            roles.append(Roles.VISUALIZER)

        artwork_support: ClientHelloArtworkSupport | None = None
        if self._supports_artwork:
            artwork_support = self._build_artwork_support()
            roles.append(Roles.ARTWORK)

        assert self._audio_handler is not None

        return SendspinClient(
            identity=args.identity,
            client_name=args.client_name,
            roles=roles,
            pairing_store=args.pairing_store,
            pairing_support=self._build_pairing_support(),
            device_info=get_device_info(
                manufacturer=args.manufacturer,
                product_name=args.product_name,
            ),
            player_support=ClientHelloPlayerSupport(
                supported_formats=self._supported_formats,
                buffer_capacity=32_000_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
            visualizer_support=visualizer_support,
            artwork_support=artwork_support,
            static_delay_ms=self._applied_delay_ms,
            state_supported_commands=[PlayerCommand.SET_STATIC_DELAY],
            initial_volume=self._audio_handler.volume,
            initial_muted=self._audio_handler.muted,
        )

    def _build_pairing_support(self) -> PairingSupport:
        """Build pairing-PIN out-channels wired into the Rich UI."""
        return PairingSupport(
            pin_display=self._show_pairing_code,
            offer_static_pin=True,
        )

    async def _show_pairing_code(self, code: str | None) -> None:
        """Render (or clear) the derived dynamic pairing PIN in the UI."""
        if self._ui is not None:
            self._ui.show_pairing_code(code)

    def _attach_client(self) -> None:
        """Attach listeners, audio handler, visualizer, and MPRIS to the current client."""
        assert self._client is not None
        assert self._audio_handler is not None

        self._listener_unsubscribes = [
            self._client.add_metadata_listener(self._handle_metadata_update),
            self._client.add_group_update_listener(self._handle_group_update),
            self._client.add_controller_state_listener(self._handle_server_state),
            self._client.add_server_command_listener(self._handle_server_command),
            self._client.add_color_listener(self._handle_color_update),
            self._client.add_pairing_abort_listener(self._handle_pairing_abort),
        ]
        self._audio_handler.attach_client(self._client)

        if self._ui is not None:
            self._ui.set_visualizer_enabled(self._visualizer_enabled)

        if self._visualizer_enabled:
            self._visualizer_handler = VisualizerHandler(
                on_frame=self._handle_visualizer_frame,
            )
            self._visualizer_handler.attach_client(self._client)
            self._beat_handler = BeatHandler(
                on_beat=self._handle_beat,
                on_schedule=self._handle_beat_schedule,
            )
            self._beat_handler.attach_client(self._client)
            self._peak_handler = PeakHandler(
                on_peak=self._handle_peak,
                on_schedule=self._handle_peak_schedule,
            )
            self._peak_handler.attach_client(self._client)
            self._listener_unsubscribes.append(
                self._client.add_stream_start_listener(self._handle_stream_start)
            )
            if self._ui is not None:
                self._ui.set_server_clock(self._server_now_us)

        if self._supports_artwork:
            self._artwork_handler = ArtworkHandler(
                on_image=self._handle_artwork_update,
            )
            self._artwork_handler.attach_client(self._client)

        if MPRIS_AVAILABLE and self._args.use_mpris:
            self._mpris = SendspinMpris(self._client)
            self._mpris.start()

    def _detach_client(self) -> None:
        """Detach listeners, audio handler, visualizer, and MPRIS from the current client."""
        assert self._audio_handler is not None

        for unsub in self._listener_unsubscribes:
            unsub()
        self._listener_unsubscribes = []
        self._audio_handler.detach_client()
        # Drop the per-server palette so a non-color server reverts to unthemed.
        if self._ui is not None:
            self._ui.reset_palette()

        if self._visualizer_handler:
            self._visualizer_handler.detach()
            self._visualizer_handler = None
        if self._beat_handler:
            self._beat_handler.detach()
            self._beat_handler = None
        if self._peak_handler:
            self._peak_handler.detach()
            self._peak_handler = None
        if self._ui is not None:
            self._ui.set_server_clock(None)
            self._ui.set_visualizer_types(frozenset())

        if self._artwork_handler is not None:
            self._artwork_handler.detach()
            self._artwork_handler = None
        if self._ui is not None:
            self._ui.state.artwork_image = None
            self._ui.state.artwork_generation += 1

        if self._mpris:
            self._mpris.stop()
            self._mpris = None

    async def run(self) -> int:  # noqa: PLR0915
        """Run the application."""
        args = self._args

        # TUI requires an interactive terminal
        if not sys.stdin.isatty():
            print(  # noqa: T201
                "Error: TUI mode requires an interactive terminal.\n"
                "Use 'sendspin daemon' for non-interactive/background operation."
            )
            return 1

        # Store reference to current task so it can be cancelled on shutdown
        main_task = asyncio.current_task()
        assert main_task is not None

        def request_shutdown() -> None:
            main_task.cancel()

        try:
            # CLI arg overrides settings for static delay
            delay = (
                args.static_delay_ms
                if args.static_delay_ms is not None
                else self._settings.static_delay_ms
            )
            self._applied_delay_ms = max(0.0, min(5000.0, delay))

            self._audio_handler = AudioStreamHandler(
                audio_device=args.audio_device,
                volume=self._settings.player_volume,
                muted=self._settings.player_muted,
                on_event=self._on_stream_event,
                on_format_change=self._handle_format_change,
                on_volume_change=self._on_volume_change,
                volume_controller=args.volume_controller,
            )
            await self._audio_handler.read_initial_volume()

            self._state.player_volume = self._audio_handler.volume
            self._state.player_muted = self._audio_handler.muted

            # Detect supported audio formats for the output device
            supported_formats = detect_supported_audio_formats(args.audio_device)
            if args.preferred_format is not None:
                supported_formats = [f for f in supported_formats if f != args.preferred_format]
                supported_formats.insert(0, args.preferred_format)
            self._supported_formats = supported_formats

            self._client = self._create_client()

            await self._audio_handler.start_volume_monitor()

            self._ui = SendspinUI(
                self._applied_delay_ms,
                player_volume=self._audio_handler.volume,
                player_muted=self._audio_handler.muted,
                use_external_volume=self._audio_handler.uses_external_volume_controller,
                visualizer_enabled=self._visualizer_enabled,
                color_mode=ColorMode.parse(self._settings.color_mode),
                on_color_mode_change=self._persist_color_mode,
            )
            self._ui.start()
            self._ui.add_event(f"Using client ID: {args.identity.peer_id}")
            self._ui.add_event(f"Using audio device: {args.audio_device.name}")

            await self._discovery.start()

            self._attach_client()

            # Start keyboard loop for interactive control
            create_task(
                keyboard_loop(
                    lambda: self._client,
                    self._state,
                    self._audio_handler,
                    self._ui,
                    self._settings,
                    self._show_server_selector,
                    self._on_server_selected,
                    request_shutdown,
                    on_toggle_visualizer=self._toggle_visualizer,
                    on_delay_changed=self._set_applied_delay,
                )
            )

            def signal_handler() -> None:
                logger.debug("Received interrupt signal, shutting down...")
                request_shutdown()

            # Signal handlers aren't supported on this platform (e.g., Windows)
            loop = asyncio.get_running_loop()
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signal.SIGINT, signal_handler)
                loop.add_signal_handler(signal.SIGTERM, signal_handler)

            # Get initial server URL
            if args.url and not args.url_from_settings:
                pass  # URL provided via CLI - selected_server already set in __init__
            elif args.url and args.url_from_settings:
                # Try last known server first, fall back to mDNS discovery
                last_url = args.url
                self._ui.add_event(f"Trying last server at {last_url}...")
                try:
                    await self._connect_cancellable(last_url)
                    self._state.selected_server = DiscoveredServer.from_url(
                        "Last connected", last_url
                    )
                    self._ui.add_event(f"Connected to {last_url}")
                    self._ui.set_connected(last_url)
                    self._settings.update(last_server_url=last_url)
                    await self._connection_loop(already_connected=True)
                    return 0
                except ServerSwitchRequested:
                    pass  # New server already set in state, fall through to connection loop
                except (TimeoutError, OSError, ClientError):
                    self._state.selected_server = None
                    self._ui.add_event("Last server unavailable, searching...")

            # No URL or last server unavailable - do mDNS discovery
            if self._state.selected_server is None:
                logger.info("Waiting for mDNS discovery of Sendspin server...")
                self._ui.add_event("Searching for Sendspin server...")
                server = await self._connection_manager.discover_server()
                self._state.selected_server = server
                self._ui.add_event(f"Found server at {server.url}")

            # Run connection loop with auto-reconnect
            await self._connection_loop()
        except asyncio.CancelledError:
            logger.debug("Connection loop cancelled")
        finally:
            if self._mpris:
                self._mpris.stop()
            if self._visualizer_handler:
                self._visualizer_handler.detach()
            if self._beat_handler:
                self._beat_handler.detach()
            if self._peak_handler:
                self._peak_handler.detach()
            if self._ui:
                self._ui.stop()
            if self._audio_handler:
                await self._audio_handler.shutdown()
            if self._client is not None:
                await self._client.disconnect()
            await self._discovery.stop()
            await self._settings.flush()

        return 0

    def _on_volume_change(self, volume: int, muted: bool) -> None:
        """Handle volume changes from any source (server command, keyboard, external)."""
        assert self._audio_handler is not None
        assert self._ui is not None

        self._state.player_volume = volume
        self._state.player_muted = muted
        self._ui.set_player_volume(volume, muted=muted)
        if not self._audio_handler.uses_external_volume_controller:
            self._settings.update(player_volume=volume, player_muted=muted)

    async def _handle_disconnect(self, message: str) -> None:
        """Update UI and reset connection-scoped audio state after a disconnect."""
        assert self._audio_handler is not None
        assert self._ui is not None

        logger.info(message)
        self._ui.add_event(message)
        self._ui.set_disconnected(message)
        if self._visualizer_handler:
            self._visualizer_handler.reset()
        self._clear_visualizer_timelines()
        await self._audio_handler.handle_disconnect()

    async def _connect_cancellable(self, url: str) -> None:
        """Connect to server. Can be cancelled by _cancel_connect().

        Wraps the connection in a tracked task so server selection can
        cancel in-progress connection attempts.

        Raises:
            ServerSwitchRequested: If cancelled due to server switch. The new
                server is already set in state; caller should continue to use it.
        """
        # Create a task for the connection so it can be cancelled
        assert self._client is not None
        self._connect_task = asyncio.create_task(self._client.connect(url))
        try:
            await self._connect_task
        except asyncio.CancelledError:
            # Check if cancelled due to server switch
            pending = self._connection_manager.consume_pending_server()
            if pending:
                self._state.selected_server = pending
                self._connection_manager.reset_backoff()
                if self._ui:
                    self._ui.add_event(f"Switching to {pending.url}...")
                    self._ui.set_disconnected(f"Switching to {pending.url}...")
                raise ServerSwitchRequested from None
            raise
        finally:
            self._connect_task = None

    def _cancel_connect(self) -> bool:
        """Cancel any in-progress connection attempt.

        Returns True if a connection was cancelled, False if no connection was in progress.
        """
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            return True
        return False

    async def _connection_loop(self, *, already_connected: bool = False) -> None:
        """
        Run the connection loop with automatic reconnection on disconnect.

        Connects to the server, waits for disconnect, cleans up, then retries
        only if the server is visible via mDNS. Reconnects immediately when
        server reappears. Uses exponential backoff (up to 5 min) for errors.

        Args:
            already_connected: If True, skip the first connection attempt
                (used when caller already established the connection).
        """
        assert self._state.selected_server
        assert self._client is not None
        assert self._audio_handler is not None
        assert self._ui is not None
        manager = self._connection_manager
        ui = self._ui
        discovery = self._discovery
        url = self._state.selected_server.url
        manager.set_last_attempted_url(url)
        skip_connect = already_connected

        while True:
            try:
                if skip_connect:
                    skip_connect = False
                else:
                    try:
                        await self._connect_cancellable(url)
                    except ServerSwitchRequested:
                        # New server already set in state, update local url and retry
                        url = self._state.selected_server.url
                        continue
                    ui.add_event(f"Connected to {url}")
                    ui.set_connected(url)
                    manager.reset_backoff()
                    manager.set_last_attempted_url(url)
                    self._settings.update(last_server_url=url)

                # Wait for disconnect
                disconnect_event: asyncio.Event = asyncio.Event()
                assert self._client is not None
                unsubscribe = self._client.add_disconnect_listener(disconnect_event.set)
                await disconnect_event.wait()
                unsubscribe()

                # Connection dropped
                await self._handle_disconnect("Connection lost")

                # Check for pending URL from server selection first
                pending_server = manager.consume_pending_server()
                if pending_server:
                    self._state.selected_server = pending_server
                    url = pending_server.url
                    manager.reset_backoff()
                    ui.add_event(f"Switching to {url}...")
                    ui.set_disconnected(f"Switching to {url}...")
                    continue

                # If URL was provided via --url, reconnect directly without mDNS
                if self._args.url:
                    ui.add_event(f"Reconnecting to {url}...")
                    ui.set_disconnected(f"Reconnecting to {url}...")
                    continue

                # Update URL from discovery
                server = servers[0] if (servers := discovery.get_servers()) else None

                # Wait for server to reappear if it's gone
                if not server:
                    ui.set_disconnected("Waiting for server...")
                    logger.info("Server offline, waiting for rediscovery...")
                    ui.add_event("Waiting for server...")

                    server = await manager.discover_server()

                self._state.selected_server = server
                url = server.url
                ui.add_event(f"Reconnecting to {url}...")
                ui.set_disconnected(f"Reconnecting to {url}...")

            except (TimeoutError, OSError, ClientError) as e:
                # Network-related errors - log cleanly
                logger.debug(
                    "Connection error (%s), retrying in %.0fs",
                    type(e).__name__,
                    manager.get_error_backoff(),
                )

                await manager.handle_error_backoff(ui)

                # Check if URL changed while sleeping
                if servers := discovery.get_servers():
                    current_url = servers[0].url
                    new_url, _ = manager.update_backoff_and_url(current_url)
                    if new_url:
                        url = new_url
            except Exception:
                # Unexpected errors - log with full traceback
                logger.exception("Unexpected error")
                break

    def _show_server_selector(self) -> None:
        assert self._ui is not None
        servers = self._discovery.get_servers()
        # Add selected server to list only if not already present (by host)
        if self._state.selected_server:
            selected_host = self._state.selected_server.host
            if not any(s.host == selected_host for s in servers):
                servers.insert(0, self._state.selected_server)
        self._ui.show_server_selector(servers)

    async def _on_server_selected(self) -> None:
        """Handle server selection by triggering reconnect."""
        assert self._ui is not None
        server = self._ui.get_selected_server()
        if server is None:
            return

        self._ui.hide_server_selector()
        # Skip reconnection if already connected to this server
        if self._state.selected_server and server.url == self._state.selected_server.url:
            return

        self._connection_manager.set_pending_server(server)
        # Cancel in-progress connection attempt if any
        if self._cancel_connect():
            # Connection task was cancelled, the CancelledError handler
            # will pick up the pending server and switch to it
            return
        # Force disconnect to trigger reconnect with new URL
        assert self._client is not None
        await self._client.disconnect()

    def _handle_metadata_update(self, payload: ServerStatePayload) -> None:
        """Handle server/state messages with metadata."""
        assert self._ui is not None
        state = self._state
        ui = self._ui
        if isinstance(payload.metadata, UndefinedField):
            return
        if payload.metadata is None or not state.update_metadata(payload.metadata):
            return

        with ui.batch_update():
            ui.set_metadata(
                title=state.title,
                artist=state.artist,
                album=state.album,
            )
            ui.set_progress(state.track_progress, state.track_duration)
            ui.set_repeat_shuffle(state.repeat_mode, state.shuffle)
        ui.add_event(state.describe())

    def _clear_visualizer_timelines(self) -> None:
        """Drop the beat and peak strips, cancelling their pending schedules."""
        if self._beat_handler:
            self._beat_handler.reset()
        if self._peak_handler:
            self._peak_handler.reset()
        if self._ui is not None:
            self._ui.clear_beats()
            self._ui.clear_peaks()

    def _handle_color_update(self, payload: ServerStatePayload) -> None:
        """Forward a color@v1 palette payload to the UI."""
        assert self._ui is not None
        if isinstance(payload.color, UndefinedField):
            return
        self._ui.update_palette(payload.color)

    def _persist_color_mode(self, mode: ColorMode) -> None:
        """Persist the user's color theme choice."""
        self._settings.update(color_mode=mode.value)

    def _handle_group_update(self, payload: GroupUpdateServerPayload) -> None:
        """Handle group update messages."""
        assert self._ui is not None
        state = self._state
        ui = self._ui
        # Track group ID changes for logging. Metadata clearing is not needed here
        # because the server always sends a metadata update (snapshot or cleared)
        # before the group update message when groups change.
        if payload.group_id is not None and payload.group_id != state.group_id:
            state.group_id = payload.group_id
            ui.add_event(f"Group ID: {payload.group_id}")

        if payload.group_name:
            ui.add_event(f"Group name: {payload.group_name}")
        with ui.batch_update():
            ui.set_group_name(payload.group_name)
            if payload.playback_state:
                changed = payload.playback_state != state.playback_state
                state.playback_state = payload.playback_state
                ui.set_playback_state(payload.playback_state)
                ui.add_event(f"Playback state: {payload.playback_state.value}")
                # No stream/clear fires on pause, so drop the frozen strips here.
                if changed and payload.playback_state != PlaybackStateType.PLAYING:
                    self._clear_visualizer_timelines()

    def _handle_server_state(self, payload: ServerStatePayload) -> None:
        """Handle server/state messages with controller state."""
        assert self._ui is not None
        state = self._state
        ui = self._ui
        if payload.controller is None or isinstance(payload.controller, UndefinedField):
            return

        controller = payload.controller
        state.supported_commands = set(controller.supported_commands)

        volume_changed = controller.volume != state.volume
        mute_changed = controller.muted != state.muted

        if volume_changed:
            state.volume = controller.volume
            ui.add_event(f"Volume: {controller.volume}%")
        if mute_changed:
            state.muted = controller.muted
            ui.add_event("Muted" if controller.muted else "Unmuted")

        if volume_changed or mute_changed:
            ui.set_volume(state.volume, muted=state.muted)

    def _handle_server_command(self, payload: ServerCommandPayload) -> None:
        """Handle server/command messages for player volume/mute control."""
        if payload.player is None:
            return

        assert self._audio_handler is not None
        assert self._ui is not None
        player_cmd: PlayerCommandPayload = payload.player

        if player_cmd.command == PlayerCommand.VOLUME and player_cmd.volume is not None:
            self._audio_handler.set_volume(player_cmd.volume, muted=self._audio_handler.muted)
            self._ui.add_event(f"Server set player volume: {player_cmd.volume}%")
        elif player_cmd.command == PlayerCommand.MUTE and player_cmd.mute is not None:
            self._audio_handler.set_volume(self._audio_handler.volume, muted=player_cmd.mute)
            self._ui.add_event(
                "Server muted player" if player_cmd.mute else "Server unmuted player"
            )
        elif (
            player_cmd.command == PlayerCommand.SET_STATIC_DELAY
            and player_cmd.static_delay_ms is not None
        ):
            # Client library already applied the delay change;
            # notify audio worker so sync correction adjusts timing gradually
            assert self._client is not None
            assert self._audio_handler is not None
            new_delay_ms = self._client.static_delay_ms
            delta_us = int((new_delay_ms - self._applied_delay_ms) * 1000)
            if delta_us != 0:
                self._audio_handler.notify_delay_change(delta_us)
            self._applied_delay_ms = new_delay_ms
            self._ui.set_delay(new_delay_ms)
            self._settings.update(static_delay_ms=new_delay_ms)
            self._ui.add_event(f"Server set delay: {player_cmd.static_delay_ms}ms")

    def _handle_format_change(
        self, codec: str | None, sample_rate: int, bit_depth: int, channels: int
    ) -> None:
        """Handle audio format changes by updating the UI."""
        assert self._ui is not None
        self._ui.set_audio_format(codec, sample_rate, bit_depth, channels)

    def _set_applied_delay(self, delay_ms: float) -> None:
        """Mirror the locally-applied static delay so SET_STATIC_DELAY deltas stay correct."""
        self._applied_delay_ms = delay_ms

    async def _toggle_visualizer(self) -> None:
        """Toggle the visualizer on/off, reconnecting with updated roles."""
        assert self._ui is not None

        self._visualizer_enabled = not self._visualizer_enabled
        self._settings.update(visualizer=self._visualizer_enabled)
        self._ui.set_visualizer_enabled(self._visualizer_enabled)

        old_client = self._client
        self._detach_client()  # detach from old (still self._client)

        self._client = self._create_client()
        self._attach_client()  # attach to new self._client

        if old_client is not None:
            # Reuse server-switch mechanism so the connection loop treats the
            # client swap as a reconnect (prevents CancelledError propagation
            # when a connect is in-flight).
            if self._state.selected_server:
                self._connection_manager.set_pending_server(self._state.selected_server)
            if not self._cancel_connect():
                await old_client.disconnect()

    def _server_now_us(self) -> int:
        """Current time in the server clock the beat and peak strips are drawn against.

        Strip events carry server-clock timestamps, so the strip's "now" must be
        in the same domain. This is the inverse of ``compute_play_time``, so an
        event lands on the playhead cell exactly when it becomes audible.
        """
        assert self._client is not None
        return self._client.compute_server_time(self._client.now_us())

    def _handle_pairing_abort(self, reason: PairAbortReason) -> None:
        """Surface a non-closing pairing abort (e.g. a mismatched code) to the user."""
        if self._ui is not None:
            self._ui.add_event(f"Pairing failed: {reason.value}")

    def _handle_stream_start(self, message: StreamStartMessage) -> None:
        """Record which visualizer types the server negotiated for this stream."""
        if self._ui is None:
            return
        config = message.payload.visualizer
        types = (
            frozenset(config.types) if isinstance(config, StreamStartVisualizer) else frozenset()
        )
        self._ui.set_visualizer_types(types)

    def _handle_artwork_update(self, image: PILImage | None) -> None:
        """Receive a decoded artwork image (or None to clear) from the handler."""
        if self._ui is None:
            return
        self._ui.state.artwork_image = image
        self._ui.state.artwork_generation += 1
        self._ui.refresh()

    def _handle_visualizer_frame(self, frame: VisualizerFrame) -> None:
        """Handle a visualizer frame from the connector."""
        if self._ui is not None:
            self._ui.set_visualizer_frame(
                frame.spectrum,
                frame.loudness,
                frame.pitch_midi_q88,
                frame.f_peak_freq,
            )

    def _handle_beat(self, beat: BeatTiming) -> None:
        """Handle a beat hitting the playhead."""
        if self._ui is not None:
            self._ui.record_beat(beat)

    def _handle_beat_schedule(self, scheduled: list[BeatTiming]) -> None:
        """Handle an updated upcoming-beat schedule."""
        if self._ui is not None:
            self._ui.set_beat_schedule(scheduled)

    def _handle_peak(self, timestamp_us: int, strength: int) -> None:
        """Handle an energy-onset peak hitting the playhead."""
        if self._ui is not None:
            self._ui.record_peak(PeakEvent(timestamp_us=timestamp_us, strength=strength))

    def _handle_peak_schedule(self, scheduled: list[tuple[int, int]]) -> None:
        """Handle an updated upcoming-peak schedule."""
        if self._ui is not None:
            self._ui.set_peak_schedule(
                [PeakEvent(timestamp_us=ts, strength=strength) for ts, strength in scheduled]
            )

    def _on_stream_event(self, event: str) -> None:
        """Handle stream lifecycle events by running hooks."""
        hook = self._args.hook_start if event == "start" else self._args.hook_stop
        if not hook:
            return
        assert self._client is not None
        server = self._state.selected_server
        server_info = self._client.server_info
        create_task(
            run_hook(
                hook,
                event=event,
                server_id=server_info.server_id if server_info else None,
                server_name=server_info.name if server_info else None,
                server_url=server.url if server else None,
                client_id=self._args.identity.peer_id,
                client_name=self._args.client_name,
            )
        )
