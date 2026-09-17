"""Daemon mode for running a Sendspin client without UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

from aiohttp import ClientError, web
from aiosendspin.client import ClientListener, PairingSupport, SendspinClient
from aiosendspin.models.core import GroupUpdateServerPayload, ServerCommandPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin_mpris import MPRIS_AVAILABLE, SendspinMpris
from aiosendspin.models.types import (
    Activity,
    GoodbyeReason,
    PairAbortReason,
    PlaybackStateType,
    PlayerCommand,
    Roles,
)

from sendspin.audio_devices import AudioDevice, detect_supported_audio_formats
from sendspin.audio_connector import AudioStreamHandler
from sendspin.hooks import run_hook
from sendspin.settings import ClientSettings
from sendspin.utils import create_task, get_device_info

if TYPE_CHECKING:
    from aiosendspin.noise import ClientPairingStore, Identity

    from sendspin.volume_controller import VolumeController

logger = logging.getLogger(__name__)


@dataclass
class DaemonArgs:
    """Configuration for the Sendspin daemon."""

    audio_device: AudioDevice
    identity: Identity
    pairing_store: ClientPairingStore
    client_name: str
    settings: ClientSettings
    url: str | None = None
    static_delay_ms: float | None = None
    listen_port: int = 8928
    use_mpris: bool = True
    preferred_format: SupportedAudioFormat | None = None
    volume_controller: VolumeController | None = None
    hook_start: str | None = None
    hook_stop: str | None = None
    manufacturer: str | None = None
    product_name: str | None = None
    interface: str | None = None


class SendspinDaemon:
    """Sendspin daemon - headless audio player mode.

    When a URL is provided, the daemon connects to that server (client-initiated).
    When no URL is provided, the daemon listens for incoming server connections
    and advertises itself via mDNS (server-initiated connections).
    """

    def __init__(self, args: DaemonArgs) -> None:
        """Initialize the daemon."""
        self._args = args
        self._client: SendspinClient | None = None
        self._listener: ClientListener | None = None
        self._audio_handler: AudioStreamHandler | None = None
        self._settings = args.settings
        self._mpris: SendspinMpris | None = None
        # Currently-applied static delay in milliseconds, mirroring
        # `SendspinClient.static_delay_ms`. Tracked separately from settings
        # because CLI overrides aren't persisted to settings, so
        # `settings.static_delay_ms` can lag the value actually given to the client.
        self._static_delay_ms: float = 0.0
        self._connection_lock: asyncio.Lock | None = None
        self._server_url: str | None = None
        self._group_update_unsubscribe: Callable[[], None] | None = None
        self._server_command_unsubscribe: Callable[[], None] | None = None
        self._pairing_abort_unsubscribe: Callable[[], None] | None = None

    def _create_client(self) -> SendspinClient:
        """Create a new SendspinClient instance."""
        assert self._audio_handler is not None
        client_roles = [Roles.PLAYER]
        if MPRIS_AVAILABLE and self._args.use_mpris:
            client_roles.extend([Roles.METADATA, Roles.CONTROLLER])

        supported_formats = detect_supported_audio_formats(self._args.audio_device)
        if self._args.preferred_format is not None:
            supported_formats = [f for f in supported_formats if f != self._args.preferred_format]
            supported_formats.insert(0, self._args.preferred_format)

        return SendspinClient(
            identity=self._args.identity,
            client_name=self._args.client_name,
            roles=client_roles,
            pairing_store=self._args.pairing_store,
            pairing_support=self._build_pairing_support(),
            device_info=get_device_info(
                manufacturer=self._args.manufacturer,
                product_name=self._args.product_name,
            ),
            player_support=ClientHelloPlayerSupport(
                supported_formats=supported_formats,
                buffer_capacity=32_000_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
            static_delay_ms=self._static_delay_ms,
            state_supported_commands=[PlayerCommand.SET_STATIC_DELAY],
            initial_volume=self._audio_handler.volume,
            initial_muted=self._audio_handler.muted,
        )

    def _build_pairing_support(self) -> PairingSupport:
        """Build pairing-PIN out-channels for headless (log-based) display."""
        return PairingSupport(
            pin_display=self._show_pairing_code,
            offer_static_pin=True,
        )

    async def _show_pairing_code(self, code: str | None) -> None:
        """Log (or clear) the derived dynamic pairing PIN."""
        if code is not None:
            logger.info("Pairing required: enter PIN %s on the server.", code)

    def _handle_pairing_abort(self, reason: PairAbortReason) -> None:
        """Log a non-closing pairing abort (e.g. a mismatched code)."""
        logger.warning("Pairing failed: %s", reason.value)

    async def run(self) -> int:
        """Run the daemon."""
        logger.info("Starting Sendspin daemon: %s", self._args.identity.peer_id)
        loop = asyncio.get_running_loop()

        # Store reference to current task so it can be cancelled on shutdown
        main_task = asyncio.current_task()
        assert main_task is not None

        def signal_handler() -> None:
            logger.debug("Received interrupt signal, shutting down...")
            main_task.cancel()

        # Register signal handlers
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)

        # CLI arg overrides settings for static delay
        delay = (
            self._args.static_delay_ms
            if self._args.static_delay_ms is not None
            else self._settings.static_delay_ms
        )
        self._static_delay_ms = max(0.0, min(5000.0, delay))

        self._audio_handler = AudioStreamHandler(
            audio_device=self._args.audio_device,
            volume=self._settings.player_volume,
            muted=self._settings.player_muted,
            on_event=self._on_stream_event,
            on_format_change=self._handle_format_change,
            on_volume_change=self._on_volume_change,
            volume_controller=self._args.volume_controller,
        )
        await self._audio_handler.read_initial_volume()
        await self._audio_handler.start_volume_monitor()

        try:
            if self._args.url is not None:
                # Client-initiated connection mode
                await self._run_client_initiated()
            else:
                # Server-initiated connection mode (listen for incoming connections)
                await self._run_server_initiated()
        except asyncio.CancelledError:
            logger.debug("Daemon cancelled")
        finally:
            if self._mpris is not None:
                self._mpris.stop()
                self._mpris = None
            if self._audio_handler is not None:
                await self._audio_handler.shutdown()
            if self._client is not None:
                await self._client.disconnect()
                self._client = None
            if self._listener is not None:
                await self._listener.stop()
                self._listener = None
            if self._settings:
                await self._settings.flush()
            logger.info("Daemon stopped")

        return 0

    def _on_volume_change(self, volume: int, muted: bool) -> None:
        """Handle volume changes from any source (server command, external, etc.)."""
        assert self._settings is not None
        assert self._audio_handler is not None

        if not self._audio_handler.uses_external_volume_controller:
            self._settings.update(player_volume=volume, player_muted=muted)

    async def _run_client_initiated(self) -> None:
        """Run in client-initiated mode, connecting to a specific URL."""
        assert self._args.url is not None
        assert self._audio_handler is not None
        client = self._create_client()
        self._server_url = self._args.url
        self._attach_client(client)
        await self._connection_loop(self._args.url)

    async def _run_server_initiated(self) -> None:
        """Run in server-initiated mode, listening for incoming connections."""
        logger.info(
            "Listening for server connections on port %d (mDNS: _sendspin._tcp.local.)",
            self._args.listen_port,
        )

        self._connection_lock = asyncio.Lock()

        self._listener = ClientListener(
            client_id=self._args.identity.peer_id,
            on_connection=self._handle_server_connection,
            port=self._args.listen_port,
            client_name=self._args.client_name,
            host=self._args.interface if self._args.interface is not None else "0.0.0.0",
        )
        await self._listener.start()

        # Keep running until cancelled
        while True:
            await asyncio.sleep(3600)

    def _attach_client(self, client: SendspinClient) -> None:
        """Attach listeners, audio handler, and MPRIS to a client."""
        assert self._audio_handler is not None
        self._client = client
        self._audio_handler.attach_client(client)
        self._server_command_unsubscribe = client.add_server_command_listener(
            self._handle_server_command
        )
        self._group_update_unsubscribe = client.add_group_update_listener(self._on_group_update)
        self._pairing_abort_unsubscribe = client.add_pairing_abort_listener(
            self._handle_pairing_abort
        )
        if MPRIS_AVAILABLE and self._args.use_mpris:
            self._mpris = SendspinMpris(client)
            self._mpris.start()

    def _detach_client(self) -> None:
        """Detach listeners, audio handler, and MPRIS from the current client."""
        if self._server_command_unsubscribe is not None:
            self._server_command_unsubscribe()
            self._server_command_unsubscribe = None
        if self._group_update_unsubscribe is not None:
            self._group_update_unsubscribe()
            self._group_update_unsubscribe = None
        if self._pairing_abort_unsubscribe is not None:
            self._pairing_abort_unsubscribe()
            self._pairing_abort_unsubscribe = None
        if self._mpris is not None:
            self._mpris.stop()
            self._mpris = None
        if self._audio_handler is not None:
            self._audio_handler.detach_client()

    async def _handle_disconnect(self) -> None:
        """Reset audio state after a connection drop."""
        if self._audio_handler is not None:
            await self._audio_handler.handle_disconnect()

    def _should_switch_to_new_server(
        self, old_client: SendspinClient, new_client: SendspinClient
    ) -> bool:
        """Decide whether to switch to a new server per the multi-server spec.

        Assumes both clients have completed their handshake.
        """
        assert new_client.server_info is not None

        # Old client may have disconnected before we acquired the lock.
        if old_client.server_info is None:
            return True

        if new_client.server_info.server_id == old_client.server_info.server_id:
            return True

        new_is_playback = Activity.PLAYBACK in new_client.activities
        old_is_playback = Activity.PLAYBACK in old_client.activities

        if new_is_playback:
            return True
        if old_is_playback:
            return False

        # Both 'discovery' — prefer last played server.
        if self._settings.last_played_server_id == new_client.server_info.server_id:
            return True

        return False

    def _on_group_update(self, payload: GroupUpdateServerPayload) -> None:
        """Track last played server for multi-server arbitration."""
        if payload.playback_state != PlaybackStateType.PLAYING:
            return
        if self._client is None or self._client.server_info is None:
            return
        server_id = self._client.server_info.server_id
        if self._settings.last_played_server_id != server_id:
            self._settings.update(last_played_server_id=server_id)

    async def _handle_server_connection(self, ws: web.WebSocketResponse) -> None:
        """Handle an incoming server connection.

        ``SendspinClient.attach_websocket()`` now blocks for the connection's
        entire lifetime once admitted (it only returns when the connection
        closes), so it's driven as a background task here and ``client.connected``
        is polled to learn when admission completes — the same pattern
        aiosendspin's own tests use for server-initiated dials.
        """
        logger.info("Server connected")
        assert self._audio_handler is not None
        assert self._connection_lock is not None
        assert self._settings is not None

        client = self._create_client()
        attach_task = create_task(client.attach_websocket(ws))

        try:
            async with asyncio.timeout(30):
                while not client.connected and not attach_task.done():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            logger.warning("Handshake with server timed out")
            attach_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await attach_task
            return

        if not client.connected:
            # attach_task finished without admitting — either rejected/failed bring-up
            # (raises, logged below) or its own internal handshake timeout elapsed
            # (aiosendspin catches that internally and returns normally, so there's no
            # exception to report; without this the whole attempt is silent).
            exc = attach_task.exception() if attach_task.done() else None
            if exc is not None:
                logger.warning("Handshake with server failed: %s", exc)
            else:
                logger.info(
                    "Incoming connection did not complete admission "
                    "(handshake timed out or was aborted); waiting for a retry."
                )
            return

        # Lock ensures we wait for any in-progress handshake to complete
        # before disconnecting the previous server
        async with self._connection_lock:
            old_client = self._client

            # Decide which server to keep.
            if old_client is not None:
                if self._should_switch_to_new_server(old_client, client):
                    assert client.server_info is not None
                    logger.info(
                        "Switching to server '%s' (activities=%s)",
                        client.server_info.name,
                        [a.value for a in client.activities],
                    )
                    self._detach_client()
                    await self._handle_disconnect()
                    await old_client.disconnect(GoodbyeReason.ANOTHER_SERVER)
                else:
                    assert old_client.server_info is not None
                    assert client.server_info is not None
                    logger.info(
                        "Keeping server '%s', rejecting '%s' (activities=%s)",
                        old_client.server_info.name,
                        client.server_info.name,
                        [a.value for a in client.activities],
                    )
                    await client.disconnect(GoodbyeReason.ANOTHER_SERVER)
                    with contextlib.suppress(Exception):
                        await attach_task
                    return

            self._attach_client(client)

        # Handshake complete, release lock so new connections can proceed.
        # attach_task only resolves once the connection actually closes.
        try:
            await attach_task
            logger.info("Server disconnected")
        except Exception:
            logger.exception("Error while connection was active")
        finally:
            # Only cleanup if we're still the active client (not replaced by new connection)
            if self._client is client:
                self._detach_client()
                await self._handle_disconnect()

    async def _connection_loop(self, url: str) -> None:
        """Run the connection loop with automatic reconnection (client-initiated mode)."""
        assert self._client is not None
        assert self._audio_handler is not None
        assert self._settings is not None
        error_backoff = 1.0
        max_backoff = 300.0

        while True:
            try:
                await self._client.connect(url)
                logger.info("Connected to %s", url)
                error_backoff = 1.0

                # Wait for disconnect
                disconnect_event: asyncio.Event = asyncio.Event()
                unsubscribe = self._client.add_disconnect_listener(disconnect_event.set)
                await disconnect_event.wait()
                unsubscribe()

                # Connection dropped
                logger.info("Disconnected from server")
                await self._handle_disconnect()

                logger.info("Reconnecting to %s", url)

            except (TimeoutError, OSError, ClientError) as e:
                logger.warning(
                    "Connection error (%s), retrying in %.0fs",
                    type(e).__name__,
                    error_backoff,
                )

                await asyncio.sleep(error_backoff)
                error_backoff = min(error_backoff * 2, max_backoff)

            except Exception:
                logger.exception("Unexpected error during connection")
                break

    def _handle_server_command(self, payload: ServerCommandPayload) -> None:
        """Handle server commands for player volume/mute control."""
        if payload.player is None or self._settings is None:
            return

        assert self._audio_handler is not None
        player_cmd = payload.player

        if player_cmd.command == PlayerCommand.VOLUME and player_cmd.volume is not None:
            self._audio_handler.set_volume(player_cmd.volume, muted=self._audio_handler.muted)
            logger.info("Server set player volume: %d%%", player_cmd.volume)
        elif player_cmd.command == PlayerCommand.MUTE and player_cmd.mute is not None:
            self._audio_handler.set_volume(self._audio_handler.volume, muted=player_cmd.mute)
            logger.info("Server %s player", "muted" if player_cmd.mute else "unmuted")
        elif (
            player_cmd.command == PlayerCommand.SET_STATIC_DELAY
            and player_cmd.static_delay_ms is not None
        ):
            # Client library already applied the delay change;
            # notify audio worker so sync correction adjusts timing gradually
            assert self._client is not None
            new_delay_ms = self._client.static_delay_ms
            delta_us = int((new_delay_ms - self._static_delay_ms) * 1000)
            if delta_us != 0:
                self._audio_handler.notify_delay_change(delta_us)
            self._static_delay_ms = new_delay_ms
            self._settings.update(static_delay_ms=new_delay_ms)
            logger.info("Server set delay: %dms", player_cmd.static_delay_ms)

    def _handle_format_change(
        self, codec: str | None, sample_rate: int, bit_depth: int, channels: int
    ) -> None:
        """Log audio format changes."""
        logger.info(
            "Audio format: %s %dHz/%d-bit/%dch",
            codec or "PCM",
            sample_rate,
            bit_depth,
            channels,
        )

    def _on_stream_event(self, event: str) -> None:
        """Handle stream lifecycle events by running hooks."""
        hook = self._args.hook_start if event == "start" else self._args.hook_stop
        if not hook:
            return
        server_info = self._client.server_info if self._client else None
        create_task(
            run_hook(
                hook,
                event=event,
                server_id=server_info.server_id if server_info else None,
                server_name=server_info.name if server_info else None,
                server_url=self._server_url,
                client_id=self._args.identity.peer_id,
                client_name=self._args.client_name,
            )
        )
