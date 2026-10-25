"""Daemon mode for running a Sendspin client without UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiohttp import ClientError, web
from aiosendspin.client import ClientListener, PairingSupport, SendspinClient
from aiosendspin.models.core import ServerCommandPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.noise.driver import HandshakeAbortedError
from aiosendspin_mpris import MPRIS_AVAILABLE, SendspinMpris
from aiosendspin.models.types import PairAbortReason, PlayerCommand, Roles

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
    output_delay_ms: float | None = None
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
        # Currently-applied output delay in milliseconds, mirroring
        # `SendspinClient.output_delay_ms`. Tracked separately from settings
        # because CLI overrides aren't persisted to settings, so
        # `settings.output_delay_ms` can lag the value actually given to the client.
        self._output_delay_ms: float = 0.0
        self._server_url: str | None = None
        self._admission_timeout_logged: bool = False

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
            ),
            output_delay_ms=self._output_delay_ms,
            state_supported_commands=[
                PlayerCommand.VOLUME,
                PlayerCommand.MUTE,
                PlayerCommand.SET_OUTPUT_DELAY,
            ],
            initial_volume=self._audio_handler.volume,
            initial_muted=self._audio_handler.muted,
        )

    def _build_pairing_support(self) -> PairingSupport:
        """Build pairing-code out-channels for headless (log-based) display."""
        return PairingSupport(
            pairing_code_display=self._show_pairing_code,
            offer_static_pairing_code=True,
        )

    async def _show_pairing_code(self, pairing_code: str | None, *, grouped: str | None) -> None:
        """Log (or clear) the derived dynamic pairing code."""
        if grouped is not None:
            logger.info("Pairing required: enter pairing code %s on the server.", grouped)

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

        # CLI arg overrides settings for output delay
        delay = (
            self._args.output_delay_ms
            if self._args.output_delay_ms is not None
            else self._settings.output_delay_ms
        )
        self._output_delay_ms = max(0.0, min(5000.0, delay))

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

        # One client receives every incoming connection, so aiosendspin applies the
        # spec's multi-server admission rules across them (connection.md, Multiple
        # servers): it ranks playback > pairing > idle, keeps a pairing attempt,
        # breaks idle ties by the last-playback server, and sends the displaced or
        # rejected server the matching goodbye or pair/abort.
        await self._seed_last_playback_server()
        client = self._create_client()
        self._attach_client(client)
        client.add_disconnect_listener(self._on_disconnect)

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
        client.add_server_command_listener(self._handle_server_command)
        client.add_pairing_abort_listener(self._handle_pairing_abort)
        if MPRIS_AVAILABLE and self._args.use_mpris:
            self._mpris = SendspinMpris(client)
            self._mpris.start()

    def _on_disconnect(self) -> None:
        """Reset audio once the admitted connection closes (server-initiated mode)."""
        create_task(self._handle_disconnect())

    async def _handle_disconnect(self) -> None:
        """Reset audio state after a connection drop."""
        if self._audio_handler is not None:
            await self._audio_handler.handle_disconnect()

    async def _seed_last_playback_server(self) -> None:
        """Carry the last-playback server over from settings into the pairing store.

        Before aiosendspin 10 the daemon tracked it in settings; the client now keeps it
        in the pairing store, where admission reads it to break ties between idle servers.
        """
        legacy = self._settings.last_played_server_id
        store = self._args.pairing_store
        if legacy is not None and await store.get_last_playback_server_id() is None:
            await store.set_last_playback_server_id(legacy)

    async def _handle_server_connection(self, ws: web.WebSocketResponse) -> None:
        """Hand an incoming server connection to the client and follow its admission.

        ``attach_websocket`` brings the connection up and arbitrates it against the
        current one. It returns early when the connection is rejected or fails bring-up;
        once admitted, it blocks until the connection closes. aiosendspin reports no
        event when an admitted connection displaces another, so a new ``server_info``
        object on the client is what shows this connection got in. The old server's
        audio is dropped then, before the new server can start a stream.
        """
        logger.info("Server connected")
        client = self._client
        assert client is not None

        previous = client.server_info
        attach_task = create_task(client.attach_websocket(ws))
        while not attach_task.done():
            current = client.server_info
            if current is not None and current is not previous:
                break
            await asyncio.sleep(0.01)
        else:
            self._log_unadmitted_connection(attach_task)
            return

        # Reaching here means the connection was admitted: re-arm the
        # first-time info log so a later wedged spell is visible again.
        self._admission_timeout_logged = False
        assert current is not None
        activities = [a.value for a in client.activities]
        if previous is not None:
            logger.info("Switching to server '%s' (activities=%s)", current.name, activities)
            await self._handle_disconnect()
        else:
            logger.info("Admitted server '%s' (activities=%s)", current.name, activities)

        try:
            await attach_task
            logger.info("Server '%s' disconnected", current.name)
        except Exception:
            logger.exception("Error while connection was active")

    def _log_unadmitted_connection(self, attach_task: asyncio.Task[None]) -> None:
        """Report an incoming connection that was rejected or never completed admission."""
        exc = attach_task.exception()
        if exc is not None:
            logger.warning("Handshake with server failed: %s", exc)
            return
        # Rejections and admission timeouts recur on every reconnect while a server
        # keeps redialing, so log the first at info (an unadmitted connection would
        # otherwise be fully silent) and repeats at debug. The flag resets once a
        # connection is admitted.
        log = logger.debug if self._admission_timeout_logged else logger.info
        log(
            "Incoming connection was not admitted (rejected for another server, "
            "handshake timed out or was aborted); waiting for a retry."
        )
        self._admission_timeout_logged = True

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

            except (
                TimeoutError,
                OSError,
                ClientError,
                RuntimeError,
                HandshakeAbortedError,
            ) as e:
                # Retry only the connection-failure set aiosendspin's own
                # client uses (HandshakeAbortedError, OSError, RuntimeError,
                # TimeoutError). RuntimeError covers "server activation
                # rejected" / "connection closed before admission". Anything
                # outside this set is a genuine bug and is left to propagate,
                # so the daemon exits non-zero and the shipped systemd unit
                # (Restart=on-failure) restarts it instead of the process
                # dying silently with exit code 0.
                logger.warning(
                    "Connection error (%s), retrying in %.0fs",
                    type(e).__name__,
                    error_backoff,
                )

                await asyncio.sleep(error_backoff)
                error_backoff = min(error_backoff * 2, max_backoff)

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
            player_cmd.command == PlayerCommand.SET_OUTPUT_DELAY
            and player_cmd.output_delay_ms is not None
        ):
            # Client library already applied the delay change;
            # notify audio worker so sync correction adjusts timing gradually
            assert self._client is not None
            new_delay_ms = self._client.output_delay_ms
            delta_us = int((new_delay_ms - self._output_delay_ms) * 1000)
            if delta_us != 0:
                self._audio_handler.notify_delay_change(delta_us)
            self._output_delay_ms = new_delay_ms
            self._settings.update(output_delay_ms=new_delay_ms)
            logger.info("Server set delay: %dms", player_cmd.output_delay_ms)

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
