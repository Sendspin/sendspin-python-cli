"""Sendspin server application."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import re
import signal
import socket
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import qrcode
from aiosendspin.noise import Identity, InMemoryServerPairingStore
from aiosendspin.server import (
    ClientAddedEvent,
    ClientRemovedEvent,
    SendspinEvent,
    SendspinServer,
    SendspinGroup,
)
from aiosendspin.server.push_stream import PushStream

from sendspin.utils import create_task

from .server import SendspinPlayerServer
from .source import AudioSource, decode_audio

if TYPE_CHECKING:
    from .chromecast import ChromecastClient

logger = logging.getLogger(__name__)

CAST_INSTALL_HINT = "Install the optional cast extra with `pip install 'sendspin[cast]'`."


def _load_chromecast_support() -> Any:
    """Import Chromecast support only when it is explicitly used."""
    try:
        from . import chromecast
    except ModuleNotFoundError as exc:
        if exc.name not in {"pychromecast", "pychromecast.discovery"}:
            raise
        raise RuntimeError(
            f"Chromecast support requires the optional 'cast' extra. {CAST_INSTALL_HINT}"
        ) from exc
    return chromecast


def print_qr_code(url: str) -> None:
    """Print a QR code to the console."""
    qr = qrcode.QRCode(
        error_correction=qrcode.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def get_local_ip() -> str:
    """Get the local IP address of this machine on the LAN."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip: str = s.getsockname()[0]
            return ip
    except Exception:
        return "localhost"


@dataclass
class ServeConfig:
    """Configuration for the serve command."""

    source: str
    source_format: str | None = None
    port: int = 8927
    name: str = "Sendspin Server"
    clients: list[str] | None = None


def _windows_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
    """Suppress ConnectionResetError on Windows during socket shutdown.

    On Windows, the ProactorEventLoop raises ConnectionResetError (WinError 10054)
    when a client disconnects and asyncio tries to shut down the socket. This is
    harmless but produces noisy error messages.
    """
    exception = context.get("exception")
    if isinstance(exception, ConnectionResetError):
        # Silently ignore - this is expected when clients disconnect
        return
    # For all other exceptions, use the default handler
    loop.default_exception_handler(context)


async def _stream_audio(stream: PushStream, source: AudioSource) -> None:
    """Push decoded PCM audio into a PushStream until cancelled."""
    try:
        async for pcm_chunk in source.generator:
            stream.prepare_audio(pcm_chunk, source.format)
            await stream.commit_audio()
            await stream.sleep_to_limit_buffer(max_buffer_us=5_000_000)
    finally:
        stream.stop()


async def run_server(config: ServeConfig) -> int:
    """Run the Sendspin server with the given audio source."""
    event_loop = asyncio.get_event_loop()

    # On Windows, suppress ConnectionResetError during client disconnect
    # Background: https://github.com/Sendspin/sendspin-cli/pull/26
    if sys.platform == "win32":
        event_loop.set_exception_handler(_windows_exception_handler)

    server = SendspinPlayerServer(
        loop=event_loop,
        identity=Identity.generate(),
        server_name=config.name,
        pairing_store=InMemoryServerPairingStore(),
    )

    client_connected = asyncio.Event()
    active_group: SendspinGroup | None = None
    play_media_task: asyncio.Task[None] | None = None
    shutdown_requested = False

    def handle_sigint() -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\nShutting down...")
        if play_media_task is not None:
            play_media_task.cancel()
        if not client_connected.is_set():
            client_connected.set()

    with suppress(NotImplementedError):
        event_loop.add_signal_handler(signal.SIGINT, handle_sigint)

    def on_server_event(server: SendspinServer, event: SendspinEvent) -> None:
        nonlocal active_group

        if isinstance(event, ClientAddedEvent):
            client = server.get_client(event.client_id)
            assert client is not None

            print("Client connected", event.client_id)

            if active_group is None:
                active_group = client.group
                client_connected.set()
                return

            create_task(active_group.add_client(client))

        if isinstance(event, ClientRemovedEvent):
            if active_group is None:
                return

            if (
                # Check no other clients left in the active group
                not [c for c in active_group.clients if c.client_id != event.client_id]
                and play_media_task is not None
            ):
                play_media_task.cancel()
                active_group = None

    server.add_event_listener(on_server_event)

    port = config.port
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if os.name == "posix" and sys.platform != "cygwin":
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
                s.bind(("", port))
                break
        except OSError as e:
            if e.errno == errno.EADDRINUSE and attempt < max_attempts - 1:
                port += 1
            else:
                raise
    else:
        raise OSError(f"Could not find available port after {max_attempts} attempts")

    await server.start_server(port=port, discover_clients=False)

    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:{port}"

    # Track connected Chromecast clients for cleanup
    chromecast_clients: list[ChromecastClient] = []

    # Connect to specified clients
    if config.clients:
        for client_url in config.clients:
            try:
                print(f"Connecting to client: {client_url}")
                if client_url.startswith("cast://"):
                    chromecast = _load_chromecast_support()

                    host, _ = chromecast.parse_cast_url(client_url)
                    # Replace non-alphanumeric chars with dashes (handles IPv4 and IPv6)
                    safe_host = re.sub(r"[^a-zA-Z0-9]", "-", host)
                    player_id = f"cast-{safe_host}"
                    cc_client = await chromecast.connect_to_chromecast(
                        url=client_url,
                        server_url=server_url,
                        player_id=player_id,
                    )
                    chromecast_clients.append(cc_client)
                    print(f"Chromecast connected: {cc_client.friendly_name}")
                else:
                    server.connect_to_client(client_url)
            except Exception as e:
                logger.warning("Failed to connect to client %s: %s", client_url, e)
                print(f"Warning: Failed to connect to client {client_url}: {e}")
    url = f"http://{local_ip}:{port}/"
    print(f"\nServer running at {url}")
    if local_ip == "localhost":
        print("Unable to print QR code because no LAN IP available\n")
        print("Open in browser to use the web player")
    else:
        print()
        print_qr_code(url)
        print()
        print("Scan QR to open in browser to use the web player")
    print("Or connect with any Sendspin client")
    print("Press Ctrl+C to quit\n")

    try:
        consecutive_errors = 0

        while not shutdown_requested:
            # Wait for a client to connect
            if not active_group:
                client_connected.clear()
                await client_connected.wait()

                if shutdown_requested:
                    break  # type: ignore[unreachable]

            assert active_group is not None

            # Decode and stream audio
            try:
                audio_source = await decode_audio(config.source, source_format=config.source_format)
                stream = active_group.start_stream()
                play_media_task = create_task(_stream_audio(stream, audio_source))
                await play_media_task
                consecutive_errors = 0
            except asyncio.CancelledError:
                pass
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except Exception as e:
                consecutive_errors += 1
                delay = min(2**consecutive_errors, 30)
                print(f"Playback error: {e}")
                logger.debug("Playback error", exc_info=True)
                print(f"Retrying in {delay}s...")
                play_media_task = create_task(asyncio.sleep(delay))
                try:
                    await play_media_task
                except asyncio.CancelledError:
                    pass

    finally:
        with suppress(Exception):
            if chromecast_clients:
                chromecast = _load_chromecast_support()

                for cc_client in chromecast_clients:
                    await chromecast.disconnect_chromecast(cc_client)

            await server.close()

    return 0


async def run_server_multi(config: ServeConfig, *, workers: int, log_level: str) -> int:
    """Run the multi-worker Sendspin server."""
    from sendspin.serve.coordinator import ServeCoordinator

    coordinator = ServeCoordinator(
        source=config.source,
        source_format=config.source_format,
        port=config.port,
        name=config.name,
        workers=workers,
        log_level=log_level,
    )
    return await coordinator.run()
