"""Chromecast connection handler for Sendspin server."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pychromecast

if TYPE_CHECKING:
    from zeroconf import Zeroconf

logger = logging.getLogger(__name__)

# Sendspin Cast receiver app ID (registered with Google)
SENDSPIN_CAST_APP_ID = "938CBF87"
SENDSPIN_CAST_NAMESPACE = "urn:x-cast:sendspin"

# Upper bound on the blocking Chromecast discovery/connection step.
_CONNECT_TIMEOUT_SECONDS = 15.0


@dataclass
class ChromecastClient:
    """Represents a connected Chromecast device."""

    host: str
    port: int
    cast: pychromecast.Chromecast
    friendly_name: str


def parse_cast_url(url: str) -> tuple[str, int]:
    """Parse a cast:// URL and return host and port.

    Args:
        url: A cast:// URL (e.g., cast://192.168.1.123:8009)

    Returns:
        Tuple of (host, port)

    Raises:
        ValueError: If the URL is invalid
    """
    parsed = urlparse(url)
    if parsed.scheme != "cast":
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}, expected 'cast'")
    if not parsed.hostname:
        raise ValueError("URL must contain a hostname")
    return parsed.hostname, parsed.port or 8009


async def connect_to_chromecast(
    url: str,
    server_url: str,
    player_id: str,
    player_name: str | None = None,
    sync_delay: int = 0,
    codec: str = "flac",
    zeroconf: Zeroconf | None = None,
) -> ChromecastClient:
    """Connect to a Chromecast device and launch the Sendspin app.

    Args:
        url: The cast:// URL of the Chromecast device
        server_url: The HTTP URL of the Sendspin server (e.g., http://192.168.1.100:8928)
        player_id: Unique player ID for this Chromecast client
        player_name: Optional friendly name for the player (defaults to Chromecast name)
        sync_delay: Sync delay in milliseconds (default 0)
        codec: Audio codec to use (default "flac")
        zeroconf: Optional shared Zeroconf instance

    Returns:
        ChromecastClient instance

    Raises:
        ConnectionError: If unable to connect to the Chromecast
        TimeoutError: If connection or app launch times out
    """
    host, port = parse_cast_url(url)
    logger.info("Connecting to Chromecast at %s:%d", host, port)

    loop = asyncio.get_running_loop()

    # Connect to Chromecast (blocking call, run in executor)
    browsers: list[Any] = []

    def _connect() -> pychromecast.Chromecast:
        chromecasts, browser = pychromecast.get_chromecasts(
            known_hosts=[host],
            zeroconf_instance=zeroconf,
        )
        # Keep a handle to the browser: asyncio.wait_for() below cannot cancel
        # this executor thread, so the timeout path needs it to stop discovery.
        browsers.append(browser)

        try:
            # Find the Chromecast matching our target host
            cast = next((cc for cc in chromecasts if cc.cast_info.host == host), None)
            if cast is None:
                raise ConnectionError(f"Could not find Chromecast at {host}")

            # Wait for socket client to initialize before stopping discovery
            cast.wait()
            return cast
        finally:
            browser.stop_discovery()

    try:
        cast = await asyncio.wait_for(
            loop.run_in_executor(None, _connect),
            timeout=_CONNECT_TIMEOUT_SECONDS,
        )
    except TimeoutError as e:
        # wait_for() cannot cancel a running executor thread, so the blocking
        # discovery may still be in flight. Stop its zeroconf browser here;
        # otherwise it (with its sockets and threads) outlives the failure.
        for browser in browsers:
            browser.stop_discovery()
        raise TimeoutError(f"Timeout connecting to Chromecast at {host}") from e

    friendly_name = cast.cast_info.friendly_name or f"Chromecast ({host})"
    logger.info("Connected to Chromecast: %s", friendly_name)

    # Launch the Sendspin Cast app
    await _launch_sendspin_app(cast, loop)

    # Send the server URL to the Chromecast
    actual_player_name = player_name or f"{friendly_name} (Sendspin)"
    await _send_sendspin_config(
        cast,
        loop,
        server_url=server_url,
        player_id=player_id,
        player_name=actual_player_name,
        sync_delay=sync_delay,
        codec=codec,
    )

    return ChromecastClient(
        host=host,
        port=port,
        cast=cast,
        friendly_name=friendly_name,
    )


async def _launch_sendspin_app(
    cast: pychromecast.Chromecast,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Launch the Sendspin Cast receiver app on the Chromecast."""
    if cast.app_id == SENDSPIN_CAST_APP_ID:
        logger.debug("Sendspin Cast App already active.")
        return

    event = asyncio.Event()
    launch_success = False
    launch_error: str | None = None

    def launched_callback(success: bool, response: dict[str, Any] | None) -> None:
        nonlocal launch_success, launch_error
        launch_success = success
        if not success:
            launch_error = str(response)
            logger.warning("Failed to launch Sendspin Cast App: %s", response)
        else:
            logger.debug("Sendspin Cast App launched successfully.")
        loop.call_soon_threadsafe(event.set)

    def launch() -> None:
        # Quit the previous app before starting Sendspin receiver
        if cast.app_id is not None:
            cast.quit_app()
        logger.info("Launching Sendspin Cast App %s", SENDSPIN_CAST_APP_ID)
        cast.socket_client.receiver_controller.launch_app(
            SENDSPIN_CAST_APP_ID,
            force_launch=True,
            callback_function=launched_callback,
        )

    await loop.run_in_executor(None, launch)
    try:
        await asyncio.wait_for(event.wait(), timeout=10.0)
    except TimeoutError as e:
        raise TimeoutError("Timeout waiting for Sendspin Cast App to launch") from e

    if not launch_success:
        raise ConnectionError(f"Failed to launch Sendspin Cast App: {launch_error}")


async def _send_sendspin_config(
    cast: pychromecast.Chromecast,
    loop: asyncio.AbstractEventLoop,
    server_url: str,
    player_id: str,
    player_name: str,
    sync_delay: int,
    codec: str,
) -> None:
    """Send the Sendspin server configuration to the Cast receiver."""
    codecs = [codec]

    def send_message() -> None:
        cast.socket_client.send_app_message(
            SENDSPIN_CAST_NAMESPACE,
            {
                "serverUrl": server_url,
                "playerId": player_id,
                "playerName": player_name,
                "syncDelay": sync_delay,
                "codecs": codecs,
            },
        )

    logger.info(
        "Sending Sendspin config to Cast receiver: url=%s, id=%s, name=%s, syncDelay=%dms",
        server_url,
        player_id,
        player_name,
        sync_delay,
    )
    await loop.run_in_executor(None, send_message)


async def disconnect_chromecast(client: ChromecastClient) -> None:
    """Disconnect from a Chromecast device.

    Args:
        client: The ChromecastClient to disconnect
    """
    loop = asyncio.get_running_loop()

    def _disconnect() -> None:
        try:
            client.cast.quit_app()
            client.cast.disconnect()
        except Exception:
            logger.debug("Error disconnecting from Chromecast", exc_info=True)

    await loop.run_in_executor(None, _disconnect)
    logger.info("Disconnected from Chromecast: %s", client.friendly_name)
