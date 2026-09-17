"""Settings persistence for the Sendspin CLI.

This module provides persistent storage for player settings. Settings are
automatically loaded from disk and saved with debouncing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

from aiosendspin.noise import FileClientPairingStore, Identity, b64url_decode, b64url_encode

logger = logging.getLogger(__name__)

# Debounce delay for saving settings
SAVE_DEBOUNCE_SECONDS = 60.0


@dataclass
class BaseSettings:
    """Base class for settings with persistence support.

    Changes are debounced and saved after 60 seconds of inactivity,
    or immediately on flush().
    """

    # Common fields
    name: str | None = None
    log_level: str | None = None
    listen_port: int | None = None

    # Internal state (not serialized)
    _settings_file: Path | None = field(default=None, repr=False, compare=False)
    _debounce_save_handle: asyncio.TimerHandle | None = field(
        default=None, repr=False, compare=False
    )

    # Fields to exclude from serialization
    _internal_fields: ClassVar[set[str]] = {"_settings_file", "_debounce_save_handle"}

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary for serialization."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._internal_fields
        }

    def _update_fields(self, updates: dict[str, Any]) -> bool:
        """Update fields and return whether any changed."""
        changed = False
        for field_name, value in updates.items():
            if value is not None and getattr(self, field_name) != value:
                setattr(self, field_name, value)
                changed = True
        return changed

    async def load(self) -> None:
        """Load settings from disk."""
        loop = asyncio.get_running_loop()
        needs_save = await loop.run_in_executor(None, self._load)
        if needs_save:
            self._schedule_save()

    async def flush(self) -> None:
        """Immediately save any pending changes to disk."""
        if self._debounce_save_handle is not None:
            self._debounce_save_handle.cancel()
            self._debounce_save_handle = None
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save)

    def _schedule_save(self) -> None:
        """Schedule a debounced save operation."""
        if self._debounce_save_handle is not None:
            self._debounce_save_handle.cancel()

        loop = asyncio.get_running_loop()
        self._debounce_save_handle = loop.call_later(
            SAVE_DEBOUNCE_SECONDS, self._debounced_save, loop
        )

    def _debounced_save(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called by the timer to save settings in executor."""
        self._debounce_save_handle = None
        loop.run_in_executor(None, self._save)

    def _load(self) -> bool:
        """Load settings from the settings file (blocking I/O).

        Returns True if a save is needed (e.g., migration applied).
        """
        raise NotImplementedError

    def _save(self) -> None:
        """Save settings to the settings file (blocking I/O)."""
        if self._settings_file is None:
            return
        try:
            self._settings_file.parent.mkdir(parents=True, exist_ok=True)
            self._settings_file.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
            logger.debug("Saved settings to %s", self._settings_file)
        except OSError as e:
            logger.warning("Failed to save settings to %s: %s", self._settings_file, e)


@dataclass
class ClientSettings(BaseSettings):
    """Settings for TUI and daemon modes."""

    player_volume: int = 25
    player_muted: bool = False
    static_delay_ms: float = 0.0
    last_server_url: str | None = None
    audio_device: str | None = None
    use_mpris: bool = True
    audio_format: str | None = None
    use_hardware_volume: bool | None = None
    hook_set_volume: str | None = None
    hook_start: str | None = None
    hook_stop: str | None = None
    visualizer: bool = False
    manufacturer: str | None = None
    product_name: str | None = None
    last_played_server_id: str | None = None
    # IP address of the network interface to use for mDNS discovery and (in daemon
    # server-initiated mode) for binding the incoming-connection listener.
    interface: str | None = None
    # User-selected color theme ("dark" or "light"), re-engaged on next palette.
    color_mode: str | None = None

    def update(
        self,
        *,
        player_volume: int | None = None,
        player_muted: bool | None = None,
        static_delay_ms: float | None = None,
        last_server_url: str | None = None,
        name: str | None = None,
        audio_device: str | None = None,
        log_level: str | None = None,
        listen_port: int | None = None,
        use_mpris: bool | None = None,
        audio_format: str | None = None,
        use_hardware_volume: bool | None = None,
        hook_set_volume: str | None = None,
        hook_start: str | None = None,
        hook_stop: str | None = None,
        visualizer: bool | None = None,
        last_played_server_id: str | None = None,
        interface: str | None = None,
        color_mode: str | None = None,
    ) -> None:
        """Update settings fields. Only changed fields trigger a save."""
        changed = False

        # Handle player_volume separately due to clamping
        if player_volume is not None:
            player_volume = max(0, min(100, player_volume))
            if self.player_volume != player_volume:
                self.player_volume = player_volume
                changed = True

        # Handle other fields generically
        changed = (
            self._update_fields(
                {
                    "player_muted": player_muted,
                    "static_delay_ms": static_delay_ms,
                    "last_server_url": last_server_url,
                    "name": name,
                    "audio_device": audio_device,
                    "log_level": log_level,
                    "listen_port": listen_port,
                    "use_mpris": use_mpris,
                    "audio_format": audio_format,
                    "use_hardware_volume": use_hardware_volume,
                    "hook_set_volume": hook_set_volume,
                    "hook_start": hook_start,
                    "hook_stop": hook_stop,
                    "visualizer": visualizer,
                    "last_played_server_id": last_played_server_id,
                    "interface": interface,
                    "color_mode": color_mode,
                }
            )
            or changed
        )

        if changed:
            self._schedule_save()

    def _load(self) -> bool:
        """Load settings from the settings file (blocking I/O)."""
        if self._settings_file is None or not self._settings_file.exists():
            logger.debug("Settings file does not exist: %s", self._settings_file)
            return False

        try:
            data = json.loads(self._settings_file.read_text())
            # Update fields from loaded data
            self.name = data.get("name")
            self.log_level = data.get("log_level")
            self.listen_port = data.get("listen_port")
            self.player_volume = data.get("player_volume", 25)
            self.player_muted = data.get("player_muted", False)
            self.static_delay_ms = data.get("static_delay_ms", 0.0)
            # Clamp to valid range; also handles old negative sign convention.
            if self.static_delay_ms < 0:
                self.static_delay_ms = min(5000.0, -self.static_delay_ms)
            elif self.static_delay_ms > 5000:
                self.static_delay_ms = 5000.0
            self.last_server_url = data.get("last_server_url")
            self.audio_device = data.get("audio_device")
            self.use_mpris = data.get("use_mpris", True)
            self.audio_format = data.get("audio_format")
            self.use_hardware_volume = data.get("use_hardware_volume")
            self.hook_set_volume = data.get("hook_set_volume")
            self.hook_start = data.get("hook_start")
            self.hook_stop = data.get("hook_stop")
            self.visualizer = data.get("visualizer", False)
            self.manufacturer = data.get("manufacturer")
            self.product_name = data.get("product_name")
            self.last_played_server_id = data.get("last_played_server_id")
            self.interface = data.get("interface")
            self.color_mode = data.get("color_mode")
            logger.info(
                "Loaded settings from %s: volume=%d%%, muted=%s",
                self._settings_file,
                self.player_volume,
                self.player_muted,
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load settings from %s: %s", self._settings_file, e)
        return False


@dataclass
class ServeSettings(BaseSettings):
    """Settings for serve mode."""

    source: str | None = None
    source_format: str | None = None
    clients: list[str] | None = None

    def update(
        self,
        *,
        name: str | None = None,
        log_level: str | None = None,
        listen_port: int | None = None,
        source: str | None = None,
        source_format: str | None = None,
        clients: list[str] | None = None,
    ) -> None:
        """Update settings fields. Only changed fields trigger a save."""
        changed = self._update_fields(
            {
                "name": name,
                "log_level": log_level,
                "listen_port": listen_port,
                "source": source,
                "source_format": source_format,
                "clients": clients,
            }
        )

        if changed:
            self._schedule_save()

    def _load(self) -> bool:
        """Load settings from the settings file (blocking I/O)."""
        if self._settings_file is None or not self._settings_file.exists():
            logger.debug("Settings file does not exist: %s", self._settings_file)
            return False

        try:
            data = json.loads(self._settings_file.read_text())
            self.name = data.get("name")
            self.log_level = data.get("log_level")
            self.listen_port = data.get("listen_port")
            self.source = data.get("source")
            self.source_format = data.get("source_format")
            self.clients = data.get("clients")
            logger.info("Loaded settings from %s", self._settings_file)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load settings from %s: %s", self._settings_file, e)
        return False


async def get_client_settings(
    mode: Literal["tui", "daemon"], config_dir: str | None = None
) -> ClientSettings:
    """Create and load client settings for TUI or daemon mode.

    Args:
        mode: The client mode ("tui" or "daemon").
        config_dir: Optional directory to store settings. Defaults to ~/.config/sendspin.

    Returns:
        ClientSettings instance with settings loaded from disk.
    """
    config_path = Path(config_dir) if config_dir else Path.home() / ".config" / "sendspin"
    settings = ClientSettings(_settings_file=config_path / f"settings-{mode}.json")
    await settings.load()
    return settings


def _config_path(config_dir: str | None) -> Path:
    return Path(config_dir) if config_dir else Path.home() / ".config" / "sendspin"


def _warn_if_upgrading_from_legacy_client_id(settings_path: Path) -> None:
    """Warn once when a fresh identity replaces a pre-encryption install's client_id.

    Before pairing support, ``client_id`` was a stable, often user-chosen string
    (``--id``); it's now derived from the generated identity's public key instead,
    so an upgrade presents as a brand-new device to any server that keyed player
    records/groups off the old client_id (see sendspin-python-cli#277).
    """
    try:
        data = json.loads(settings_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    old_client_id = data.get("client_id")
    if old_client_id:
        logger.warning(
            "Generating a new client identity to replace the old client_id %r. "
            "Servers (e.g. Music Assistant) will see this as a new/different player "
            "than before this upgrade — you may need to re-add it to any zones, "
            "groups, or Home Assistant entities that referenced the old one, and "
            "pair it again if the server requires pairing.",
            old_client_id,
        )


def _load_identity(path: Path, settings_path: Path) -> Identity:
    """Load a persisted identity, generating and persisting a fresh one if absent."""
    try:
        data = json.loads(path.read_text())
        return Identity.from_private_bytes(b64url_decode(data["private_key"]))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError) as e:
        if not isinstance(e, FileNotFoundError):
            logger.warning("Failed to load identity from %s, generating a new one: %s", path, e)
        else:
            _warn_if_upgrading_from_legacy_client_id(settings_path)
        identity = Identity.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"private_key": b64url_encode(identity.private_bytes)}))
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        return identity


async def get_client_identity(
    mode: Literal["tui", "daemon"], config_dir: str | None = None
) -> Identity:
    """Load this client's persistent Noise identity, generating one on first run.

    The identity's public key (``identity.peer_id``) is the protocol ``client_id``
    servers see, so it must stay stable across restarts for pairing records and
    "last played server" arbitration to keep working.
    """
    config_path = _config_path(config_dir)
    path = config_path / f"identity-{mode}.json"
    settings_path = config_path / f"settings-{mode}.json"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _load_identity, path, settings_path)


async def get_client_pairing_store(
    mode: Literal["tui", "daemon"], config_dir: str | None = None
) -> FileClientPairingStore:
    """Open this client's persistent pairing-record store, seeding one on first run."""
    path = _config_path(config_dir) / f"pairing-store-{mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return await FileClientPairingStore.open(path)


async def get_serve_settings(config_dir: str | None = None) -> ServeSettings:
    """Create and load serve settings.

    Args:
        config_dir: Optional directory to store settings. Defaults to ~/.config/sendspin.

    Returns:
        ServeSettings instance with settings loaded from disk.
    """
    config_path = Path(config_dir) if config_dir else Path.home() / ".config" / "sendspin"
    settings = ServeSettings(_settings_file=config_path / "settings-serve.json")
    await settings.load()
    return settings
