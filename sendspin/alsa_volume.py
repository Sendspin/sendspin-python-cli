"""ALSA mixer volume control backend for Linux."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import sys
from typing import TYPE_CHECKING

from sendspin.volume_controller import VolumeChangeCallback

if TYPE_CHECKING:
    from sendspin.audio_devices import AudioDevice

logger = logging.getLogger(__name__)

AVAILABLE = sys.platform.startswith("linux") and shutil.which("amixer") is not None

_HW_CARD_RE = re.compile(r"\bhw:(\d+)")
_CARD_NAME_RE = re.compile(r"\bCARD=([^,\s]+)")

_SCONTROL_RE = re.compile(r"Simple mixer control '([^']+)'")
_VOLUME_RE = re.compile(r"\[(\d+)%\]")
_SWITCH_RE = re.compile(r"\[(on|off)\]")

_POLL_INTERVAL_S = 1.0

# Well-known ALSA mixer element names, in priority order.
# When multiple elements have playback volume, prefer these over others.
# - Digital: HiFiBerry DAC+/DAC2/Amp2, most I2S DAC HATs (PCM5122), TAS58xx amplifiers (Louder Raspberry)
# - Master: HiFiBerry Amp+, generic ALSA cards, many USB interfaces
# - PCM: bcm2835 headphones, some USB DACs
_PREFERRED_ELEMENTS: tuple[str, ...] = ("Digital", "Master", "PCM")

# PortAudio's ALSA host API enumerates the system default device using a
# bare alias with no card info ("sysdefault", "default"), distinct from the
# fully-qualified hints (e.g. "sysdefault:CARD=vc4hdmi") that `aplay -L`
# reports for the same underlying device.
_BARE_DEFAULT_NAMES = ("sysdefault", "default")


def _resolve_bare_default_card(bare_name: str) -> str | None:
    """Resolve a bare ALSA default alias to a card name via ``aplay -L``.

    PortAudio may report the default output device as a bare alias like
    "sysdefault" with no card info, while ``aplay -L`` (ALSA's own device-hint
    listing) reports the same device fully-qualified, e.g.
    "sysdefault:CARD=vc4hdmi". Cross-referencing recovers the card without
    guessing at ALSA's own default-card resolution rules (env vars, config
    overrides, etc.).

    Only resolves when exactly one hint matches ``<bare_name>:CARD=`` — on a
    system with multiple cards each exposing their own default hint, the
    match is ambiguous and can't be safely disambiguated from names alone.
    """
    from sendspin.audio_devices import list_alsa_devices

    prefix = f"{bare_name}:CARD="
    matches = [name for name, _ in list_alsa_devices() if name.startswith(prefix)]
    if len(matches) != 1:
        return None
    m = _CARD_NAME_RE.search(matches[0])
    return m.group(1) if m else None


def parse_alsa_card(device_name: str) -> int | str | None:
    """Extract the ALSA card identifier from a device name.

    PortAudio names hardware devices like:
      "snd_rpi_hifiberry_dacplus: ... (hw:1,0)"
    which gives a numeric card index.

    Raw ALSA device names (e.g. from ``--audio-device plughw:CARD=vc4hdmi,DEV=0``,
    used to reach the ``plug``/``dmix`` conversion layer that ``hw:N,M`` bypasses)
    identify the card by string name instead:
      "plughw:CARD=vc4hdmi,DEV=0"
    ``amixer -c`` accepts either form directly (it resolves a name via
    ``snd_card_get_index()`` internally), so both are returned as-is.

    Returns the card index or name, or None for virtual devices
    (pipewire, pulse, default, etc.) that don't reference a specific card.
    """
    m = _HW_CARD_RE.search(device_name)
    if m:
        return int(m.group(1))
    m = _CARD_NAME_RE.search(device_name)
    if m:
        return m.group(1)
    return None


async def _has_playback_volume(card: int | str, element: str) -> bool:
    """Check if an ALSA mixer element has playback volume capability.

    Accepts both ``pvolume`` (standard playback volume, e.g. HiFiBerry DAC+,
    USB interfaces) and ``volume`` (used by TAS58xx-based amplifier HATs such
    as the Sonocotta Louder Raspberry — reported as ``volume`` in stereo mode
    or ``volume volume-joined`` in mono mode).

    The bare ``volume`` capability is ambiguous: ALSA also tags bidirectional
    analog gain stages with it when a codec driver registers them via
    SOC_SINGLE_TLV/SOC_DOUBLE_TLV without a "Playback"/"Capture" qualifier in
    the control name (e.g. RT5616's "ADC Boost", "IN1 Boost", "IN2 Boost" — all
    capture-side mic/line boost gain, disconnected from the playback chain).
    Such elements report both a ``Playback channels:`` and a ``Capture
    channels:`` line. A bare ``volume`` match is therefore only accepted when
    the element is genuinely output-only (no ``Capture channels:`` line), which
    is the real TAS58xx case. The unambiguous ``pvolume`` branch is unaffected.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "amixer",
            "-M",
            "-c",
            str(card),
            "sget",
            element,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except FileNotFoundError:
        return False
    if proc.returncode != 0:
        return False
    output = stdout.decode()
    caps = set(output.split())
    if "pvolume" in caps:
        return True
    return "volume" in caps and "Capture channels:" not in output


async def find_mixer_element(card: int | str) -> str | None:
    """Discover the playback volume mixer element on an ALSA card.

    Runs ``amixer -c <card> scontrols``, then checks each element for
    playback volume capability (``pvolume`` or ``volume``).
    Prefers well-known element
    names (Digital, Master, PCM) when multiple elements have playback
    volume.  Returns the best match, or None if no element has playback
    volume control.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "amixer",
            "-c",
            str(card),
            "scontrols",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except FileNotFoundError:
        logger.debug("amixer not found on this system")
        return None

    if proc.returncode != 0:
        logger.debug("amixer -c %s scontrols failed (exit %d)", card, proc.returncode)
        return None

    available: list[str] = _SCONTROL_RE.findall(stdout.decode())
    if not available:
        logger.debug("ALSA card %s has no mixer controls", card)
        return None

    seen: set[str] = set()
    volume_elements: list[str] = []
    for element in available:
        if element in seen:
            continue
        seen.add(element)
        if await _has_playback_volume(card, element):
            volume_elements.append(element)

    if not volume_elements:
        logger.debug(
            "ALSA card %s: no playback volume element among %s",
            card,
            sorted(seen),
        )
        return None

    # Prefer well-known element names used by common DAC HATs.
    for preferred in _PREFERRED_ELEMENTS:
        if preferred in volume_elements:
            logger.debug("ALSA card %s: selected preferred mixer element %r", card, preferred)
            return preferred

    # Fallback: first element with playback volume (e.g. USB DACs with non-standard names).
    selected = volume_elements[0]
    logger.debug("ALSA card %s: selected mixer element %r", card, selected)
    return selected


async def async_check_alsa_available(
    audio_device: AudioDevice,
) -> tuple[int | str, str] | None:
    """Check if ALSA mixer volume control is available for a device.

    Falls back to resolving bare default aliases ("sysdefault", "default")
    against ``aplay -L`` hints when the device name itself carries no card
    info (see ``_resolve_bare_default_card``).

    Returns ``(card, mixer_element)`` if available, or None.
    """
    if not AVAILABLE:
        return None

    card = parse_alsa_card(audio_device.name)
    if card is None and audio_device.name in _BARE_DEFAULT_NAMES:
        card = _resolve_bare_default_card(audio_device.name)
    if card is None:
        return None

    element = await find_mixer_element(card)
    if element is None:
        return None

    return card, element


class AlsaVolumeController:
    """Controls audio volume directly via ALSA mixer using amixer.

    This bypasses PulseAudio/PipeWire and sets the hardware mixer element
    on the ALSA card, giving true hardware volume control on DAC HATs.
    """

    def __init__(self, card: int | str, element: str) -> None:
        self._card = str(card)
        self._element = element
        self._watch_task: asyncio.Task[None] | None = None

    async def set_state(self, volume: int, *, muted: bool) -> None:
        """Set ALSA mixer volume and mute state."""
        if not 0 <= volume <= 100:
            raise ValueError(f"Volume must be 0-100, got {volume}")

        mute_arg = "mute" if muted else "unmute"
        proc = await asyncio.create_subprocess_exec(
            "amixer",
            "-M",
            "-c",
            self._card,
            "sset",
            self._element,
            "playback",
            f"{volume}%",
            mute_arg,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"amixer sset failed (exit {proc.returncode}): "
                f"{stderr.decode().strip() if stderr else '(empty)'}"
            )

    async def get_state(self) -> tuple[int, bool]:
        """Read ALSA mixer volume and mute state."""
        proc = await asyncio.create_subprocess_exec(
            "amixer",
            "-M",
            "-c",
            self._card,
            "sget",
            self._element,
            "playback",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"amixer sget failed (exit {proc.returncode}): "
                f"{stderr.decode().strip() if stderr else '(empty)'}"
            )

        output = stdout.decode()
        vol_match = _VOLUME_RE.search(output)
        switch_match = _SWITCH_RE.search(output)

        if vol_match is None:
            raise RuntimeError(f"Could not parse volume from amixer output: {output!r}")

        volume = int(vol_match.group(1))
        muted = switch_match.group(1) == "off" if switch_match else False
        return volume, muted

    async def start_monitoring(self, callback: VolumeChangeCallback) -> None:
        """Start polling for external ALSA volume changes."""
        if self._watch_task is not None:
            return
        self._watch_task = asyncio.get_running_loop().create_task(self._poll_loop(callback))

    async def stop_monitoring(self) -> None:
        """Stop the monitoring poll loop."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None

    async def _poll_loop(self, callback: VolumeChangeCallback) -> None:
        """Poll amixer for volume changes and invoke callback on change."""
        while True:
            try:
                previous = await self.get_state()
            except RuntimeError:
                logger.debug("Failed to read initial ALSA volume, retrying...")
                await asyncio.sleep(2)
                continue
            break

        while True:
            await asyncio.sleep(_POLL_INTERVAL_S)
            try:
                current = await self.get_state()
            except RuntimeError:
                continue
            if current != previous:
                logger.debug("ALSA volume changed externally: %s -> %s", previous, current)
                previous = current
                volume, muted = current
                callback(volume, muted)
