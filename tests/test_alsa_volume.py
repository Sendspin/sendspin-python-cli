from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import NoReturn
from types import SimpleNamespace

import pytest

import sendspin.audio_devices as _audio_devices_mod
import sendspin.alsa_volume as _alsa_mod
from sendspin.alsa_volume import (
    AlsaVolumeController,
    _has_playback_volume,
    async_check_alsa_available,
    find_mixer_element,
    parse_alsa_card,
)


# -- Helpers ------------------------------------------------------------------


class _FakeProcess:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


_AmixerExecFactory = Callable[..., Awaitable[_FakeProcess]]


def _amixer_exec(stdout: str, returncode: int = 0) -> _AmixerExecFactory:
    """Return an async factory that produces a fake amixer process."""
    proc = _FakeProcess(returncode=returncode, stdout=stdout.encode())

    async def factory(*args: object, **kwargs: object) -> _FakeProcess:
        return proc

    return factory


# -- parse_alsa_card ----------------------------------------------------------


def test_parse_card_from_hw_device() -> None:
    name = "snd_rpi_hifiberry_dacplus: HiFiBerry DAC+ HiFi pcm512x-hifi-0 (hw:1,0)"
    assert parse_alsa_card(name) == 1


def test_parse_card_from_hw0() -> None:
    name = "bcm2835 Headphones: - (hw:0,0)"
    assert parse_alsa_card(name) == 0


def test_parse_card_returns_none_for_virtual_device() -> None:
    assert parse_alsa_card("pipewire") is None
    assert parse_alsa_card("default") is None
    assert parse_alsa_card("pulse") is None
    assert parse_alsa_card("dmix") is None


def test_parse_card_from_plughw_card_name() -> None:
    """Raw ALSA device names use CARD=<name> instead of a numeric hw:N index."""
    assert parse_alsa_card("plughw:CARD=vc4hdmi,DEV=0") == "vc4hdmi"


def test_parse_card_from_dmix_card_name() -> None:
    assert parse_alsa_card("dmix:CARD=vc4hdmi,DEV=0") == "vc4hdmi"


def test_parse_card_from_hw_card_name() -> None:
    assert parse_alsa_card("hw:CARD=Amp,DEV=0") == "Amp"


# -- find_mixer_element -------------------------------------------------------


async def test_find_mixer_element_digital(monkeypatch) -> None:
    """HiFiBerry DAC+ exposes 'Digital' as the volume control."""
    scontrols = (
        "Simple mixer control 'Analogue',0\n"
        "Simple mixer control 'Analogue Playback Boost',0\n"
        "Simple mixer control 'Auto Mute',0\n"
        "Simple mixer control 'Auto Mute Mono',0\n"
        "Simple mixer control 'Digital',0\n"
        "Simple mixer control 'DSP Program',0\n"
    )
    sget_with_pvolume = "  Capabilities: pvolume pswitch\n"
    sget_without_pvolume = "  Capabilities: enum\n"

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols.encode())
        # On real HiFiBerry DAC+ (PCM5122), both Analogue and Digital
        # have pvolume. The preference logic should pick Digital.
        if "Analogue" in argv or "Digital" in argv:
            return _FakeProcess(stdout=sget_with_pvolume.encode())
        return _FakeProcess(stdout=sget_without_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert await find_mixer_element(1) == "Digital"


async def test_find_mixer_element_prefers_digital_over_analogue(monkeypatch) -> None:
    """When both Analogue and Digital have pvolume, Digital is preferred."""
    scontrols = "Simple mixer control 'Analogue',0\nSimple mixer control 'Digital',0\n"
    sget_with_pvolume = "  Capabilities: pvolume pswitch\n"

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols.encode())
        return _FakeProcess(stdout=sget_with_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert await find_mixer_element(1) == "Digital"


async def test_find_mixer_element_master(monkeypatch) -> None:
    """HiFiBerry Amp+ uses 'Master' for volume."""
    scontrols = "Simple mixer control 'Channels',0\nSimple mixer control 'Master',0\n"
    sget_pvolume = "  Capabilities: pvolume pswitch\n"
    sget_no_pvolume = "  Capabilities: enum\n"

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols.encode())
        if "Master" in argv:
            return _FakeProcess(stdout=sget_pvolume.encode())
        return _FakeProcess(stdout=sget_no_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert await find_mixer_element(1) == "Master"


async def test_find_mixer_element_none_when_no_controls(monkeypatch) -> None:
    """PCM5102A boards have no mixer controls."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec(""))
    assert await find_mixer_element(1) is None


async def test_find_mixer_element_none_on_amixer_failure(monkeypatch) -> None:
    """Gracefully handle amixer failure (e.g., card not found)."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec("", returncode=1))
    assert await find_mixer_element(99) is None


async def test_find_mixer_element_none_when_amixer_not_found(monkeypatch) -> None:
    """Gracefully handle amixer not being installed."""

    async def not_found(*args: object, **kwargs: object) -> NoReturn:
        raise FileNotFoundError("amixer")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", not_found)
    assert await find_mixer_element(1) is None


async def test_find_mixer_element_capability_scan_fallback(monkeypatch) -> None:
    """Falls back to capability scan for USB devices with non-standard names."""
    scontrols_output = (
        "Simple mixer control 'Mic',0\n"
        "Simple mixer control 'Mic',1\n"
        "Simple mixer control 'UMC202HD 192k Output',0\n"
        "Simple mixer control 'UMC202HD 192k Output',1\n"
    )
    sget_mic = "Simple mixer control 'Mic',0\n  Capabilities: cvolume cswitch\n"
    sget_output = (
        "Simple mixer control 'UMC202HD 192k Output',0\n"
        "  Capabilities: pvolume pswitch\n"
        "  Playback channels: Front Left - Front Right\n"
    )

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols_output.encode())
        # sget calls: return capabilities based on element name
        if "Mic" in argv:
            return _FakeProcess(stdout=sget_mic.encode())
        return _FakeProcess(stdout=sget_output.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert await find_mixer_element(3) == "UMC202HD 192k Output"


async def test_find_mixer_element_capability_scan_skips_capture_only(monkeypatch) -> None:
    """Capability scan skips elements that only have capture volume."""
    scontrols_output = "Simple mixer control 'Mic',0\n"
    sget_mic = "Simple mixer control 'Mic',0\n  Capabilities: cvolume cswitch\n"

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols_output.encode())
        return _FakeProcess(stdout=sget_mic.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert await find_mixer_element(3) is None


# -- AlsaVolumeController.set_state ------------------------------------------


async def test_set_state_calls_amixer_sset(monkeypatch) -> None:
    """set_state runs amixer sset with percentage and mute/unmute."""
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ctrl = AlsaVolumeController(card=1, element="Digital")
    await ctrl.set_state(75, muted=False)
    assert calls == [("amixer", "-M", "-c", "1", "sset", "Digital", "playback", "75%", "unmute")]


async def test_set_state_with_string_card_name(monkeypatch) -> None:
    """set_state passes a string card name straight through to amixer -c.

    amixer -c resolves a card name via snd_card_get_index() internally, so
    this works without translating the name to a numeric index ourselves.
    """
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ctrl = AlsaVolumeController(card="vc4hdmi", element="PCM")
    await ctrl.set_state(75, muted=False)
    assert calls == [("amixer", "-M", "-c", "vc4hdmi", "sset", "PCM", "playback", "75%", "unmute")]


async def test_set_state_muted(monkeypatch) -> None:
    """When muted, amixer is called with 'mute'."""
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ctrl = AlsaVolumeController(card=1, element="Digital")
    await ctrl.set_state(50, muted=True)
    assert calls == [("amixer", "-M", "-c", "1", "sset", "Digital", "playback", "50%", "mute")]


# -- AlsaVolumeController.get_state ------------------------------------------


async def test_get_state_parses_amixer_output(monkeypatch) -> None:
    """get_state parses volume percentage and on/off from amixer sget."""
    amixer_output = (
        "Simple mixer control 'Digital',0\n"
        "  Capabilities: pvolume pswitch\n"
        "  Playback channels: Front Left - Front Right\n"
        "  Limits: Playback 0 - 207\n"
        "  Front Left: Playback 155 [74%] [-15.60dB] [on]\n"
        "  Front Right: Playback 155 [74%] [-15.60dB] [on]\n"
    )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec(amixer_output))
    ctrl = AlsaVolumeController(card=1, element="Digital")
    volume, muted = await ctrl.get_state()
    assert volume == 74
    assert muted is False


async def test_get_state_detects_muted(monkeypatch) -> None:
    """get_state detects [off] as muted."""
    amixer_output = (
        "Simple mixer control 'PCM',0\n"
        "  Capabilities: pvolume pvolume-joined pswitch pswitch-joined\n"
        "  Playback channels: Mono\n"
        "  Limits: Playback -10239 - 400\n"
        "  Mono: Playback -4919 [50%] [-49.19dB] [off]\n"
    )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec(amixer_output))
    ctrl = AlsaVolumeController(card=0, element="PCM")
    volume, muted = await ctrl.get_state()
    assert volume == 50
    assert muted is True


async def test_get_state_mono_channel(monkeypatch) -> None:
    """get_state handles mono devices (single channel line)."""
    amixer_output = (
        "Simple mixer control 'PCM',0\n"
        "  Capabilities: pvolume pvolume-joined pswitch pswitch-joined\n"
        "  Playback channels: Mono\n"
        "  Limits: Playback -10239 - 400\n"
        "  Mono: Playback 0 [96%] [0.00dB] [on]\n"
    )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec(amixer_output))
    ctrl = AlsaVolumeController(card=0, element="PCM")
    volume, muted = await ctrl.get_state()
    assert volume == 96
    assert muted is False


async def test_get_state_requests_playback_only(monkeypatch) -> None:
    """get_state scopes amixer query to playback channels."""
    calls: list[tuple[str, ...]] = []
    amixer_output = (
        "Simple mixer control 'Headset',0\n"
        "  Capabilities: pvolume pswitch cvolume cswitch\n"
        "  Playback channels: Front Left - Front Right\n"
        "  Capture channels: Front Left - Front Right\n"
        "  Front Left: Playback 20 [80%] [on]\n"
        "  Front Right: Playback 20 [80%] [on]\n"
    )

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        return _FakeProcess(stdout=amixer_output.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ctrl = AlsaVolumeController(card=0, element="Headset")
    volume, muted = await ctrl.get_state()
    assert calls == [("amixer", "-M", "-c", "0", "sget", "Headset", "playback")]
    assert volume == 80
    assert muted is False


# -- AlsaVolumeController.start_monitoring ------------------------------------


async def test_monitoring_detects_external_change(monkeypatch) -> None:
    """Monitor loop calls callback when volume changes between polls."""
    got_callback = asyncio.Event()
    callback_received: list[tuple[int, bool]] = []

    states = [
        (50, False),  # initial read
        (50, False),  # first poll — no change
        (75, False),  # second poll — changed -> triggers callback
    ]
    state_iter = iter(states)

    def on_change(volume: int, muted: bool) -> None:
        callback_received.append((volume, muted))
        got_callback.set()

    monkeypatch.setattr(_alsa_mod, "_POLL_INTERVAL_S", 0.0)
    ctrl = AlsaVolumeController(card=0, element="PCM")

    async def fake_get() -> tuple[int, bool]:
        try:
            return next(state_iter)
        except StopIteration:
            # Keep returning last state after sequence ends
            return (75, False)

    ctrl.get_state = fake_get  # type: ignore[assignment]

    await ctrl.start_monitoring(on_change)
    try:
        await asyncio.wait_for(got_callback.wait(), timeout=2.0)
    finally:
        await ctrl.stop_monitoring()

    assert callback_received == [(75, False)]


# -- async_check_alsa_available -----------------------------------------------


async def test_alsa_available_for_hw_device_with_mixer(monkeypatch) -> None:
    """Returns (card, element) for a hw: device with mixer controls."""
    scontrols = "Simple mixer control 'Digital',0\n"
    sget_pvolume = "  Capabilities: pvolume pswitch\n"

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols.encode())
        return _FakeProcess(stdout=sget_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    device = SimpleNamespace(name="HiFiBerry DAC+: pcm512x (hw:1,0)", is_default=False)
    result = await async_check_alsa_available(device)
    assert result == (1, "Digital")


async def test_alsa_available_for_plughw_card_name_device(monkeypatch) -> None:
    """Returns (card_name, element) for a plughw:CARD=<name> device.

    This is the raw ALSA device path (e.g. --audio-device plughw:CARD=vc4hdmi,DEV=0),
    used to reach the plug/dmix conversion layer that hw:N,M bypasses.
    """
    scontrols = "Simple mixer control 'Digital',0\n"
    sget_pvolume = "  Capabilities: pvolume pswitch\n"
    calls: list[tuple[object, ...]] = []

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols.encode())
        return _FakeProcess(stdout=sget_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    device = SimpleNamespace(name="plughw:CARD=vc4hdmi,DEV=0", is_default=False)
    result = await async_check_alsa_available(device)
    assert result == ("vc4hdmi", "Digital")
    assert calls[0] == ("amixer", "-c", "vc4hdmi", "scontrols")


# Real `aplay -L` hint output from an ODROID-N2 (single card, reported in a
# user issue): PortAudio's own device enumeration reports this same default
# output device as the bare "sysdefault", with no CARD= info at all.
_ODROIDN2_APLAY_L = [
    ("null", "Discard all samples (playback) or generate zero samples (capture)"),
    ("hw:CARD=ODROIDN2,DEV=0", "ODROID-N2,"),
    ("plughw:CARD=ODROIDN2,DEV=0", "ODROID-N2,"),
    ("sysdefault:CARD=ODROIDN2", "ODROID-N2,"),
    ("dmix:CARD=ODROIDN2,DEV=0", "ODROID-N2,"),
]


async def test_alsa_available_for_bare_sysdefault_via_aplay_l(monkeypatch) -> None:
    """Resolves bare 'sysdefault' (no CARD= info) via aplay -L hints."""
    scontrols = "Simple mixer control 'Digital',0\n"
    sget_pvolume = "  Capabilities: pvolume pswitch\n"
    calls: list[tuple[object, ...]] = []

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        if "scontrols" in argv:
            return _FakeProcess(stdout=scontrols.encode())
        return _FakeProcess(stdout=sget_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    monkeypatch.setattr(_audio_devices_mod, "list_alsa_devices", lambda: _ODROIDN2_APLAY_L)
    device = SimpleNamespace(name="sysdefault", is_default=True)
    result = await async_check_alsa_available(device)
    assert result == ("ODROIDN2", "Digital")
    assert calls[0] == ("amixer", "-c", "ODROIDN2", "scontrols")


async def test_alsa_not_available_for_bare_sysdefault_when_ambiguous(monkeypatch) -> None:
    """Refuses to guess when multiple cards each expose a sysdefault hint."""
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    monkeypatch.setattr(
        _audio_devices_mod,
        "list_alsa_devices",
        lambda: [
            ("sysdefault:CARD=ODROIDN2", "ODROID-N2,"),
            ("sysdefault:CARD=USBDAC", "USB DAC,"),
        ],
    )
    device = SimpleNamespace(name="sysdefault", is_default=True)
    assert await async_check_alsa_available(device) is None


async def test_alsa_not_available_for_bare_sysdefault_when_no_hint(monkeypatch) -> None:
    """Falls back to None when aplay -L has no matching sysdefault hint."""
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    monkeypatch.setattr(_audio_devices_mod, "list_alsa_devices", lambda: [])
    device = SimpleNamespace(name="sysdefault", is_default=True)
    assert await async_check_alsa_available(device) is None


async def test_alsa_not_available_for_virtual_device(monkeypatch) -> None:
    """Returns None for virtual devices (no hw: in name)."""
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    device = SimpleNamespace(name="pipewire", is_default=False)
    assert await async_check_alsa_available(device) is None


async def test_alsa_not_available_when_no_mixer_controls(monkeypatch) -> None:
    """Returns None when card has no mixer elements (PCM5102A)."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec(""))
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    device = SimpleNamespace(name="DAC Zero (hw:2,0)", is_default=False)
    assert await async_check_alsa_available(device) is None


# -- Simulated HiFiBerry DAC+ HAT scenario -----------------------------------
# Reproduces the exact setup from the GitHub issues: a Raspberry Pi with a
# HiFiBerry DAC+ HAT on card 1, using the real device names and amixer output
# that users reported.


# Real amixer output from a HiFiBerry DAC+ (PCM5122 chip)
_HIFIBERRY_SCONTROLS = (
    "Simple mixer control 'Analogue',0\n"
    "Simple mixer control 'Analogue Playback Boost',0\n"
    "Simple mixer control 'Auto Mute',0\n"
    "Simple mixer control 'Auto Mute Mono',0\n"
    "Simple mixer control 'Digital',0\n"
    "Simple mixer control 'DSP Program',0\n"
)

_HIFIBERRY_SGET_74 = (
    "Simple mixer control 'Digital',0\n"
    "  Capabilities: pvolume pswitch\n"
    "  Playback channels: Front Left - Front Right\n"
    "  Limits: Playback 0 - 207\n"
    "  Front Left: Playback 155 [74%] [-15.60dB] [on]\n"
    "  Front Right: Playback 155 [74%] [-15.60dB] [on]\n"
)

# The exact PortAudio device name from the issue logs
_HIFIBERRY_DEVICE_NAME = "snd_rpi_hifiberry_dacplus: HiFiBerry DAC+ HiFi pcm512x-hifi-0 (hw:1,0)"


async def test_hifiberry_dac_discovery(monkeypatch) -> None:
    """Full discovery flow for a HiFiBerry DAC+ HAT on card 1."""
    sget_with_pvolume = "  Capabilities: pvolume pswitch\n"
    sget_without_pvolume = "  Capabilities: enum\n"

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=_HIFIBERRY_SCONTROLS.encode())
        # Both Analogue and Digital have pvolume on real hardware.
        if "Analogue" in argv or "Digital" in argv:
            return _FakeProcess(stdout=sget_with_pvolume.encode())
        return _FakeProcess(stdout=sget_without_pvolume.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    device = SimpleNamespace(name=_HIFIBERRY_DEVICE_NAME, is_default=False)
    result = await async_check_alsa_available(device)
    assert result == (1, "Digital")


async def test_hifiberry_dac_set_and_get_volume(monkeypatch) -> None:
    """Set and read volume on a simulated HiFiBerry DAC+ HAT."""
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        calls.append(argv)
        return _FakeProcess(stdout=_HIFIBERRY_SGET_74.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ctrl = AlsaVolumeController(card=1, element="Digital")

    # Server sends volume command -> controller sets ALSA mixer
    await ctrl.set_state(74, muted=False)
    assert calls[-1] == (
        "amixer",
        "-M",
        "-c",
        "1",
        "sset",
        "Digital",
        "playback",
        "74%",
        "unmute",
    )

    # Read back the volume
    volume, muted = await ctrl.get_state()
    assert calls[-1] == ("amixer", "-M", "-c", "1", "sget", "Digital", "playback")
    assert volume == 74
    assert muted is False


# -- TAS58xx / Sonocotta Louder Raspberry HAT --------------------------------
# The TAS58xx driver reports "volume" (or "volume volume-joined") instead of
# the standard "pvolume" capability.

_TAS58XX_SCONTROLS = "Simple mixer control 'Analog Gain',0\nSimple mixer control 'Digital',0\n"

_TAS58XX_SGET_DIGITAL_MONO = (
    "Simple mixer control 'Digital',0\n"
    "  Capabilities: volume volume-joined\n"
    "  Playback channels: Mono\n"
    "  Limits: 0 - 127\n"
    "  Mono: 73 [57%]\n"
)

_TAS58XX_SGET_DIGITAL_STEREO = (
    "Simple mixer control 'Digital',0\n"
    "  Capabilities: volume\n"
    "  Playback channels: Front Left - Front Right\n"
    "  Limits: 0 - 127\n"
    "  Front Left: 73 [57%]\n"
    "  Front Right: 73 [57%]\n"
)


def _tas58xx_exec(digital_output: str) -> _AmixerExecFactory:
    """Build a fake amixer exec for TAS58xx scenarios."""

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=_TAS58XX_SCONTROLS.encode())
        if "Digital" in argv:
            return _FakeProcess(stdout=digital_output.encode())
        return _FakeProcess(stdout=b"")

    return fake_exec


@pytest.mark.parametrize(
    "digital_output",
    [
        _TAS58XX_SGET_DIGITAL_MONO,
        _TAS58XX_SGET_DIGITAL_STEREO,
    ],
    ids=["mono", "stereo"],
)
async def test_find_mixer_element_tas58xx(monkeypatch, digital_output: str) -> None:
    """TAS58xx 'volume' capability is detected."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _tas58xx_exec(digital_output))
    assert await find_mixer_element(2) == "Digital"


async def test_louder_raspberry_discovery(monkeypatch) -> None:
    """Full discovery flow for a Sonocotta Louder Raspberry HAT on card 2."""
    monkeypatch.setattr(_alsa_mod, "AVAILABLE", True)
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _tas58xx_exec(_TAS58XX_SGET_DIGITAL_MONO)
    )
    device = SimpleNamespace(
        name="Louder-Raspberry: bcm2835-i2s-tas58xx-amplifier tas58xx-amplifier-0 (hw:2,0)",
        is_default=False,
    )
    assert await async_check_alsa_available(device) == (2, "Digital")


async def test_louder_raspberry_get_volume(monkeypatch) -> None:
    """get_state reads back the correct volume from a TAS58xx Digital control."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _amixer_exec(_TAS58XX_SGET_DIGITAL_MONO))
    ctrl = AlsaVolumeController(card=2, element="Digital")
    volume, muted = await ctrl.get_state()
    assert volume == 57
    assert muted is False


# -- Realtek RT5616 analog codec (NanoPC-T6 / RK3588) -------------------------
# The mainline RT5616 codec driver registers several capture-side analog boost
# gain stages ("ADC Boost", "IN1 Boost", "IN2 Boost") with bare `volume`
# capability and *both* Playback and Capture channels. These must NOT be
# detected as playback-volume controls — they are disconnected from the
# playback chain and adjusting them has zero effect on output loudness.
# The genuine playback elements (DAC1, HP, OUT) all carry `pvolume`.

# False positives: capture-side boost/preamp gain tagged with bare `volume`
# plus a `Capture channels:` line.
_RT5616_SGET_ADC_BOOST = (
    "Simple mixer control 'ADC Boost',0\n"
    "  Capabilities: volume\n"
    "  Playback channels: Front Left - Front Right\n"
    "  Capture channels: Front Left - Front Right\n"
    "  Limits: 0 - 3\n"
    "  Front Left: 0 [0%] [0.00dB]\n"
    "  Front Right: 0 [0%] [0.00dB]\n"
)

_RT5616_SGET_IN1_BOOST = (
    "Simple mixer control 'IN1 Boost',0\n"
    "  Capabilities: volume volume-joined\n"
    "  Playback channels: Mono\n"
    "  Capture channels: Mono\n"
    "  Limits: 0 - 8\n"
    "  Mono: 0 [0%] [0.00dB]\n"
)

_RT5616_SGET_IN2_BOOST = (
    "Simple mixer control 'IN2 Boost',0\n"
    "  Capabilities: volume volume-joined\n"
    "  Playback channels: Mono\n"
    "  Capture channels: Mono\n"
    "  Limits: 0 - 8\n"
    "  Mono: 0 [0%] [0.00dB]\n"
)

# Capture-only control: already correctly excluded (cvolume, no playback).
_RT5616_SGET_ADC = (
    "Simple mixer control 'ADC',0\n"
    "  Capabilities: cvolume cswitch\n"
    "  Capture channels: Front Left - Front Right\n"
    "  Limits: Capture 0 - 127\n"
    "  Front Left: Capture 47 [37%] [0.00dB] [on]\n"
    "  Front Right: Capture 47 [37%] [0.00dB] [on]\n"
)

# True playback-volume elements (must remain detected).
_RT5616_SGET_DAC1 = (
    "Simple mixer control 'DAC1',0\n"
    "  Capabilities: pvolume\n"
    "  Playback channels: Front Left - Front Right\n"
    "  Limits: Playback 0 - 175\n"
    "  Front Left: Playback 175 [100%] [0.00dB]\n"
    "  Front Right: Playback 175 [100%] [0.00dB]\n"
)

_RT5616_SGET_HP = (
    "Simple mixer control 'HP',0\n"
    "  Capabilities: pvolume pswitch\n"
    "  Playback channels: Front Left - Front Right\n"
    "  Limits: Playback 0 - 39\n"
    "  Front Left: Playback 39 [100%] [0.00dB] [on]\n"
    "  Front Right: Playback 39 [100%] [0.00dB] [on]\n"
)

_RT5616_SGET_OUT = (
    "Simple mixer control 'OUT',0\n"
    "  Capabilities: pvolume pswitch\n"
    "  Playback channels: Front Left - Front Right\n"
    "  Limits: Playback 0 - 39\n"
    "  Front Left: Playback 39 [100%] [0.00dB] [on]\n"
    "  Front Right: Playback 39 [100%] [0.00dB] [on]\n"
)

# Enumeration order as reported by `amixer -c2 scontrols`: the capture-side
# boost stages enumerate *before* the real playback elements, which is what
# previously caused "ADC Boost" to win.
_RT5616_SCONTROLS = (
    "Simple mixer control 'ADC Boost',0\n"
    "Simple mixer control 'IN1 Boost',0\n"
    "Simple mixer control 'IN2 Boost',0\n"
    "Simple mixer control 'ADC',0\n"
    "Simple mixer control 'DAC1',0\n"
    "Simple mixer control 'HP',0\n"
    "Simple mixer control 'OUT',0\n"
)

_RT5616_SGET_BY_ELEMENT = {
    "ADC Boost": _RT5616_SGET_ADC_BOOST,
    "IN1 Boost": _RT5616_SGET_IN1_BOOST,
    "IN2 Boost": _RT5616_SGET_IN2_BOOST,
    "ADC": _RT5616_SGET_ADC,
    "DAC1": _RT5616_SGET_DAC1,
    "HP": _RT5616_SGET_HP,
    "OUT": _RT5616_SGET_OUT,
}


def _rt5616_exec() -> _AmixerExecFactory:
    """Build a fake amixer exec over the full RT5616 control set."""

    async def fake_exec(*argv: object, **kwargs: object) -> _FakeProcess:
        if "scontrols" in argv:
            return _FakeProcess(stdout=_RT5616_SCONTROLS.encode())
        for element, output in _RT5616_SGET_BY_ELEMENT.items():
            if element in argv:
                return _FakeProcess(stdout=output.encode())
        return _FakeProcess(stdout=b"")

    return fake_exec


@pytest.mark.parametrize(
    "element",
    ["ADC", "ADC Boost", "IN1 Boost", "IN2 Boost"],
)
async def test_rt5616_rejects_capture_side_controls(monkeypatch, element: str) -> None:
    """Capture-side boost gain (bare `volume` + Capture channels) is rejected."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _rt5616_exec())
    assert await _has_playback_volume(2, element) is False


@pytest.mark.parametrize(
    "element",
    ["DAC1", "HP", "OUT"],
)
async def test_rt5616_accepts_playback_controls(monkeypatch, element: str) -> None:
    """Genuine pvolume playback elements remain detected."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _rt5616_exec())
    assert await _has_playback_volume(2, element) is True


async def test_rt5616_find_mixer_element_returns_dac1(monkeypatch) -> None:
    """find_mixer_element picks DAC1, not the earlier-enumerated 'ADC Boost'."""
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _rt5616_exec())
    assert await find_mixer_element(2) == "DAC1"
