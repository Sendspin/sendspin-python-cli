"""Tests for ALSA device resolution in audio_devices.py."""

from __future__ import annotations

from unittest.mock import patch

import sounddevice

import sendspin.audio_devices as _mod
from sendspin.audio_devices import (
    AudioDevice,
    _check_format,
    _try_alsa_device,
    detect_supported_audio_formats,
)


def test_try_alsa_device_returns_device_when_portaudio_accepts():
    """Device recognized by PortAudio is returned with queried info."""
    info = {
        "name": "jsm_speaker_left",
        "max_output_channels": 128,
        "default_samplerate": 48000.0,
    }
    with (
        patch.object(sounddevice, "check_output_settings"),
        patch.object(sounddevice, "query_devices", return_value=info),
    ):
        result = _try_alsa_device("jsm_speaker_left")

    assert result is not None
    assert result.alsa_device_name == "jsm_speaker_left"
    assert result.output_channels == 128
    assert result.sample_rate == 48000.0


def test_try_alsa_device_returns_none_on_portaudio_error():
    """PortAudioError from check_output_settings means device can't be opened."""
    with patch.object(
        sounddevice,
        "check_output_settings",
        side_effect=sounddevice.PortAudioError("err", -1, ""),
    ):
        assert _try_alsa_device("nonexistent") is None


def test_try_alsa_device_accepts_hw_card_format_known_to_alsa():
    """hw:CARD=...,DEV=... that PortAudio doesn't know but ALSA lists should work."""
    alsa_list = [
        (
            "hw:CARD=sndrpihifiberry,DEV=0",
            "snd_rpi_hifiberry_dacplus, HiFiBerry DAC+ HiFi",
        ),
    ]
    with (
        patch.object(
            sounddevice,
            "check_output_settings",
            side_effect=ValueError("not found"),
        ),
        patch.object(
            sounddevice,
            "query_devices",
            side_effect=ValueError("not found"),
        ),
        patch.object(_mod, "list_alsa_devices", return_value=alsa_list),
    ):
        result = _try_alsa_device("hw:CARD=sndrpihifiberry,DEV=0")

    assert result is not None
    assert result.alsa_device_name == "hw:CARD=sndrpihifiberry,DEV=0"
    assert result.output_channels == 2
    assert result.sample_rate == 48000.0


def test_try_alsa_device_returns_none_for_unknown_hw_card():
    """hw:CARD=...,DEV=... not in ALSA list should return None."""
    with (
        patch.object(
            sounddevice,
            "check_output_settings",
            side_effect=ValueError("not found"),
        ),
        patch.object(_mod, "list_alsa_devices", return_value=[]),
    ):
        assert _try_alsa_device("hw:CARD=nonexistent,DEV=0") is None


def test_try_alsa_device_uses_safe_defaults_when_query_fails():
    """PortAudio accepts the device but can't query info → safe defaults."""
    with (
        patch.object(sounddevice, "check_output_settings"),
        patch.object(
            sounddevice,
            "query_devices",
            side_effect=sounddevice.PortAudioError("err", -1, ""),
        ),
    ):
        result = _try_alsa_device("jsm_speaker_left")

    assert result is not None
    assert result.output_channels == 2
    assert result.sample_rate == 48000.0


def test_try_alsa_device_accepts_alsa_only_device():
    """ALSA-only device (e.g. bluealsa) not known to PortAudio should work."""
    alsa_list = [("bluealsa", "Bluetooth Audio")]
    with (
        patch.object(
            sounddevice,
            "check_output_settings",
            side_effect=ValueError("not found"),
        ),
        patch.object(
            sounddevice,
            "query_devices",
            side_effect=ValueError("not found"),
        ),
        patch.object(_mod, "list_alsa_devices", return_value=alsa_list),
    ):
        result = _try_alsa_device("bluealsa")

    assert result is not None
    assert result.alsa_device_name == "bluealsa"


def _alsa_only_device() -> AudioDevice:
    """A device resolved by name that PortAudio cannot resolve itself."""
    return AudioDevice(
        index=None,
        name="hw:CARD=sndrpihifiberry,DEV=0",
        output_channels=2,
        sample_rate=48000.0,
        is_default=False,
        alsa_device_name="hw:CARD=sndrpihifiberry,DEV=0",
    )


def test_check_format_reports_unsupported_for_unknown_device_name():
    """PortAudio raises ValueError for raw ALSA names; treat it as unsupported, not fatal."""
    with patch.object(
        sounddevice,
        "check_output_settings",
        side_effect=ValueError("No output device matching"),
    ):
        assert _check_format(_alsa_only_device(), 48000, 2, "int16") is False


def test_detect_supported_audio_formats_falls_back_for_alsa_only_device():
    """Startup must not raise for a device PortAudio cannot probe."""
    with patch.object(
        sounddevice,
        "check_output_settings",
        side_effect=ValueError("No output device matching"),
    ):
        formats = detect_supported_audio_formats(_alsa_only_device())

    assert formats, "expected safe-default formats for an unprobeable device"
