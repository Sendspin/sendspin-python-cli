"""Tests for source encoding and signal helpers."""

from __future__ import annotations

import math
import struct

import pytest
from aiosendspin.models.source import ClientStreamStartSource
from aiosendspin.models.types import AudioCodec
from aiosendspin.server.roles.source.group import SourceDecoder

from sendspin.source_utils import SourceEncoder, calc_level

RATE = 48000
CHANNELS = 2
STRIDE = 2 * CHANNELS


def _sine_pcm(duration_ms: int, freq: float = 440.0) -> bytes:
    samples = RATE * duration_ms // 1000
    buffer = bytearray()
    for i in range(samples):
        value = int(0.3 * math.sin(2 * math.pi * freq * i / RATE) * 32767)
        buffer.extend(struct.pack("<h", value) * CHANNELS)
    return bytes(buffer)


def test_calc_level_silence_is_zero() -> None:
    """Silence has zero level; empty input is safe."""
    assert calc_level(b"") == 0.0
    assert calc_level(b"\x00\x00" * 100) == 0.0


def test_calc_level_signal_is_positive() -> None:
    """A real signal produces a positive normalized level."""
    assert 0.0 < calc_level(_sine_pcm(20)) <= 1.0


def test_pcm_encoder_passes_through() -> None:
    """PCM encoding passes bytes through unchanged with no header."""
    encoder = SourceEncoder(
        codec=AudioCodec.PCM, channels=CHANNELS, sample_rate=RATE, frame_samples=960
    )
    assert encoder.codec_header is None
    pcm = _sine_pcm(20)
    assert encoder.encode(pcm, 5000) == [(pcm, 5000)]
    assert encoder.encode(b"", 6000) == []


@pytest.mark.parametrize("codec", [AudioCodec.FLAC, AudioCodec.OPUS])
def test_compressed_encoder_produces_header_and_frames(codec: AudioCodec) -> None:
    """FLAC/Opus encoding yields a codec header and encoded frames."""
    encoder = SourceEncoder(
        codec=codec, channels=CHANNELS, sample_rate=RATE, frame_samples=RATE * 20 // 1000
    )
    assert encoder.codec_header is not None
    frames = encoder.encode(_sine_pcm(200), 1_000_000)
    frames.extend((tail, 0) for tail in encoder.flush())
    assert frames
    assert all(isinstance(data, bytes) and data for data, _ in frames)


@pytest.mark.parametrize("codec", [AudioCodec.PCM, AudioCodec.FLAC, AudioCodec.OPUS])
def test_encode_round_trips_through_server_decoder(codec: AudioCodec) -> None:
    """Frames encoded by the CLI decode back to PCM on the server side."""
    pcm = _sine_pcm(500)
    encoder = SourceEncoder(
        codec=codec, channels=CHANNELS, sample_rate=RATE, frame_samples=RATE * 20 // 1000
    )
    frame_bytes = RATE * 20 // 1000 * STRIDE
    encoded: list[bytes] = []
    offset = 0
    ts = 1_000_000
    while offset < len(pcm):
        block = pcm[offset : offset + frame_bytes]
        offset += frame_bytes
        encoded.extend(data for data, _ in encoder.encode(block, ts))
        ts += 20_000
    encoded.extend(encoder.flush())

    decoder = SourceDecoder(
        ClientStreamStartSource(
            codec=codec,
            channels=CHANNELS,
            sample_rate=RATE,
            bit_depth=16,
            codec_header=encoder.codec_header,
        )
    )
    decoded = bytearray()
    for frame in encoded:
        for chunk in decoder.decode(frame):
            decoded += chunk

    in_samples = len(pcm) // STRIDE
    out_samples = len(decoded) // STRIDE
    # PCM is exact; FLAC/Opus carry small codec delay/padding, so allow a tolerance.
    assert out_samples == pytest.approx(in_samples, abs=RATE // 10)
