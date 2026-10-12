"""Encoding and signal helpers for the Sendspin source role.

A source client captures 16-bit PCM from a local input and streams it to the
server. ``SourceEncoder`` encodes that PCM to the negotiated codec (PCM
passthrough, FLAC, or Opus) using PyAV, exposing the raw codec header so the
server can initialize its decoder. ``calc_level`` provides a simple RMS level
used for optional line sensing.
"""

from __future__ import annotations

import array
import base64
import logging
import sys
from fractions import Fraction
from types import ModuleType
from typing import TYPE_CHECKING

from aiosendspin.models.types import AudioCodec

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)

# Source capture is fixed at 16-bit signed PCM (matches sounddevice int16 capture
# and keeps codec init simple). The server can still receive other depths from
# other source implementations.
BYTES_PER_SAMPLE = 2
_MAX_INT16 = 32767.0


def calc_level(pcm: bytes) -> float:
    """Return a normalized RMS level (0.0-1.0) for 16-bit interleaved PCM."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        norm = sample / _MAX_INT16
        total += norm * norm
    rms = (total / len(samples)) ** 0.5
    return float(min(1.0, rms))


class SourceEncoder:
    """Encode 16-bit interleaved PCM to the negotiated source codec via PyAV."""

    def __init__(
        self,
        *,
        codec: AudioCodec,
        channels: int,
        sample_rate: int,
        frame_samples: int,
    ) -> None:
        """Create an encoder.

        Args:
            codec: Target codec (pcm, flac, or opus).
            channels: Number of channels.
            sample_rate: Sample rate in Hz.
            frame_samples: Preferred samples per frame (used for PCM and as a
                fallback when the codec does not report its own frame size).
        """
        self._codec = codec
        self._channels = channels
        self._sample_rate = sample_rate
        self._stride = BYTES_PER_SAMPLE * channels
        self._layout = "mono" if channels == 1 else "stereo"
        self._buffer = bytearray()
        self._buffer_head_us: int | None = None
        self._pts = 0
        self._encoder: av.AudioCodecContext | None = None
        self._codec_header: str | None = None
        self._frame_samples = frame_samples

        if codec == AudioCodec.PCM:
            return

        av_mod = _get_av()
        codec_name = "libopus" if codec == AudioCodec.OPUS else "flac"
        encoder = av_mod.AudioCodecContext.create(codec_name, "w")
        encoder.sample_rate = sample_rate
        encoder.format = "s16"
        encoder.layout = self._layout
        with av_mod.logging.Capture():
            encoder.open()
        if encoder.frame_size:
            self._frame_samples = encoder.frame_size
        if encoder.extradata:
            self._codec_header = base64.b64encode(bytes(encoder.extradata)).decode("ascii")
        self._encoder = encoder

    @property
    def codec_header(self) -> str | None:
        """Base64 raw codec header (extradata) for client_stream/start, if any."""
        return self._codec_header

    @property
    def frame_samples(self) -> int:
        """Preferred number of samples per captured frame."""
        return self._frame_samples

    def encode(self, pcm: bytes, capture_timestamp_us: int) -> list[tuple[bytes, int]]:
        """Encode captured PCM into ``(frame_bytes, capture_timestamp_us)`` pairs.

        For PCM the input passes through unchanged. For FLAC/Opus the PCM is
        buffered and emitted in codec-sized frames; each emitted frame is stamped
        with the capture time of its first sample.
        """
        if self._encoder is None:
            return [(pcm, capture_timestamp_us)] if pcm else []

        if not self._buffer:
            self._buffer_head_us = capture_timestamp_us
        self._buffer.extend(pcm)

        results: list[tuple[bytes, int]] = []
        chunk_size = self._frame_samples * self._stride
        while len(self._buffer) >= chunk_size:
            block = bytes(self._buffer[:chunk_size])
            del self._buffer[:chunk_size]
            assert self._buffer_head_us is not None
            frame_ts = self._buffer_head_us
            self._buffer_head_us += round(self._frame_samples * 1_000_000 / self._sample_rate)
            for encoded in self._encode_block(block):
                results.append((encoded, frame_ts))
        return results

    def flush(self) -> list[bytes]:
        """Flush the codec, returning any trailing frames."""
        if self._encoder is None:
            return []
        return [data for packet in self._encoder.encode(None) if (data := bytes(packet))]

    def _encode_block(self, block: bytes) -> list[bytes]:
        assert self._encoder is not None
        av_mod = _get_av()
        frame = av_mod.AudioFrame(format="s16", layout=self._layout, samples=self._frame_samples)
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self._sample_rate)
        self._pts += self._frame_samples
        frame.planes[0].update(block)
        return [data for packet in self._encoder.encode(frame) if (data := bytes(packet))]


def _get_av() -> ModuleType:
    """Import PyAV lazily so non-source commands do not require it eagerly."""
    import av  # noqa: PLC0415

    return av
