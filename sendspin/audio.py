"""Audio playback for the Sendspin CLI with time synchronization.

This module provides an AudioPlayer that handles time-synchronized audio playback
with DAC-level timing precision. It manages buffering, scheduled start times,
and sync error correction to maintain sync between server and client timelines.
"""

from __future__ import annotations

import collections
import concurrent.futures
import logging
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final, Protocol, cast

import sounddevice
from aiosendspin.client.time_sync import SendspinTimeFilter
from sounddevice import CallbackFlags

from sendspin.audio_devices import SOUNDDEVICE_DTYPE_MAP, AudioDevice

try:
    from sendspin._volume import apply_volume as _c_apply_volume
except ImportError:
    _c_apply_volume = None
    logging.getLogger(__name__).info(
        "C volume extension unavailable; falling back to numpy (slower)"
    )
    import numpy as np

if TYPE_CHECKING:
    from aiosendspin.client import AudioFormat, PCMFormat

logger = logging.getLogger(__name__)


class AudioTimeInfo(Protocol):
    """Protocol for audio timing information from sounddevice callback.

    Provides DAC (Digital-to-Analog Converter) and other timing metrics
    needed for precise playback synchronization.
    """

    outputBufferDacTime: float  # noqa: N815
    """DAC time when the output buffer will be played (in seconds)."""


class PlaybackState(Enum):
    """State machine for audio playback lifecycle.

    Tracks the playback progression from initialization through active playback.
    """

    INITIALIZING = auto()
    """Waiting for first audio chunk and sync info."""

    WAITING_FOR_START = auto()
    """Buffer filled, scheduled start time computed, awaiting start gate."""

    PLAYING = auto()
    """Audio actively playing with sync corrections."""

    REANCHORING = auto()
    """Sync error exceeded threshold, resetting and waiting to restart."""


@dataclass
class _QueuedChunk:
    """Represents a queued audio chunk with timing information."""

    server_timestamp_us: int
    """Server timestamp when this chunk should start playing."""
    audio_data: bytes | bytearray
    """Raw PCM audio bytes."""


class AudioPlayer:
    """
    Audio player for the Sendspin CLI with time synchronization support.

    This player accepts audio chunks with server timestamps and dynamically
    computes playback times using a time synchronization function. This allows
    for accurate synchronization even when the time base changes during playback.

    Attributes:
        _compute_client_time: Function that converts server timestamps to client
            timestamps (monotonic time), accounting for clock drift, offset,
            and static delay.
        _compute_server_time: Function that converts client timestamps (monotonic
            loop time) to server timestamps (inverse of _compute_client_time).
    """

    _compute_client_time: Callable[[int], int]
    _compute_server_time: Callable[[int], int]

    _MIN_CHUNKS_TO_START: Final[int] = 16
    """Minimum chunks buffered before starting playback to absorb network jitter."""
    _MIN_CHUNKS_TO_MAINTAIN: Final[int] = 8
    """Minimum chunks to maintain during playback to avoid underruns."""
    _MICROSECONDS_PER_SECOND: Final[int] = 1_000_000
    """Conversion factor for time calculations."""
    _DAC_PER_LOOP_MIN: Final[float] = 0.999
    """Minimum DAC-to-loop time ratio to prevent wild extrapolation."""
    _DAC_PER_LOOP_MAX: Final[float] = 1.001
    """Maximum DAC-to-loop time ratio to prevent wild extrapolation."""

    # Sync error correction: playback speed adjustment range
    _MAX_SPEED_CORRECTION: Final[float] = 0.002
    """Maximum playback speed deviation for sync correction (0.002 = ±0.2% speed variation)."""

    # Sync error correction: secondary thresholds (rarely need adjustment)
    _CORRECTION_DEADBAND_US: Final[int] = 2_000
    """Sync error threshold below which no correction is applied (2 ms)."""
    _REANCHOR_THRESHOLD_US: Final[int] = 500_000
    """Sync error threshold above which re-anchoring is triggered (500 ms)."""
    _REANCHOR_COOLDOWN_US: Final[int] = 5_000_000
    """Minimum time between re-anchor events (5 seconds)."""
    _MIN_BUFFER_DURATION_US: Final[int] = 200_000
    """Minimum buffer duration (200ms) to start playback and absorb network jitter."""

    # Audio stream configuration
    _BLOCKSIZE: Final[int] = 2048
    """Audio block size (~46ms at 44.1kHz)."""

    # Time synchronization thresholds
    _EARLY_START_THRESHOLD_US: Final[int] = 700_000
    """Threshold for detecting early start due to fallback mapping (700ms)."""
    _START_TIME_UPDATE_THRESHOLD_US: Final[int] = 5_000
    """Minimum threshold for updating start time to avoid churn (5ms)."""

    # Sync correction planning
    _CORRECTION_TARGET_SECONDS: Final[float] = 8.0
    """Target window to fix sync error through micro-corrections (8 seconds)."""
    _CORRECTION_START_GRACE_US: Final[int] = 750_000
    """Delay sync corrections after startup so DAC/time-sync estimates can settle."""

    def __init__(
        self,
        compute_client_time: Callable[[int], int],
        compute_server_time: Callable[[int], int],
        now_us: Callable[[], int] | None = None,
        is_clock_synced: Callable[[], bool] | None = None,
    ) -> None:
        """
        Initialize the audio player.

        Args:
            compute_client_time: Function that converts server timestamps to client
                timestamps (monotonic clock time), accounting for clock drift, offset,
                and static delay.
            compute_server_time: Function that converts client timestamps (monotonic
                clock time) to server timestamps, accounting for clock drift,
                offset, and static delay as the inverse of compute_client_time.
            now_us: Function returning current monotonic time in microseconds.
                Must be in the same clock domain as compute_client_time.
                Defaults to time.monotonic().
            is_clock_synced: Returns True when the time filter has converged.
                Used to tell a real mid-stream join (start time near now, clock
                trustworthy) apart from the unsynced fallback in compute_play_time
                (now + 500ms). Defaults to always-True.
        """
        self._compute_client_time = compute_client_time
        self._compute_server_time = compute_server_time
        self._now_us = now_us or (lambda: int(time.monotonic() * 1_000_000))
        self._is_clock_synced = is_clock_synced or (lambda: True)
        self._format: PCMFormat | None = None
        self._queue: queue.Queue[_QueuedChunk] = queue.Queue()
        self._stream: sounddevice.RawOutputStream | None = None
        self._closed = False
        self._stream_started = False
        self._stream_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sendspin-audio"
        )
        self._first_real_chunk = True  # Flag to initialize timing from first chunk

        self._volume: int = 100  # 0-100 range
        self._muted: bool = False

        self._output_latency_us: int = 0

        # Partial chunk tracking (to avoid discarding partial chunks)
        self._current_chunk: _QueuedChunk | None = None
        self._current_chunk_offset = 0

        # Track expected next chunk timestamp for intelligent gap/overlap handling
        self._expected_next_timestamp: int | None = None

        # Underrun tracking
        self._underrun_count = 0
        self._last_buffer_warning_us = 0

        # Track queued audio duration instead of just item count
        self._queued_duration_us = 0

        # DAC timing for accurate playback position tracking
        self._dac_loop_calibrations: collections.deque[tuple[int, int]] = collections.deque(
            maxlen=100
        )
        # Recent [(dac_time_us, loop_time_us), ...] pairs for DAC-Loop mapping
        self._last_known_playback_position_us: int = 0
        # Current playback position in server timestamp space
        self._last_dac_calibration_time_us: int = 0
        # Last loop time when we calibrated DAC-Loop mapping

        # Playback state machine
        self._playback_state: PlaybackState = PlaybackState.INITIALIZING
        """Current playback state (INITIALIZING, WAITING_FOR_START, PLAYING, REANCHORING)."""

        # Scheduled start anchoring
        self._scheduled_start_loop_time_us: int | None = None
        self._scheduled_start_dac_time_us: int | None = None
        self._playback_started_loop_time_us: int = 0

        # Server timeline cursor for the next input frame to be consumed
        self._server_ts_cursor_us: int = 0
        self._server_ts_cursor_remainder: int = 0  # fractional accumulator for microseconds

        # First-chunk and re-anchor tracking
        self._first_server_timestamp_us: int | None = None
        self._early_start_suspect: bool = False
        self._has_reanchored: bool = False
        self._force_reanchor: bool = True

        # Low-overhead drift/sync correction scheduling (sample drop/insert)
        self._insert_every_n_frames: int = 0
        self._drop_every_n_frames: int = 0
        self._frames_until_next_insert: int = 0
        self._frames_until_next_drop: int = 0
        self._last_output_frame: bytes = b""

        # Sync error smoothing (Kalman filter) and re-anchor cooldown
        self._sync_error_filter = SendspinTimeFilter(process_std_dev=0.01, forget_factor=1.001)
        self._sync_error_filtered_us: float = 0.0  # Cached filtered error value
        self._last_reanchor_loop_time_us: int = 0
        self._last_sync_error_log_us: int = 0  # Rate limit sync error logging
        self._frames_inserted_since_log: int = 0  # Track inserts for logging
        self._frames_dropped_since_log: int = 0  # Track drops for logging
        self._callback_time_total_us: int = 0  # Total callback time for averaging
        self._callback_count: int = 0  # Number of callbacks for averaging

        # Thread-safe flag for deferred operations (audio thread → main thread)
        self._clear_requested: bool = False

    def set_format(self, audio_format: AudioFormat, device: AudioDevice) -> None:
        """Configure the audio output format.

        Args:
            pcm_format: PCM audio format specification.
            device: Audio device to use.
        """
        pcm_format = audio_format.pcm_format
        self._format = pcm_format
        self._close_stream()

        # Reset state on format change
        self._stream_started = False
        self._first_real_chunk = True

        # Low latency settings for accurate playback (chunks arrive 5+ seconds early)
        self._stream = sounddevice.RawOutputStream(
            samplerate=pcm_format.sample_rate,
            channels=pcm_format.channels,
            dtype=SOUNDDEVICE_DTYPE_MAP[pcm_format.bit_depth],
            blocksize=self._BLOCKSIZE,
            callback=self._audio_callback,
            latency="high",
            device=device.device_id,
        )
        self._output_latency_us = int(self._stream.latency * self._MICROSECONDS_PER_SECOND)
        logger.info(
            "Audio stream configured: codec=%s, sample_rate=%d, channels=%d, bit_depth=%d, blocksize=%d, latency=high, output_latency=%.1f ms, device=%s",
            audio_format.codec.value,
            pcm_format.sample_rate,
            pcm_format.channels,
            pcm_format.bit_depth,
            self._BLOCKSIZE,
            self._output_latency_us / 1000.0,
            device.device_id,
        )

    @property
    def volume(self) -> int:
        """Get the current volume level (0-100)."""
        return self._volume

    @property
    def muted(self) -> bool:
        """Get the current mute state."""
        return self._muted

    def set_volume(self, volume: int, *, muted: bool) -> None:
        """
        Set the player volume and mute state.

        Args:
            volume: Volume level 0-100.
            muted: Whether audio is muted.
        """
        self._volume = max(0, min(100, volume))
        self._muted = muted

    def apply_delay_change(self, delta_us: int) -> None:  # noqa: ARG002
        """Accept a static delay notification while preserving source timestamps.

        The live aiosendspin clock conversions already apply the updated delay.
        The source cursor continues to track the audio frames being consumed;
        existing sync correction follows the changed target on subsequent chunks.
        Keeping this notification side-effect-free also preserves buffered audio
        when queued notifications arrive after further delay changes.

        Args:
            delta_us: Reported delay change in microseconds, already reflected
                by the clock conversion callbacks.
        """

    def is_drained(self) -> bool:
        """Return True when the internal audio queue is empty.

        Thread-safe: called from the worker thread while the PortAudio
        callback thread updates ``_current_chunk``.  Also returns True
        when the stream is not actively playing (nothing to drain).
        """
        # Chunks may be buffered before the stream has started (waiting for
        # the startup buffer to fill); treat them as not-yet-drained so
        # format switches don't skip the drain loop and play stale PCM.
        if not self._queue.empty():
            return False
        if not self._stream_started:
            return True
        return self._current_chunk is None

    def stop(self) -> None:
        """Stop playback and release resources."""
        self._closed = True
        self._close_stream()
        self._stream_executor.shutdown(wait=True)

    def close_stream(self) -> None:
        """Drop queued audio and fully close the stream to release the device.

        Unlike clear(), which only stops the stream (leaving the device FD open),
        this fully closes the PortAudio stream. Call when the server signals
        end-of-stream; the stream will be recreated by set_format() when the
        next track begins.
        """
        if self._closed:
            return
        self.clear()
        self._close_stream()

    def clear(self) -> None:
        """Drop all queued audio chunks."""
        if self._closed:
            return
        # Clear deferred operation flag
        self._clear_requested = False

        # Stop the audio stream (but don't close it) to release ALSA device
        # This allows the device to transition to 'closed' state when paused.
        # The stream restarts when new chunks arrive in submit().
        self._stream_started = False
        stream = self._stream
        if stream is not None:
            self._stream_executor.submit(self._call_stream, stream.stop)

        # Drain all queued chunks
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # Reset playback state
        self._playback_state = PlaybackState.INITIALIZING
        self._first_real_chunk = True
        self._current_chunk = None
        self._current_chunk_offset = 0
        self._expected_next_timestamp = None
        self._underrun_count = 0
        self._queued_duration_us = 0
        # Reset timing calibration for fresh start
        self._dac_loop_calibrations.clear()
        self._last_known_playback_position_us = 0
        self._last_dac_calibration_time_us = 0
        self._scheduled_start_loop_time_us = None
        self._scheduled_start_dac_time_us = None
        self._playback_started_loop_time_us = 0
        self._server_ts_cursor_us = 0
        self._server_ts_cursor_remainder = 0
        self._first_server_timestamp_us = None
        self._early_start_suspect = False
        self._has_reanchored = False
        self._force_reanchor = True
        self._insert_every_n_frames = 0
        self._drop_every_n_frames = 0
        self._frames_until_next_insert = 0
        self._frames_until_next_drop = 0
        self._last_output_frame = b""
        self._sync_error_filter.reset()
        self._sync_error_filtered_us = 0.0
        self._last_reanchor_loop_time_us = 0
        self._last_sync_error_log_us = 0
        self._frames_inserted_since_log = 0
        self._frames_dropped_since_log = 0
        self._callback_time_total_us = 0
        self._callback_count = 0

    def _audio_callback(  # noqa: PLR0915
        self,
        outdata: memoryview,
        frames: int,
        time: AudioTimeInfo,
        status: CallbackFlags,
    ) -> None:
        """
        Audio callback invoked by sounddevice when output buffer needs filling.

        Args:
            outdata: Output buffer to fill with audio data.
            frames: Number of frames requested.
            time: CFFI cdata structure with timing info (outputBufferDacTime, etc).
            status: Status flags (underrun, overflow, etc.).
        """
        callback_start_us = self._now_us()

        assert self._format is not None

        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")

        if status:
            # Detect underflow and request re-anchor (processed by main thread)
            if status.input_underflow or status.output_underflow:
                logger.warning("Audio underflow detected; requesting re-anchor")
                self._clear_requested = True
                # Fill buffer with silence and return early to avoid glitches
                self._fill_silence(output_buffer, 0, bytes_needed)
                return
            logger.debug("Audio callback status: %s", status)

        # Capture exact DAC output time and update playback position
        self._update_playback_position_from_dac(time)

        # Reanchor: snap read cursor to DAC-derived server time so the
        # cursor tracks actual playback position, not bytes-read position.
        if (
            self._playback_state == PlaybackState.PLAYING
            and self._last_known_playback_position_us > 0
            and self._server_ts_cursor_us > 0
            and self._force_reanchor
        ):
            self._server_ts_cursor_us = self._last_known_playback_position_us
            self._server_ts_cursor_remainder = 0
            self._force_reanchor = False
            self._insert_every_n_frames = 0
            self._drop_every_n_frames = 0
            self._frames_until_next_insert = 0
            self._frames_until_next_drop = 0
            self._sync_error_filter.reset()
            self._sync_error_filtered_us = 0.0

        bytes_written = 0

        try:
            # Pre-start gating: fill silence until scheduled start time
            if self._playback_state == PlaybackState.WAITING_FOR_START:
                bytes_written = self._handle_start_gating(
                    output_buffer, bytes_written, frames, time
                )

            # If still waiting after gating, fill remaining buffer with silence
            if self._playback_state == PlaybackState.WAITING_FOR_START:
                if bytes_written < bytes_needed:
                    silence_bytes = bytes_needed - bytes_written
                    self._fill_silence(output_buffer, bytes_written, silence_bytes)
                    bytes_written += silence_bytes
            else:
                frame_size = self._format.frame_size

                # Thread-safe snapshot of correction schedule (prevent mid-callback changes)
                insert_every_n = self._insert_every_n_frames
                drop_every_n = self._drop_every_n_frames

                # Fast path: no sync corrections needed - use bulk operations
                if insert_every_n == 0 and drop_every_n == 0:
                    # Bulk read all frames at once - 15-25x faster than frame-by-frame
                    frames_data = self._read_input_frames_bulk(frames)
                    frames_bytes = len(frames_data)
                    output_buffer[bytes_written : bytes_written + frames_bytes] = frames_data
                    bytes_written += frames_bytes
                else:
                    # Slow path: sync corrections active - process in optimized segments
                    # Reset cadence counters if needed
                    if self._frames_until_next_insert <= 0 and insert_every_n > 0:
                        self._frames_until_next_insert = insert_every_n
                    if self._frames_until_next_drop <= 0 and drop_every_n > 0:
                        self._frames_until_next_drop = drop_every_n

                    if not self._last_output_frame:
                        self._last_output_frame = b"\x00" * frame_size

                    insert_counter = self._frames_until_next_insert
                    drop_counter = self._frames_until_next_drop
                    frames_remaining = frames

                    while frames_remaining > 0:
                        # Calculate frames until next correction event
                        frames_until_insert = (
                            insert_counter if insert_every_n > 0 else frames_remaining + 1
                        )
                        frames_until_drop = (
                            drop_counter if drop_every_n > 0 else frames_remaining + 1
                        )

                        # Find next event and process segment before it
                        next_event_in = min(
                            frames_until_insert, frames_until_drop, frames_remaining
                        )

                        if next_event_in > 0:
                            # Bulk read segment of normal frames
                            segment_data = self._read_input_frames_bulk(next_event_in)
                            segment_bytes = len(segment_data)
                            output_buffer[bytes_written : bytes_written + segment_bytes] = (
                                segment_data
                            )
                            bytes_written += segment_bytes
                            frames_remaining -= next_event_in
                            insert_counter -= next_event_in
                            drop_counter -= next_event_in

                        # Handle correction event if at boundary
                        if frames_remaining > 0:
                            if drop_counter <= 0 and drop_every_n > 0:
                                # Drop one input frame, then output the following frame. This
                                # advances the source cursor by two frames while rendering one,
                                # avoiding the old duplicate-then-skip artifact.
                                _ = self._read_one_input_frame()
                                replacement_frame = self._read_one_input_frame()
                                if replacement_frame is None:
                                    replacement_frame = self._last_output_frame
                                drop_counter = drop_every_n
                                self._frames_dropped_since_log += 1
                                output_buffer[bytes_written : bytes_written + frame_size] = (
                                    replacement_frame
                                )
                                self._last_output_frame = replacement_frame
                                bytes_written += frame_size
                                frames_remaining -= 1
                                insert_counter -= 1
                            elif insert_counter <= 0 and insert_every_n > 0:
                                # Insert frame: output duplicate WITHOUT reading
                                # This makes playback catch up to cursor (cursor doesn't advance)
                                insert_counter = insert_every_n
                                self._frames_inserted_since_log += 1
                                output_buffer[bytes_written : bytes_written + frame_size] = (
                                    self._last_output_frame
                                )
                                bytes_written += frame_size
                                frames_remaining -= 1
                                drop_counter -= 1

                    # Write cadence state back
                    self._frames_until_next_insert = insert_counter
                    self._frames_until_next_drop = drop_counter

        except Exception:
            logger.exception("Error in audio callback")
            # Fill rest with silence on error
            if bytes_written < bytes_needed:
                silence_bytes = bytes_needed - bytes_written
                output_buffer[bytes_written : bytes_written + silence_bytes] = (
                    b"\x00" * silence_bytes
                )
            # Reset partial chunk state on error
            self._current_chunk = None
            self._current_chunk_offset = 0

        # Apply volume scaling to the output
        self._apply_volume(output_buffer)

        # Track callback execution time for performance monitoring
        callback_end_us = self._now_us()
        self._callback_time_total_us += callback_end_us - callback_start_us
        self._callback_count += 1

    def _update_playback_position_from_dac(self, time: AudioTimeInfo) -> None:
        """Capture DAC and loop time simultaneously, update playback position."""
        try:
            dac_time_us = int(time.outputBufferDacTime * 1_000_000)
            loop_time_us = self._now_us()

            # Store complete calibration pair atomically
            self._dac_loop_calibrations.append((dac_time_us, loop_time_us))
            self._last_dac_calibration_time_us = loop_time_us

            # Update playback position in server time using latest calibration
            try:
                # Estimate the loop time that corresponds to the captured DAC time
                loop_at_dac_us = self._estimate_loop_time_for_dac_time(dac_time_us)
                if loop_at_dac_us == 0:
                    loop_at_dac_us = loop_time_us
                estimated_position = self._compute_server_time(loop_at_dac_us)
                self._last_known_playback_position_us = estimated_position
            except Exception:
                logger.exception("Failed to estimate playback position")

            # If we haven't set the DAC-anchored start yet, approximate it now
            if self._scheduled_start_dac_time_us is None and self._first_server_timestamp_us:
                try:
                    est_dac = self._estimate_dac_time_for_server_timestamp(
                        self._first_server_timestamp_us
                    )
                    if est_dac:
                        self._scheduled_start_dac_time_us = est_dac
                except Exception:
                    logger.exception("Failed to estimate DAC start time")
                    self._scheduled_start_dac_time_us = self._scheduled_start_loop_time_us

        except (AttributeError, TypeError):
            # time object may not have expected attributes in all backends
            logger.debug("Could not extract timing info from callback")

    def _initialize_current_chunk(self) -> None:
        """Load next chunk from queue and initialize read position.

        Updates server timestamp cursor if needed.
        """
        self._current_chunk = self._queue.get_nowait()
        self._current_chunk_offset = 0
        # Initialize server cursor if needed
        if self._server_ts_cursor_us == 0:
            self._server_ts_cursor_us = self._current_chunk.server_timestamp_us

    def _read_one_input_frame(self) -> bytes | None:
        """Read and consume a single audio frame from the queue.

        Returns frame bytes or None if no data available.
        Updates internal cursor and buffer duration when chunks are exhausted.
        """
        if self._format is None or self._format.frame_size == 0:
            return None

        frame_size = self._format.frame_size

        # Ensure we have a current chunk
        if self._current_chunk is None:
            try:
                self._initialize_current_chunk()
            except queue.Empty:
                return None

        chunk = self._current_chunk
        assert chunk is not None
        data = chunk.audio_data
        if self._current_chunk_offset >= len(data):
            # Should not happen, but guard
            self._advance_finished_chunk()
            return None

        start = self._current_chunk_offset
        end = start + frame_size
        end = min(end, len(data))
        frame = data[start:end]

        # Advance offsets and timeline cursor
        self._current_chunk_offset = end
        self._advance_server_cursor_frames(1)

        # If chunk finished, advance and update buffered duration tracking
        if self._current_chunk_offset >= len(data):
            self._advance_finished_chunk()

        # Ensure full frame size by padding nulls if needed (shouldn't occur normally)
        if len(frame) < frame_size:
            frame = frame + b"\x00" * (frame_size - len(frame))
        return frame

    def _read_input_frames_bulk(self, n_frames: int) -> bytes:
        """Read N frames efficiently in bulk, handling chunk boundaries.

        Returns concatenated frame data. Much faster than calling
        _read_one_input_frame() N times due to reduced overhead.
        """
        if self._format is None or n_frames <= 0:
            return b""

        frame_size = self._format.frame_size
        total_bytes_needed = n_frames * frame_size
        result = bytearray(total_bytes_needed)
        bytes_written = 0

        while bytes_written < total_bytes_needed:
            # Get frames from current chunk
            if self._current_chunk is None:
                try:
                    self._initialize_current_chunk()
                except queue.Empty:
                    # No more data - pad with silence
                    silence_bytes = total_bytes_needed - bytes_written
                    result[bytes_written:] = b"\x00" * silence_bytes
                    break

            # Calculate how much we can read from current chunk
            assert self._current_chunk is not None
            chunk_data = self._current_chunk.audio_data
            available_bytes = len(chunk_data) - self._current_chunk_offset
            bytes_to_read = min(available_bytes, total_bytes_needed - bytes_written)

            # Bulk copy from chunk to result
            result[bytes_written : bytes_written + bytes_to_read] = chunk_data[
                self._current_chunk_offset : self._current_chunk_offset + bytes_to_read
            ]

            # Update state
            self._current_chunk_offset += bytes_to_read
            bytes_written += bytes_to_read
            frames_read = bytes_to_read // frame_size
            self._advance_server_cursor_frames(frames_read)

            # Check if chunk finished
            if self._current_chunk_offset >= len(chunk_data):
                self._advance_finished_chunk()

        # Save last frame for potential duplication
        if bytes_written >= frame_size:
            self._last_output_frame = bytes(result[bytes_written - frame_size : bytes_written])

        return bytes(result)

    def _advance_finished_chunk(self) -> None:
        """Update durations and state when current chunk is fully consumed."""
        assert self._format is not None
        if self._current_chunk is None:
            return
        data = self._current_chunk.audio_data
        chunk_frames = len(data) // self._format.frame_size
        chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
        self._queued_duration_us = max(0, self._queued_duration_us - chunk_duration_us)
        self._current_chunk = None
        self._current_chunk_offset = 0

    def _advance_server_cursor_frames(self, frames: int) -> None:
        """Advance server timeline cursor by a number of frames."""
        if self._format is None or frames <= 0:
            return
        # Accumulate microseconds precisely: add 1e6 per frame, carry by sample_rate
        self._server_ts_cursor_remainder += frames * 1_000_000
        sr = self._format.sample_rate
        if self._server_ts_cursor_remainder >= sr:
            inc_us = self._server_ts_cursor_remainder // sr
            self._server_ts_cursor_remainder = self._server_ts_cursor_remainder % sr
            self._server_ts_cursor_us += int(inc_us)

    def _skip_input_frames(self, frames_to_skip: int) -> None:
        """Discard frames from the input to reduce buffer depth quickly."""
        if self._format is None or frames_to_skip <= 0:
            return
        frame_size = self._format.frame_size
        while frames_to_skip > 0:
            if self._current_chunk is None:
                try:
                    self._current_chunk = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._current_chunk_offset = 0
                if self._server_ts_cursor_us == 0:
                    self._server_ts_cursor_us = self._current_chunk.server_timestamp_us
            data = self._current_chunk.audio_data
            rem_bytes = len(data) - self._current_chunk_offset
            rem_frames = rem_bytes // frame_size
            if rem_frames <= 0:
                self._advance_finished_chunk()
                continue
            take = min(rem_frames, frames_to_skip)
            self._current_chunk_offset += take * frame_size
            self._advance_server_cursor_frames(take)
            frames_to_skip -= take
            if self._current_chunk_offset >= len(data):
                self._advance_finished_chunk()

    def _estimate_dac_time_for_server_timestamp(self, server_timestamp_us: int) -> int:
        """Estimate when a server timestamp will play out (in DAC time).

        Maps: server_ts → loop_time → dac_time
        """
        # Need at least one calibration point
        if self._last_dac_calibration_time_us == 0:
            return 0

        # Convert server timestamp to client loop time
        loop_time_us = self._compute_client_time(server_timestamp_us)

        # Find calibration point closest to this loop time
        if not self._dac_loop_calibrations:
            return 0

        # Use most recent calibration and previous one (if available) to estimate slope
        dac_ref_us, loop_ref_us = self._dac_loop_calibrations[-1]
        dac_prev_us, loop_prev_us = (0, 0)
        if len(self._dac_loop_calibrations) >= 2:
            dac_prev_us, loop_prev_us = self._dac_loop_calibrations[-2]

        if loop_ref_us == 0:
            # Calibration not yet filled in
            return 0

        # Estimate DAC-per-Loop slope if possible, else assume 1.0
        dac_per_loop = 1.0
        if loop_prev_us and dac_prev_us and (loop_ref_us != loop_prev_us):
            dac_per_loop = (dac_ref_us - dac_prev_us) / (loop_ref_us - loop_prev_us)
            # Clamp to sane bounds to avoid wild extrapolation
            dac_per_loop = max(self._DAC_PER_LOOP_MIN, min(self._DAC_PER_LOOP_MAX, dac_per_loop))

        return round(dac_ref_us + (loop_time_us - loop_ref_us) * dac_per_loop)

    def _estimate_loop_time_for_dac_time(self, dac_time_us: int) -> int:
        """Estimate loop time corresponding to a DAC time using recent calibrations."""
        if not self._dac_loop_calibrations:
            return 0
        dac_ref_us, loop_ref_us = self._dac_loop_calibrations[-1]
        if loop_ref_us == 0:
            return 0
        dac_prev_us, loop_prev_us = (0, 0)
        if len(self._dac_loop_calibrations) >= 2:
            dac_prev_us, loop_prev_us = self._dac_loop_calibrations[-2]
        loop_per_dac = 1.0
        if dac_prev_us and (dac_ref_us != dac_prev_us):
            loop_per_dac = (loop_ref_us - loop_prev_us) / (dac_ref_us - dac_prev_us)
            loop_per_dac = max(self._DAC_PER_LOOP_MIN, min(self._DAC_PER_LOOP_MAX, loop_per_dac))
        return round(loop_ref_us + (dac_time_us - dac_ref_us) * loop_per_dac)

    def _get_current_playback_position_us(self) -> int:
        """Get the current playback position in server timestamp space."""
        return self._last_known_playback_position_us

    def get_timing_metrics(self) -> dict[str, float]:
        """Return current timing metrics for monitoring."""
        return {
            "playback_position_us": float(self._get_current_playback_position_us()),
            "buffered_audio_us": float(self._queued_duration_us),
            "dac_samples_recorded": len(self._dac_loop_calibrations),
            "output_latency_us": float(self._output_latency_us),
        }

    def _log_chunk_timing(self, _server_timestamp_us: int) -> None:
        """Log sync error and buffer status for debugging sync issues."""
        if self._sync_error_filter.is_synchronized:
            now_us = self._now_us()
            if now_us - self._last_sync_error_log_us >= 1_000_000:
                self._last_sync_error_log_us = now_us
                # Calculate playback speed relative to source timeline.
                # Drops skip source frames (track advances faster), inserts repeat
                # frames (track advances slower). Reflect that in the speed metric.
                if self._format is not None:
                    expected_frames = self._format.sample_rate
                    track_frames = (
                        expected_frames
                        + self._frames_dropped_since_log
                        - self._frames_inserted_since_log
                    )
                    playback_speed_percent = (track_frames / expected_frames) * 100.0
                    # Distinct output frames rendered (for info):
                    normal_frames = (
                        expected_frames
                        - self._frames_dropped_since_log
                        + self._frames_inserted_since_log
                    )
                else:
                    playback_speed_percent = 100.0
                    normal_frames = 0

                # Calculate average callback execution time
                avg_callback_us = self._callback_time_total_us / max(self._callback_count, 1)

                logger.debug(
                    "Sync error: %.1f ms, buffer: %.2f s, speed: %.2f%%, "
                    "played: %d, inserted: %d, dropped: %d, callback: %.1f µs",
                    self._sync_error_filtered_us / 1000.0,
                    self._queued_duration_us / 1_000_000,
                    playback_speed_percent,
                    normal_frames,
                    self._frames_inserted_since_log,
                    self._frames_dropped_since_log,
                    avg_callback_us,
                )
                # Reset counters for next logging period
                self._frames_inserted_since_log = 0
                self._frames_dropped_since_log = 0
                self._callback_time_total_us = 0
                self._callback_count = 0

    def _smooth_sync_error(self, error_us: int) -> None:
        """Update Kalman filtered sync error to optimally track error and drift."""
        now_us = self._now_us()
        # Use fixed max_error representing expected jitter/noise (5ms)
        max_error_us = 5_000
        self._sync_error_filter.update(
            measurement=error_us,
            max_error=max_error_us,
            time_added=now_us,
        )
        # Cache filtered offset for use in correction logic
        self._sync_error_filtered_us = self._sync_error_filter.offset

    def _fill_silence(self, output_buffer: memoryview, offset: int, num_bytes: int) -> None:
        """Fill output buffer range with silence."""
        if num_bytes > 0:
            output_buffer[offset : offset + num_bytes] = b"\x00" * num_bytes

    def _apply_volume(self, output_buffer: memoryview) -> None:
        """
        Apply volume scaling to the output buffer.

        Scales audio samples by the current volume level, supporting multiple bit depths.
        """
        muted = self._muted
        volume = self._volume

        if muted or volume == 0:
            # Fill with silence
            self._fill_silence(output_buffer, 0, len(output_buffer))
            return

        if volume == 100:
            return

        # Power curve for natural volume control (gentler at high volumes)
        amplitude = (volume / 100.0) ** 1.5

        bit_depth = self._format.bit_depth if self._format else 16
        num_bytes = len(output_buffer)

        if _c_apply_volume is not None:
            # Fast path: fixed-point volume scaling via C extension
            bytes_per_sample = bit_depth // 8
            scale = int(amplitude * 4294967296)  # 2**32 fixed-point
            _c_apply_volume(output_buffer, bytes_per_sample, scale)
        elif bit_depth == 24:
            self._apply_volume_24bit(output_buffer, num_bytes, amplitude)
        else:
            if bit_depth == 32:
                dtype_str = "int32"
                clip_min, clip_max = -2147483648, 2147483647
            else:  # 16-bit default
                dtype_str = "int16"
                clip_min, clip_max = -32768, 32767

            samples = np.frombuffer(output_buffer[:num_bytes], dtype=dtype_str).copy()
            scaled = np.clip(samples.astype(np.float64) * amplitude, clip_min, clip_max)
            output_buffer[:num_bytes] = scaled.astype(dtype_str).tobytes()

    def _apply_volume_24bit(
        self, output_buffer: memoryview, num_bytes: int, amplitude: float
    ) -> None:
        """Apply volume scaling to packed 24-bit audio data (numpy fallback)."""

        num_samples = num_bytes // 3
        if num_samples == 0:
            return

        raw = np.frombuffer(output_buffer, dtype=np.uint8, count=num_bytes).reshape(-1, 3)
        samples_i32 = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        samples_i32 = np.where(
            samples_i32 & 0x800000, samples_i32 | np.int32(-0x1000000), samples_i32
        )
        scaled = np.clip(samples_i32.astype(np.float64) * amplitude, -8388608, 8388607).astype(
            np.int32
        )
        result = np.empty((num_samples, 3), dtype=np.uint8)
        result[:, 0] = scaled & 0xFF
        result[:, 1] = (scaled >> 8) & 0xFF
        result[:, 2] = (scaled >> 16) & 0xFF
        output_buffer[:num_bytes] = result.tobytes()

    def _compute_and_set_loop_start(self, server_timestamp_us: int) -> None:
        """Compute and set scheduled start time from server timestamp."""
        try:
            self._scheduled_start_loop_time_us = self._compute_client_time(server_timestamp_us)
        except Exception:
            logger.exception("Failed to compute client time for start")
            self._scheduled_start_loop_time_us = self._now_us()

    def _handle_start_gating(
        self,
        output_buffer: memoryview,
        bytes_written: int,
        frames: int,
        time: AudioTimeInfo | None = None,
    ) -> int:
        """Handle pre-start gating using DAC or loop time. Returns bytes written."""
        assert self._format is not None

        # Try DAC-based gating first if time info available
        use_dac_gating = False
        dac_now_us = 0
        if time is not None and self._scheduled_start_dac_time_us is not None:
            try:
                dac_now_us = int(time.outputBufferDacTime * self._MICROSECONDS_PER_SECOND)
                if dac_now_us > 0:
                    use_dac_gating = True
            except (AttributeError, TypeError):
                pass

        if use_dac_gating:
            # DAC-based gating: precise hardware timing
            assert self._scheduled_start_dac_time_us is not None
            delta_us = self._scheduled_start_dac_time_us - dac_now_us
            target_time_us = self._scheduled_start_dac_time_us
            current_time_us = dac_now_us
            can_drop_frames = True  # DAC gating allows frame dropping when late
        elif self._scheduled_start_loop_time_us is not None:
            # Loop-based gating: fallback when DAC timing unavailable
            loop_now_us = self._now_us()
            delta_us = self._scheduled_start_loop_time_us - loop_now_us
            target_time_us = self._scheduled_start_loop_time_us
            current_time_us = loop_now_us
            can_drop_frames = False  # Loop gating waits for DAC calibration
        else:
            return bytes_written

        if delta_us > 0:
            # Not yet time to start: fill with silence
            frames_until_start = int(
                (delta_us * self._format.sample_rate + 999_999) // self._MICROSECONDS_PER_SECOND
            )
            frames_to_silence = min(frames_until_start, frames)
            silence_bytes = frames_to_silence * self._format.frame_size
            self._fill_silence(output_buffer, bytes_written, silence_bytes)
            bytes_written += silence_bytes
        elif delta_us < 0 and can_drop_frames:
            # Late: fast-forward by dropping input frames (DAC gating only)
            if not (self._early_start_suspect and not self._has_reanchored):
                frames_to_drop = int(
                    ((-delta_us) * self._format.sample_rate + 999_999)
                    // self._MICROSECONDS_PER_SECOND
                )
                self._skip_input_frames(frames_to_drop)
                self._set_playing()

        # If we've reached/overrun the scheduled time, arm playback
        if current_time_us >= target_time_us:
            self._set_playing()

        return bytes_written

    def _set_playing(self) -> None:
        """Transition to PLAYING and start the correction grace window once."""
        if self._playback_state != PlaybackState.PLAYING:
            self._playback_started_loop_time_us = self._now_us()
        self._playback_state = PlaybackState.PLAYING

    def _update_correction_schedule(self, error_us: int) -> None:
        """Plan occasional sample drop/insert to correct sync error.

        Uses simple proportional control: correction rate is proportional to error.
        The feedback loop naturally handles both clock drift and accumulated error.

        Positive error means DAC/server playback is ahead of our read cursor;
        schedule drops to catch up. Negative error means we're ahead; schedule
        inserts to slow down. Large errors trigger re-anchoring instead of
        aggressive correction to avoid artifacts.
        """
        if self._format is None or self._format.sample_rate <= 0:
            return

        # Smooth the error to avoid reacting to jitter
        self._smooth_sync_error(error_us)

        abs_err = abs(self._sync_error_filtered_us)

        if self._playback_started_loop_time_us:
            since_start_us = self._now_us() - self._playback_started_loop_time_us
            if since_start_us < self._CORRECTION_START_GRACE_US:
                self._insert_every_n_frames = 0
                self._drop_every_n_frames = 0
                return

        # Do nothing within deadband
        if abs_err <= self._CORRECTION_DEADBAND_US:
            self._insert_every_n_frames = 0
            self._drop_every_n_frames = 0
            return

        # Re-anchor if error is very large and cooldown has elapsed.
        now_loop_us = self._now_us()
        if (
            abs_err > self._REANCHOR_THRESHOLD_US
            and self._playback_state == PlaybackState.PLAYING
            and now_loop_us - self._last_reanchor_loop_time_us > self._REANCHOR_COOLDOWN_US
        ):
            logger.info("Sync error %.1f ms too large; scheduling reanchor", abs_err / 1000.0)
            self._last_reanchor_loop_time_us = now_loop_us
            self._force_reanchor = True
            return

        # Simple proportional control: correction rate proportional to error
        # Target is to fix error within _CORRECTION_TARGET_SECONDS
        frames_error = abs_err * self._format.sample_rate / 1_000_000.0
        desired_corrections_per_sec = frames_error / self._CORRECTION_TARGET_SECONDS

        # Cap at maximum allowed correction rate (4%)
        max_corrections_per_sec = self._format.sample_rate * self._MAX_SPEED_CORRECTION
        corrections_per_sec = min(desired_corrections_per_sec, max_corrections_per_sec)

        # Convert to interval between corrections
        if corrections_per_sec > 0:
            interval_frames = int(self._format.sample_rate / corrections_per_sec)
            interval_frames = max(interval_frames, 1)
        else:
            interval_frames = int(1.0 / max(self._MAX_SPEED_CORRECTION, 0.001))

        # Determine direction based on sign of sync error
        if self._sync_error_filtered_us > 0:
            # We are behind (DAC ahead) -> drop to catch up
            self._drop_every_n_frames = interval_frames
            self._insert_every_n_frames = 0
        else:
            # We are ahead -> insert to slow down
            self._insert_every_n_frames = interval_frames
            self._drop_every_n_frames = 0

    def submit(self, server_timestamp_us: int, payload: bytes | bytearray) -> None:  # noqa: PLR0915
        """
        Queue an audio payload for playback, intelligently handling gaps and overlaps.

        Fills gaps with silence and trims overlaps to ensure a continuous stream.

        Args:
            server_timestamp_us: Server timestamp when this audio should play.
            payload: Raw PCM audio bytes.
        """
        if self._closed:
            return

        # Handle deferred operations from audio thread
        if self._clear_requested:
            self._clear_requested = False
            self.clear()
            logger.info("Cleared audio queue after underflow (deferred from audio thread)")

        if self._format is None:
            logger.debug("Audio format missing; dropping audio chunk")
            return
        if self._format.frame_size == 0:
            return
        if len(payload) % self._format.frame_size != 0:
            logger.warning(
                "Dropping audio chunk with invalid size: %s bytes (frame size %s)",
                len(payload),
                self._format.frame_size,
            )
            return

        now_us = self._now_us()

        # On first real chunk, schedule start time aligned to server timeline
        if self._scheduled_start_loop_time_us is None:
            self._compute_and_set_loop_start(server_timestamp_us)
            # Best-effort DAC schedule; refined later as calibrations accumulate
            est_dac = self._estimate_dac_time_for_server_timestamp(server_timestamp_us)
            # Only set DAC time when we can estimate it; otherwise use loop-based gating
            self._scheduled_start_dac_time_us = est_dac if est_dac else None
            self._playback_state = PlaybackState.WAITING_FOR_START
            self._first_server_timestamp_us = server_timestamp_us
            # Near-now start with unsynced clock = `compute_play_time` fallback;
            # suppress catch-up. Synced near-now is a real mid-stream join.
            # Cast: we just set this via _compute_and_set_loop_start so it's not None
            scheduled_start = cast("int", self._scheduled_start_loop_time_us)
            if (
                scheduled_start - now_us <= self._EARLY_START_THRESHOLD_US
                and not self._is_clock_synced()
            ):
                self._early_start_suspect = True

        # While waiting to start, keep the scheduled loop start updated as time sync improves
        elif (
            self._playback_state == PlaybackState.WAITING_FOR_START
            and self._first_server_timestamp_us is not None
        ):
            try:
                updated_loop_start = self._compute_client_time(self._first_server_timestamp_us)
                # Only update if it moves significantly to avoid churn
                if (
                    abs(updated_loop_start - (self._scheduled_start_loop_time_us or 0))
                    > self._START_TIME_UPDATE_THRESHOLD_US
                ):
                    self._scheduled_start_loop_time_us = updated_loop_start
                    est_dac = self._estimate_dac_time_for_server_timestamp(
                        self._first_server_timestamp_us
                    )
                    self._scheduled_start_dac_time_us = est_dac if est_dac else None
            except Exception:
                logger.exception("Failed to update start time")

        # After calibration, if we have both a DAC-derived playback position and a
        # server-timeline cursor, compute sync error and schedule micro-corrections.
        # Only compute sync error when actively playing (not during initial buffering)
        if (
            self._playback_state == PlaybackState.PLAYING
            and self._last_known_playback_position_us > 0
            and self._server_ts_cursor_us > 0
        ):
            sync_error_us = self._last_known_playback_position_us - self._server_ts_cursor_us
            self._update_correction_schedule(sync_error_us)

        # Log timing information (verbose, for debugging latency issues)
        self._log_chunk_timing(server_timestamp_us)

        # Initialize expected next timestamp on first chunk
        if self._expected_next_timestamp is None:
            self._expected_next_timestamp = server_timestamp_us
        # Handle gap: insert silence to fill the gap
        elif server_timestamp_us > self._expected_next_timestamp:
            gap_us = server_timestamp_us - self._expected_next_timestamp
            gap_frames = (gap_us * self._format.sample_rate) // 1_000_000
            silence_bytes = gap_frames * self._format.frame_size
            silence = b"\x00" * silence_bytes
            self._queue.put_nowait(
                _QueuedChunk(
                    server_timestamp_us=self._expected_next_timestamp,
                    audio_data=silence,
                )
            )
            # Account for inserted silence in buffer duration
            silence_duration_us = (gap_frames * 1_000_000) // self._format.sample_rate
            self._queued_duration_us += silence_duration_us
            logger.debug(
                "Gap: %.1f ms filled with silence",
                gap_us / 1000.0,
            )
            self._expected_next_timestamp = server_timestamp_us

        # Handle overlap: trim the start of the chunk
        elif server_timestamp_us < self._expected_next_timestamp:
            overlap_us = self._expected_next_timestamp - server_timestamp_us
            overlap_frames = (overlap_us * self._format.sample_rate) // 1_000_000
            trim_bytes = overlap_frames * self._format.frame_size
            if trim_bytes < len(payload):
                payload = payload[trim_bytes:]
                server_timestamp_us = self._expected_next_timestamp
                logger.debug(
                    "Overlap: %.1f ms trimmed",
                    overlap_us / 1000.0,
                )
            else:
                # Entire chunk is overlap, skip it
                logger.debug(
                    "Overlap: %.1f ms (chunk skipped, already played)",
                    overlap_us / 1000.0,
                )
                return

        # Queue the chunk
        chunk_duration_us = 0
        if len(payload) > 0:
            # Compute duration from the post-trim payload
            chunk_frames = len(payload) // self._format.frame_size
            chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
            chunk = _QueuedChunk(
                server_timestamp_us=server_timestamp_us,
                audio_data=payload,
            )
            self._queue.put_nowait(chunk)
            # Track duration of queued audio
            self._queued_duration_us += chunk_duration_us
            # Update expected position for next chunk
            self._expected_next_timestamp = server_timestamp_us + chunk_duration_us

        # Start stream once we have enough buffer to avoid immediate underflow
        if (
            not self._stream_started
            and self._stream is not None
            and (
                self._queued_duration_us >= self._MIN_BUFFER_DURATION_US
                or self._queue.qsize() >= self._MIN_CHUNKS_TO_START
            )
        ):
            self._stream_started = True
            self._stream_executor.submit(self._call_stream, self._stream.start)
            logger.info(
                "Stream STARTED: %d chunks, %.2f seconds buffered",
                self._queue.qsize(),
                self._queued_duration_us / 1_000_000,
            )

    def _close_stream(self) -> None:
        """Close the audio output stream."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            self._stream_executor.submit(self._call_stream, stream.stop, stream.close)

    @staticmethod
    def _call_stream(
        *calls: Callable[[], object],
    ) -> None:
        """Run stream operations with exception logging (runs in executor thread)."""
        for call in calls:
            try:
                call()
            except Exception:
                logger.exception("Stream operation %s failed", call.__name__)
