"""Visualizer connector for bridging Sendspin client to the TUI visualizer."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

from aiosendspin.models.visualizer import BeatTiming, VisualizerFrame

if TYPE_CHECKING:
    from aiosendspin.client import SendspinClient

logger = logging.getLogger(__name__)


class VisualizerHandler:
    """Bridges between SendspinClient visualizer data and the TUI.

    Receives VisualizerFrame batches from the client, converts timestamps
    to client time, and provides the latest frame for rendering.
    """

    def __init__(
        self,
        on_frame: Callable[[VisualizerFrame], None],
    ) -> None:
        """Initialize the visualizer handler.

        Args:
            on_frame: Callback invoked with the latest frame for display.
        """
        self._on_frame = on_frame
        self._client: SendspinClient | None = None
        self._unsubscribes: list[Callable[[], None]] = []
        self._pending: deque[tuple[int, VisualizerFrame]] = deque()
        self._timer: asyncio.TimerHandle | None = None
        # Latest periodic values, each carried on its own single-type frame.
        # Attached to the next emitted spectrum frame so they ride the same
        # playhead schedule instead of being dropped.
        self._latest_loudness: int | None = None
        self._latest_f_peak_freq: int | None = None

    def attach_client(self, client: SendspinClient) -> None:
        """Attach to a SendspinClient and register listeners."""
        self._client = client
        self._unsubscribes = [
            client.add_visualizer_listener(self._on_visualizer_data),
            client.add_stream_end_listener(self._on_stream_end),
            client.add_stream_clear_listener(self._on_stream_clear),
        ]

    def reset(self) -> None:
        """Clear pending frames and cancel scheduled emissions."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending.clear()
        self._latest_loudness = None
        self._latest_f_peak_freq = None
        self._on_frame(VisualizerFrame(timestamp_us=0))

    def detach(self) -> None:
        """Detach from the client and unregister listeners."""
        for unsub in self._unsubscribes:
            unsub()
        self._unsubscribes = []
        self.reset()
        self._client = None

    def _on_visualizer_data(self, frames: list[VisualizerFrame]) -> None:
        """Handle incoming visualizer frames.

        Each v1 frame carries one field. Spectrum frames are queued on the
        playhead schedule; loudness rides along as the latest value attached to
        the next emitted spectrum frame.
        """
        if not frames:
            return

        if self._client is None:
            return

        # Queue frames by synced server timestamps (independent of local audio delay).
        for frame in frames:
            if frame.loudness is not None:
                self._latest_loudness = frame.loudness
            if frame.f_peak_freq is not None:
                self._latest_f_peak_freq = frame.f_peak_freq
            if frame.spectrum is None:
                continue
            play_time_us = self._client.compute_play_time(frame.timestamp_us)
            self._pending.append((play_time_us, frame))

        if not self._pending:
            return
        self._schedule_next()

    def _on_stream_end(self, roles: list[str] | None) -> None:
        """Handle stream end for visualizer role."""
        if roles is not None and "visualizer" not in roles:
            return
        # reset() also drops the last-seen loudness and f_peak so a stale readout from
        # the previous stream can't survive into the next one.
        self.reset()

    def _on_stream_clear(self, roles: list[str] | None) -> None:
        """Handle stream clear for visualizer role."""
        if roles is not None and "visualizer" not in roles:
            return
        self.reset()

    def _schedule_next(self) -> None:
        """Schedule emission of the next due visualizer frame."""
        if self._client is None or not self._pending:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        now_us = self._client.now_us()
        next_play_us = self._pending[0][0]
        delay_s = max(0.0, (next_play_us - now_us) / 1_000_000.0)
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(delay_s, self._emit_due_frames)

    def _emit_due_frames(self) -> None:
        """Emit the newest frame whose play time is due."""
        self._timer = None
        if self._client is None or not self._pending:
            return

        now_us = self._client.now_us()
        latest_due: VisualizerFrame | None = None
        while self._pending and self._pending[0][0] <= now_us:
            _play_us, frame = self._pending.popleft()
            latest_due = frame

        if latest_due is not None:
            if latest_due.loudness is None:
                latest_due.loudness = self._latest_loudness
            latest_due.f_peak_freq = self._latest_f_peak_freq
            self._on_frame(latest_due)

        if self._pending:
            self._schedule_next()


class BeatHandler:
    """Bridges between SendspinClient beat events and the TUI.

    Beat binary messages arrive ahead of playhead; each beat is scheduled to
    fire at its synced server time so on_beat() is invoked exactly when a beat
    is audible. BeatTiming carries the downbeat flag through to the UI so the
    timeline strip can render bar starts differently.
    """

    def __init__(
        self,
        on_beat: Callable[[BeatTiming], None],
        on_schedule: Callable[[list[BeatTiming]], None] | None = None,
    ) -> None:
        """Initialize the beat handler.

        :param on_beat: Invoked when a scheduled beat is due (BeatTiming).
        :param on_schedule: Invoked with the full list of upcoming beats whenever
            the pending schedule changes (for timeline rendering).
        """
        self._on_beat = on_beat
        self._on_schedule = on_schedule
        self._client: SendspinClient | None = None
        self._unsubscribes: list[Callable[[], None]] = []
        self._pending: deque[BeatTiming] = deque()
        self._timer: asyncio.TimerHandle | None = None

    def attach_client(self, client: SendspinClient) -> None:
        """Attach to a SendspinClient and register listeners.

        Beats arrive through the unified visualizer callback on the v1
        wire: a `VisualizerFrame` whose `is_downbeat` is set carries a
        beat event.
        """
        self._client = client
        self._unsubscribes = [
            client.add_visualizer_listener(self._on_visualizer_data),
            client.add_stream_end_listener(self._on_stream_end),
            client.add_stream_clear_listener(self._on_stream_clear),
        ]

    def reset(self) -> None:
        """Clear pending beats and cancel scheduled emissions."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending.clear()
        if self._on_schedule is not None:
            self._on_schedule([])

    def detach(self) -> None:
        """Detach from the client and unregister listeners."""
        for unsub in self._unsubscribes:
            unsub()
        self._unsubscribes = []
        self.reset()
        self._client = None

    def pending_beats(self) -> list[BeatTiming]:
        """Snapshot of upcoming beats still waiting to fire."""
        return list(self._pending)

    def _on_visualizer_data(self, frames: list[VisualizerFrame]) -> None:
        """Handle incoming visualizer frames, picking out beat events.

        v1 wire: each binary carries one type. Beat frames are identified
        by `is_downbeat is not None`; other types are ignored here.
        Duplicates are filtered by timestamp so a repeated batch doesn't
        draw twice on the timeline.
        """
        if self._client is None:
            return
        existing_ts = {beat.timestamp_us for beat in self._pending}
        added = False
        for frame in frames:
            if frame.is_downbeat is None:
                continue
            if frame.timestamp_us in existing_ts:
                continue
            existing_ts.add(frame.timestamp_us)
            self._pending.append(
                BeatTiming(timestamp_us=frame.timestamp_us, is_downbeat=frame.is_downbeat)
            )
            added = True
        if not added:
            return
        if self._on_schedule is not None:
            self._on_schedule(list(self._pending))
        self._schedule_next()

    def _on_stream_end(self, roles: list[str] | None) -> None:
        """Handle stream end for visualizer role."""
        if roles is not None and "visualizer" not in roles:
            return
        self._cancel_timer()
        self._pending.clear()
        if self._on_schedule is not None:
            self._on_schedule([])

    def _on_stream_clear(self, roles: list[str] | None) -> None:
        """Handle stream clear for visualizer role."""
        if roles is not None and "visualizer" not in roles:
            return
        self._cancel_timer()
        self._pending.clear()
        if self._on_schedule is not None:
            self._on_schedule([])

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        """Schedule emission of the next due beat."""
        if self._client is None or not self._pending:
            return
        self._cancel_timer()

        now_us = self._client.now_us()
        play_us = self._client.compute_play_time(self._pending[0].timestamp_us)
        delay_s = max(0.0, (play_us - now_us) / 1_000_000.0)
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(delay_s, self._emit_due_beats)

    def _emit_due_beats(self) -> None:
        """Emit all beats whose play time is due, then reschedule for the next."""
        self._timer = None
        if self._client is None or not self._pending:
            return
        now_us = self._client.now_us()
        while self._pending and (
            self._client.compute_play_time(self._pending[0].timestamp_us) <= now_us
        ):
            beat = self._pending.popleft()
            self._on_beat(beat)
        if self._on_schedule is not None:
            self._on_schedule(list(self._pending))
        if self._pending:
            self._schedule_next()


class PeakHandler:
    """Bridges between SendspinClient energy-onset (peak) events and the TUI.

    Peaks arrive as single-type visualizer frames carrying ``peak_strength``
    (0-255). Unlike spectrum frames they are events, not periodic samples, so
    each peak is scheduled to fire at its synced play time rather than being
    coalesced. Pending peaks ``(timestamp_us, strength)`` feed the peak strip.
    """

    def __init__(
        self,
        on_peak: Callable[[int, int], None],
        on_schedule: Callable[[list[tuple[int, int]]], None] | None = None,
    ) -> None:
        """Initialize the peak handler.

        :param on_peak: Invoked as ``(timestamp_us, strength)`` when a peak is due.
        :param on_schedule: Invoked with the full pending list whenever it changes.
        """
        self._on_peak = on_peak
        self._on_schedule = on_schedule
        self._client: SendspinClient | None = None
        self._unsubscribes: list[Callable[[], None]] = []
        self._pending: deque[tuple[int, int]] = deque()
        self._timer: asyncio.TimerHandle | None = None

    def attach_client(self, client: SendspinClient) -> None:
        """Attach to a SendspinClient and register listeners."""
        self._client = client
        self._unsubscribes = [
            client.add_visualizer_listener(self._on_visualizer_data),
            client.add_stream_end_listener(self._on_stream_end),
            client.add_stream_clear_listener(self._on_stream_clear),
        ]

    def reset(self) -> None:
        """Clear pending peaks and cancel scheduled emissions."""
        self._cancel_timer()
        self._pending.clear()
        if self._on_schedule is not None:
            self._on_schedule([])

    def detach(self) -> None:
        """Detach from the client and unregister listeners."""
        for unsub in self._unsubscribes:
            unsub()
        self._unsubscribes = []
        self.reset()
        self._client = None

    def _on_visualizer_data(self, frames: list[VisualizerFrame]) -> None:
        """Queue incoming peak frames, ignoring all other visualizer types."""
        if self._client is None:
            return
        existing_ts = {ts for ts, _ in self._pending}
        added = False
        for frame in frames:
            if frame.peak_strength is None:
                continue
            if frame.timestamp_us in existing_ts:
                continue
            existing_ts.add(frame.timestamp_us)
            self._pending.append((frame.timestamp_us, frame.peak_strength))
            added = True
        if not added:
            return
        if self._on_schedule is not None:
            self._on_schedule(list(self._pending))
        self._schedule_next()

    def _on_stream_end(self, roles: list[str] | None) -> None:
        """Handle stream end for visualizer role."""
        if roles is not None and "visualizer" not in roles:
            return
        self.reset()

    def _on_stream_clear(self, roles: list[str] | None) -> None:
        """Handle stream clear for visualizer role."""
        if roles is not None and "visualizer" not in roles:
            return
        self.reset()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        """Schedule emission of the next due peak."""
        if self._client is None or not self._pending:
            return
        self._cancel_timer()

        now_us = self._client.now_us()
        play_us = self._client.compute_play_time(self._pending[0][0])
        delay_s = max(0.0, (play_us - now_us) / 1_000_000.0)
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(delay_s, self._emit_due_peaks)

    def _emit_due_peaks(self) -> None:
        """Emit all peaks whose play time is due, then reschedule for the next."""
        self._timer = None
        if self._client is None or not self._pending:
            return
        now_us = self._client.now_us()
        while self._pending and self._client.compute_play_time(self._pending[0][0]) <= now_us:
            timestamp_us, strength = self._pending.popleft()
            self._on_peak(timestamp_us, strength)
        if self._on_schedule is not None:
            self._on_schedule(list(self._pending))
        if self._pending:
            self._schedule_next()
