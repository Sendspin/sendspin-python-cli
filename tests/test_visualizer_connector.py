from __future__ import annotations

import asyncio

from aiosendspin.models.visualizer import VisualizerFrame

from sendspin.visualizer_connector import VisualizerHandler


class _FakeClient:
    def __init__(self) -> None:
        self.visualizer_listeners: list[object] = []
        self.stream_start_listeners: list[object] = []
        self.stream_end_listeners: list[object] = []
        self.stream_clear_listeners: list[object] = []

    def compute_play_time(self, timestamp_us: int) -> int:
        return timestamp_us

    def now_us(self) -> int:
        # Far ahead so any queued frame is immediately due.
        return 10_000_000

    def add_visualizer_listener(self, callback: object):
        return self._add(self.visualizer_listeners, callback)

    def add_stream_start_listener(self, callback: object):
        return self._add(self.stream_start_listeners, callback)

    def add_stream_end_listener(self, callback: object):
        return self._add(self.stream_end_listeners, callback)

    def add_stream_clear_listener(self, callback: object):
        return self._add(self.stream_clear_listeners, callback)

    @staticmethod
    def _add(callbacks: list[object], callback: object):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)


def test_loudness_only_frame_reaches_emitted_spectrum_frame() -> None:
    """Loudness arrives on its own single-type frame and must not be dropped:
    its value rides along on the next emitted spectrum frame."""

    async def exercise() -> None:
        received: list[VisualizerFrame] = []
        handler = VisualizerHandler(on_frame=received.append)
        client = _FakeClient()
        handler.attach_client(client)
        on_data = client.visualizer_listeners[0]

        on_data([VisualizerFrame(timestamp_us=1, loudness=40000)])
        on_data([VisualizerFrame(timestamp_us=2, spectrum=[1, 2, 3])])
        await asyncio.sleep(0.02)  # let the scheduled emission fire

        assert received, "no frame was emitted"
        emitted = received[-1]
        assert emitted.spectrum == [1, 2, 3]
        assert emitted.loudness == 40000

    asyncio.run(exercise())


def test_stream_clear_drops_stale_f_peak() -> None:
    """stream/clear resets the last-seen f_peak so it can't ride into the next stream."""

    async def exercise() -> None:
        received: list[VisualizerFrame] = []
        handler = VisualizerHandler(on_frame=received.append)
        client = _FakeClient()
        handler.attach_client(client)
        on_data = client.visualizer_listeners[0]
        on_clear = client.stream_clear_listeners[0]

        on_data([VisualizerFrame(timestamp_us=1, f_peak_freq=440)])
        on_clear(None)
        on_data([VisualizerFrame(timestamp_us=2, spectrum=[1, 2, 3])])
        await asyncio.sleep(0.02)  # let the scheduled emission fire

        emitted = received[-1]
        assert emitted.spectrum == [1, 2, 3]
        assert emitted.f_peak_freq is None

    asyncio.run(exercise())
