"""Tests for the Chromecast serve connector.

``pychromecast`` lives in the optional ``cast`` extra and is not installed by the
CI test job, so skip the module when it is unavailable.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pychromecast")

import sendspin.serve.chromecast as chromecast  # noqa: E402


class _RecordingBrowser:
    def __init__(self) -> None:
        self.stopped = False

    def stop_discovery(self) -> None:
        self.stopped = True


def test_connect_stops_discovery_when_blocking_connect_times_out(monkeypatch) -> None:
    """A discovery that outlives the timeout must not leak its zeroconf browser."""
    browser = _RecordingBrowser()

    cast = MagicMock()
    cast.cast_info.host = "10.0.0.5"
    # Outlast the (patched) timeout, then finish so the executor can shut down.
    cast.wait.side_effect = lambda: time.sleep(0.5)

    def fake_get_chromecasts(**kwargs: object) -> tuple[list[object], _RecordingBrowser]:
        return [cast], browser

    monkeypatch.setattr(chromecast.pychromecast, "get_chromecasts", fake_get_chromecasts)
    monkeypatch.setattr(chromecast, "_CONNECT_TIMEOUT_SECONDS", 0.05)

    async def exercise() -> None:
        with pytest.raises(TimeoutError):
            await chromecast.connect_to_chromecast(
                "cast://10.0.0.5:8009",
                "http://10.0.0.1:8928",
                "player-1",
            )

    asyncio.run(exercise())

    # Asserted while the blocking thread is still sleeping: only the timeout
    # path can have stopped discovery by now.
    assert browser.stopped is True
