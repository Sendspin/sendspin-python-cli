"""Coordinator for multi-worker serve mode.

The coordinator is the main process. It spawns worker subprocesses, decodes
the audio source, and fans out PCM chunks with shared timestamps to all workers.
Workers are standalone HTTP+WS servers; put a reverse proxy in front for load balancing.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue as _queue
import signal
import time
from contextlib import suppress

from aiosendspin.server.push_stream import DEFAULT_INITIAL_DELAY_US

from sendspin.serve import get_local_ip
from sendspin.serve.ipc import (
    AudioChunk,
    Shutdown,
    WorkerClientConnected,
    WorkerClientCount,
    WorkerError,
    WorkerListening,
)
from sendspin.serve.source import decode_audio
from sendspin.serve.worker import worker_main

logger = logging.getLogger(__name__)

# Keep at least 5s of buffer on workers/connected clients
MAX_BUFFER_AHEAD_US = 5_000_000


def server_now_us() -> int:
    """Current time in microseconds on the aiosendspin server clock domain.

    The aiosendspin server timestamps audio with ``RawMonotonicClock``
    (``CLOCK_MONOTONIC_RAW`` on Linux), which — unlike
    ``time.monotonic()`` — is not slewed by NTP/adjtime
    (``aiosendspin/clock.py``). ``play_start_us`` is an absolute timestamp
    consumed verbatim by workers/clients, so it must be sampled from the
    same clock domain; ``time.monotonic()`` drifts away from it by every
    NTP adjustment since boot (measured 102 s apart on one Linux host).
    Interval-only measurements (stats/health timers) stay on
    ``time.monotonic()``, which is the correct clock for durations.
    """
    if hasattr(time, "CLOCK_MONOTONIC_RAW"):
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) // 1_000
    return time.monotonic_ns() // 1_000


class ServeCoordinator:
    """Orchestrates multi-worker serve mode."""

    def __init__(
        self,
        *,
        source: str,
        source_format: str | None,
        port: int,
        name: str,
        workers: int,
        log_level: str,
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.port = port
        self.name = name
        self.workers = workers
        self.log_level = log_level
        self.worker_ports = [port + i for i in range(workers)]

        # IPC
        self._ctx = mp.get_context("spawn")
        self._audio_queues: list[mp.Queue] = []  # type: ignore[type-arg]
        self._status_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        self._processes: list[mp.process.BaseProcess] = []
        self._total_listeners: mp.sharedctypes.Synchronized[int] = self._ctx.Value("i", 0)

        # State
        self._client_counts: dict[int, int] = {}
        self._shutdown_requested = False
        self._run_task: asyncio.Task[int] | None = None
        self._failed_workers: set[int] = set()
        self._reported_crashed: set[int] = set()

    async def run(self) -> int:
        """Main coordinator loop."""
        loop = asyncio.get_running_loop()

        with suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGINT, self._handle_sigint)

        self._run_task = asyncio.current_task()

        try:
            self._spawn_workers()
            ready = await self._wait_for_workers_listening()
            if ready == 0:
                print("Error: all workers failed to start")  # noqa: T201
                return 1
            self._print_banner()

            while not self._shutdown_requested:
                # Wait for at least one client on any worker
                await self._consume_status_until_client()
                if self._shutdown_requested:
                    break  # type: ignore[unreachable]

                # Decode and fan out audio until all clients disconnect
                await self._stream_audio_loop()

        except asyncio.CancelledError:
            pass
        finally:
            self._run_task = None
            await self._shutdown()

        return 0

    def _handle_sigint(self) -> None:
        if self._shutdown_requested:
            # Second Ctrl+C — force exit
            return
        self._shutdown_requested = True
        print("\nShutting down...")  # noqa: T201
        # Cancel the run task to break out of blocking audio decode
        if self._run_task is not None:
            self._run_task.cancel()

    def _spawn_workers(self) -> None:
        """Spawn worker subprocesses."""
        for i in range(self.workers):
            audio_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
            self._audio_queues.append(audio_queue)

            p = self._ctx.Process(
                target=worker_main,
                args=(
                    i,
                    self.worker_ports[i],
                    audio_queue,
                    self._status_queue,
                    self._total_listeners,
                    self.log_level,
                ),
            )
            p.start()
            self._processes.append(p)

        logger.info("Spawned %d worker processes", self.workers)

    async def _wait_for_workers_listening(self) -> int:
        """Wait for all workers to report status. Returns count of healthy workers."""
        loop = asyncio.get_running_loop()
        listening_count = 0
        error_count = 0
        failed_workers: set[int] = set()

        while (listening_count + error_count) < self.workers:
            msg = await loop.run_in_executor(None, self._status_queue.get)

            if isinstance(msg, WorkerListening):
                listening_count += 1
                logger.info(
                    "Worker %d listening on port %d (%d/%d)",
                    msg.worker_id,
                    msg.port,
                    listening_count,
                    self.workers,
                )
            elif isinstance(msg, WorkerError):
                error_count += 1
                failed_workers.add(msg.worker_id)
                logger.error("Worker %d error during startup: %s", msg.worker_id, msg.error)

        self._failed_workers = failed_workers
        self._reported_crashed.update(failed_workers)

        if failed_workers and self._audio_queues:
            # Remove audio queues for failed workers so we don't push into dead queues
            for wid in sorted(failed_workers, reverse=True):
                if wid < len(self._audio_queues):
                    self._audio_queues.pop(wid)

        return listening_count

    async def _consume_status_until_client(self) -> None:
        """Process status messages until at least one client connects or shutdown."""
        loop = asyncio.get_running_loop()

        def _get_with_timeout() -> object | None:
            try:
                result: object = self._status_queue.get(timeout=0.5)
            except _queue.Empty:
                return None
            return result

        while not self._shutdown_requested:
            msg = await loop.run_in_executor(None, _get_with_timeout)
            if msg is None:
                continue
            self._handle_status_message(msg)
            if isinstance(msg, WorkerClientConnected):
                return

    def _handle_status_message(self, msg: object) -> None:
        """Handle a single status message from a worker."""
        if isinstance(msg, WorkerClientConnected):
            logger.info("Client %s connected to worker %d", msg.client_id, msg.worker_id)
        elif isinstance(msg, WorkerClientCount):
            self._client_counts[msg.worker_id] = msg.count
            self._total_listeners.value = sum(self._client_counts.values())
        elif isinstance(msg, WorkerError):
            logger.error("Worker %d error: %s", msg.worker_id, msg.error)

    async def _drain_status_queue(self) -> None:
        """Non-blocking drain of pending status messages."""
        while True:
            try:
                msg = self._status_queue.get_nowait()
                self._handle_status_message(msg)
            except _queue.Empty:
                break

    def _check_worker_health(self) -> None:
        """Check for crashed workers - shut down if any died."""
        for i, proc in enumerate(self._processes):
            if i in self._failed_workers:
                continue
            if not proc.is_alive() and i not in self._reported_crashed:
                self._reported_crashed.add(i)
                port = self.worker_ports[i]
                print(f"[health] Worker {i} (port {port}) crashed, shutting down")  # noqa: T201
                self._shutdown_requested = True
                if self._run_task is not None:
                    self._run_task.cancel()
                return

    def _log_client_stats(self) -> None:
        """Print per-worker client counts to console."""
        total = sum(self._client_counts.values())
        parts = [f"W{wid}={count}" for wid, count in sorted(self._client_counts.items())]
        summary = ", ".join(parts) if parts else "no workers reporting"
        print(f"[stats] {total} clients connected ({summary})")  # noqa: T201

    async def _stream_audio_loop(self) -> None:
        """Decode audio and fan out PCM chunks to all workers.

        Returns when all clients disconnect or shutdown is requested.
        """
        consecutive_errors = 0
        last_stats_time = time.monotonic()

        while not self._shutdown_requested:
            try:
                audio_source = await decode_audio(self.source, source_format=self.source_format)
                fmt = audio_source.format

                play_start_us = server_now_us() + DEFAULT_INITIAL_DELAY_US

                async for pcm_chunk in audio_source.generator:
                    if self._shutdown_requested:
                        break  # type: ignore[unreachable]

                    await self._drain_status_queue()

                    # Pause if all clients disconnected
                    if self._total_listeners.value == 0:
                        logger.info("All clients disconnected, pausing playback")
                        return

                    now = time.monotonic()
                    if now - last_stats_time >= 30.0:
                        self._check_worker_health()
                        self._log_client_stats()
                        last_stats_time = now

                    frame_stride = (fmt.bit_depth // 8) * fmt.channels
                    sample_count = len(pcm_chunk) // frame_stride
                    chunk_duration_us = int(sample_count * 1_000_000 / fmt.sample_rate)

                    chunk_msg = AudioChunk(
                        pcm_bytes=pcm_chunk,
                        sample_rate=fmt.sample_rate,
                        bit_depth=fmt.bit_depth,
                        channels=fmt.channels,
                        play_start_us=play_start_us,
                    )
                    for queue in self._audio_queues:
                        queue.put(chunk_msg)

                    play_start_us += chunk_duration_us

                    now_us = server_now_us()
                    ahead_us = play_start_us - DEFAULT_INITIAL_DELAY_US - now_us
                    if ahead_us > MAX_BUFFER_AHEAD_US:
                        await asyncio.sleep((ahead_us - MAX_BUFFER_AHEAD_US) / 1_000_000)

                consecutive_errors = 0

            except asyncio.CancelledError:
                return
            except FileNotFoundError as e:
                print(f"Error: {e}")  # noqa: T201
                self._shutdown_requested = True
                return
            except Exception as e:  # noqa: BLE001
                consecutive_errors += 1
                delay = min(2**consecutive_errors, 30)
                print(f"Playback error: {e}")  # noqa: T201
                logger.debug("Playback error", exc_info=True)
                print(f"Retrying in {delay}s...")  # noqa: T201
                await asyncio.sleep(delay)

    def _print_banner(self) -> None:
        """Print worker URLs."""
        local_ip = get_local_ip()

        healthy_workers = [i for i in range(self.workers) if i not in self._failed_workers]

        print(f"\nMulti-worker server running ({len(healthy_workers)} workers)")  # noqa: T201
        for i in healthy_workers:
            port = self.worker_ports[i]
            print(f"  Worker {i}: http://{local_ip}:{port}/")  # noqa: T201
        print("\nPlace a reverse proxy in front of the worker ports for load balancing.")  # noqa: T201
        print("Press Ctrl+C to quit\n")  # noqa: T201

    async def _shutdown(self) -> None:
        """Gracefully shut down all workers."""
        for queue in self._audio_queues:
            with suppress(Exception):
                queue.put(Shutdown())

        for p in self._processes:
            p.join(timeout=5.0)

        for p in self._processes:
            if p.is_alive():
                p.terminate()
        for p in self._processes:
            p.join(timeout=2.0)
