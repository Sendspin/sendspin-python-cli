"""Worker subprocess for multi-worker serve mode.

Each worker runs a SendspinPlayerServer on its assigned port, groups incoming
clients, and receives PCM chunks from the coordinator via a multiprocessing Queue.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from contextlib import suppress

from aiosendspin.noise import Identity, InMemoryServerPairingStore
from aiosendspin.server import (
    ClientAddedEvent,
    ClientRemovedEvent,
    SendspinEvent,
    SendspinGroup,
    SendspinServer,
)
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.push_stream import PushStream, StreamStoppedError

from sendspin.serve.ipc import (
    AudioChunk,
    Shutdown,
    WorkerClientConnected,
    WorkerClientCount,
    WorkerError,
    WorkerListening,
)
from sendspin.serve.server import SendspinPlayerServer
from sendspin.utils import create_task

logger = logging.getLogger(__name__)


class ServeWorker:
    """A single server worker running in a subprocess."""

    def __init__(
        self,
        *,
        worker_id: int,
        port: int,
        audio_queue: mp.Queue,  # type: ignore[type-arg]
        status_queue: mp.Queue,  # type: ignore[type-arg]
        total_listeners: Synchronized[int],
    ) -> None:
        self.worker_id = worker_id
        self.port = port
        self._audio_queue = audio_queue
        self._status_queue = status_queue
        self._total_listeners = total_listeners
        self._server: SendspinPlayerServer | None = None
        self._active_group: SendspinGroup | None = None
        self._stream: PushStream | None = None

    async def run(self) -> None:
        """Main worker loop: start server, consume audio queue."""
        loop = asyncio.get_running_loop()

        try:
            await self._start_server()
            self._status_queue.put(WorkerListening(worker_id=self.worker_id, port=self.port))

            # Consume audio chunks from coordinator
            while True:
                msg = await loop.run_in_executor(None, self._audio_queue.get)

                if isinstance(msg, Shutdown):
                    logger.info("[W%d] Shutdown received", self.worker_id)
                    break

                if isinstance(msg, AudioChunk):
                    stream = self._get_stream()
                    if stream is None:
                        continue
                    fmt = AudioFormat(
                        sample_rate=msg.sample_rate,
                        bit_depth=msg.bit_depth,
                        channels=msg.channels,
                    )
                    try:
                        stream.prepare_audio(msg.pcm_bytes, fmt)
                        await stream.commit_audio(play_start_us=msg.play_start_us)
                    except StreamStoppedError:
                        # All clients disconnected — stream was stopped.
                        # Clear both so the next client starts a fresh group/stream.
                        self._stream = None
                        self._active_group = None

        except Exception as e:
            logger.exception("[W%d] Worker error", self.worker_id)
            self._status_queue.put(WorkerError(worker_id=self.worker_id, error=str(e)))
        finally:
            await self._shutdown_server()

    async def _start_server(self) -> None:
        """Start the SendspinPlayerServer on this worker's port."""
        loop = asyncio.get_running_loop()
        self._server = SendspinPlayerServer(
            loop=loop,
            identity=Identity.generate(),
            server_name=f"Sendspin Worker {self.worker_id}",
            pairing_store=InMemoryServerPairingStore(),
            total_listeners=self._total_listeners,
        )
        self._server.add_event_listener(self._on_server_event)
        await self._server.start_server(
            port=self.port,
            advertise_addresses=[],
            discover_clients=False,
        )
        logger.info("[W%d] Listening on port %d", self.worker_id, self.port)

    def _on_server_event(self, server: SendspinServer, event: SendspinEvent) -> None:
        """Handle client connect/disconnect events."""
        if isinstance(event, ClientAddedEvent):
            client = server.get_client(event.client_id)
            if client is None:
                return

            logger.info("[W%d] Client connected: %s", self.worker_id, event.client_id)

            if self._active_group is None:
                self._active_group = client.group
                self._stream = self._active_group.start_stream()
            else:
                create_task(self._active_group.add_client(client))

            self._status_queue.put(
                WorkerClientConnected(worker_id=self.worker_id, client_id=event.client_id)
            )
            self._report_client_count()

        elif isinstance(event, ClientRemovedEvent):
            logger.info("[W%d] Client disconnected: %s", self.worker_id, event.client_id)
            # If no clients remain, tear down so the next client starts fresh
            if self._active_group is not None and not [
                c for c in self._active_group.clients if c.client_id != event.client_id
            ]:
                if self._stream is not None:
                    with suppress(Exception):
                        self._stream.stop()
                    self._stream = None
                self._active_group = None
            self._report_client_count()

    def _report_client_count(self) -> None:
        """Send current client count to coordinator."""
        if self._server is None:
            return
        count = len(self._server.connected_clients)
        self._status_queue.put(WorkerClientCount(worker_id=self.worker_id, count=count))

    def _get_stream(self) -> PushStream | None:
        """Get the active PushStream, or None if no clients are connected."""
        return self._stream

    async def _shutdown_server(self) -> None:
        """Gracefully shut down the server."""
        if self._stream is not None:
            with suppress(Exception):
                self._stream.stop()
        if self._server is not None:
            with suppress(Exception):
                await self._server.close()


def worker_main(
    worker_id: int,
    port: int,
    audio_queue: mp.Queue,  # type: ignore[type-arg]
    status_queue: mp.Queue,  # type: ignore[type-arg]
    total_listeners: Synchronized[int],
    log_level: str,
) -> None:
    """Entry point for the worker subprocess."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=f"%(asctime)s %(levelname)s [W{worker_id}] %(message)s",
    )
    worker = ServeWorker(
        worker_id=worker_id,
        port=port,
        audio_queue=audio_queue,
        status_queue=status_queue,
        total_listeners=total_listeners,
    )
    asyncio.run(worker.run())
