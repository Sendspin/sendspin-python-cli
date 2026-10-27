"""Custom SendspinServer with embedded web player."""

from __future__ import annotations

from importlib.resources import files
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Any

from aiohttp import web
from aiosendspin.noise import InMemoryServerPairingStore, TrustedUnpairedClient
from aiosendspin.server import (
    ClientConnectedEvent,
    ClientDisconnectedEvent,
    SendspinEvent,
    SendspinServer,
)

from sendspin.utils import create_task


class PartyPairingStore(InMemoryServerPairingStore):
    """Pairing store that approves every client for unpaired playback.

    A party admits anyone who opens the page, so no operator approves clients.
    """

    async def trusted_unpaired(self, client_id: str) -> TrustedUnpairedClient | None:
        """Return the stored approval, or a new approval for any other client."""
        return await super().trusted_unpaired(client_id) or TrustedUnpairedClient(client_id)


class SendspinPlayerServer(SendspinServer):
    """SendspinServer that serves an embedded web player at /."""

    def __init__(
        self,
        *,
        total_listeners: Synchronized[int] | None = None,
        allow_unencrypted: bool = False,
        **kwargs: Any,
    ) -> None:
        # aiosendspin refuses an unencrypted client whose id the store already approves.
        pairing_store = InMemoryServerPairingStore() if allow_unencrypted else PartyPairingStore()
        super().__init__(pairing_store=pairing_store, allow_unencrypted=allow_unencrypted, **kwargs)
        self._total_listeners = total_listeners
        if allow_unencrypted:
            self.add_event_listener(self._trust_encrypted_clients_while_connected)

    def _trust_encrypted_clients_while_connected(
        self, server: SendspinServer, event: SendspinEvent
    ) -> None:
        if isinstance(event, ClientDisconnectedEvent):
            # Forget disconnected clients so the store doesn't grow with every visitor.
            create_task(self.untrust_unpaired(event.client_id))
            return
        if not isinstance(event, ClientConnectedEvent):
            return
        client = self.get_client(event.client_id)
        if client is not None and client.connection is not None and client.connection.is_encrypted:
            # Not eager, so the trust refresh runs once the connection is bound to the client.
            create_task(self.trust_unpaired(event.client_id), eager_start=False)

    def _create_web_application(self) -> web.Application:
        """Create web app with embedded player and static file serving."""
        app = super()._create_web_application()

        # Get path to web assets directory
        web_path = Path(str(files("sendspin.serve.web")))

        total_listeners = self._total_listeners
        server_ref = self

        # Serve index.html at root
        async def index_handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(web_path / "index.html")

        async def status_handler(request: web.Request) -> web.Response:
            if total_listeners is not None:
                count = total_listeners.value
            else:
                count = len(server_ref.connected_clients)
            return web.json_response(
                {"total_clients": count},
                headers={"Access-Control-Allow-Origin": "*"},
            )

        app.router.add_get("/", index_handler)
        app.router.add_get("/api/status", status_handler)

        # Serve other static files (css, js)
        app.router.add_static("/", web_path)

        return app
