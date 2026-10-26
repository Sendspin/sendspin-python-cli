"""Custom SendspinServer with embedded web player."""

from __future__ import annotations

from importlib.resources import files
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Any

from aiohttp import web
from aiosendspin.noise import InMemoryServerPairingStore, TrustedUnpairedClient
from aiosendspin.server import SendspinServer


class PartyPairingStore(InMemoryServerPairingStore):
    """Pairing store that approves every client for unpaired playback.

    A party admits anyone who opens the page, so no operator approves clients.
    """

    async def trusted_unpaired(self, client_id: str) -> TrustedUnpairedClient | None:
        """Return the stored approval, or a new approval for any other client."""
        return await super().trusted_unpaired(client_id) or TrustedUnpairedClient(client_id)


class SendspinPlayerServer(SendspinServer):
    """SendspinServer that serves an embedded web player at /."""

    def __init__(self, *, total_listeners: Synchronized[int] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._total_listeners = total_listeners

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
