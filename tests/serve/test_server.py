"""Tests for SendspinPlayerServer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from aiosendspin.noise import Identity
from aiosendspin.server import ClientConnectedEvent, ClientDisconnectedEvent

from sendspin.serve.server import SendspinPlayerServer


async def test_allow_unencrypted_trusts_only_encrypted_clients() -> None:
    server = SendspinPlayerServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="Test",
        allow_unencrypted=True,
    )
    clients = {
        client_id: MagicMock(
            connection=MagicMock(is_encrypted=encrypted, refresh_trusted_unpaired=AsyncMock())
        )
        for client_id, encrypted in (("encrypted", True), ("legacy", False))
    }

    with patch.object(server, "get_client", side_effect=clients.get):
        for client_id in clients:
            server._signal_event(ClientConnectedEvent(client_id))
        await asyncio.sleep(0)

    assert await server.pairing_store.trusted_unpaired("encrypted") is not None
    assert await server.pairing_store.trusted_unpaired("legacy") is None

    with patch.object(server, "get_client", return_value=None):
        server._signal_event(ClientDisconnectedEvent("encrypted", None))
        await asyncio.sleep(0)

    assert await server.pairing_store.trusted_unpaired("encrypted") is None
    await server.close()
