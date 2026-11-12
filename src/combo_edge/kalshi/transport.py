"""Transports: real (httpx + websockets) and in-process mock. The client is identical for both."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Response:
    status: int
    data: Any
    text: str = ""


class WsConnection(Protocol):
    async def send(self, text: str) -> None: ...
    async def recv(self) -> str: ...
    async def close(self) -> None: ...


class Transport(Protocol):
    async def request(self, method: str, url: str, *, params: dict | None, json_body: Any, headers: dict) -> Response: ...
    async def ws_connect(self, url: str, headers: dict) -> WsConnection: ...
    async def aclose(self) -> None: ...


class HttpxTransport:
    def __init__(self, timeout: float = 15.0):
        import httpx

        self._client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "combo-edge/2"})

    async def request(self, method: str, url: str, *, params: dict | None, json_body: Any, headers: dict) -> Response:
        r = await self._client.request(method, url, params=params, json=json_body, headers=headers)
        try:
            data = r.json() if r.content else None
        except json.JSONDecodeError:
            data = None
        return Response(r.status_code, data, r.text[:2000])

    async def ws_connect(self, url: str, headers: dict) -> WsConnection:
        from websockets.asyncio.client import connect

        ws = await connect(url, additional_headers=headers, ping_interval=20, ping_timeout=20, max_size=2**23)
        return _RealWs(ws)

    async def aclose(self) -> None:
        await self._client.aclose()


class _RealWs:
    def __init__(self, ws: Any):
        self._ws = ws

    async def send(self, text: str) -> None:
        await self._ws.send(text)

    async def recv(self) -> str:
        msg = await self._ws.recv()
        return msg if isinstance(msg, str) else msg.decode()

    async def close(self) -> None:
        await self._ws.close()
