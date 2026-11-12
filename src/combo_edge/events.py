"""Tiny in-process pub/sub used for live dashboard updates (server-sent events).

Scans running in another process (CLI, launchd) write progress to the ``runs`` table; the web
server's poller turns those into the same events."""

from __future__ import annotations

import asyncio
import json
from typing import Any


class Bus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, event: str, data: Any = None) -> None:
        payload = json.dumps(data if data is not None else {}, default=str)
        for q in list(self._subs):
            try:
                q.put_nowait((event, payload))
            except asyncio.QueueFull:
                pass


bus = Bus()


def toast(message: str, level: str = "info") -> None:
    bus.publish("toast", {"message": message, "level": level})
