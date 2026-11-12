"""Authenticated websocket manager with reconnect and gap detection via ``seq``.

Any disconnect or sequence gap is recorded as a "blink" interval; the quote collector falls back to
REST for every RFQ whose quote window overlapped a blink.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from combo_edge.kalshi.auth import Signer
from combo_edge.kalshi.transport import Transport
from combo_edge.settings import EnvConfig

log = logging.getLogger(__name__)
WS_SIGN_PATH = "/trade-api/ws/v2"
Handler = Callable[[dict, float], Awaitable[None] | None]


class WsManager:
    def __init__(
        self,
        env: EnvConfig,
        transport: Transport,
        signer: Signer | None,
        *,
        channels: tuple[str, ...] = ("communications",),
        user_filter_self: bool = True,
    ):
        self.env = env
        self.transport = transport
        self.signer = signer
        self.channels = channels
        self.user_filter_self = user_filter_self
        self.handlers: dict[str, list[Handler]] = defaultdict(list)
        self.connected = asyncio.Event()
        self.blinks: list[list[float | None]] = []  # [down_ts, up_ts or None]
        self.last_seq: dict[int, int] = {}
        self.messages = 0
        self.reconnects = 0
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None
        self._conn: Any = None
        self._stop = False
        self._cmd_id = 0

    def on(self, msg_type: str, fn: Handler) -> None:
        self.handlers[msg_type].append(fn)

    async def start(self, wait: float = 10.0) -> bool:
        self._stop = False
        self._task = asyncio.create_task(self._run(), name="ws-manager")
        try:
            await asyncio.wait_for(self.connected.wait(), timeout=wait)
            return True
        except asyncio.TimeoutError:
            return False

    async def stop(self) -> None:
        self._stop = True
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def blinked_between(self, t0: float, t1: float) -> bool:
        for down, up in self.blinks:
            up_ts = up if up is not None else float("inf")
            if down <= t1 and up_ts >= t0:  # type: ignore[operator]
                return True
        return not self.connected.is_set()

    def _mark_down(self, why: str) -> None:
        if self.connected.is_set():
            self.connected.clear()
            self.blinks.append([time.time(), None])
        self.last_error = why

    def _mark_up(self) -> None:
        if self.blinks and self.blinks[-1][1] is None:
            self.blinks[-1][1] = time.time()
        self.connected.set()

    async def _subscribe(self) -> None:
        self._cmd_id += 1
        params: dict[str, Any] = {"channels": list(self.channels)}
        if self.user_filter_self and "communications" in self.channels:
            params["user_filter"] = "self"
        await self._conn.send(json.dumps({"id": self._cmd_id, "cmd": "subscribe", "params": params}))

    async def _run(self) -> None:
        backoff = 0.5
        while not self._stop:
            try:
                headers = self.signer.headers("GET", WS_SIGN_PATH) if self.signer else {}
                self._conn = await self.transport.ws_connect(self.env.ws_url, headers)
                self.last_seq.clear()
                await self._subscribe()
                backoff = 0.5
                while not self._stop:
                    raw = await self._conn.recv()
                    await self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # disconnects and protocol errors
                self._mark_down(f"{type(e).__name__}: {e}")
                if self._stop:
                    break
                self.reconnects += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def _dispatch(self, raw: str) -> None:
        now = time.time()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        mtype = msg.get("type", "")
        if mtype == "subscribed":
            self._mark_up()
        elif mtype == "error":
            self.last_error = json.dumps(msg.get("msg"))[:300]
            log.warning("ws error: %s", self.last_error)
        sid, seq = msg.get("sid"), msg.get("seq")
        if isinstance(sid, int) and isinstance(seq, int):
            prev = self.last_seq.get(sid)
            if prev is not None and seq != prev + 1:
                # gap: record a zero-length blink so overlapping quote windows re-check via REST
                self.blinks.append([now, now])
            self.last_seq[sid] = seq
        self.messages += 1
        for fn in self.handlers.get(mtype, []) + self.handlers.get("*", []):
            try:
                r = fn(msg, now)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                log.exception("ws handler failed for %s", mtype)

    def status(self) -> dict:
        return {
            "connected": self.connected.is_set(),
            "messages": self.messages,
            "reconnects": self.reconnects,
            "blinks": len(self.blinks),
            "last_error": self.last_error,
        }
