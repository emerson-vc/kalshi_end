"""Read-only feed recorder (24/7 under launchd).

Subscribes to the public ``rfq_created`` stream (all users) and ``multivariate_market_lifecycle``.
NFL same-game combos other people created go into the combo index for free (so scans can skip a
POST), with daily demand counts per combo and which collections NFL same-game combos use. RFQ
batch children are skipped (they carry a placeholder creator). Lifecycle ``created`` events carry
no legs, so they are only counted.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any

from combo_edge.storage.db import DB
from combo_edge.timeutil import local_day

log = logging.getLogger(__name__)
PLACEHOLDER = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def game_code(event_ticker: str) -> str:
    parts = event_ticker.split("-")
    return parts[1] if len(parts) > 1 else ""


class Recorder:
    def __init__(self, db: DB, env: str):
        self.db = db
        self.env = env
        self.seen = 0
        self.nfl_same_game = 0
        self.lifecycle_created = 0

    def on_rfq_created(self, msg: dict, ts: float) -> None:
        m = msg.get("msg") or {}
        self.seen += 1
        if m.get("rfq_batch_id"):
            return
        legs = m.get("mve_selected_legs") or []
        coll = m.get("mve_collection_ticker")
        ticker = m.get("market_ticker")
        if len(legs) < 2 or not coll or not ticker:
            return
        evs = [l.get("event_ticker", "") for l in legs]
        if not all(e.startswith("KXNFL") for e in evs):
            return
        same_game = len({game_code(e) for e in evs}) == 1
        day = local_day(ts)
        if same_game:
            self.nfl_same_game += 1
            self.db.execute("INSERT INTO feed_collections(collection,day,nfl_same_game) VALUES(?,?,1) ON CONFLICT(collection,day) DO UPDATE SET nfl_same_game=nfl_same_game+1", (coll, day))
        from combo_edge.sports.base import legs_key

        key = legs_key(coll, [{"market_ticker": l["market_ticker"], "side": l.get("side", "yes")} for l in legs])
        self.db.execute(
            "INSERT INTO combos(ticker,env,collection,legs_key,legs_json,game_id,source,created_by_us,first_seen_ts,last_seen_ts) "
            "VALUES(?,?,?,?,?,?,?,0,?,?) ON CONFLICT(ticker) DO UPDATE SET last_seen_ts=excluded.last_seen_ts",
            (ticker, self.env, coll, key, __import__("json").dumps(legs), game_code(evs[0]), "feed", ts, ts),
        )
        obscured = int(m.get("creator_id") == PLACEHOLDER)
        self.db.execute(
            "INSERT INTO feed_demand(combo_ticker,day,count,obscured) VALUES(?,?,1,?) ON CONFLICT(combo_ticker,day) DO UPDATE SET count=count+1, obscured=obscured+excluded.obscured",
            (ticker, day, obscured),
        )

    def on_lifecycle(self, msg: dict, ts: float) -> None:
        if (msg.get("msg") or {}).get("event_type") == "created":
            self.lifecycle_created += 1

    def status(self) -> dict:
        return {"seen": self.seen, "nfl_same_game": self.nfl_same_game, "lifecycle_created": self.lifecycle_created}


async def run_recorder(session: Any, db: DB, *, duration_s: float | None = None) -> dict:
    from combo_edge.kalshi.ws import WsManager

    rec = Recorder(db, session.env.name)
    ws = WsManager(session.env, session.transport, session.signer, channels=("communications", "multivariate_market_lifecycle"), user_filter_self=False)
    ws.on("rfq_created", rec.on_rfq_created)
    ws.on("multivariate_market_lifecycle", rec.on_lifecycle)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    await ws.start(wait=15)
    start = time.time()
    try:
        while not stop.is_set():
            db.set_state("recorder_heartbeat", {"ts": time.time(), "pid": __import__("os").getpid(), **rec.status(), "ws": ws.status()})
            if duration_s and time.time() - start >= duration_s:
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=min(30.0, duration_s or 30.0))
            except asyncio.TimeoutError:
                pass
    finally:
        await ws.stop()
    return rec.status()
