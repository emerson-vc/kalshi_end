"""Combo resolution: local index first, then the feed index, then POST — within the per-run and
weekly creation caps. Every POST is counted (Kalshi presumably returns the existing combo for a
repeat, but that is undocumented, so it counts against the 5,000/week cap either way)."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from combo_edge.kalshi.client import KalshiError
from combo_edge.pipeline import budget
from combo_edge.sports.nfl.candidates import Candidate, class_key
from combo_edge.storage.db import DB, dumps

PREFERRED_SAME_GAME = ("KXMVECROSSCATEGORY-R",)


class ComboResolver:
    def __init__(self, session: Any, db: DB, settings: Any, universe: Any, *, run_id: int | None, max_new: int | None = None):
        self.session = session
        self.db = db
        self.settings = settings
        self.universe = universe
        self.run_id = run_id
        self.env = session.env.name
        self.max_new = int(max_new if max_new is not None else settings.get("max_new_combos_per_run"))
        self.new_this_run = 0
        self.counts: Counter[str] = Counter()
        self.cap_hit = False
        self._feed_pref = self._feed_collection_preference()

    def _feed_collection_preference(self) -> list[str]:
        rows = self.db.query("SELECT collection, SUM(nfl_same_game) n FROM feed_collections GROUP BY collection ORDER BY n DESC")
        return [r["collection"] for r in rows if r["n"]]

    def choose_collection(self, c: Candidate) -> str | None:
        options = self.universe.collections_for([c.a.leg.event_ticker, c.b.leg.event_ticker])
        if not options:
            return None
        same_game = True
        def rank(name: str) -> tuple:
            feed_rank = self._feed_pref.index(name) if name in self._feed_pref else 99
            pref = 0 if name in PREFERRED_SAME_GAME else 1
            bad = int(same_game and ("SHARD1" in name or "MULTIGAME" in name))
            return (bad, feed_rank, pref, name)
        return sorted(options, key=rank)[0]

    def lookup(self, key: str) -> tuple[str | None, str | None]:
        row = self.db.one("SELECT ticker, source FROM combos WHERE env=? AND legs_key=? ORDER BY CASE source WHEN 'ours' THEN 0 WHEN 'import' THEN 1 ELSE 2 END LIMIT 1", (self.env, key))
        return (row["ticker"], row["source"]) if row else (None, None)

    async def resolve(self, c: Candidate) -> str:
        coll = self.choose_collection(c)
        if coll is None:
            self.counts["no_collection"] += 1
            return "no_collection"
        c.collection = coll
        key = c.key(coll)
        ticker, source = self.lookup(key)
        if ticker:
            c.combo_ticker = ticker
            self.db.execute("UPDATE combos SET last_seen_ts=? WHERE ticker=?", (time.time(), ticker))
            status = "found_local" if source in ("ours", "import") else "found_feed"
            self.counts[status] += 1
            return status
        if self.new_this_run >= self.max_new or budget.creations_left(self.settings, self.db, self.env) <= 0:
            self.cap_hit = True
            self.counts["cap_reached"] += 1
            return "cap_reached"
        self.new_this_run += 1
        sel = c.selections()
        try:
            res = await self.session.client.create_combo(coll, sel)
        except KalshiError as e:
            self._record_creation(coll, key, None, "rejected" if e.status == 400 else "error", e.status, e.message or str(e))
            if e.status == 400:
                ck = class_key(c.a, c.b)
                self.db.execute(
                    "INSERT INTO rejected_classes(class_key,example_json,message,count,first_ts,last_ts) VALUES(?,?,?,1,?,?) "
                    "ON CONFLICT(class_key) DO UPDATE SET count=count+1,last_ts=excluded.last_ts",
                    (ck, dumps({"legs": [c.a.wording, c.b.wording], "tickers": [s["market_ticker"] for s in sel]}), e.message[:300], time.time(), time.time()),
                )
                self.counts["rejected"] += 1
                return "rejected"
            if e.status == 429:
                self.cap_hit = True
                self.counts["cap_reached"] += 1
                return "cap_reached"
            self.counts["error"] += 1
            return "create_error"
        ticker = res.get("market_ticker")
        self._record_creation(coll, key, ticker, "created", 200, "")
        if not ticker:
            self.counts["error"] += 1
            return "create_error"
        c.combo_ticker = ticker
        now = time.time()
        self.db.insert("combos", {
            "ticker": ticker, "env": self.env, "collection": coll, "legs_key": key, "legs_json": sel, "game_id": c.game.id,
            "source": "ours", "created_by_us": 1, "first_seen_ts": now, "last_seen_ts": now,
        }, ignore=True)
        self.counts["created"] += 1
        return "created"

    def _record_creation(self, coll: str, key: str, ticker: str | None, status: str, http: int, msg: str) -> None:
        self.db.insert("creations", {
            "env": self.env, "ts": time.time(), "run_id": self.run_id, "collection": coll, "legs_key": key,
            "combo_ticker": ticker, "status": status, "http_status": http, "message": msg[:300],
        })


def rejected_class_set(db: DB) -> set[str]:
    return {r["class_key"] for r in db.query("SELECT class_key FROM rejected_classes")}


def feed_demand(db: DB, ticker: str | None) -> int:
    if not ticker:
        return 0
    return int(db.scalar("SELECT COALESCE(SUM(count),0) FROM feed_demand WHERE combo_ticker=?", (ticker,), 0))
