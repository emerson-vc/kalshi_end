"""In-process mock of the Kalshi exchange (the *server* side) for offline mode and tests.

It models what v1 learned the hard way:
* some makers quote only identified RFQs; the rest quote anonymous RFQs ~3x worse;
* a maker's new quote replaces (and cancels) its previous one;
* REST lists every quote on an RFQ, cancelled ones included, best price first on the main side;
* deleting an RFQ kills an accepted-but-unconfirmed quote;
* 409 for a second open RFQ on one market, 429 ``open_rfq_limit_exceeded`` at 100 open RFQs;
* combo classes where one leg forces another are rejected;
* makers that never confirm, makers whose contract counts imply the wrong side;
* websocket drops and sequence gaps.

Fixtures are real recorded public API responses (see fixtures/), shifted so games are upcoming.
This file is the simulated exchange; it is the only place outside the execution package that knows
the accept route, because it has to serve it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from combo_edge import money
from combo_edge.kalshi.transport import Response
from combo_edge.paths import MOCK_FIXTURES

PLACEHOLDER_CREATOR = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
MY_COMMS_ID = "mock-me-0001"


@dataclass
class MockMaker:
    id: str
    quotes_anonymous: bool = True
    style: str = "independent"  # independent | correlated
    margin: float = 0.10  # multiplicative markup on YES price
    anon_markup: float = 2.0  # anonymous YES price multiplier (worse for the taker)
    latency: tuple[float, float] = (0.05, 0.6)
    replace_prob: float = 0.3
    confirms: bool = True
    wrong_side_counts: bool = False


def default_makers() -> list[MockMaker]:
    return [
        MockMaker("mkr-alpha", True, "independent", 0.05, 2.6),
        MockMaker("mkr-bravo", True, "correlated", 0.25, 2.2),
        MockMaker("mkr-charlie", False, "independent", 0.02),
        MockMaker("mkr-delta", False, "independent", 0.08),
        MockMaker("mkr-echo", False, "correlated", 0.15),
    ]


@dataclass
class _Rfq:
    id: str
    market_ticker: str
    target_cost: float
    obscure: bool
    creator: str
    created_ts: float
    status: str = "open"
    cancelled_ts: float | None = None
    tasks: list = field(default_factory=list)


class MockWs:
    def __init__(self, ex: "MockExchange"):
        self.ex = ex
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.channels: set[str] = set()
        self.user_filter_self = False
        self.sid = 0
        self.seq = 0
        self.closed = False

    async def send(self, text: str) -> None:
        cmd = json.loads(text)
        if cmd.get("cmd") == "subscribe":
            params = cmd.get("params", {})
            self.channels |= set(params.get("channels", []))
            self.user_filter_self = params.get("user_filter") == "self"
            self.sid = self.ex._next_sid()
            await self.queue.put(json.dumps({"type": "subscribed", "id": cmd.get("id"), "msg": {"channel": ",".join(self.channels), "sid": self.sid}}))

    async def recv(self) -> str:
        if self.closed:
            raise ConnectionError("mock websocket closed")
        msg = await self.queue.get()
        if msg is None:
            self.closed = True
            raise ConnectionError("mock websocket dropped")
        return msg

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    def push(self, mtype: str, msg: dict, *, skip_seq: bool = False) -> None:
        if self.closed:
            return
        self.seq += 2 if skip_seq else 1
        self.queue.put_nowait(json.dumps({"type": mtype, "sid": self.sid, "seq": self.seq, "msg": msg, "sending_ts_ms": int(time.time() * 1000)}))


class MockExchange:
    def __init__(
        self,
        *,
        makers: list[MockMaker] | None = None,
        seed: int = 7,
        time_scale: float = 1.0,
        fixtures_dir: Path = MOCK_FIXTURES,
        first_kickoff_in_h: float = 30.0,
        confirm_window_s: float = 3.0,
        exec_timer_s: float = 1.0,
    ):
        self.rng = random.Random(seed)
        self.makers = makers if makers is not None else default_makers()
        self.time_scale = time_scale
        self.confirm_window_s = confirm_window_s * time_scale
        self.exec_timer_s = exec_timer_s * time_scale
        self.markets: dict[str, dict] = {}
        self.events: dict[str, dict] = {}
        self.milestones: list[dict] = []
        self.collections: list[dict] = []
        self.combos: dict[str, dict] = {}
        self.rfqs: dict[str, _Rfq] = {}
        self.quotes: dict[str, dict] = {}
        self.fills: list[dict] = []
        self.ws_conns: list[MockWs] = []
        self.creations = 0
        self.creation_cap = 5000
        self.open_rfq_limit = 100
        self.api_orders = 0
        self.calls: list[tuple[str, str]] = []
        self.accepts: list[dict] = []
        self.fail_next: dict[str, int] = {}
        self._sid = 0
        self._order_seq = 0
        self._load(fixtures_dir, first_kickoff_in_h)

    # ------------------------------------------------------------------ fixtures
    def _load(self, d: Path, first_in_h: float) -> None:
        ms = json.loads((d / "milestones.json").read_text())["milestones"]
        evs = json.loads((d / "events.json").read_text())["events"]
        self.collections = json.loads((d / "collections.json").read_text())["multivariate_contracts"]
        starts = [_parse_ts(m["start_date"]) for m in ms]
        shift = (time.time() + first_in_h * 3600) - min(starts)
        for m in ms:
            m = json.loads(json.dumps(m))
            m["start_date"] = _iso(_parse_ts(m["start_date"]) + shift)
            self.milestones.append(m)
        for e in evs:
            self.events[e["event_ticker"]] = e
            for mk in e.get("markets", []):
                mk.setdefault("event_ticker", e["event_ticker"])
                mk["price_ranges"] = [{"start": "0.0000", "end": "1.0000", "step": "0.0100"}]
                self.markets[mk["ticker"]] = mk

    # ------------------------------------------------------------------ helpers
    def _next_sid(self) -> int:
        self._sid += 1
        return self._sid

    def ws_connect(self) -> MockWs:
        c = MockWs(self)
        self.ws_conns = [w for w in self.ws_conns if not w.closed]
        self.ws_conns.append(c)
        return c

    def drop_ws(self) -> None:
        for w in self.ws_conns:
            w.queue.put_nowait(None)

    def _broadcast(self, mtype: str, msg: dict, *, owner_only: bool, owner: str, skip_seq: bool = False) -> None:
        for w in self.ws_conns:
            if w.closed or "communications" not in w.channels:
                continue
            if owner_only and owner != MY_COMMS_ID:
                continue
            if not owner_only and w.user_filter_self and owner != MY_COMMS_ID:
                continue
            w.push(mtype, msg, skip_seq=skip_seq)

    def leg_prob(self, ticker: str) -> float:
        m = self.markets.get(ticker, {})
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if bid > 0 and 0 < ask < 1:
            return (bid + ask) / 2
        if 0 < ask < 1:
            return max(ask - 0.02, 0.01)
        if bid > 0:
            return min(bid + 0.02, 0.98)
        return 0.3

    def true_joint(self, legs: list[dict]) -> tuple[float, float]:
        """(independent product, 'true' joint) for the mock's maker pricing."""
        ps = []
        for l in legs:
            p = self.leg_prob(l["market_ticker"])
            ps.append(p if l["side"] == "yes" else 1 - p)
        indep = math.prod(ps)
        if len(legs) == 2:
            a, b = legs
            sa, sb = a["market_ticker"].split("-")[0], b["market_ticker"].split("-")[0]
            pair = {sa, sb}
            tail_team = ({"KXNFLPASSTDS", "KXNFLTEAMTOTAL"} == pair) or ({"KXNFLTD", "KXNFLTEAMTOTAL"} == pair)
            yds = {"KXNFLPASSYDS", "KXNFLRECYDS"} == pair
            mlh = {"KXNFLGAME", "KXNFL1HSPREAD"} == pair
            if (tail_team or yds or mlh) and a["side"] == b["side"] == "yes":
                cond = 0.9 if tail_team else (0.85 if yds else 0.85)
                pa = min(ps)
                return indep, min(pa * cond, min(ps))
        return indep, indep

    # ------------------------------------------------------------------ dispatch
    async def handle(self, method: str, path: str, params: dict | None, body: Any, headers: dict) -> Response:
        path = path.split("?")[0]
        path = re.sub(r"^.*?/trade-api/v2", "", path)
        params = params or {}
        self.calls.append((method, path))
        if "KALSHI-ACCESS-KEY" not in headers and not path.startswith(("/milestones", "/events", "/markets", "/multivariate_event_collections")):
            return Response(401, {"error": {"code": "unauthorized", "message": "missing auth"}})
        key = f"{method} {path}"
        for pat, n in list(self.fail_next.items()):
            if re.match(pat, key) and n > 0:
                self.fail_next[pat] = n - 1
                return Response(503, {"error": {"code": "service_unavailable", "message": "mock outage"}})
        try:
            return await self._route(method, path, params, body)
        except _Err as e:
            return Response(e.status, {"error": {"code": e.code, "message": e.message, "details": e.details}})

    async def _route(self, method: str, path: str, params: dict, body: Any) -> Response:
        m = method
        if m == "GET" and path == "/account/limits":
            return Response(200, {"usage_tier": "basic", "read": {"refill_rate": 200, "bucket_capacity": 200}, "write": {"refill_rate": 100, "bucket_capacity": 100}, "grants": []})
        if m == "GET" and path == "/account/endpoint_costs":
            return Response(200, {"default_cost": 10, "endpoint_costs": [{"method": "GET", "path": "/trade-api/v2/communications/rfqs/{rfq_id}/quotes/{quote_id}", "cost": 2}]})
        if m == "POST" and path == "/account/api_usage_level/upgrade":
            if self.api_orders < 1:
                raise _Err(403, "forbidden", "None of your last 100 orders were created via the API")
            return Response(201, {})
        if m == "GET" and path == "/communications/id":
            return Response(200, {"communications_id": MY_COMMS_ID})
        if m == "GET" and path == "/exchange/status":
            return Response(200, {"exchange_active": True, "trading_active": True})
        if m == "GET" and path == "/milestones":
            out = [x for x in self.milestones if (not params.get("type") or x["type"] == params["type"])]
            if params.get("minimum_start_date"):
                lo = _parse_ts(params["minimum_start_date"])
                out = [x for x in out if _parse_ts(x["start_date"]) >= lo]
            return Response(200, {"milestones": out, "cursor": ""})
        if m == "GET" and path == "/events":
            tickers = [t for t in str(params.get("tickers") or params.get("event_ticker") or "").split(",") if t]
            evs = [self._event_payload(t, params) for t in tickers if t in self.events]
            return Response(200, {"events": evs, "cursor": ""})
        if m == "GET" and path.startswith("/events/"):
            t = path.split("/")[2]
            if t not in self.events:
                raise _Err(404, "not_found", "event not found")
            return Response(200, {"event": self._event_payload(t, params)})
        if m == "GET" and path == "/markets":
            tickers = [t for t in str(params.get("tickers", "")).split(",") if t]
            out = [self._market_payload(t) for t in tickers if t in self.markets or t in self.combos]
            return Response(200, {"markets": out, "cursor": ""})
        if m == "GET" and path.startswith("/markets/"):
            t = path.split("/")[2]
            if t not in self.markets and t not in self.combos:
                raise _Err(404, "not_found", "market not found")
            return Response(200, {"market": self._market_payload(t)})
        if m == "GET" and path == "/multivariate_event_collections":
            return Response(200, {"multivariate_contracts": self.collections, "cursor": ""})
        if m == "GET" and path.startswith("/multivariate_event_collections/"):
            t = path.split("/")[2]
            for c in self.collections:
                if c["collection_ticker"] == t:
                    return Response(200, {"multivariate_contract": c})
            raise _Err(404, "not_found", "collection not found")
        if m == "POST" and path.startswith("/multivariate_event_collections/"):
            return Response(200, self._create_combo(path.split("/")[2], body or {}))
        if m == "POST" and path == "/communications/rfqs":
            return Response(201, {"id": self._create_rfq(body or {})})
        if m == "GET" and path == "/communications/rfqs":
            out = [self._rfq_payload(r) for r in self.rfqs.values() if (not params.get("status") or r.status == params["status"])]
            if params.get("user_filter") == "self":
                out = [r for r in out if r["creator_id"] == MY_COMMS_ID]
            return Response(200, {"rfqs": out, "cursor": ""})
        mm = re.match(r"^/communications/rfqs/([^/]+)$", path)
        if mm and m == "GET":
            r = self.rfqs.get(mm.group(1))
            if not r:
                raise _Err(404, "not_found", "rfq not found")
            return Response(200, {"rfq": self._rfq_payload(r)})
        if mm and m == "DELETE":
            self._delete_rfq(mm.group(1))
            return Response(204, None)
        if m == "GET" and path == "/communications/quotes":
            return Response(200, {"quotes": self._list_quotes(params.get("rfq_id")), "cursor": ""})
        mq = re.match(r"^/communications/rfqs/([^/]+)/quotes/([^/]+)$", path)
        if mq and m == "GET":
            q = self.quotes.get(mq.group(2))
            if not q or q["rfq_id"] != mq.group(1):
                raise _Err(404, "not_found", "quote not found")
            return Response(200, {"quote": self._quote_payload(q)})
        ma = re.match(r"^/communications/rfqs/([^/]+)/quotes/([^/]+)/accept$", path)
        if ma and m == "PUT":
            self._accept(ma.group(1), ma.group(2), (body or {}).get("accepted_side"))
            return Response(204, None)
        if m == "GET" and path == "/portfolio/fills":
            out = self.fills
            if params.get("order_id"):
                out = [f for f in out if f["order_id"] == params["order_id"]]
            if params.get("ticker"):
                ts = set(str(params["ticker"]).split(","))
                out = [f for f in out if f["ticker"] in ts]
            if params.get("min_ts"):
                out = [f for f in out if f["ts"] >= int(params["min_ts"])]
            return Response(200, {"fills": out, "cursor": ""})
        if m == "GET" and path == "/portfolio/balance":
            return Response(200, {"balance": 100000, "portfolio_value": 0})
        if m == "GET" and path in ("/portfolio/settlements", "/portfolio/positions"):
            return Response(200, {"settlements": [], "market_positions": [], "cursor": ""})
        raise _Err(404, "not_found", f"mock has no route for {m} {path}")

    # ------------------------------------------------------------------ payloads
    def _event_payload(self, t: str, params: dict) -> dict:
        e = dict(self.events[t])
        if str(params.get("with_nested_markets", "false")).lower() != "true":
            e.pop("markets", None)
        return e

    def _market_payload(self, t: str) -> dict:
        if t in self.markets:
            return self.markets[t]
        c = self.combos[t]
        return {
            "ticker": t, "event_ticker": c["event_ticker"], "status": "active", "mve_collection_ticker": c["collection"],
            "mve_selected_legs": c["legs"], "price_ranges": money.DEFAULT_COMBO_RANGES, "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0000", "volume_fp": "0.00", "result": "",
        }

    def _rfq_payload(self, r: _Rfq) -> dict:
        c = self.combos.get(r.market_ticker, {})
        return {
            "id": r.id, "creator_id": r.creator, "market_ticker": r.market_ticker, "contracts_fp": "0.00",
            "target_cost_dollars": f"{r.target_cost:.4f}", "status": r.status, "created_ts": _iso(r.created_ts),
            "mve_collection_ticker": c.get("collection"), "mve_selected_legs": c.get("legs"), "rest_remainder": False,
        }

    def _quote_payload(self, q: dict) -> dict:
        out = {k: v for k, v in q.items() if not k.startswith("_")}
        out["rfq_creator_id"] = PLACEHOLDER_CREATOR if q["_obscure"] else MY_COMMS_ID
        return out

    # ------------------------------------------------------------------ combos
    def _create_combo(self, collection: str, body: dict) -> dict:
        coll = next((c for c in self.collections if c["collection_ticker"] == collection), None)
        if coll is None:
            raise _Err(404, "not_found", "collection not found")
        legs = body.get("selected_markets") or []
        if len(legs) < 2:
            raise _Err(400, "invalid_parameters", "Select at least 2 markets to create a combo.")
        allowed = {a["ticker"]: a for a in coll["associated_events"]}
        per_event: dict[str, int] = {}
        for l in legs:
            if l["event_ticker"] not in allowed:
                raise _Err(400, "invalid_parameters", "A selected market's event is not in this collection.", l["market_ticker"])
            if l["market_ticker"] not in self.markets:
                raise _Err(400, "invalid_parameters", "A selected market is inactive.", l["market_ticker"])
            per_event[l["event_ticker"]] = per_event.get(l["event_ticker"], 0) + 1
            smax = allowed[l["event_ticker"]].get("size_max")
            if smax is not None and per_event[l["event_ticker"]] > smax:
                raise _Err(400, "invalid_parameters", "Too many markets selected from one event.", l["event_ticker"])
        if collection.endswith("SHARD1-R"):
            games = {l["event_ticker"].split("-")[1] for l in legs}
            if len(games) < len(legs):
                raise _Err(400, "invalid_parameters", "This collection only accepts legs from different games.")
        reason = self._forced_reason(legs)
        if reason:
            raise _Err(400, "invalid_parameters", reason, ",".join(l["market_ticker"] for l in legs))
        self.creations += 1
        if self.creations > self.creation_cap:
            raise _Err(429, "too_many_requests", "weekly combo creation limit reached")
        key = "|".join(sorted(f"{l['market_ticker']}:{l['side']}" for l in legs))
        h = hashlib.sha1((collection + key).encode()).hexdigest()[:10].upper()
        series = "KXMVECROSSCATEGORY0" if collection == "KXMVECROSSCATEGORY-R" else collection.removesuffix("-R")
        ticker = f"{series}-S2026{h}-{h[:6]}"
        if ticker not in self.combos:
            self.combos[ticker] = {"legs": [dict(l) for l in legs], "collection": collection, "event_ticker": f"{series}-S2026{h}"}
        return {"event_ticker": self.combos[ticker]["event_ticker"], "market_ticker": ticker}

    def _forced_reason(self, legs: list[dict]) -> str | None:
        """Reject combos where one leg logically forces another (same class Kalshi rejects)."""
        info = [(l, self.markets[l["market_ticker"]]) for l in legs]
        for la, ma in info:
            for lb, mb in info:
                if la is lb or la["side"] != "yes" or lb["side"] != "yes":
                    continue
                sa, sb = la["market_ticker"].split("-")[0], lb["market_ticker"].split("-")[0]
                ta = (ma.get("custom_strike") or {}).get("football_team")
                tb = (mb.get("custom_strike") or {}).get("football_team")
                if sa in ("KXNFLPASSTDS", "KXNFLTD") and sb == "KXNFLTEAMTOTAL" and ta == tb:
                    k = math.floor(float(ma["floor_strike"])) + 1
                    if 6 * k > float(mb["floor_strike"]):
                        return "One selected market implies another; this combination isn't available."
                if sa == "KXNFLSPREAD" and sb == "KXNFLGAME" and ta == tb:
                    return "One selected market implies another; this combination isn't available."
        return None

    # ------------------------------------------------------------------ RFQs and quotes
    def _create_rfq(self, body: dict) -> str:
        t = body.get("market_ticker")
        if "rest_remainder" not in body:
            raise _Err(400, "invalid_parameters", "rest_remainder is required")
        if t not in self.combos and t not in self.markets:
            raise _Err(400, "rfq_market_not_open", "market is not open")
        if sum(1 for r in self.rfqs.values() if r.status == "open" and r.creator == MY_COMMS_ID) >= self.open_rfq_limit:
            raise _Err(429, "open_rfq_limit_exceeded", "You already have 100 open RFQs")
        if any(r.status == "open" and r.market_ticker == t and r.creator == MY_COMMS_ID for r in self.rfqs.values()):
            raise _Err(409, "conflict", "An open RFQ already exists on this market")
        target = float(body.get("target_cost_dollars") or 0)
        r = _Rfq(str(uuid.uuid4()), t, target, bool(body.get("obscure_creator_id")), MY_COMMS_ID, time.time())
        self.rfqs[r.id] = r
        self._broadcast("rfq_created", {"id": r.id, "creator_id": PLACEHOLDER_CREATOR if r.obscure else MY_COMMS_ID, "market_ticker": t, "target_cost_dollars": f"{target:.4f}", "created_ts": _iso(r.created_ts), **({"mve_collection_ticker": self.combos[t]["collection"], "mve_selected_legs": self.combos[t]["legs"]} if t in self.combos else {})}, owner_only=False, owner=MY_COMMS_ID)
        if t in self.combos:
            for mk in self.makers:
                if r.obscure and not mk.quotes_anonymous:
                    continue
                r.tasks.append(asyncio.ensure_future(self._maker_flow(r, mk)))
        return r.id

    def _maker_price(self, r: _Rfq, mk: MockMaker, improve: float = 1.0) -> float:
        legs = self.combos[r.market_ticker]["legs"]
        indep, true = self.true_joint(legs)
        base = indep if mk.style == "independent" else true
        yes = base * (1 + mk.margin) * improve
        if r.obscure:
            yes *= mk.anon_markup
        yes = min(max(yes, 0.0010), 0.97)
        return money.snap(yes, money.DEFAULT_COMBO_RANGES, up=True)

    async def _maker_flow(self, r: _Rfq, mk: MockMaker) -> None:
        try:
            await asyncio.sleep(self.rng.uniform(*mk.latency) * self.time_scale)
            if r.status != "open":
                return
            self._post_quote(r, mk, self._maker_price(r, mk))
            if self.rng.random() < mk.replace_prob:
                await asyncio.sleep(self.rng.uniform(*mk.latency) * self.time_scale)
                if r.status == "open":
                    self._post_quote(r, mk, self._maker_price(r, mk, improve=0.97))
        except asyncio.CancelledError:
            return

    def _post_quote(self, r: _Rfq, mk: MockMaker, yes_price: float) -> dict:
        now = time.time()
        for q in self.quotes.values():  # replacement cancels the maker's previous quote
            if q["rfq_id"] == r.id and q["creator_id"] == mk.id and q["status"] == "open":
                q["status"] = "cancelled"
                q["cancelled_ts"] = _iso(now)
                q["cancellation_reason"] = "replaced"
        no_bid = round(1 - yes_price, 4)
        yes_bid = max(round(yes_price - max(0.002, yes_price * 0.4), 4), 0.0)
        c_buy_yes = money.contracts_for_target(r.target_cost, 1 - no_bid)
        c_buy_no = money.contracts_for_target(r.target_cost, 1 - yes_bid) if yes_bid > 0 else 0.0
        if mk.wrong_side_counts:
            c_buy_yes, c_buy_no = c_buy_no, c_buy_yes
        q = {
            "id": str(uuid.uuid4()), "rfq_id": r.id, "creator_id": mk.id, "market_ticker": r.market_ticker,
            "contracts_fp": "0.00", "yes_bid_dollars": f"{yes_bid:.4f}", "no_bid_dollars": f"{no_bid:.4f}",
            "created_ts": _iso(now), "updated_ts": _iso(now), "status": "open", "rfq_target_cost_dollars": f"{r.target_cost:.4f}",
            "yes_contracts_fp": f"{c_buy_no:.2f}", "no_contracts_fp": f"{c_buy_yes:.2f}", "rest_remainder": False,
            "_obscure": r.obscure, "_maker": mk, "_created": now,
        }
        self.quotes[q["id"]] = q
        self._broadcast(
            "quote_created",
            {"quote_id": q["id"], "rfq_id": r.id, "quote_creator_id": mk.id, "rfq_creator_id": PLACEHOLDER_CREATOR if r.obscure else MY_COMMS_ID,
             "market_ticker": r.market_ticker, "yes_bid_dollars": q["yes_bid_dollars"], "no_bid_dollars": q["no_bid_dollars"],
             "yes_contracts_offered_fp": q["yes_contracts_fp"], "no_contracts_offered_fp": q["no_contracts_fp"],
             "rfq_target_cost_dollars": q["rfq_target_cost_dollars"], "created_ts": q["created_ts"]},
            owner_only=True, owner=MY_COMMS_ID,
        )
        return q

    def _list_quotes(self, rfq_id: str | None) -> list[dict]:
        qs = [q for q in self.quotes.values() if rfq_id is None or q["rfq_id"] == rfq_id]
        qs.sort(key=lambda q: (-float(q["no_bid_dollars"]), q["_created"]))  # target-cost RFQ: main side is "no"
        return [self._quote_payload(q) for q in qs]

    def _delete_rfq(self, rfq_id: str) -> None:
        r = self.rfqs.get(rfq_id)
        if not r:
            raise _Err(404, "not_found", "rfq not found")
        if r.status != "open":
            return
        r.status = "closed"
        r.cancelled_ts = time.time()
        for t in r.tasks:
            t.cancel()
        for q in self.quotes.values():
            if q["rfq_id"] == rfq_id and q["status"] in ("open", "accepted"):
                q["status"] = "cancelled"
                q["cancellation_reason"] = "rfq_deleted"
                q["cancelled_ts"] = _iso(time.time())
        self._broadcast("rfq_deleted", {"id": r.id, "creator_id": MY_COMMS_ID, "market_ticker": r.market_ticker, "deleted_ts": _iso(time.time())}, owner_only=False, owner=MY_COMMS_ID)

    def _accept(self, rfq_id: str, quote_id: str, side: str | None) -> None:
        r = self.rfqs.get(rfq_id)
        q = self.quotes.get(quote_id)
        if not r or r.status != "open":
            raise _Err(400, "RFQ_CLOSED", "RFQ was deleted, expired, or already executed")
        if not q or q["rfq_id"] != rfq_id:
            raise _Err(404, "not_found", "quote not found")
        if q["status"] != "open":
            raise _Err(400, "invalid_parameters", "quote is not open")
        if side not in ("yes", "no"):
            raise _Err(400, "invalid_parameters", "accepted_side must be yes or no")
        q["status"] = "accepted"
        q["accepted_side"] = side
        q["accepted_ts"] = _iso(time.time())
        contracts = q["no_contracts_fp"] if side == "no" else q["yes_contracts_fp"]
        self.accepts.append({"rfq_id": rfq_id, "quote_id": quote_id, "side": side, "contracts": contracts})
        self._broadcast(
            "quote_accepted",
            {"quote_id": quote_id, "rfq_id": rfq_id, "quote_creator_id": q["creator_id"], "rfq_creator_id": MY_COMMS_ID,
             "market_ticker": r.market_ticker, "yes_bid_dollars": q["yes_bid_dollars"], "no_bid_dollars": q["no_bid_dollars"],
             "accepted_side": side, "contracts_accepted_fp": contracts, "yes_contracts_offered_fp": q["yes_contracts_fp"],
             "no_contracts_offered_fp": q["no_contracts_fp"], "rfq_target_cost_dollars": q["rfq_target_cost_dollars"]},
            owner_only=True, owner=MY_COMMS_ID,
        )
        asyncio.ensure_future(self._confirm_flow(r, q, side, float(contracts)))

    async def _confirm_flow(self, r: _Rfq, q: dict, side: str, contracts: float) -> None:
        mk: MockMaker = q["_maker"]
        if not mk.confirms:
            await asyncio.sleep(self.confirm_window_s)
            if q["status"] == "accepted":
                q["status"] = "cancelled"
                q["cancellation_reason"] = "confirm_timeout"
            return
        await asyncio.sleep(self.confirm_window_s * 0.2)
        if q["status"] != "accepted":
            return  # RFQ deleted before the maker confirmed: killed
        q["status"] = "confirmed"
        q["confirmed_ts"] = _iso(time.time())
        await asyncio.sleep(self.exec_timer_s)
        q["status"] = "executed"
        q["executed_ts"] = _iso(time.time())
        r.status = "closed"
        self._order_seq += 1
        oid = f"mock-order-{self._order_seq}"
        q["rfq_creator_order_id"] = oid
        self.api_orders += 1
        yes_price = 1 - float(q["no_bid_dollars"]) if side == "no" else float(q["yes_bid_dollars"])
        outcome = "yes" if side == "no" else "no"
        price_paid = yes_price if outcome == "yes" else 1 - float(q["yes_bid_dollars"])
        self.fills.append({
            "fill_id": str(uuid.uuid4()), "trade_id": "", "order_id": oid, "ticker": r.market_ticker, "market_ticker": r.market_ticker,
            "outcome_side": outcome, "book_side": "bid" if outcome == "yes" else "ask", "count_fp": f"{contracts:.2f}",
            "yes_price_dollars": f"{yes_price:.4f}", "no_price_dollars": f"{1 - yes_price:.4f}", "is_taker": True,
            "fee_cost": f"{money.trade_fee(contracts, price_paid):.6f}", "created_time": _iso(time.time()), "ts": int(time.time()),
        })
        self._broadcast(
            "quote_executed",
            {"quote_id": q["id"], "rfq_id": r.id, "quote_creator_id": mk.id, "rfq_creator_id": MY_COMMS_ID, "order_id": oid,
             "client_order_id": "", "market_ticker": r.market_ticker, "executed_ts": q["executed_ts"]},
            owner_only=True, owner=MY_COMMS_ID,
        )

    # ------------------------------------------------------------------ public feed (other users)
    def emit_public_rfq(self, legs: list[dict], *, collection: str = "KXMVECROSSCATEGORY-R", batch: bool = False, obscure: bool = True) -> str:
        res = self._create_combo(collection, {"selected_markets": legs})
        self.creations -= 1  # someone else's creation
        rid = str(uuid.uuid4())
        msg = {"id": rid, "creator_id": PLACEHOLDER_CREATOR if obscure else "other-user", "market_ticker": res["market_ticker"],
               "created_ts": _iso(time.time()), "mve_collection_ticker": collection, "mve_selected_legs": legs}
        if batch:
            msg["rfq_batch_id"] = "batch-" + rid[:8]
        self._broadcast("rfq_created", msg, owner_only=False, owner="other-user")
        return res["market_ticker"]

    def settle(self, combo_ticker: str, result: str) -> None:
        self.combos[combo_ticker]["result"] = result


class _Err(Exception):
    def __init__(self, status: int, code: str, message: str, details: str = ""):
        self.status, self.code, self.message, self.details = status, code, message, details


class MockTransport:
    """Transport that routes the real client's calls into a MockExchange."""

    def __init__(self, exchange: MockExchange):
        self.exchange = exchange

    async def request(self, method: str, url: str, *, params: dict | None, json_body: Any, headers: dict) -> Response:
        await asyncio.sleep(0)
        return await self.exchange.handle(method, url, params, json_body, headers)

    async def ws_connect(self, url: str, headers: dict) -> MockWs:
        if "KALSHI-ACCESS-KEY" not in headers:
            raise ConnectionError("401: websocket requires authentication")
        return self.exchange.ws_connect()

    async def aclose(self) -> None:
        return None


def _parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
