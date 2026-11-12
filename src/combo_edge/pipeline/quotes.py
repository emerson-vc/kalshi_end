"""RFQs and quotes, with identity on everything.

* The authenticated websocket is opened before any RFQ and kept up for the run.
* Quotes are matched to RFQs by id; quotes that arrive before the create call returns are buffered.
* A maker's new quote replaces its previous one: per maker we keep the newest *open* quote by the
  maker's ``created_ts`` (never by local receive time — v1's bug).
* If the socket blinked during an RFQ's window (or for every identified RFQ), quotes are re-read over
  REST, which lists every quote (cancelled included) best-first; only ``open`` quotes are kept.
* Every RFQ logs time to first quote, maker count, best and second-best payout, and its outcome:
  quoted | no_quotes_timeout | rejected_combo | conflict_409 | create_error | aborted.
* Identity comes from how the RFQ was sent, never from a quote.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from combo_edge import money
from combo_edge.kalshi.client import KalshiError
from combo_edge.pipeline import budget
from combo_edge.storage.db import DB, dumps
from combo_edge.timeutil import parse_ts

log = logging.getLogger(__name__)
Identity = Literal["anonymous", "id"]


@dataclass
class QuoteObs:
    id: str
    rfq_id: str
    maker_id: str
    yes_bid: float
    no_bid: float
    yes_contracts: float | None
    no_contracts: float | None
    created_ts: float
    received_ts: float
    source: str
    status: str = "open"
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def yes_cost(self) -> float | None:
        """What a taker pays per YES contract: 1 − the maker's NO bid."""
        return (1.0 - self.no_bid) if self.no_bid > 0 else None

    @property
    def payout(self) -> float | None:
        c = self.yes_cost
        return money.payout(c) if c else None

    def age(self, now: float | None = None) -> float:
        return (now or time.time()) - self.received_ts


@dataclass
class RfqState:
    rfq_id: str
    identity: Identity
    sent_ts: float
    target_cost: float
    quotes: dict[str, QuoteObs] = field(default_factory=dict)
    first_quote_ts: float | None = None
    accepted: list[dict] = field(default_factory=list)
    executed: list[dict] = field(default_factory=list)
    new_quote: asyncio.Event = field(default_factory=asyncio.Event)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    used_rest: bool = False

    def best_open(self) -> list[QuoteObs]:
        latest: dict[str, QuoteObs] = {}
        for q in self.quotes.values():
            cur = latest.get(q.maker_id)
            if cur is None or q.created_ts > cur.created_ts or (q.created_ts == cur.created_ts and q.received_ts > cur.received_ts):
                latest[q.maker_id] = q
        live = [q for q in latest.values() if q.status == "open" and q.yes_cost is not None]
        return sorted(live, key=lambda q: (q.yes_cost, q.created_ts))


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class QuoteBook:
    def __init__(self) -> None:
        self.rfqs: dict[str, RfqState] = {}
        self._early: dict[str, list[tuple[str, dict, float]]] = {}

    def attach(self, ws: Any) -> None:
        ws.on("quote_created", self.on_quote_created)
        ws.on("quote_accepted", self.on_quote_accepted)
        ws.on("quote_executed", self.on_quote_executed)

    def register(self, rfq_id: str, identity: Identity, sent_ts: float, target_cost: float) -> RfqState:
        st = RfqState(rfq_id, identity, sent_ts, target_cost)
        self.rfqs[rfq_id] = st
        for kind, msg, ts in self._early.pop(rfq_id, []):
            getattr(self, f"_apply_{kind}")(st, msg, ts)
        return st

    def _route(self, kind: str, msg: dict, ts: float) -> None:
        m = msg.get("msg") or {}
        rid = m.get("rfq_id")
        if not rid:
            return
        st = self.rfqs.get(rid)
        if st is None:
            self._early.setdefault(rid, []).append((kind, m, ts))
            return
        getattr(self, f"_apply_{kind}")(st, m, ts)

    def on_quote_created(self, msg: dict, ts: float) -> None:
        self._route("created", msg, ts)

    def on_quote_accepted(self, msg: dict, ts: float) -> None:
        self._route("accepted", msg, ts)

    def on_quote_executed(self, msg: dict, ts: float) -> None:
        self._route("executed", msg, ts)

    def _apply_created(self, st: RfqState, m: dict, ts: float) -> None:
        q = QuoteObs(
            id=m["quote_id"], rfq_id=st.rfq_id, maker_id=m.get("quote_creator_id") or "?",
            yes_bid=_f(m.get("yes_bid_dollars")) or 0.0, no_bid=_f(m.get("no_bid_dollars")) or 0.0,
            yes_contracts=_f(m.get("yes_contracts_offered_fp")), no_contracts=_f(m.get("no_contracts_offered_fp")),
            created_ts=parse_ts(m.get("created_ts")) or ts, received_ts=ts, source="ws", raw=m,
        )
        # replacement: the maker's older open quotes are cancelled by the new one
        for old in st.quotes.values():
            if old.maker_id == q.maker_id and old.created_ts <= q.created_ts and old.status == "open":
                old.status = "cancelled"
        st.quotes[q.id] = q
        st.first_quote_ts = st.first_quote_ts or ts
        st.new_quote.set()

    def _apply_accepted(self, st: RfqState, m: dict, ts: float) -> None:
        st.accepted.append({**m, "_ts": ts})
        q = st.quotes.get(m.get("quote_id", ""))
        if q:
            q.status = "accepted"
        st.event.set()

    def _apply_executed(self, st: RfqState, m: dict, ts: float) -> None:
        st.executed.append({**m, "_ts": ts})
        q = st.quotes.get(m.get("quote_id", ""))
        if q:
            q.status = "executed"
        st.event.set()

    def apply_rest(self, st: RfqState, quotes: list[dict], now: float) -> None:
        """REST is authoritative for status: rebuild per-maker state from the full list."""
        st.used_rest = True
        for r in quotes:
            qid = r["id"]
            created = parse_ts(r.get("created_ts")) or now
            existing = st.quotes.get(qid)
            q = QuoteObs(
                id=qid, rfq_id=st.rfq_id, maker_id=r.get("creator_id") or "?",
                yes_bid=_f(r.get("yes_bid_dollars")) or 0.0, no_bid=_f(r.get("no_bid_dollars")) or 0.0,
                yes_contracts=_f(r.get("yes_contracts_fp")), no_contracts=_f(r.get("no_contracts_fp")),
                created_ts=created, received_ts=existing.received_ts if existing else now,
                source=existing.source if existing else "rest", status=r.get("status", "open"), raw=r,
            )
            st.quotes[qid] = q
            if st.first_quote_ts is None and q.status in ("open", "accepted", "confirmed", "executed"):
                st.first_quote_ts = now


@dataclass
class RfqOutcome:
    rfq_id: str | None
    combo_ticker: str
    identity: Identity
    outcome: str
    quotes: list[QuoteObs] = field(default_factory=list)
    best: QuoteObs | None = None
    error: str | None = None
    decision: Any = None
    state: RfqState | None = None

    @property
    def best_payout(self) -> float | None:
        return self.best.payout if self.best else None

    @property
    def n_makers(self) -> int:
        return len({q.maker_id for q in self.quotes})


DecideFn = Callable[["RfqOutcome", "RfqRunner"], Awaitable[Any]]


class RfqRunner:
    """Sends RFQs, collects quotes, logs everything, and always cleans up (cancel or pending-cancel)."""

    def __init__(self, session: Any, db: DB, settings: Any, book: QuoteBook, *, run_id: int | None, abort: Callable[[], bool] | None = None,
                 window_s: float | None = None, settle_s: float = 1.5):
        self.session = session
        self.client = session.client
        self.db = db
        self.settings = settings
        self.book = book
        self.run_id = run_id
        self.env = session.env.name
        self.abort = abort or (lambda: False)
        self.window_s = float(window_s if window_s is not None else settings.get("quote_window_s"))
        self.settle_s = settle_s
        self._slots = asyncio.Semaphore(int(settings.get("max_open_rfqs")))
        self.stats = {"sent": 0, "anonymous": 0, "id": 0, "quoted": 0}

    def _blinked(self, t0: float, t1: float) -> bool:
        ws = self.session.ws
        return True if ws is None else ws.blinked_between(t0, t1)

    async def run_one(self, combo_ticker: str, identity: Identity, *, stake: float, purpose: str, candidate_id: int | None = None,
                      kickoff_ts: float | None = None, feed_demand: int | None = None, decide: DecideFn | None = None) -> RfqOutcome:
        if self.abort():
            return RfqOutcome(None, combo_ticker, identity, "aborted")
        if identity == "id":
            budget.check_id_budget(self.settings, self.db, self.env)
        async with self._slots:
            return await self._run(combo_ticker, identity, stake, purpose, candidate_id, kickoff_ts, feed_demand, decide)

    async def _run(self, combo_ticker: str, identity: Identity, stake: float, purpose: str, candidate_id: int | None,
                   kickoff_ts: float | None, feed_demand: int | None, decide: DecideFn | None) -> RfqOutcome:
        sent = time.time()
        try:
            rfq_id = await self.client.create_rfq(combo_ticker, target_cost=stake, anonymous=(identity == "anonymous"))
        except KalshiError as e:
            outcome = "conflict_409" if e.status == 409 else ("rejected_combo" if e.code in ("rfq_market_not_open",) else "create_error")
            self._log_failed(combo_ticker, identity, purpose, stake, candidate_id, kickoff_ts, outcome, str(e), sent)
            return RfqOutcome(None, combo_ticker, identity, outcome, error=str(e))
        self.stats["sent"] += 1
        self.stats[identity] += 1
        st = self.book.register(rfq_id, identity, sent, stake)
        self.db.insert("rfqs", {
            "id": rfq_id, "env": self.env, "run_id": self.run_id, "candidate_id": candidate_id, "combo_ticker": combo_ticker,
            "identity": identity, "purpose": purpose, "stake": stake, "created_ts": sent,
            "hours_to_kickoff": ((kickoff_ts - sent) / 3600) if kickoff_ts else None, "kickoff_ts": kickoff_ts,
            "status": "open", "feed_demand": feed_demand,
        })
        out = RfqOutcome(rfq_id, combo_ticker, identity, "no_quotes_timeout", state=st)
        executed = False
        try:
            deadline = sent + self.window_s
            decided = False
            while True:
                now = time.time()
                if self.abort():
                    out.outcome = "aborted"
                    break
                remaining = deadline - now
                if remaining <= 0:
                    break
                # identified RFQs with a decision hook decide shortly after the first quote, while fresh
                if decide and not decided and st.first_quote_ts and now >= st.first_quote_ts + self.settle_s:
                    await self._refresh(st, sent, force=True)
                    self._fill(out, st)
                    out.decision = await decide(out, self)
                    decided = True
                    if getattr(out.decision, "executed", False) or getattr(out.decision, "final", False):
                        executed = getattr(out.decision, "executed", False)
                        break
                st.new_quote.clear()
                try:
                    await asyncio.wait_for(st.new_quote.wait(), timeout=min(remaining, 0.25))
                except asyncio.TimeoutError:
                    pass
            if out.outcome != "aborted":
                await self._refresh(st, sent, force=(identity == "id"))
                self._fill(out, st)
                if decide and not decided and not self.abort():
                    out.decision = await decide(out, self)
                    executed = getattr(out.decision, "executed", False)
                out.outcome = "quoted" if out.quotes else "no_quotes_timeout"
                if out.quotes:
                    self.stats["quoted"] += 1
        finally:
            if not executed:
                await self.cancel(rfq_id)
            self._log_rfq(out, st, sent)
        return out

    async def _refresh(self, st: RfqState, sent: float, *, force: bool) -> None:
        if force or self._blinked(sent, time.time()):
            try:
                rows = await self.client.list_quotes(st.rfq_id)
                self.book.apply_rest(st, rows, time.time())
            except KalshiError as e:
                log.warning("REST quote fallback failed for %s: %s", st.rfq_id, e)

    def _fill(self, out: RfqOutcome, st: RfqState) -> None:
        out.quotes = st.best_open()
        out.best = out.quotes[0] if out.quotes else None

    async def cancel(self, rfq_id: str) -> bool:
        try:
            await self.client.delete_rfq(rfq_id)
            self.db.update("rfqs", "id", rfq_id, {"status": "closed", "deleted_ts": time.time()})
            self.db.execute("DELETE FROM pending_cancels WHERE rfq_id=?", (rfq_id,))
            return True
        except Exception as e:  # keep it for the end-of-run / startup sweep
            self.db.insert("pending_cancels", {"rfq_id": rfq_id, "env": self.env, "added_ts": time.time(), "attempts": 1, "last_error": str(e)[:300]}, replace=True)
            self.db.log("warn", "rfq", f"cancel failed for {rfq_id}: {e}")
            return False

    def _log_failed(self, ticker: str, identity: str, purpose: str, stake: float, cand: int | None, kickoff: float | None, outcome: str, err: str, sent: float) -> None:
        import uuid

        self.db.insert("rfqs", {
            "id": f"failed-{uuid.uuid4()}", "env": self.env, "run_id": self.run_id, "candidate_id": cand, "combo_ticker": ticker,
            "identity": identity, "purpose": purpose, "stake": stake, "created_ts": sent, "kickoff_ts": kickoff,
            "hours_to_kickoff": ((kickoff - sent) / 3600) if kickoff else None, "status": "never_opened", "outcome": outcome, "error": err[:500],
        })

    def _log_rfq(self, out: RfqOutcome, st: RfqState, sent: float) -> None:
        all_q = list(st.quotes.values())
        live = out.quotes
        ttfq = ((st.first_quote_ts - sent) * 1000) if st.first_quote_ts else None
        self.db.update("rfqs", "id", out.rfq_id, {
            "outcome": out.outcome, "time_to_first_quote_ms": ttfq, "n_makers": len({q.maker_id for q in all_q}),
            "best_payout": live[0].payout if live else None, "second_payout": live[1].payout if len(live) > 1 else None,
            "best_yes_cost": live[0].yes_cost if live else None, "ws_blinked": int(self._blinked(sent, time.time())),
            "used_rest": int(st.used_rest),
        })
        rows = []
        for q in all_q:
            check = side_check(q, st.target_cost)
            rows.append((q.id, out.rfq_id, st.identity, q.maker_id, q.yes_bid, q.no_bid, q.yes_contracts, q.no_contracts,
                         q.created_ts, q.received_ts, (q.received_ts - q.created_ts) * 1000, q.source, q.status, q.yes_cost,
                         q.payout, check, dumps(q.raw)))
        if rows:
            self.db.executemany(
                "INSERT OR REPLACE INTO quotes(id,rfq_id,identity,maker_id,yes_bid,no_bid,yes_contracts,no_contracts,created_ts,"
                "received_ts,latency_ms,source,status,yes_cost,payout,side_check,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def side_check(q: QuoteObs, target_cost: float) -> str:
    """Does ``no_contracts_offered`` equal what the target cost buys at 1 − no_bid (fee-inclusive)?

    'buy_yes_on_no' → accepting side "no" buys YES at 1 − no_bid (v1's inference);
    'buy_no_on_no'  → the counts imply the opposite mapping; 'unknown' when there is nothing to check."""
    if not q.no_contracts or q.no_bid <= 0 or not target_cost:
        return "unknown"
    yes_cost = 1 - q.no_bid
    expect_yes = money.contracts_for_target(target_cost, yes_cost)
    expect_no = money.contracts_for_target(target_cost, q.no_bid)
    tol = max(0.02, 0.005 * expect_yes)
    if abs(q.no_contracts - expect_yes) <= tol:
        return "buy_yes_on_no"
    if abs(q.no_contracts - expect_no) <= max(0.02, 0.005 * expect_no):
        return "buy_no_on_no"
    return "mismatch"


async def sweep_pending_cancels(session: Any, db: DB) -> dict:
    """Retry failed cancels; on startup also cancel any open RFQs *this tool* created (never the app's)."""
    env = session.env.name
    done = failed = 0
    ids = {r["rfq_id"] for r in db.query("SELECT rfq_id FROM pending_cancels WHERE env=?", (env,))}
    ids |= {r["id"] for r in db.query("SELECT id FROM rfqs WHERE env=? AND status='open'", (env,))}
    for rid in ids:
        try:
            await session.client.delete_rfq(rid)
            db.update("rfqs", "id", rid, {"status": "closed", "deleted_ts": time.time()})
            db.execute("DELETE FROM pending_cancels WHERE rfq_id=?", (rid,))
            done += 1
        except Exception as e:
            db.execute("UPDATE pending_cancels SET attempts=attempts+1, last_error=? WHERE rfq_id=?", (str(e)[:300], rid))
            failed += 1
    return {"cancelled": done, "failed": failed}
