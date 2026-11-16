"""The execution module — the only code that can accept a quote.

Modes: off | paper | real (paper by default). In real mode every quote that passes the gates is
bought automatically. Real requires a verified side mapping and, on the first real bet, a $1 canary
verified against GET /portfolio/fills. ``COMBO_EDGE_FORBID_LIVE=1`` makes real impossible.

Accept flow (real): RFQ-scoped accept while the RFQ is open → check the accepted contract count
against the expected side mapping (mismatch: delete the RFQ before the maker confirms and turn
execution off) → wait for the maker to confirm within the 3 s window (``maker_no_confirm``, no
retry) → verify the fill (YES side, contract count).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from combo_edge import money
from combo_edge.execution.capability import _mint, accept_path
from combo_edge.execution.gates import Exposure, GateInput, GateResult, evaluate
from combo_edge.jobs.killswitch import is_killed
from combo_edge.pipeline.quotes import QuoteObs, RfqOutcome, side_check
from combo_edge.storage.db import DB
from combo_edge.timeutil import start_of_local_day

ACCEPT_SIDE_FOR_YES = "no"  # accepting the maker's NO bid buys YES at 1 − no_bid (verified per quote)


@dataclass
class Timing:
    confirm_window_s: float = 3.0
    exec_timer_s: float = 1.0
    slack_s: float = 2.0


@dataclass
class Decision:
    action: str  # buy | skip | requote
    reason: str
    chip: str
    executed: bool = False
    final: bool = False
    requote: bool = False
    requote_stake: float | None = None
    gate: GateResult | None = None
    decision_id: int | None = None
    execution_id: int | None = None


@dataclass
class DecisionContext:
    candidate: Any  # Candidate (fair, game, players, family)
    planned_stake: float
    run_id: int | None
    dry_run: bool = False
    family_ok: bool = True
    family_note: str = ""
    extra: dict = field(default_factory=dict)


def side_mapping_status(db: DB, env: str) -> dict:
    rows = db.query("SELECT side_check, COUNT(*) n FROM quotes q JOIN rfqs r ON r.id=q.rfq_id WHERE r.env=? GROUP BY side_check", (env,))
    c = {r["side_check"]: r["n"] for r in rows}
    verified_by_canary = bool(db.get_state(f"canary_passed:{env}", False))
    ok = (c.get("buy_yes_on_no", 0) >= 3 and c.get("buy_no_on_no", 0) == 0) or verified_by_canary
    if db.get_state(f"side_mapping_broken:{env}", False):
        ok = False
    return {"verified": ok, "counts": c, "canary_passed": verified_by_canary}


def can_enable_real(db: DB, env_cfg: Any) -> tuple[bool, str]:
    if env_cfg.forbid_live:
        return False, "COMBO_EDGE_FORBID_LIVE=1 in .env makes Real impossible"
    if is_killed(db):
        return False, "kill switch is on; resume first"
    if env_cfg.name == "mock":
        return True, "mock exchange (no real money)"
    st = side_mapping_status(db, env_cfg.name)
    if not st["verified"]:
        return False, "side mapping not verified yet (needs 3 consistent contract-count checks; run a scan in Paper first)"
    return True, "ok (the first Real bet is a $1 canary)"


def set_mode(db: DB, env_cfg: Any, mode: str) -> tuple[bool, str]:
    if mode not in ("off", "paper", "real"):
        return False, "mode must be off, paper or real"
    if mode == "real":
        ok, why = can_enable_real(db, env_cfg)
        if not ok:
            return False, why
    db.set_state("execution_mode", mode)
    db.log("info", "execution", f"execution mode set to {mode}")
    return True, mode


def exposure(db: DB, env: str, book: str) -> Exposure:
    rows = db.query(
        "SELECT combo_ticker, game_id, players_json, stake, ts FROM executions WHERE env=? AND book=? AND status IN ('filled','paper_filled','pending')",
        (env, book),
    )
    e = Exposure()
    sod = start_of_local_day()
    for r in rows:
        e.bought_combos.add(r["combo_ticker"])
        e.bets_by_game[r["game_id"]] = e.bets_by_game.get(r["game_id"], 0) + 1
        for p in json.loads(r["players_json"] or "[]"):
            e.bets_by_player[p] = e.bets_by_player.get(p, 0) + 1
        if r["ts"] >= sod:
            e.spent_today += r["stake"] or 0.0
    return e


class Executor:
    def __init__(self, session: Any, db: DB, settings: Any, *, timing: Timing | None = None):
        self.session = session
        self.client = session.client
        self.db = db
        self.s = settings
        self.env = session.env.name
        self.timing = timing or Timing()
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> str:
        m = self.db.get_state("execution_mode", "paper")
        if m == "real" and self.session.env.forbid_live:
            return "paper"
        return m

    def canary_pending(self) -> bool:
        return self.mode == "real" and not self.db.get_state(f"canary_passed:{self.env}", False) and self.env != "mock"

    def planned_stake(self, default: float) -> float:
        return 1.0 if self.canary_pending() else default

    async def consider(self, out: RfqOutcome, ctx: DecisionContext) -> Decision:
        async with self._lock:  # caps are checked and booked atomically
            return await self._consider(out, ctx)

    async def _consider(self, out: RfqOutcome, ctx: DecisionContext) -> Decision:
        c = ctx.candidate
        q = out.best
        if is_killed(self.db):
            return self._record(out, ctx, None, Decision("skip", "kill switch on", "Off", final=True))
        if q is None:
            return self._record(out, ctx, None, Decision("skip", "no quotes", "No quotes"))
        mode = self.mode
        book = "real" if mode == "real" else "paper"
        gi = GateInput(
            identity=out.identity, payout=q.payout or 0.0, fair_joint=c.fair.fair_joint,
            fair_lower=c.fair.lower(max(q.payout or 1.0, c.fair.fair_payout), self.s), family_ok=ctx.family_ok,
            family_note=ctx.family_note, quote_age_s=q.age(), combo_ticker=out.combo_ticker, game_id=c.game.id,
            players=c.players, rfq_stake=ctx.planned_stake, model_disagrees=list(getattr(c.fair, "model_disagrees", None) or []),
            n_makers=out.n_makers, near_lock=getattr(c.fair, "near_lock", None), thin_legs=list(getattr(c.fair, "thin_legs", None) or []),
        )
        g = evaluate(gi, exposure(self.db, self.env, book), self.s)
        if g.stale and not ctx.extra.get("requoted"):
            return self._record(out, ctx, g, Decision("requote", g.reason, "Stale: refresh", requote=True, requote_stake=ctx.planned_stake, gate=g, final=True))
        if g.size_mismatch and not ctx.extra.get("requoted"):
            return self._record(out, ctx, g, Decision("requote", g.reason, "Passes gates", requote=True, requote_stake=g.stake, gate=g, final=True))
        if not g.passes:
            return self._record(out, ctx, g, Decision("skip", g.reason, g.chip, gate=g))
        if ctx.dry_run:
            return self._record(out, ctx, g, Decision("skip", "dry run: would buy", "Passes gates", gate=g, final=True))
        if mode == "off":
            return self._record(out, ctx, g, Decision("skip", "execution is off", "Passes gates", gate=g, final=True))
        check = side_check(q, ctx.planned_stake)
        if check in ("buy_no_on_no", "mismatch"):
            return self._record(out, ctx, g, Decision("skip", f"contract count check failed ({check}); not buying", "Below gate (side check)", gate=g))
        if mode == "paper":
            d = self._record(out, ctx, g, Decision("buy", "paper buy", "Paper bought", gate=g, final=True))
            self._book_execution(d, out, ctx, q, book="paper", status="paper_filled", stake=g.stake,
                                 contracts=money.contracts_for_target(g.stake, q.yes_cost or 0), order_id=None, fee=None)
            return d
        # real
        if self.canary_pending() and abs(ctx.planned_stake - 1.0) > 1e-9:
            return self._record(out, ctx, g, Decision("requote", "first Real bet is a $1 canary", "Passes gates", requote=True, requote_stake=1.0, gate=g, final=True))
        if not side_mapping_status(self.db, self.env)["verified"] and self.env != "mock" and not self.canary_pending():
            return self._record(out, ctx, g, Decision("skip", "side mapping not verified", "Below gate (side check)", gate=g))
        d = self._record(out, ctx, g, Decision("buy", "accepting", "Passes gates", gate=g, final=True))
        return await self._accept_real(d, out, ctx, q, g)

    async def _accept_real(self, d: Decision, out: RfqOutcome, ctx: DecisionContext, q: QuoteObs, g: GateResult) -> Decision:
        st = out.state
        rfq_id = out.rfq_id or ""
        expected = money.contracts_for_target(ctx.planned_stake, q.yes_cost or 0)
        is_canary = self.canary_pending()
        cap = _mint(rfq_id, q.id)
        try:
            await self.client.request("PUT", accept_path(rfq_id, q.id), json_body={"accepted_side": ACCEPT_SIDE_FOR_YES}, capability=cap)
        except Exception as e:
            return self._finish(d, "accept_error", f"accept failed: {e}", "Below gate (accept error)")
        exec_id = self._book_execution(d, out, ctx, q, book="real", status="pending", stake=ctx.planned_stake, contracts=expected,
                                       order_id=None, fee=None, is_canary=is_canary)
        # 1) accepted contract count must match buying YES
        acc = await self._wait(lambda: st.accepted[-1] if st and st.accepted else None, 1.5)
        accepted_count = None
        if acc:
            try:
                accepted_count = float(acc.get("contracts_accepted_fp"))
            except (TypeError, ValueError):
                accepted_count = None
        else:
            try:
                rq = await self.client.get_rfq_quote(rfq_id, q.id)
                accepted_count = float(rq.get("no_contracts_fp") if rq.get("accepted_side") == "no" else rq.get("yes_contracts_fp") or 0)
            except Exception:
                accepted_count = None
        if accepted_count is not None and abs(accepted_count - expected) > max(0.02, 0.005 * expected):
            try:
                await self.client.delete_rfq(rfq_id)  # kill it before the maker confirms
            finally:
                self.db.set_state("execution_mode", "off")
                self.db.set_state(f"side_mapping_broken:{self.env}", True)
                self.db.log("error", "execution", f"side mismatch on {rfq_id}: accepted {accepted_count} vs expected {expected}; RFQ deleted, execution OFF")
            self._update_exec(exec_id, "side_mismatch")
            return self._finish(d, "side_mismatch", "accepted contract count didn't match buying YES: RFQ deleted, execution turned off", "Below gate (side check)")
        # 2) maker must confirm within the window; execution follows after the timer
        t = self.timing
        ex = await self._wait(lambda: st.executed[-1] if st and st.executed else None, t.confirm_window_s + t.exec_timer_s + t.slack_s)
        if not ex:
            try:
                rq = await self.client.get_rfq_quote(rfq_id, q.id)
                status = rq.get("status")
                order_id = rq.get("rfq_creator_order_id")
            except Exception:
                status, order_id = None, None
            if status != "executed":
                self._update_exec(exec_id, "maker_no_confirm")
                return self._finish(d, "maker_no_confirm", "maker didn't confirm in time (no retry)", "Below gate (maker no-confirm)")
        else:
            order_id = ex.get("order_id")
        # 3) verify the fill
        fills = []
        for _ in range(6):
            try:
                fills = await self.client.fills(order_id=order_id) if order_id else []
            except Exception:
                fills = []
            if fills:
                break
            await asyncio.sleep(0.5)
        count = sum(float(f.get("count_fp") or 0) for f in fills)
        sides = {f.get("outcome_side") for f in fills}
        fee = sum(float(f.get("fee_cost") or 0) for f in fills)
        price = float(fills[0]["yes_price_dollars"]) if fills else q.yes_cost
        if fills and (sides != {"yes"} or abs(count - expected) > max(0.02, 0.005 * expected)):
            self.db.set_state("execution_mode", "off")
            self.db.set_state(f"side_mapping_broken:{self.env}", True)
            self.db.log("error", "execution", f"fill check failed for order {order_id}: sides={sides} count={count} expected={expected}; execution OFF")
            self._update_exec(exec_id, "filled", order_id=order_id, fee=fee, contracts=count, yes_price=price, note="FILL CHECK FAILED")
            return self._finish(d, "fill_check_failed", "fill didn't match YES / expected count: execution turned off", "Bought", executed=True)
        self._update_exec(exec_id, "filled", order_id=order_id, fee=fee, contracts=count or expected, yes_price=price)
        if is_canary and fills:
            self.db.set_state(f"canary_passed:{self.env}", True)
            self.db.set_state("upgrade_tier_offer", True)
            self.db.log("info", "execution", "canary passed: $1 real bet filled on YES with the expected count")
        return self._finish(d, "filled", "bought", "Bought", executed=True)

    async def _wait(self, probe: Any, timeout: float) -> Any:
        end = time.time() + timeout
        while time.time() < end:
            v = probe()
            if v:
                return v
            await asyncio.sleep(0.005)
        return probe()

    def _finish(self, d: Decision, status: str, reason: str, chip: str, executed: bool = False) -> Decision:
        d.reason, d.chip, d.executed = reason, chip, executed
        if d.decision_id:
            self.db.update("decisions", "id", d.decision_id, {"reason": f"{status}: {reason}"})
        return d

    def _record(self, out: RfqOutcome, ctx: DecisionContext, g: GateResult | None, d: Decision) -> Decision:
        c = ctx.candidate
        q = out.best
        d.decision_id = self.db.insert("decisions", {
            "run_id": ctx.run_id, "ts": time.time(), "env": self.env, "mode": "dry" if ctx.dry_run else self.mode,
            "rfq_id": out.rfq_id, "quote_id": q.id if q else None, "combo_ticker": out.combo_ticker,
            "candidate_id": c.db_id, "action": d.action, "reason": d.reason, "payout": q.payout if q else None,
            "fair_joint": c.fair.fair_joint, "fair_lower": c.fair.lower(max(q.payout or 1, c.fair.fair_payout), self.s) if q else None,
            "ev": g.ev if g else None, "ev_lower": g.ev_lower if g else None, "required_ev": g.required_ev if g else None,
            "stake": g.stake if g else None, "quote_age_s": q.age() if q else None,
            "details_json": {"chip": d.chip, "identity": out.identity, "n_makers": out.n_makers, "planned_stake": ctx.planned_stake},
        })
        return d

    def _book_execution(self, d: Decision, out: RfqOutcome, ctx: DecisionContext, q: QuoteObs, *, book: str, status: str, stake: float,
                        contracts: float, order_id: str | None, fee: float | None, is_canary: bool = False) -> int:
        c = ctx.candidate
        eid = self.db.insert("executions", {
            "decision_id": d.decision_id, "env": self.env, "book": book, "ts": time.time(), "rfq_id": out.rfq_id, "quote_id": q.id,
            "combo_ticker": out.combo_ticker, "game_id": c.game.id, "players_json": c.players, "family": c.family or "exploration",
            "stake": stake, "contracts": contracts, "yes_price": q.yes_cost, "payout": q.payout, "fair_joint": c.fair.fair_joint,
            "ev": money.ev(c.fair.fair_joint, q.payout or 0), "status": status, "order_id": order_id, "fee": fee,
            "is_canary": int(is_canary), "kickoff_ts": c.game.start_ts,
            "legs_json": [{"wording": c.a.wording, "ticker": c.a.leg.ticker}, {"wording": c.b.wording, "ticker": c.b.leg.ticker}],
        })
        d.execution_id = eid
        return eid

    def _update_exec(self, eid: int, status: str, **kw: Any) -> None:
        fields: dict[str, Any] = {"status": status}
        for k in ("order_id", "fee", "contracts", "yes_price"):
            if kw.get(k) is not None:
                fields[k] = kw[k]
        if kw.get("note"):
            fields["details_json"] = {"note": kw["note"]}
        self.db.update("executions", "id", eid, fields)
