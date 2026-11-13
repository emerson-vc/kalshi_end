"""One scan: every game in the current NFL gameweek.

legs found → candidates → selected → looked up/created → anonymous quotes → ID shortlist →
ID quotes → bought. Progress is written to the ``runs`` row and published for the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from combo_edge import events
from combo_edge.execution.executor import DecisionContext, Executor, Timing
from combo_edge.jobs.killswitch import is_killed
from combo_edge.kalshi.session import Session, build_session
from combo_edge.learning.bandit import Bandit
from combo_edge.learning.maker_model import MakerModel, anon_maker_count, candidate_features, quote_features
from combo_edge.pipeline import budget, portfolio
from combo_edge.pipeline.budget import IdBudgetExhausted
from combo_edge.pipeline.quotes import QuoteBook, RfqOutcome, RfqRunner, sweep_pending_cancels
from combo_edge.pipeline.resolve import ComboResolver, feed_demand, rejected_class_set
from combo_edge.pipeline.selection import select
from combo_edge.settings import Settings, env_config
from combo_edge.sports.nfl import calibration, nflverse
from combo_edge.sports.nfl.candidates import Candidate, generate, price_all
from combo_edge.sports.nfl.fair import FairModel
from combo_edge.sports.nfl.parser import coverage
from combo_edge.sports.nfl.universe import load_week
from combo_edge.storage.db import DB, get_db

log = logging.getLogger(__name__)
STAGES = [
    ("legs", "Legs found"), ("candidates", "Candidates"), ("selected", "Selected"), ("resolved", "Looked up / created"),
    ("anon_quotes", "Anonymous quotes"), ("shortlist", "ID shortlist"), ("id_quotes", "ID quotes"), ("bought", "Bought"),
]


@dataclass
class ScanParams:
    dry_run: bool = False
    budget: int | None = None
    max_new: int | None = None
    stake: float | None = None
    id_shortlist: int | None = None
    kind: str = "manual"
    game_ids: list[str] | None = None
    sim_paths: int | None = None
    window_s: float | None = None


class ScanBusy(Exception):
    pass


@dataclass
class Progress:
    run_id: int
    db: DB
    stages: dict[str, dict] = field(default_factory=dict)
    message: str = ""

    def __post_init__(self) -> None:
        self.stages = {k: {"label": lbl, "count": None, "state": "pending"} for k, lbl in STAGES}

    def set(self, key: str, count: int | None = None, state: str = "running", message: str | None = None) -> None:
        st = self.stages[key]
        if count is not None:
            st["count"] = count
        st["state"] = state
        if message is not None:
            self.message = message
        payload = {"run_id": self.run_id, "stages": self.stages, "message": self.message}
        self.db.update("runs", "id", self.run_id, {"progress_json": payload, "heartbeat_ts": time.time()})
        events.bus.publish("pipeline", payload)


def acquire_lock(db: DB, run_id: int) -> None:
    cur = db.get_state("scan_running")
    if cur and cur.get("run_id") != run_id:
        r = db.one("SELECT heartbeat_ts, status FROM runs WHERE id=?", (cur.get("run_id"),))
        if r and r["status"] == "running" and r["heartbeat_ts"] and time.time() - r["heartbeat_ts"] < 180:
            raise ScanBusy(f"another scan (run {cur['run_id']}) is running")
    db.set_state("scan_running", {"run_id": run_id, "pid": os.getpid(), "ts": time.time()})


def release_lock(db: DB, run_id: int) -> None:
    cur = db.get_state("scan_running")
    if cur and cur.get("run_id") == run_id:
        db.set_state("scan_running", None)


def scan_running(db: DB) -> dict | None:
    cur = db.get_state("scan_running")
    if not cur:
        return None
    r = db.one("SELECT heartbeat_ts, status FROM runs WHERE id=?", (cur.get("run_id"),))
    if r and r["status"] == "running" and r["heartbeat_ts"] and time.time() - r["heartbeat_ts"] < 180:
        return cur
    return None


async def run_scan(params: ScanParams, *, session: Session | None = None, db: DB | None = None, settings: Settings | None = None,
                   timing: Timing | None = None, stop: asyncio.Event | None = None) -> dict:
    db = db or get_db()
    base = settings or Settings(db)
    s = base.with_overrides(stake=params.stake, creation_budget_per_run=params.budget, max_new_combos_per_run=params.max_new,
                            id_shortlist_per_run=params.id_shortlist, sim_paths=params.sim_paths, quote_window_s=params.window_s)
    own_session = session is None
    session = session or build_session(env_config(), fraction=float(s.get("rate_budget_fraction")))
    env = session.env.name
    if is_killed(db):
        raise RuntimeError("kill switch is on — resume it on the System page or with `combo-edge resume`")
    run_id = db.insert("runs", {"env": env, "kind": params.kind, "started_ts": time.time(), "status": "running",
                                "params_json": asdict(params), "pid": os.getpid(), "heartbeat_ts": time.time()})
    try:
        acquire_lock(db, run_id)
    except ScanBusy:
        db.update("runs", "id", run_id, {"status": "refused", "finished_ts": time.time(), "error": "another scan is running"})
        raise
    prog = Progress(run_id, db)
    abort = lambda: is_killed(db) or bool(stop and stop.is_set())  # noqa: E731
    summary: dict[str, Any] = {"run_id": run_id, "env": env, "dry_run": params.dry_run, "mode": db.get_state("execution_mode", "paper")}
    events.bus.publish("run_started", {"run_id": run_id})
    try:
        lim = await session.client.configure_limiter(float(s.get("rate_budget_fraction")))
        db.update("runs", "id", run_id, {"limits_before_json": lim.get("limits") or {}})
        db.insert("limits_log", {"env": env, "ts": time.time(), "run_id": run_id, "phase": "before", "data_json": lim.get("limits") or {}})
        summary["cleanup_before"] = await sweep_pending_cancels(session, db)
        book = QuoteBook()
        await session.open_ws(user_filter_self=True)
        if session.ws:
            book.attach(session.ws)
        summary["ws_connected"] = bool(session.ws and session.ws.connected.is_set())

        # 1. universe
        prog.set("legs", message="Loading this week's games and legs")
        uni = await load_week(session.client, time.time(), game_ids=params.game_ids)
        n_legs = sum(len(g.legs) for g in uni.games)
        _store_universe(db, env, run_id, uni)
        prog.set("legs", n_legs, "done", f"{len(uni.games)} games, {n_legs} legs")
        summary.update(games=len(uni.games), legs=n_legs, universe_errors=uni.errors)

        # 2. candidates
        prog.set("candidates")
        positions, shares = {}, {}
        if nflverse.available():
            try:
                positions, shares = nflverse.player_positions(), nflverse.team_pass_shares()
            except Exception as e:  # history is optional
                log.warning("nflverse unavailable: %s", e)
        id_model = MakerModel(db, "id")
        try:
            summary["maker_model"] = id_model.fit()
        except Exception as e:  # a bad fit must never stop the scan: fall back to the built-in estimate
            log.exception("maker model fit failed")
            db.log("error", "maker_model", f"fit failed, using the built-in estimate: {type(e).__name__}: {e}")
            id_model.model, id_model.columns = None, []
            summary["maker_model"] = {"rows": id_model.metrics.get("rows", 0), "fitted": False, "status": "error",
                                      "error": f"{type(e).__name__}: {e}"}
        rejected = rejected_class_set(db)
        cands: list[Candidate] = []
        per_game: dict[str, int] = {}
        for g in uni.games:
            if abort():
                break
            if not g.legs:
                per_game[g.id] = 0
                continue
            fm = FairModel(g, s, positions=positions, pass_shares=shares)
            gc = generate(g, fm, s, rejected=rejected)
            price_all(gc, fm, s, id_model)
            per_game[g.id] = len(gc)
            cands.extend(gc)
            await asyncio.sleep(0)
        cands = cands[: int(s.get("max_candidates_per_run"))]
        cov = coverage(uni.games, per_game)
        summary["coverage"] = cov
        _store_candidates(db, run_id, cands, s)
        prog.set("candidates", len(cands), "done")

        # 3. select
        bandit = Bandit(db)
        chosen = select(cands, s, int(s.get("creation_budget_per_run")), bandit)
        for c in chosen:
            db.update("candidates", "id", c.db_id, {"selected": 1})
        prog.set("selected", len(chosen), "done")

        # 4. resolve
        prog.set("resolved")
        resolver = ComboResolver(session, db, s, uni, run_id=run_id, max_new=params.max_new)
        resolved: list[Candidate] = []
        for i, c in enumerate(chosen):
            if abort():
                break
            st = await resolver.resolve(c)
            db.update("candidates", "id", c.db_id, {"combo_ticker": c.combo_ticker, "status": st})
            if c.combo_ticker:
                c.extra["existed"] = st.startswith("found")
                resolved.append(c)
            if i % 10 == 0:
                prog.set("resolved", len(resolved))
        prog.set("resolved", len(resolved), "done")
        summary["resolution"] = dict(resolver.counts)

        stake = float(s.get("stake"))
        runner = RfqRunner(session, db, s, book, run_id=run_id, abort=abort)
        executor = Executor(session, db, s, timing=timing)

        # 5. anonymous screen
        prog.set("anon_quotes")
        anon: dict[int, RfqOutcome] = {}
        done = 0

        async def screen(c: Candidate) -> None:
            nonlocal done
            fd = feed_demand(db, c.combo_ticker)
            c.extra["feed_demand"] = fd
            out = await runner.run_one(c.combo_ticker, "anonymous", stake=stake, purpose="screen", candidate_id=c.db_id,
                                       kickoff_ts=c.game.start_ts, feed_demand=fd, features=quote_features(c, feed_demand=fd))
            anon[id(c)] = out
            done += 1
            if done % 5 == 0:
                prog.set("anon_quotes", sum(1 for o in anon.values() if o.quotes))

        await asyncio.gather(*(screen(c) for c in resolved))
        prog.set("anon_quotes", sum(1 for o in anon.values() if o.quotes), "done")

        # 6. ID shortlist
        mode = db.get_state("execution_mode", "paper")
        id_left = budget.id_left(s, db, env)
        n_short = min(int(s.get("id_shortlist_per_run")), id_left)
        shortlist = build_shortlist(resolved, anon, id_model, s, n_short)
        prog.set("shortlist", len(shortlist), "done", "" if id_left else "ID budget used up for today")
        summary["id_budget_left_before"] = id_left

        # 7. ID quotes, deciding while each RFQ is open
        prog.set("id_quotes")
        id_outcomes: list[tuple[Candidate, RfqOutcome]] = []
        bought = 0
        planned_default = float(s.get("max_stake_per_combo"))
        for c in shortlist:
            if abort():
                break
            fam_ok, fam_note = calibration.family_passes(c.family)
            planned = executor.planned_stake(_planned_stake(c, s, planned_default))
            if bool(s.get("require_history_calibration")) and fam_note.startswith("no history"):
                fam_ok = False
            ctx = DecisionContext(c, planned, run_id, dry_run=params.dry_run, family_ok=fam_ok, family_note=fam_note)
            fd = feed_demand(db, c.combo_ticker)
            feats = quote_features(c, feed_demand=fd, anon=anon.get(id(c)))
            try:
                out = await runner.run_one(c.combo_ticker, "id", stake=planned, purpose="shortlist", candidate_id=c.db_id,
                                           kickoff_ts=c.game.start_ts, feed_demand=fd, features=feats,
                                           decide=lambda o, r, ctx=ctx: executor.consider(o, ctx))
                d = out.decision
                if d is not None and d.requote:
                    ctx.planned_stake = d.requote_stake or planned
                    ctx.extra["requoted"] = True
                    out = await runner.run_one(c.combo_ticker, "id", stake=ctx.planned_stake, purpose="requote_stale", candidate_id=c.db_id,
                                               kickoff_ts=c.game.start_ts, feed_demand=fd, features=feats,
                                               decide=lambda o, r, ctx=ctx: executor.consider(o, ctx))
            except IdBudgetExhausted as e:
                summary["id_budget_note"] = str(e)
                events.toast("ID budget used up for today: identified RFQs stopped", "warn")
                break
            id_outcomes.append((c, out))
            d = out.decision
            if d is not None and d.action == "buy" and (d.executed or "Paper" in d.chip):
                bought += 1
                events.toast(f"{'Bought' if d.executed else 'Paper bought'}: {c.label} at {out.best_payout:.1f}x", "success")
            prog.set("id_quotes", sum(1 for _, o in id_outcomes if o.quotes))
            prog.set("bought", bought)
        prog.set("id_quotes", sum(1 for _, o in id_outcomes if o.quotes), "done")
        prog.set("bought", bought, "done")

        # 8. learning bookkeeping
        gaps: dict[str, list[float]] = {}
        for c, o in id_outcomes:
            if o.best_payout:
                ev = c.fair.fair_joint * o.best_payout - 1
                bandit.observe(c.family or c.stratum, ev)
                if c.exploration:
                    gaps.setdefault(c.stratum, []).append(ev)
        summary["promoted"] = bandit.promotions(gaps)
        from combo_edge.learning.requote import add_to_panels

        add_to_panels(db, env, [(c, o) for c, o in id_outcomes if o.quotes], [(c, anon[id(c)]) for c in resolved if id(c) in anon and anon[id(c)].quotes], s)
        if env != "mock":
            try:
                summary["fills_synced"] = await portfolio.sync_fills(session, db)
            except Exception as e:
                summary["fills_error"] = str(e)
        summary.update(
            candidates=len(cands), selected=len(chosen), resolved=len(resolved),
            anon_sent=sum(1 for o in anon.values() if o.rfq_id), anon_quoted=sum(1 for o in anon.values() if o.quotes),
            shortlist=len(shortlist), id_sent=len(id_outcomes), id_quoted=sum(1 for _, o in id_outcomes if o.quotes), bought=bought,
            creations_used_week=budget.creations_this_week(db, env), aborted=abort(),
            rfq_outcomes=_outcome_counts(db, run_id),
        )
        status = "aborted" if abort() else "done"
    except Exception as e:
        log.exception("scan failed")
        summary["error"] = f"{type(e).__name__}: {e}"
        status = "error"
        db.log("error", "scan", summary["error"])
    finally:
        try:
            summary["cleanup_after"] = await sweep_pending_cancels(session, db)
            lim_after = await session.client.get_limits()
            db.update("runs", "id", run_id, {"limits_after_json": lim_after})
            db.insert("limits_log", {"env": env, "ts": time.time(), "run_id": run_id, "phase": "after", "data_json": lim_after})
        except Exception:
            pass
        db.update("runs", "id", run_id, {"status": status, "finished_ts": time.time(), "summary_json": summary, "error": summary.get("error")})
        release_lock(db, run_id)
        if own_session:
            await session.close()
        elif session.ws:
            await session.ws.stop()
            session.ws = None
        events.bus.publish("run_finished", {"run_id": run_id, "status": status})
        events.toast(f"Scan {status}: {summary.get('bought', 0)} bought, {summary.get('id_quoted', 0)} ID quotes", "error" if status == "error" else "success")
    return summary


def _planned_stake(c: Candidate, s: Any, default: float) -> float:
    from combo_edge.execution.gates import required_ev, size_stake

    pay = c.pred_id_payout
    req = required_ev(pay, s.get("payout_curve"), float(s.get("longshot_payout")), float(s.get("longshot_required_ev")))
    lower = c.fair.lower(max(pay, c.fair.fair_payout), s)
    st = size_stake(c.fair.fair_joint, lower, pay, c.pred_ev, req, s)
    return float(min(max(st, float(s.get("min_stake"))), default))


def build_shortlist(resolved: list[Candidate], anon: dict[int, RfqOutcome], model: MakerModel, s: Any, n: int, rng: random.Random | None = None) -> list[Candidate]:
    if n <= 0:
        return []
    rng = rng or random.Random(int(time.time()) // 3600)
    lo, hi = (float(x) for x in s.get("payout_band"))
    scored = []
    straight = []
    for c in resolved:
        o = anon.get(id(c))
        if o is not None and o.outcome in ("rejected_combo", "conflict_409", "create_error", "aborted"):
            continue
        ap = o.best_payout if o else None
        pay, src = model.predict_id_payout(c, ap, anon_maker_count(o))
        c.pred_id_payout, c.pred_source = pay, src
        lower = c.fair.lower(max(pay, c.fair.fair_payout), s)
        c.pred_ev = c.fair.fair_joint * pay - 1
        c.pred_ev_lower = lower * pay - 1
        if ap and (c.fair.lower(max(ap, c.fair.fair_payout), s) * ap - 1) > 0:
            straight.append(c)  # already +EV anonymously
        else:
            scored.append(c)
    straight.sort(key=lambda c: -c.pred_ev_lower)
    scored.sort(key=lambda c: -c.pred_ev_lower)
    n_random = int(round(n * float(s.get("id_random_share"))))
    out = straight[:n]
    need_top = max(0, n - len(out) - n_random)
    out += scored[:need_top]
    picked = {id(c) for c in out}
    mid_band = [c for c in scored if id(c) not in picked and lo <= c.pred_id_payout <= hi]
    rng.shuffle(mid_band)
    for c in mid_band:
        if len(out) >= n:
            break
        c.extra["random_pick"] = True
        out.append(c)
    for c in scored:
        if len(out) >= n:
            break
        if id(c) not in {id(x) for x in out}:
            out.append(c)
    return out[:n]


def _store_universe(db: DB, env: str, run_id: int, uni: Any) -> None:
    now = time.time()
    snaps = []
    for g in uni.games:
        db.insert("games", {"id": g.id, "env": env, "title": g.title, "start_ts": g.start_ts, "away_name": g.away_name, "home_name": g.home_name,
                            "away_code": g.away_code, "home_code": g.home_code, "away_team_id": g.away_team_id, "home_team_id": g.home_team_id,
                            "season": g.season, "week": g.week, "event_tickers_json": g.event_tickers, "updated_ts": now}, replace=True)
        game_vol = sum((l.volume or 0) for l in g.legs)
        rows = []
        for l in g.legs:
            rows.append((l.ticker, env, g.id, l.event_ticker, l.series, l.stat, l.team, l.team_id, l.player, l.player_id, l.line, l.title,
                         l.wording, int(l.protected), None, now))
            sn = l.snapshot()
            snaps.append((run_id, l.ticker, now, sn["yes_bid"], sn["yes_ask"], sn["no_bid"], sn["no_ask"], sn["spread"], sn["bid_size"],
                          sn["ask_size"], sn["volume"], sn["volume_24h"], sn["open_interest"], sn["last_price"], sn["previous_price"],
                          sn["move_since_open"], game_vol, sn["prob_market"], sn["prob_model"], sn["prob_fair"], sn["extrapolated"], sn["two_sided"]))
        if rows:
            db.executemany("INSERT OR REPLACE INTO legs(ticker,env,game_id,event_ticker,series,stat,team_code,team_id,player,player_id,line,title,wording,protected,extra_json,updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    if snaps:
        db.executemany("INSERT INTO leg_snapshots(run_id,ticker,ts,yes_bid,yes_ask,no_bid,no_ask,spread,bid_size,ask_size,volume,volume_24h,open_interest,last_price,previous_price,move_since_open,game_volume,prob_market,prob_model,prob_fair,extrapolated,two_sided) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", snaps)


def _store_candidates(db: DB, run_id: int, cands: list[Candidate], s: Any) -> None:
    from combo_edge.storage.db import dumps

    for c in cands:
        f = candidate_features(c)
        c.extra["features"] = f
        c.db_id = db.insert("candidates", {
            "run_id": run_id, "game_id": c.game.id, "legs_key": "|".join(sorted(f"{s['market_ticker']}:{s['side']}" for s in c.selections())),
            "family": c.family, "stratum": c.stratum, "leg_a": c.a.leg.ticker, "leg_b": c.b.leg.ticker, "side_a": c.a.side, "side_b": c.b.side,
            "p_a": c.a.p, "p_b": c.b.p, "fair_joint": c.fair.fair_joint, "fair_lower": c.fair.lower(max(c.pred_id_payout, c.fair.fair_payout), s),
            "leg_payout_a": c.leg_payout_a, "leg_payout_b": c.leg_payout_b, "leg_product": c.leg_product, "pred_id_payout": c.pred_id_payout,
            "pred_ev": c.pred_ev, "pred_ev_lower": c.pred_ev_lower, "selected": 0, "exploration": int(c.exploration),
            "explanation_json": dumps({**c.fair.explanation, "u_base": c.fair.u_base, "wording_a": c.a.wording, "wording_b": c.b.wording}),
            "features_json": dumps(f),
        })


def _outcome_counts(db: DB, run_id: int) -> dict:
    return {f"{r['identity']}:{r['outcome']}": r["n"] for r in db.query("SELECT identity, outcome, COUNT(*) n FROM rfqs WHERE run_id=? GROUP BY identity, outcome", (run_id,))}
