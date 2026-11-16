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
from collections import Counter
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
from combo_edge.pipeline.selection import apply_candidate_cap, is_core, is_trial, select
from combo_edge.settings import Settings, env_config
from combo_edge.sports.nfl import calibration, nflverse
from combo_edge.sports.nfl.candidates import FAMILIES, Candidate, generate, price_all
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
        generated: list[Candidate] = []
        per_game: dict[str, int] = {}
        game_rows: dict[str, dict] = {}
        # every game first (no kickoff-order cut), then one ranking across all games
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
            n_core = sum(1 for c in gc if is_core(c))
            game_rows[g.id] = {"matchup": f"{g.away_code} @ {g.home_code}", "kickoff_ts": g.start_ts, "generated": len(gc),
                               "core": n_core, "exploration": len(gc) - n_core}
            log.info("candidates %s @ %s: %d core, %d exploration", g.away_code, g.home_code, n_core, len(gc) - n_core)
            generated.extend(gc)
            await asyncio.sleep(0)
        cands, cut = apply_candidate_cap(generated, int(s.get("max_candidates_per_run")))
        for c in cands:
            game_rows[c.game.id]["kept"] = game_rows[c.game.id].get("kept", 0) + 1
        cov = coverage(uni.games, per_game)
        summary["coverage"] = cov
        summary["candidate_cap"] = {"generated": len(generated), "kept": len(cands), "cut": len(cut), "cap": int(s.get("max_candidates_per_run"))}
        _store_candidates(db, run_id, cands, s)
        _store_candidates(db, run_id, cut, s, slim=True)
        prog.set("candidates", len(cands), "done", f"{sum(1 for c in cands if is_core(c))} core + "
                 f"{sum(1 for c in cands if not is_core(c))} exploration across {len(game_rows)} games")

        # 3. select: existing combos are re-quoted free; the budget counts new creations only
        resolver = ComboResolver(session, db, s, uni, run_id=run_id, max_new=params.max_new)
        summary["prelookup"] = dict(resolver.prelookup(cands))
        new_budget = min(int(s.get("creation_budget_per_run")), resolver.max_new, budget.creations_left(s, db, env))
        bandit = Bandit(db)
        chosen = select(cands, s, new_budget, bandit)
        _store_selection(db, cands, chosen)
        sel_counts: dict[str, int] = {}
        for c in chosen:
            k = c.family if is_core(c) else "exploration"
            sel_counts[k] = sel_counts.get(k, 0) + 1
        cand_counts: dict[str, int] = {}
        for c in cands:
            k = c.family if is_core(c) else "exploration"
            cand_counts[k] = cand_counts.get(k, 0) + 1
        summary["families"] = [
            {"family": f, "kind": "trial" if is_trial(f, s) else "core", "candidates": cand_counts.get(f, 0), "selected": sel_counts.get(f, 0)}
            for f in list(FAMILIES) + ["exploration"]
        ]
        summary["families"][-1]["kind"] = "exploration"
        summary["near_locked"] = sum(1 for c in cands if c.fair and c.fair.near_lock)
        summary["fair_sources"] = dict(Counter((c.fair.source or c.fair.method).split(" n=")[0] for c in cands if c.fair))
        summary["selection"] = {"new_budget": new_budget, "by_family": sel_counts,
                                "core_existing": sum(1 for c in chosen if c.extra["sel"]["pool"] == "core_existing"),
                                "core_new": sum(1 for c in chosen if c.extra["sel"]["pool"].startswith("core_new")),
                                "exploration": sum(1 for c in chosen if c.extra["sel"]["pool"] == "exploration")}
        prog.set("selected", len(chosen), "done")

        # 4. resolve
        prog.set("resolved")
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
        for c in cands:
            if c.combo_ticker:
                game_rows[c.game.id]["covered"] = game_rows[c.game.id].get("covered", 0) + int(is_core(c))
        core_total = sum(1 for c in cands if is_core(c))
        core_covered = sum(1 for c in cands if is_core(c) and c.combo_ticker)
        summary["core_coverage"] = {"covered": core_covered, "total": core_total,
                                    "pct": (core_covered / core_total) if core_total else None}
        summary["per_game"] = sorted(game_rows.values(), key=lambda r: r["kickoff_ts"] or 0)

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
        n_short = shortlist_size(s, id_left)
        shortlist = build_shortlist(resolved, anon, id_model, s, n_short)
        _store_shortlist(db, resolved, n_short)
        prog.set("shortlist", len(shortlist), "done", "" if n_short else
                 f"ID budget: {id_left} asks left today, {int(s.get('id_reserve') or 0)} kept back for asking by hand")
        summary["id_budget_left_before"] = id_left
        summary["id_shortlist_size"] = n_short

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
            creations_used_week=budget.creations_this_week(db, env), creation_cap=budget.creation_cap(s),
            created_this_run=resolver.new_this_run, aborted=abort(),
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
    from combo_edge.execution.gates import required_ev_for, size_stake

    pay = c.pred_id_payout
    req = required_ev_for(pay, s)
    lower = c.fair.lower(max(pay, c.fair.fair_payout), s)
    st = size_stake(c.fair.fair_joint, lower, pay, c.pred_ev, req, s)
    return float(min(max(st, float(s.get("min_stake"))), default))


def shortlist_size(s: Any, id_left: int) -> int:
    """ID asks left today minus the hand-asking reserve, capped by id_shortlist_per_run."""
    return max(0, min(int(s.get("id_shortlist_per_run")), id_left - int(s.get("id_reserve") or 0)))


FAILED_SCREEN = ("rejected_combo", "conflict_409", "create_error", "aborted")


def shortlist_score(c: Candidate, s: Any, quoted: bool) -> float:
    """Model edge vs the market, at the fair lower bound: half "fair vs the product of the leg prices",
    half EV at the predicted ID payout. A quote on the anonymous screen proves makers will quote the combo
    (small plus); the anonymous price itself only ever raises the predicted ID payout (a floor)."""
    lower = c.fair.lower(max(c.pred_id_payout, c.fair.fair_payout), s)
    market_edge = lower * c.leg_product - 1
    return 0.5 * market_edge + 0.5 * c.pred_ev_lower + (0.05 if quoted else 0.0)


def shortlist_tier(c: Candidate, anon_payout: float | None) -> int:
    """1: near-locked and the anonymous quote or the predicted ID payout beats leg A alone; 2: QB tail stat +
    own team wins; 3: everything else (ordered by the score)."""
    nl = (c.fair.near_lock if c.fair else None) or None
    if nl and nl.get("leg_alone_payout"):
        best = max(anon_payout or 0.0, c.pred_id_payout or 0.0)
        if best > nl["leg_alone_payout"]:
            return 1
    if c.family == "qb_tail_team_win":
        return 2
    return 3


def build_shortlist(resolved: list[Candidate], anon: dict[int, RfqOutcome], model: MakerModel, s: Any, n: int, rng: random.Random | None = None) -> list[Candidate]:
    rng = rng or random.Random(int(time.time()) // 3600)
    lo, hi = (float(x) for x in s.get("payout_band"))
    per_game = int(s.get("id_asks_per_game_per_run") or 6)
    per_player = int(s.get("id_asks_per_player_per_run") or 4)
    scored: list[Candidate] = []
    for c in resolved:
        o = anon.get(id(c))
        if o is not None and o.outcome in FAILED_SCREEN:
            c.extra["short"] = {"rank": None, "reason": f"anonymous RFQ failed ({o.outcome}), so not asked under your ID"}
            continue
        ap = o.best_payout if o else None
        pay, src = model.predict_id_payout(c, ap, anon_maker_count(o))  # max(model, anonymous): a floor only
        c.pred_id_payout, c.pred_source = pay, src
        lower = c.fair.lower(max(pay, c.fair.fair_payout), s)
        c.pred_ev = c.fair.fair_joint * pay - 1
        c.pred_ev_lower = lower * pay - 1
        c.extra["short_score"] = shortlist_score(c, s, bool(o and o.quotes))
        c.extra["short_tier"] = shortlist_tier(c, ap)
        scored.append(c)
    scored.sort(key=lambda c: (c.extra["short_tier"], -c.extra["short_score"]))
    for i, c in enumerate(scored, 1):
        c.extra["short"] = {"rank": i, "reason": "below the shortlist cutoff"}
    if n <= 0:
        for c in scored:
            c.extra["short"]["reason"] = "no ID asks available for the scan today (after the hand-asking reserve)"
        return []
    out: list[Candidate] = []
    games: Counter = Counter()
    players: Counter = Counter()

    def fits(c: Candidate) -> str | None:
        if games[c.game.id] >= per_game:
            return f"already {per_game} ID asks on this game this scan (id_asks_per_game_per_run)"
        for p in c.players:
            if players[p] >= per_player:
                return f"already {per_player} ID asks on this player this scan (id_asks_per_player_per_run)"
        return None

    def take(c: Candidate, how: str) -> None:
        out.append(c)
        games[c.game.id] += 1
        for p in c.players:
            players[p] += 1
        c.extra["short"]["reason"] = how
        if how == "random pick":
            c.extra["random_pick"] = True

    n_random = int(round(n * float(s.get("id_random_share"))))
    n_top = n - n_random
    for c in scored:
        if len(out) >= n_top:
            break
        why = fits(c)
        if why:
            c.extra["short"]["reason"] = why
        else:
            take(c, "shortlisted")
    picked = {id(c) for c in out}
    mid_band = [c for c in scored if id(c) not in picked and lo <= c.pred_id_payout <= hi]
    rng.shuffle(mid_band)
    for c in mid_band:
        if len(out) >= n:
            break
        if not fits(c):
            take(c, "random pick")
            picked.add(id(c))
    for c in scored:
        if len(out) >= n:
            break
        if id(c) not in picked and not fits(c):
            take(c, "shortlisted")
            picked.add(id(c))
    for c in out:
        c.extra["short"]["cutoff"] = n
    for c in scored:
        c.extra["short"].setdefault("cutoff", n)
    return out


def _store_selection(db: DB, cands: list[Candidate], chosen: list[Candidate]) -> None:
    from combo_edge.storage.db import dumps

    picked = {id(c) for c in chosen}
    rows = []
    for c in cands:
        sel = c.extra.get("sel") or {}
        rows.append((int(id(c) in picked), sel.get("pool"), sel.get("rank"), sel.get("cutoff"), c.extra.get("prestatus"), dumps(sel), c.db_id))
    db.executemany("UPDATE candidates SET selected=?, sel_pool=?, sel_rank=?, sel_cutoff=?, prestatus=?, sel_json=? WHERE id=?", rows)


def _store_shortlist(db: DB, resolved: list[Candidate], n: int) -> None:
    from combo_edge.storage.db import dumps

    db.executemany("UPDATE candidates SET short_rank=?, short_cutoff=?, short_reason=?, short_score=? WHERE id=?", [
        ((c.extra.get("short") or {}).get("rank"), n, (c.extra.get("short") or {}).get("reason"), c.extra.get("short_score"), c.db_id)
        for c in resolved
    ])


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


def _store_candidates(db: DB, run_id: int, cands: list[Candidate], s: Any, *, slim: bool = False) -> None:
    """``slim`` (candidates cut by the cap): enough for the trace tool, without explanation or features."""
    from combo_edge.storage.db import dumps

    for c in cands:
        common = {"core": int(is_core(c)), "rank_global": c.extra.get("rank_global"), "cut_reason": c.extra.get("cut_reason")}
        if slim:
            c.db_id = db.insert("candidates", {
                "run_id": run_id, "game_id": c.game.id, "family": c.family, "stratum": c.stratum, "leg_a": c.a.leg.ticker, "leg_b": c.b.leg.ticker,
                "side_a": c.a.side, "side_b": c.b.side, "p_a": c.a.p, "p_b": c.b.p, "fair_joint": c.fair.fair_joint, "leg_product": c.leg_product,
                "pred_id_payout": c.pred_id_payout, "pred_ev": c.pred_ev, "pred_ev_lower": c.pred_ev_lower, "selected": 0,
                "exploration": int(c.exploration), **common,
            })
            continue
        f = candidate_features(c)
        c.extra["features"] = f
        c.db_id = db.insert("candidates", {
            **common,
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
