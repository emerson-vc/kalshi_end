"""“Why did it miss this?” — follow one 2-leg combo through a scan and report where it stopped.

Stages: generated → candidate cap → selected → looked up / created → anonymous quote → ID shortlist →
ID quote → gates. Everything after "generated" is read from what the scan stored (candidates, rfqs,
decisions). When the pair isn't in the run's candidates at all, the game is re-generated from the
current markets (read-only) to name the rule that excluded it, so that part reflects today's legs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from combo_edge.storage.db import DB

OK_STATUSES = ("found_local", "found_feed", "created")
STATUS_TEXT = {
    "no_collection": "no Kalshi combo collection lists both legs, so it can't be created",
    "cap_reached": "the creation cap was reached (per-run or weekly) before this combo's turn",
    "rejected": "Kalshi rejected the combo when it was created (its class is now skipped)",
    "create_error": "creating the combo failed with an error",
}


@dataclass
class LegSpec:
    ticker: str
    side: str = "yes"


def parse_leg(text: str) -> LegSpec:
    """``TICKER`` or ``TICKER:no`` / ``TICKER:yes``."""
    t = text.strip()
    side = "yes"
    if ":" in t:
        t, side = t.rsplit(":", 1)
        side = side.strip().lower()
        if side not in ("yes", "no"):
            raise ValueError(f"side must be yes or no (got {side!r})")
    if not t:
        raise ValueError("empty leg ticker")
    return LegSpec(t.strip(), side)


def latest_run_with_candidates(db: DB, env: str | None = None) -> int | None:
    sql = "SELECT MAX(c.run_id) FROM candidates c JOIN runs r ON r.id=c.run_id"
    if env:
        return db.scalar(sql + " WHERE r.env=?", (env,))
    return db.scalar(sql)


def find_candidate(db: DB, run_id: int, a: LegSpec, b: LegSpec) -> dict | None:
    return db.one(
        "SELECT * FROM candidates WHERE run_id=? AND ((leg_a=? AND side_a=? AND leg_b=? AND side_b=?) OR (leg_a=? AND side_a=? AND leg_b=? AND side_b=?)) "
        "ORDER BY id LIMIT 1",
        (run_id, a.ticker, a.side, b.ticker, b.side, b.ticker, b.side, a.ticker, a.side),
    )


def _wording(db: DB, leg: LegSpec) -> str:
    r = db.one("SELECT wording FROM legs WHERE ticker=?", (leg.ticker,))
    w = (r or {}).get("wording") or leg.ticker
    return w if leg.side == "yes" else f"No: {w}"


async def explain_not_generated(session: Any, db: DB, settings: Any, a: LegSpec, b: LegSpec) -> list[str]:
    """Re-generate the game from current markets and name the rule that keeps this pair out."""
    from combo_edge.pipeline.resolve import rejected_class_set
    from combo_edge.sports.base import ComboLeg
    from combo_edge.sports.nfl.candidates import core_family_for, generate
    from combo_edge.sports.nfl.fair import FairModel
    from combo_edge.sports.nfl.universe import load_week

    games = {r["ticker"]: r["game_id"] for r in db.query("SELECT ticker, game_id FROM legs WHERE ticker IN (?,?)", (a.ticker, b.ticker))}
    ga, gb = games.get(a.ticker), games.get(b.ticker)
    if ga and gb and ga != gb:
        return ["the legs are in different games: scans only build same-game combos"]
    uni = await load_week(session.client, time.time(), game_ids=[ga or gb] if (ga or gb) else None)
    g = next((g for g in uni.games if any(l.ticker == a.ticker for l in g.legs) and any(l.ticker == b.ticker for l in g.legs)), None)
    if g is None:
        missing = [x.ticker for x in (a, b) if not any(l.ticker == x.ticker for gg in uni.games for l in gg.legs)]
        if missing:
            return [f"not found among this gameweek's parsed legs: {', '.join(missing)} (closed, not yet listed, or an unparsed market type)"]
        return ["the legs are in different games: scans only build same-game combos"]
    la = next(l for l in g.legs if l.ticker == a.ticker)
    lb = next(l for l in g.legs if l.ticker == b.ticker)
    fm = FairModel(g, settings, n_paths=3000)
    trace = {"key": sorted([(a.ticker, a.side), (b.ticker, b.side)])}
    generate(g, fm, settings, rejected=rejected_class_set(db), trace=trace)
    attempts = trace.get("attempts") or []
    results = {r for _, r in attempts}
    if "added" in results:
        fam = next((f for f, r in attempts if r == "added"), None)
        return [f"with today's markets it IS generated ({fam or 'exploration'}); the legs or their prices changed since that run"]
    if "forced" in results:
        return ["one leg forces or excludes the other (e.g. 4+ pass TDs with a team total under 24), so Kalshi refuses it (forced())"]
    if "rejected_class" in results:
        return ["Kalshi rejected a combo of this class before (rejected_classes), so it is skipped"]
    _, why = core_family_for(g, fm.roster, ComboLeg(la, a.side), ComboLeg(lb, b.side), settings, fm)
    return why + ["exploration is a random sample of other pairs and didn't draw this one"]


async def trace_combo(db: DB, a: LegSpec, b: LegSpec, *, run_id: int | None = None, session: Any = None, settings: Any = None,
                      env: str | None = None) -> dict:
    run_id = run_id or latest_run_with_candidates(db, env)
    out: dict[str, Any] = {"run_id": run_id, "legs": [_wording(db, a), _wording(db, b)], "steps": [], "verdict": ""}
    steps: list[dict] = out["steps"]

    def step(stage: str, ok: bool | None, detail: str) -> None:
        steps.append({"stage": stage, "ok": ok, "detail": detail})

    def stop(verdict: str) -> dict:
        out["verdict"] = verdict
        return out

    if run_id is None:
        step("Run", False, "no scan with stored candidates yet")
        return stop("no scan to trace")
    run = db.one("SELECT summary_json, started_ts, status FROM runs WHERE id=?", (run_id,)) or {}
    summary = json.loads(run.get("summary_json") or "{}") if run else {}
    n_all = int(db.scalar("SELECT COUNT(*) FROM candidates WHERE run_id=?", (run_id,), 0))
    row = find_candidate(db, run_id, a, b)
    if row is None:
        reasons = ["not in this run's candidates"]
        if session is not None and settings is not None:
            try:
                reasons = await explain_not_generated(session, db, settings, a, b)
            except Exception as e:  # the explanation is best effort
                reasons = [f"not in this run's candidates (couldn't re-check the rules: {type(e).__name__}: {e})"]
        step("Generated", False, "; ".join(reasons))
        return stop("not generated")
    fam = row["family"] or "exploration"
    step("Generated", True, f"{'core family ' + fam if row.get('core') else 'exploration (' + (row['stratum'] or '') + ')'}; "
                            f"predicted ID payout {row['pred_id_payout']:.1f}x, EV at lower bound {row['pred_ev_lower']:+.0%}")

    cap = (summary.get("candidate_cap") or {})
    if row.get("cut_reason") == "candidate_cap":
        step("Candidate cap", False, f"cut by the candidate cap: ranked {row.get('rank_global')} of {cap.get('generated', n_all)} by EV at "
                                     f"the lower bound; {cap.get('kept', '?')} kept (cap {cap.get('cap', '?')}; core pairs are never cut)")
        return stop("cut by the candidate cap")
    step("Candidate cap", True, f"kept (ranked {row.get('rank_global')} of {cap.get('generated', n_all)})")

    sel = json.loads(row.get("sel_json") or "{}")
    pool = sel.get("pool") or row.get("sel_pool") or "?"
    if not row["selected"]:
        if pool.startswith("core_new"):
            if sel.get("rank") is None:
                detail = sel.get("note") or "not creatable"
            else:
                detail = (f"new combo, ranked {sel.get('rank')} in {pool.split(':', 1)[-1]}; only the first {sel.get('cutoff')} were created "
                          f"this run (new-combo budget {(summary.get('selection') or {}).get('new_budget', '?')}, spread across games)")
        elif pool == "exploration":
            detail = f"exploration {sel.get('note') or 'not sampled'} (exploration budget {sel.get('cutoff')})"
        else:
            detail = f"not selected ({pool})"
        step("Selected", False, f"{detail}; predicted {row['pred_id_payout']:.1f}x")
        return stop("not selected")
    step("Selected", True, {"core_existing": "existing core combo: re-quoted every run"}.get(pool, f"selected ({pool}, rank {sel.get('rank')})"))

    status = row.get("status")
    if status not in OK_STATUSES:
        step("Looked up / created", False, STATUS_TEXT.get(status or "", f"not resolved ({status})"))
        return stop(status or "not resolved")
    step("Looked up / created", True, {"found_local": "already existed (yours)", "found_feed": "already existed (seen in the public feed)",
                                       "created": "created this run"}[status] + f": {row.get('combo_ticker')}")

    an = db.one("SELECT outcome, best_payout, n_makers FROM rfqs WHERE candidate_id=? AND identity='anonymous' ORDER BY created_ts DESC LIMIT 1", (row["id"],))
    if an is None:
        step("Anonymous quote", None, "no anonymous screen RFQ recorded (the scan stopped early?)")
    elif an["best_payout"]:
        step("Anonymous quote", True, f"{an['best_payout']:.1f}x from {an['n_makers']} maker(s) (used only as proof of quoting and a floor)")
    else:
        step("Anonymous quote", None, f"{an['outcome']}: no anonymous quote (not held against it)")

    reason = row.get("short_reason") or ""
    if reason not in ("shortlisted", "random pick"):
        rank = row.get("short_rank")
        step("ID shortlist", False, f"{reason or 'not shortlisted'}" + (f" (rank {rank}, shortlist size {row.get('short_cutoff')})" if rank else ""))
        return stop("not shortlisted for an ID quote")
    step("ID shortlist", True, f"{reason}, rank {row.get('short_rank')} of shortlist size {row.get('short_cutoff')}")

    idq = db.one("SELECT outcome, best_payout, n_makers FROM rfqs WHERE candidate_id=? AND identity='id' ORDER BY created_ts DESC LIMIT 1", (row["id"],))
    if idq is None:
        step("ID quote", False, "no ID RFQ recorded (ID budget ran out or the scan stopped)")
        return stop("no ID quote")
    if not idq["best_payout"]:
        step("ID quote", False, f"{idq['outcome']}: no maker quoted it under your ID")
        return stop("no ID quote")
    step("ID quote", True, f"{idq['best_payout']:.1f}x from {idq['n_makers']} maker(s)")
    d = db.one("SELECT action, reason, ev, ev_lower, required_ev FROM decisions WHERE candidate_id=? ORDER BY id DESC LIMIT 1", (row["id"],))
    if d is None:
        step("Gates", None, "no decision recorded")
        return stop("quoted, no decision recorded")
    nums = (f"EV {d['ev']:+.0%} vs +{d['required_ev']:.0%} required; EV at lower bound {d['ev_lower']:+.0%}"
            if d["ev"] is not None and d["required_ev"] is not None and d["ev_lower"] is not None else "")
    if d["action"] == "buy":
        step("Gates", True, f"passed: {d['reason']}" + (f" ({nums})" if nums else ""))
        return stop("bought / booked")
    step("Gates", False, f"{d['reason']}" + (f" ({nums})" if nums else ""))
    return stop("failed a gate")
