"""Re-quote panel: re-RFQ a fixed panel of already-created combos at 144/96/72/48/24/12/6/1 hours
before kickoff. A small panel under your ID (fits the ID budget), a larger one anonymous. This is
where edge-versus-time and CLV come from. It never creates combos and never buys (it has no
access to the executor)."""

from __future__ import annotations

import time
from typing import Any

from combo_edge.pipeline import budget
from combo_edge.pipeline.budget import IdBudgetExhausted
from combo_edge.storage.db import DB

BUCKET_TOLERANCE = 0.5  # fraction of the gap to the next bucket


def add_to_panels(db: DB, env: str, id_quoted: list[tuple[Any, Any]], anon_quoted: list[tuple[Any, Any]], s: Any) -> None:
    now = time.time()
    for identity, items, cap in (("id", id_quoted, int(s.get("requote_id_panel_size"))), ("anonymous", anon_quoted, int(s.get("requote_anon_panel_size")))):
        def active() -> int:
            return int(db.scalar("SELECT COUNT(*) FROM requote_panel WHERE env=? AND identity=? AND active=1 AND kickoff_ts>?", (env, identity, now), 0))
        for c, _o in sorted(items, key=lambda x: -(x[0].pred_ev_lower or 0)):
            if active() >= cap:
                break
            if c.game.start_ts <= now + 3600:
                continue
            db.insert("requote_panel", {"combo_ticker": c.combo_ticker, "identity": identity, "env": env, "game_id": c.game.id,
                                        "kickoff_ts": c.game.start_ts, "candidate_id": c.db_id, "added_ts": now, "active": 1}, ignore=True)


def due_bucket(hours_to_kickoff: float, buckets: list[float], done: set[float]) -> float | None:
    bs = sorted(buckets, reverse=True)
    for i, b in enumerate(bs):
        nxt = bs[i + 1] if i + 1 < len(bs) else 0.0
        lo = b - (b - nxt) * BUCKET_TOLERANCE
        if lo <= hours_to_kickoff <= b + 0.25 and b not in done:
            return b
    return None


async def requote_due(session: Any, db: DB, s: Any) -> dict:
    from combo_edge.pipeline.quotes import QuoteBook, RfqRunner, sweep_pending_cancels

    env = session.env.name
    now = time.time()
    if db.get_state("schedulers_paused", False):
        return {"skipped": "schedulers paused"}
    db.execute("UPDATE requote_panel SET active=0 WHERE kickoff_ts<=?", (now,))
    panel = db.query("SELECT * FROM requote_panel WHERE env=? AND active=1", (env,))
    buckets = [float(b) for b in s.get("requote_buckets_h")]
    due = []
    for p in panel:
        h = (p["kickoff_ts"] - now) / 3600
        done = {r["bucket_h"] for r in db.query("SELECT bucket_h FROM requote_obs WHERE combo_ticker=? AND identity=?", (p["combo_ticker"], p["identity"]))}
        b = due_bucket(h, buckets, done)
        if b is not None:
            due.append((p, b, h))
    if not due:
        return {"due": 0}
    book = QuoteBook()
    await session.client.configure_limiter(float(s.get("rate_budget_fraction")))
    await session.open_ws(user_filter_self=True)
    if session.ws:
        book.attach(session.ws)
    runner = RfqRunner(session, db, s, book, run_id=None)
    res: dict[str, Any] = {"due": len(due), "sent": 0, "quoted": 0, "id_budget": None}
    for p, b, h in due:
        if p["identity"] == "id" and budget.id_left(s, db, env) <= 0:
            res["id_budget"] = "used up"
            continue
        try:
            out = await runner.run_one(p["combo_ticker"], p["identity"], stake=float(s.get("stake")), purpose="requote",
                                       candidate_id=p["candidate_id"], kickoff_ts=p["kickoff_ts"])
        except IdBudgetExhausted:
            res["id_budget"] = "used up"
            continue
        res["sent"] += 1
        fair = db.scalar("SELECT fair_joint FROM candidates WHERE id=?", (p["candidate_id"],))
        db.insert("requote_obs", {"combo_ticker": p["combo_ticker"], "identity": p["identity"], "rfq_id": out.rfq_id, "bucket_h": b,
                                  "hours_to_kickoff": h, "ts": time.time(), "payout": out.best_payout, "n_makers": out.n_makers, "fair_joint": fair})
        if out.quotes:
            res["quoted"] += 1
    await sweep_pending_cancels(session, db)
    return res


def clv_for(db: DB, combo_ticker: str, identity: str, bought_payout: float) -> float | None:
    """CLV = our payout vs the price nearest kickoff: bought ÷ closing − 1 (positive is good)."""
    r = db.one("SELECT payout FROM requote_obs WHERE combo_ticker=? AND identity=? AND payout IS NOT NULL ORDER BY hours_to_kickoff ASC LIMIT 1",
               (combo_ticker, identity))
    if not r or not r["payout"]:
        return None
    return bought_payout / r["payout"] - 1
