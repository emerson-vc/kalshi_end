"""Fade monitor: once a week, quote a few fixed control combos both ways (anonymous, then under
your ID seconds later). If the ID/anonymous payout gap shrinks steadily, makers are starting to
treat you like anonymous flow — a red warning goes in the header and on Insights."""

from __future__ import annotations

import math
import time
from typing import Any

from combo_edge.storage.db import DB

WEEK = 7 * 86400


def due(db: DB, env: str) -> bool:
    last = db.scalar("SELECT MAX(ts) FROM fade_obs WHERE env=?", (env,), 0) or 0
    return time.time() - last >= WEEK - 3600


def controls(db: DB, env: str, n: int) -> list[dict]:
    """Fixed control set: the ID-panel combos with the latest kickoff (kept until they expire)."""
    rows = db.query(
        "SELECT p.combo_ticker, c.leg_product FROM requote_panel p LEFT JOIN candidates c ON c.id=p.candidate_id "
        "WHERE p.env=? AND p.identity='id' AND p.active=1 AND p.kickoff_ts>? ORDER BY p.kickoff_ts DESC LIMIT ?",
        (env, time.time() + 3 * 3600, n),
    )
    return rows


async def fade_check(session: Any, db: DB, s: Any, *, force: bool = False) -> dict:
    from combo_edge.pipeline import budget
    from combo_edge.pipeline.quotes import QuoteBook, RfqRunner

    env = session.env.name
    if not force and not due(db, env):
        return {"skipped": "not due"}
    ctrl = controls(db, env, int(s.get("fade_controls")))
    if not ctrl:
        return {"skipped": "no control combos yet (run a scan first)"}
    book = QuoteBook()
    await session.client.configure_limiter(float(s.get("rate_budget_fraction")))
    await session.open_ws(user_filter_self=True)
    if session.ws:
        book.attach(session.ws)
    runner = RfqRunner(session, db, s, book, run_id=None)
    out = []
    for c in ctrl:
        if budget.id_left(s, db, env) <= 0:
            break
        a = await runner.run_one(c["combo_ticker"], "anonymous", stake=float(s.get("stake")), purpose="fade")
        i = await runner.run_one(c["combo_ticker"], "id", stake=float(s.get("stake")), purpose="fade")
        db.insert("fade_obs", {"env": env, "ts": time.time(), "combo_ticker": c["combo_ticker"], "anon_payout": a.best_payout,
                               "id_payout": i.best_payout, "leg_product": c["leg_product"], "anon_makers": a.n_makers,
                               "id_makers": i.n_makers, "anon_rfq": a.rfq_id, "id_rfq": i.rfq_id})
        out.append({"combo": c["combo_ticker"], "anon": a.best_payout, "id": i.best_payout})
    return {"controls": out, "trend": trend(db, env)}


def trend(db: DB, env: str) -> dict:
    rows = db.query("SELECT ts, anon_payout, id_payout, leg_product FROM fade_obs WHERE env=? AND anon_payout>0 AND id_payout>0 ORDER BY ts", (env,))
    weeks: dict[int, list[float]] = {}
    vs_prod: dict[int, list[float]] = {}
    for r in rows:
        w = int(r["ts"] // WEEK)
        weeks.setdefault(w, []).append(math.log(r["id_payout"] / r["anon_payout"]))
        if r["leg_product"]:
            vs_prod.setdefault(w, []).append(r["id_payout"] / r["leg_product"])
    series = [{"week": w, "gap": math.exp(sum(v) / len(v)), "id_vs_product": (sum(vs_prod.get(w, [0])) / len(vs_prod[w])) if vs_prod.get(w) else None, "n": len(v)}
              for w, v in sorted(weeks.items())]
    warn = False
    slope = None
    if len(series) >= 4:
        xs = list(range(len(series)))
        ys = [math.log(p["gap"]) for p in series]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        last3_down = all(series[k]["gap"] < series[k - 1]["gap"] for k in range(len(series) - 3, len(series)))
        warn = slope < -0.05 and (last3_down or series[-1]["gap"] < 0.75 * series[0]["gap"])
    return {"series": series, "slope_log_gap_per_week": slope, "warning": warn}
