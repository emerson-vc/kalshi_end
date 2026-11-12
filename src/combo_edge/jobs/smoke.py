"""Live smoke test: for a few proven-family combos, send one anonymous and one identified RFQ
seconds apart at a tiny target cost, record the best quotes, cancel everything. Never buys —
this module has no access to the executor."""

from __future__ import annotations

import time
from typing import Any

from combo_edge.kalshi.session import build_session
from combo_edge.pipeline.quotes import QuoteBook, RfqRunner, sweep_pending_cancels
from combo_edge.pipeline.resolve import ComboResolver
from combo_edge.settings import EnvConfig, Settings
from combo_edge.sports.nfl.candidates import generate, price_all
from combo_edge.sports.nfl.fair import FairModel
from combo_edge.sports.nfl.universe import load_week
from combo_edge.storage.db import get_db


async def smoke(env: EnvConfig, *, n_combos: int = 5, stake: float = 5.0, max_new: int = 10) -> dict:
    db = get_db()
    s = Settings(db).with_overrides(sim_paths=8000, quote_window_s=5.0)
    session = build_session(env)
    rows: list[dict] = []
    try:
        lim = await session.client.configure_limiter(0.6)
        await sweep_pending_cancels(session, db)
        book = QuoteBook()
        ok = await session.open_ws(user_filter_self=True, wait=15)
        if session.ws:
            book.attach(session.ws)
        uni = await load_week(session.client, time.time())
        cands = []
        for g in uni.games:
            if not g.has_player_props:
                continue
            fm = FairModel(g, s)
            cs = [c for c in generate(g, fm, s) if c.family == "qb_ptd_team_total" and not c.exploration]
            price_all(cs, fm, s)
            cs.sort(key=lambda c: abs(c.pred_id_payout - 20))
            cands.extend(cs[:2])
        cands.sort(key=lambda c: abs(c.pred_id_payout - 20))
        resolver = ComboResolver(session, db, s, uni, run_id=None, max_new=max_new)
        runner = RfqRunner(session, db, s, book, run_id=None)
        for c in cands:
            if len(rows) >= n_combos:
                break
            st = await resolver.resolve(c)
            if not c.combo_ticker:
                continue
            a = await runner.run_one(c.combo_ticker, "anonymous", stake=stake, purpose="smoke", kickoff_ts=c.game.start_ts)
            i = await runner.run_one(c.combo_ticker, "id", stake=stake, purpose="smoke", kickoff_ts=c.game.start_ts)
            rows.append({
                "combo": c.label, "resolution": st, "leg_product": round(c.leg_product, 1), "fair": round(c.fair.fair_payout, 1),
                "anon_best": round(a.best_payout, 2) if a.best_payout else None, "anon_makers": a.n_makers, "anon_outcome": a.outcome,
                "id_best": round(i.best_payout, 2) if i.best_payout else None, "id_makers": i.n_makers, "id_outcome": i.outcome,
            })
        cleanup = await sweep_pending_cancels(session, db)
        return {"env": env.name, "ws_connected": ok, "tier": (lim.get("limits") or {}).get("usage_tier"), "rows": rows,
                "creations": dict(resolver.counts), "rfqs_sent": runner.stats, "cleanup": cleanup,
                "open_after": len(await session.client.my_open_rfqs())}
    finally:
        await session.close()
