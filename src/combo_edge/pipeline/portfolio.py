"""Fills (for take rate and verification) and settlements (for P&L)."""

from __future__ import annotations

import time
from typing import Any

from combo_edge.storage.db import DB, dumps
from combo_edge.timeutil import parse_ts


async def sync_fills(session: Any, db: DB) -> int:
    env = session.env.name
    last = db.get_state(f"fills_synced_ts:{env}", 0)
    fills = await session.client.fills(min_ts=int(last) - 3600 if last else int(time.time() - 35 * 86400))
    n = 0
    for f in fills:
        ticker = f.get("ticker") or f.get("market_ticker") or ""
        n += 1
        db.insert("fills", {
            "fill_id": f.get("fill_id") or f.get("trade_id"), "env": env, "ticker": ticker, "order_id": f.get("order_id"),
            "ts": float(f.get("ts") or parse_ts(f.get("created_time"))), "outcome_side": f.get("outcome_side") or f.get("side"),
            "count": float(f.get("count_fp") or 0), "yes_price": float(f.get("yes_price_dollars") or 0),
            "no_price": float(f.get("no_price_dollars") or 0), "fee": float(f.get("fee_cost") or 0),
            "is_taker": int(bool(f.get("is_taker"))), "is_combo": int(ticker.startswith("KXMVE")), "raw_json": dumps(f),
        }, replace=True)
    db.set_state(f"fills_synced_ts:{env}", time.time())
    return n


async def sync_settlements(session: Any, db: DB) -> int:
    """Mark open bets won/lost from market results (both books)."""
    env = session.env.name
    rows = db.query("SELECT id, combo_ticker, stake, contracts FROM executions WHERE env=? AND result IS NULL AND status IN ('filled','paper_filled')", (env,))
    if not rows:
        return 0
    tickers = sorted({r["combo_ticker"] for r in rows})
    markets = {m["ticker"]: m for m in await session.client.markets(tickers)}
    n = 0
    for r in rows:
        res = (markets.get(r["combo_ticker"]) or {}).get("result")
        if res not in ("yes", "no"):
            continue
        pnl = (r["contracts"] or 0) * (1.0 if res == "yes" else 0.0) - (r["stake"] or 0)
        db.update("executions", "id", r["id"], {"result": "won" if res == "yes" else "lost", "pnl": pnl, "settled_ts": time.time()})
        db.insert("settlements", {"combo_ticker": r["combo_ticker"], "env": env, "result": res, "settled_ts": time.time(), "value": 1.0 if res == "yes" else 0.0}, replace=True)
        n += 1
    return n
