"""Budgets that protect the account: identified RFQs per day, combo creations per week, and the
take rate (fills ÷ identified RFQs) that tells whether makers still see a retail customer."""

from __future__ import annotations

import time
from typing import Any

from combo_edge.storage.db import DB
from combo_edge.timeutil import start_of_local_day


class IdBudgetExhausted(Exception):
    pass


def execution_mode(db: DB) -> str:
    return db.get_state("execution_mode", "paper")


def id_cap(settings: Any, db: DB) -> int:
    if execution_mode(db) == "real":
        return int(settings.get("id_rfq_daily_cap"))
    return int(settings.get("id_rfq_daily_cap_paper"))


def id_asks_today(db: DB, env: str) -> int:
    return int(db.scalar(
        "SELECT COUNT(*) FROM rfqs WHERE env=? AND identity='id' AND created_ts>=?",
        (env, start_of_local_day()), 0,
    ))


def id_left(settings: Any, db: DB, env: str) -> int:
    return max(0, id_cap(settings, db) - id_asks_today(db, env))


def check_id_budget(settings: Any, db: DB, env: str, n: int = 1) -> None:
    left = id_left(settings, db, env)
    if left < n:
        raise IdBudgetExhausted(
            f"ID budget used up for today ({id_cap(settings, db)} identified RFQs in {execution_mode(db)} mode); "
            "identified RFQs stop until tomorrow"
        )


def creations_this_week(db: DB, env: str, now: float | None = None) -> int:
    now = now or time.time()
    return int(db.scalar("SELECT COUNT(*) FROM creations WHERE env=? AND ts>=?", (env, now - 7 * 86400), 0))


def creation_cap(settings: Any) -> int:
    return int(settings.get("weekly_creation_cap")) - int(settings.get("weekly_creation_reserve"))


def creations_left(settings: Any, db: DB, env: str) -> int:
    return max(0, creation_cap(settings) - creations_this_week(db, env))


def open_rfqs(db: DB, env: str) -> int:
    return int(db.scalar("SELECT COUNT(*) FROM rfqs WHERE env=? AND status='open'", (env,), 0))


def spent_today(db: DB, env: str, book: str) -> float:
    return float(db.scalar(
        "SELECT COALESCE(SUM(stake),0) FROM executions WHERE env=? AND book=? AND ts>=? AND status IN ('filled','paper_filled','pending')",
        (env, book, start_of_local_day()), 0.0,
    ))


def take_rate(db: DB, env: str, days: int) -> dict:
    since = time.time() - days * 86400
    fills = int(db.scalar("SELECT COUNT(DISTINCT order_id) FROM fills WHERE env=? AND is_combo=1 AND ts>=?", (env, since), 0))
    asks = int(db.scalar("SELECT COUNT(*) FROM rfqs WHERE env=? AND identity='id' AND created_ts>=?", (env, since), 0))
    return {"days": days, "fills": fills, "asks": asks, "rate": (fills / asks) if asks else None}


def gauges(settings: Any, db: DB, env: str) -> dict:
    mode = execution_mode(db)
    book = "real" if mode == "real" else "paper"
    tr7 = take_rate(db, env, 7)
    return {
        "env": env, "mode": mode,
        "spent_today": spent_today(db, env, book), "daily_cap": float(settings.get("daily_spend_cap")),
        "creations_week": creations_this_week(db, env), "creation_cap": int(settings.get("weekly_creation_cap")),
        "creation_usable": creation_cap(settings),
        "id_today": id_asks_today(db, env), "id_cap": id_cap(settings, db),
        "take_rate_7d": tr7, "take_rate_1d": take_rate(db, env, 1), "take_rate_30d": take_rate(db, env, 30),
        "take_rate_warn": tr7["rate"] is not None and tr7["asks"] >= 10 and tr7["rate"] < float(settings.get("take_rate_warn")),
        "open_rfqs": open_rfqs(db, env), "bankroll": float(settings.get("bankroll")), "take_rate_warn_level": float(settings.get("take_rate_warn")),
    }
