"""Daily automatic scan: once a day, at the best time the timing data has found (10:00 local until
it learns), covering every game in the current NFL gameweek; buys if the toggle is on Real."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from combo_edge.learning.reports import best_scan_hour
from combo_edge.storage.db import DB
from combo_edge.timeutil import local_day


def scheduled_hour(db: DB, env: str, s: Any) -> int:
    return best_scan_hour(db, env, int(s.get("daily_scan_default_hour")))


def next_scan_ts(db: DB, env: str, s: Any, now: float | None = None) -> float | None:
    if not s.get("daily_scan_enabled"):
        return None
    now = now or time.time()
    hour = scheduled_hour(db, env, s)
    d = datetime.fromtimestamp(now)
    target = d.replace(hour=hour, minute=0, second=0, microsecond=0)
    if db.get_state(f"daily_scan_done:{env}") == local_day(now) or target.timestamp() <= now - 3 * 3600:
        target += timedelta(days=1)
    return target.timestamp()


def should_run(db: DB, env: str, s: Any, now: float | None = None) -> tuple[bool, str]:
    now = now or time.time()
    if not s.get("daily_scan_enabled"):
        return False, "daily scan disabled in settings"
    if db.get_state("schedulers_paused", False):
        return False, "schedulers paused (kill switch)"
    if db.get_state(f"daily_scan_done:{env}") == local_day(now):
        return False, "already ran today"
    hour = scheduled_hour(db, env, s)
    if datetime.fromtimestamp(now).hour < hour:
        return False, f"waiting until {hour:02d}:00"
    return True, "due"


async def tick(db: DB, s: Any) -> dict:
    from combo_edge.pipeline.scan import ScanBusy, ScanParams, run_scan
    from combo_edge.settings import env_config

    env = env_config().name
    ok, why = should_run(db, env, s)
    if not ok:
        return {"ran": False, "reason": why}
    db.set_state(f"daily_scan_done:{env}", local_day())
    try:
        summary = await run_scan(ScanParams(kind="daily"), db=db, settings=s)
    except ScanBusy as e:
        db.set_state(f"daily_scan_done:{env}", None)
        return {"ran": False, "reason": str(e)}
    return {"ran": True, "summary": {k: summary.get(k) for k in ("run_id", "games", "id_quoted", "bought", "error")}}
