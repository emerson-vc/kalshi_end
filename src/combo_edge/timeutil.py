"""Time helpers. The NFL gameweek runs Thursday–Monday; it ends Tuesday morning US Eastern."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def now() -> float:
    return time.time()


def parse_ts(s: str | float | int | None) -> float:
    if s is None or s == "":
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def gameweek_end(ts: float) -> float:
    """End of the current NFL gameweek: the next Tuesday 06:00 ET strictly after ``ts``."""
    d = datetime.fromtimestamp(ts, tz=ET)
    days = (1 - d.weekday()) % 7  # Tuesday = 1
    cand = (d + timedelta(days=days)).replace(hour=6, minute=0, second=0, microsecond=0)
    if cand.timestamp() <= ts:
        cand += timedelta(days=7)
    return cand.timestamp()


def local_day(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time()).strftime("%Y-%m-%d")


def start_of_local_day(ts: float | None = None) -> float:
    d = datetime.fromtimestamp(ts if ts is not None else time.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def week_start_utc(ts: float | None = None) -> float:
    """Rolling 7-day window start used for the weekly combo-creation cap."""
    return (ts if ts is not None else time.time()) - 7 * 86400


def hours_to(ts_future: float, ts_now: float | None = None) -> float:
    return (ts_future - (ts_now if ts_now is not None else time.time())) / 3600.0


def fmt_kickoff(ts: float) -> str:
    d = datetime.fromtimestamp(ts, tz=ET)
    return d.strftime("%a %b %-d, %-I:%M %p ET")


def fmt_ago(ts: float | None) -> str:
    if not ts:
        return "never"
    s = time.time() - ts
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"
