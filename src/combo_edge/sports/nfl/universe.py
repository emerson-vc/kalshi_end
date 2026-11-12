"""List every open NFL game in the current gameweek and every combo-eligible leg."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from combo_edge.sports.base import Game
from combo_edge.sports.nfl.ladders import fit_game
from combo_edge.sports.nfl.parser import build_game
from combo_edge.timeutil import gameweek_end, iso, parse_ts

log = logging.getLogger(__name__)


@dataclass
class Universe:
    games: list[Game]
    collections: list[dict]
    eligible_events: dict[str, set[str]] = field(default_factory=dict)  # collection → event tickers
    errors: list[str] = field(default_factory=list)

    def collections_for(self, event_tickers: list[str]) -> list[str]:
        return [c for c, evs in self.eligible_events.items() if all(e in evs for e in event_tickers)]


async def load_week(client: Any, now: float, *, game_ids: list[str] | None = None) -> Universe:
    end = gameweek_end(now)
    milestones = await client.milestones(type="football_game", minimum_start_date=iso(now - 3600))
    nfl = [
        m for m in milestones
        if (m.get("details") or {}).get("league") == "NFL" and now + 60 < parse_ts(m["start_date"]) <= end
    ]
    if game_ids:
        nfl = [m for m in nfl if m["id"] in game_ids]
    collections = await client.collections(status="open")
    eligible: dict[str, set[str]] = {}
    for c in collections:
        evs = {a["ticker"] for a in c.get("associated_events") or []}
        if any(e.startswith("KXNFL") for e in evs):
            eligible[c["collection_ticker"]] = evs
    all_eligible = set().union(*eligible.values()) if eligible else set()
    uni = Universe(games=[], collections=collections, eligible_events=eligible)
    for m in sorted(nfl, key=lambda m: m["start_date"]):
        tickers = [t for t in m.get("related_event_tickers") or [] if t in all_eligible]
        if not tickers:
            uni.errors.append(f"{m.get('title')}: no combo-eligible events yet")
        try:
            events = await client.events(tickers) if tickers else []
        except Exception as e:  # keep going: one bad game must not stop the scan
            uni.errors.append(f"{m.get('title')}: {e}")
            events = []
        g = build_game(m, events)
        fit_game(g)
        uni.games.append(g)
    return uni
