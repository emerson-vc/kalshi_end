"""Leg names in Kalshi-app wording, e.g. "Jayden Daniels: 4+ passing TDs" and
"WAS Commanders over 24.5 points". Team names come from the milestone title, as in the app."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from combo_edge.sports.base import Game, Leg

STAT_NAMES = {
    "pass_td": "passing TDs", "pass_yds": "passing yards", "rec_yds": "receiving yards", "rush_yds": "rushing yards",
    "receptions": "receptions", "td": "TDs", "rush_rec_yds": "rushing + receiving yards", "pass_att": "pass attempts",
    "pass_comp": "completions", "rush_att": "rush attempts",
}


def _num(x: float | None) -> str:
    if x is None:
        return "?"
    return f"{x:g}"


def leg_wording(leg: "Leg", g: "Game") -> str:
    team = g.team_name(leg.team) if leg.team else None
    s = leg.stat
    if leg.player:
        title = leg.title or ""
        if ":" in title:
            w = re.sub(r"\btouchdowns\b", "TDs", title)
            w = re.sub(r"\btouchdown\b", "TD", w)
            w = w.replace("rushing and receiving yards combined", "rushing + receiving yards")
            return w.strip()
        return f"{leg.player}: {leg.k}+ {STAT_NAMES.get(s, s)}" + (" (Protected)" if leg.protected else "")
    if s == "ml":
        return f"{team} to win"
    if s == "spread":
        return f"{team} win by over {_num(leg.line)} points"
    if s == "team_total":
        return f"{team} over {_num(leg.line)} points"
    if s == "game_total":
        return f"Over {_num(leg.line)} total points"
    if s == "h1_spread":
        return f"{team} win 1st half by over {_num(leg.line)} points"
    if s == "h1_winner":
        return f"{team} win the 1st half" if team else "1st half tied"
    if s == "h1_team_total":
        return f"{team} over {_num(leg.line)} 1st-half points"
    if s == "h1_total":
        return f"Over {_num(leg.line)} 1st-half points"
    return leg.title or leg.ticker


def kalshi_game_url(g: "Game") -> str:
    """Game page in the Kalshi app/site (the main game-line event)."""
    ev = next((t for t in g.event_tickers if t.startswith("KXNFLGAME-")), None)
    if ev:
        return f"https://kalshi.com/markets/kxnflgame/{ev.lower()}"
    return "https://kalshi.com/sports/football/nfl"
