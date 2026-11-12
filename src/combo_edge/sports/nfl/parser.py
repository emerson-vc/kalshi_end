"""Parse Kalshi NFL markets into legs. Never crashes on an unknown market: it is returned as
unparsed (with an example) for the coverage report."""

from __future__ import annotations

import logging
import re
from typing import Any

from combo_edge.sports.base import Game, Leg, _f
from combo_edge.sports.nfl.wording import leg_wording

log = logging.getLogger(__name__)

SERIES_STAT: dict[str, str] = {
    "KXNFLGAME": "ml",
    "KXNFLSPREAD": "spread",
    "KXNFLTOTAL": "game_total",
    "KXNFLTEAMTOTAL": "team_total",
    "KXNFLPASSTDS": "pass_td",
    "KXNFLPASSYDS": "pass_yds",
    "KXNFLRECYDS": "rec_yds",
    "KXNFLRSHYDS": "rush_yds",
    "KXNFLREC": "receptions",
    "KXNFLTD": "td",
    "KXNFLANYTD": "td",
    "KXNFL2TD": "td",
    "KXNFL1H": "h1_winner",
    "KXNFL1HSPREAD": "h1_spread",
    "KXNFL1HTEAMTOTAL": "h1_team_total",
    "KXNFL1HTOTAL": "h1_total",
    "KXNFLRRYDS": "rush_rec_yds",
    "KXNFLWINMARGIN": "win_margin",
    "KXNFLPASSATT": "pass_att",
    "KXNFLPASSCOMP": "pass_comp",
    "KXNFLRSHATT": "rush_att",
}
PLAYER_STATS = {"pass_td", "pass_yds", "rec_yds", "rush_yds", "receptions", "td", "rush_rec_yds", "pass_att", "pass_comp", "rush_att"}
_DATE_RX = re.compile(r"^\d{2}[A-Z]{3}\d{2}")


def series_of(event_ticker: str) -> tuple[str, bool]:
    s = event_ticker.split("-")[0]
    protected = False
    if s.endswith("PROT"):
        s, protected = s[: -len("PROT")], True
    return s, protected


def game_code(event_ticker: str) -> str:
    parts = event_ticker.split("-")
    return parts[1] if len(parts) > 1 else ""


def team_codes_from_events(events: list[dict]) -> dict[str, str]:
    """team_id → team code (e.g. NYG), read from game-line markets."""
    out: dict[str, str] = {}
    for e in events:
        s, _ = series_of(e["event_ticker"])
        if s not in ("KXNFLGAME", "KXNFLTEAMTOTAL", "KXNFLSPREAD", "KXNFL1H"):
            continue
        for m in e.get("markets") or []:
            tid = (m.get("custom_strike") or {}).get("football_team")
            suffix = m["ticker"].split("-")[-1]
            code = re.match(r"^[A-Z]+", suffix)
            if tid and code and code.group(0) not in ("TIE",):
                out.setdefault(tid, code.group(0))
    return out


def split_matchup(code: str, away_hint: str | None, team_codes: set[str]) -> tuple[str, str]:
    body = _DATE_RX.sub("", code)
    for c in sorted(team_codes, key=len, reverse=True):
        if body.startswith(c) and body[len(c):] in team_codes:
            return c, body[len(c):]
    if away_hint and body.startswith(away_hint):
        return away_hint, body[len(away_hint):]
    half = len(body) // 2
    return body[:half], body[half:]


def build_game(milestone: dict, events: list[dict]) -> Game:
    from combo_edge.timeutil import parse_ts

    d = milestone.get("details") or {}
    title = milestone.get("title", "")
    away_name, _, home_name = title.partition(" at ")
    tmap = team_codes_from_events(events)
    code = game_code(d.get("main_game_event_ticker") or (milestone.get("primary_event_tickers") or ["-"])[0])
    away_code = tmap.get(d.get("away_team_id", ""))
    home_code = tmap.get(d.get("home_team_id", ""))
    if not (away_code and home_code):
        a, h = split_matchup(code, away_name.split(" ")[0] if away_name else None, set(tmap.values()))
        away_code, home_code = away_code or a, home_code or h
    season = (d.get("season") or {}).get("year")
    week = (d.get("season") or {}).get("week")
    g = Game(
        id=milestone["id"], title=title, start_ts=parse_ts(milestone["start_date"]),
        away_name=away_name.strip() or away_code, home_name=home_name.strip() or home_code,
        away_code=away_code, home_code=home_code, away_team_id=d.get("away_team_id"), home_team_id=d.get("home_team_id"),
        season=season, week=week, event_tickers=[e["event_ticker"] for e in events],
    )
    id_to_code = {d.get("away_team_id"): away_code, d.get("home_team_id"): home_code, **tmap}
    for e in events:
        for m in e.get("markets") or []:
            try:
                leg = parse_market(m, e, g, id_to_code)
            except Exception as ex:  # never crash on odd markets
                log.debug("parse failure %s: %s", m.get("ticker"), ex)
                leg = None
            if leg is None:
                g.unparsed.append({"ticker": m.get("ticker"), "title": m.get("title"), "series": series_of(e["event_ticker"])[0]})
            else:
                g.legs.append(leg)
    return g


def parse_market(m: dict, e: dict, g: Game, id_to_code: dict[str | None, str]) -> Leg | None:
    if m.get("status") not in (None, "active", "open", "initialized"):
        return None
    series, protected = series_of(e["event_ticker"])
    stat = SERIES_STAT.get(series)
    if stat is None:
        return None
    cs = m.get("custom_strike") or {}
    team_id = cs.get("football_team")
    team = id_to_code.get(team_id)
    player_id = cs.get("football_player")
    title = m.get("title") or ""
    player = None
    if stat in PLAYER_STATS:
        player = title.split(":")[0].strip() if ":" in title else (m.get("yes_sub_title") or "").split(":")[0].strip()
        if not player:
            return None
    line = _f(m.get("floor_strike"))
    if stat in ("ml", "h1_winner"):
        suffix = m["ticker"].split("-")[-1]
        if suffix == "TIE":
            team = None
        line = None
    elif stat == "win_margin":
        line = None
    elif line is None:
        return None
    if stat in ("spread", "team_total", "ml", "h1_spread", "h1_team_total") and team is None:
        return None
    leg = Leg(
        ticker=m["ticker"], event_ticker=e["event_ticker"], series=series, game_id=g.id, stat=stat, line=line,
        team=team, team_id=team_id, player=player, player_id=player_id, protected=protected, title=title, wording="",
        yes_bid=_price(m.get("yes_bid_dollars")), yes_ask=_price(m.get("yes_ask_dollars")),
        no_bid=_price(m.get("no_bid_dollars")), no_ask=_price(m.get("no_ask_dollars")),
        bid_size=_f(m.get("yes_bid_size_fp")), ask_size=_f(m.get("yes_ask_size_fp")),
        volume=_f(m.get("volume_fp")), volume_24h=_f(m.get("volume_24h_fp")), open_interest=_f(m.get("open_interest_fp")),
        last_price=_price(m.get("last_price_dollars")), previous_price=_price(m.get("previous_price_dollars")), raw=m,
    )
    leg.wording = leg_wording(leg, g)
    return leg


def _price(x: Any) -> float | None:
    v = _f(x)
    if v is None:
        return None
    return v


def coverage(games: list[Game], candidate_counts: dict[str, int]) -> dict[str, Any]:
    """Coverage report: games with props but zero candidates (red), unparsed legs with examples."""
    rows = []
    unparsed: dict[str, dict] = {}
    for g in games:
        n = candidate_counts.get(g.id, 0)
        rows.append({
            "game_id": g.id, "matchup": g.matchup, "title": g.title, "legs": len(g.legs),
            "player_props": g.has_player_props, "candidates": n, "red": g.has_player_props and n == 0,
            "unparsed": len(g.unparsed),
        })
        for u in g.unparsed:
            s = unparsed.setdefault(u["series"], {"series": u["series"], "count": 0, "examples": []})
            s["count"] += 1
            if len(s["examples"]) < 3:
                s["examples"].append(u["title"] or u["ticker"])
    return {"games": rows, "unparsed": sorted(unparsed.values(), key=lambda x: -x["count"])}
