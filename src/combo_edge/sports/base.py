"""Sport-agnostic types. Only NFL is implemented (sports/nfl), but the pipeline talks to these."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


@dataclass
class Leg:
    ticker: str
    event_ticker: str
    series: str
    game_id: str
    stat: str
    line: float | None
    team: str | None
    team_id: str | None
    player: str | None
    player_id: str | None
    protected: bool
    title: str
    wording: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    volume_24h: float | None = None
    open_interest: float | None = None
    last_price: float | None = None
    previous_price: float | None = None
    raw: dict = field(default_factory=dict, repr=False)
    # filled by ladders/fair model
    p_market: float | None = None
    p_model: float | None = None
    p_fair: float | None = None
    extrapolated: bool = False
    disagreement: bool = False

    @property
    def k(self) -> int | None:
        """Count threshold for 'k+' markets (line 3.5 → 4)."""
        return None if self.line is None else int(math.floor(self.line)) + 1

    @property
    def two_sided(self) -> bool:
        return bool(self.yes_bid and self.yes_bid > 0 and self.yes_ask and 0 < self.yes_ask < 1 and self.yes_ask >= self.yes_bid)

    @property
    def mid(self) -> float | None:
        if self.two_sided:
            return (self.yes_bid + self.yes_ask) / 2  # type: ignore[operator]
        return None

    @property
    def spread(self) -> float | None:
        return (self.yes_ask - self.yes_bid) if self.two_sided else None  # type: ignore[operator]

    @property
    def ladder_key(self) -> tuple:
        return (self.stat, self.team, self.player_id or self.player, self.protected)

    def snapshot(self) -> dict:
        return {
            "ticker": self.ticker, "yes_bid": self.yes_bid, "yes_ask": self.yes_ask, "no_bid": self.no_bid,
            "no_ask": self.no_ask, "spread": self.spread, "bid_size": self.bid_size, "ask_size": self.ask_size,
            "volume": self.volume, "volume_24h": self.volume_24h, "open_interest": self.open_interest,
            "last_price": self.last_price, "previous_price": self.previous_price,
            "move_since_open": (self.last_price - self.previous_price) if (self.last_price is not None and self.previous_price is not None) else None,
            "prob_market": self.p_market, "prob_model": self.p_model, "prob_fair": self.p_fair,
            "extrapolated": int(self.extrapolated), "two_sided": int(self.two_sided),
        }


@dataclass
class ComboLeg:
    leg: Leg
    side: str  # yes | no

    @property
    def wording(self) -> str:
        return self.leg.wording if self.side == "yes" else f"No: {self.leg.wording}"

    @property
    def p(self) -> float:
        p = self.leg.p_fair if self.leg.p_fair is not None else (self.leg.p_market or 0.5)
        return p if self.side == "yes" else 1 - p

    def selection(self) -> dict:
        return {"market_ticker": self.leg.ticker, "event_ticker": self.leg.event_ticker, "side": self.side}


@dataclass
class Game:
    id: str
    title: str
    start_ts: float
    away_name: str
    home_name: str
    away_code: str
    home_code: str
    away_team_id: str | None
    home_team_id: str | None
    season: int | None
    week: int | None
    event_tickers: list[str]
    legs: list[Leg] = field(default_factory=list)
    unparsed: list[dict] = field(default_factory=list)
    env: Any = None  # GameEnv, filled by the fair model

    def team_name(self, code: str | None) -> str:
        if code == self.away_code:
            return self.away_name
        if code == self.home_code:
            return self.home_name
        return code or "?"

    def opponent(self, code: str | None) -> str | None:
        if code == self.away_code:
            return self.home_code
        if code == self.home_code:
            return self.away_code
        return None

    @property
    def has_player_props(self) -> bool:
        return any(l.player for l in self.legs)

    @property
    def matchup(self) -> str:
        return f"{self.away_code} @ {self.home_code}"


def legs_key(collection: str, selections: list[dict]) -> str:
    return collection + "|" + "|".join(sorted(f"{s['market_ticker']}:{s['side']}" for s in selections))


class SportPlugin(Protocol):
    name: str

    async def load_week(self, client: Any, now: float) -> list[Game]: ...
