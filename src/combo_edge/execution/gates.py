"""Execution gates and sizing (pure functions, fully unit-tested).

Gates, all on an ID quote:
* required EV after fees rises with payout (curve interpolated in log-payout; above 50x: +100%);
* EV must stay positive at the fair model's lower bound;
* the family's conditional passes its history calibration check;
* the quote is at most 3 s old;
* the stake fits every cap: $5 per combo, $30 per day, 3 bets per game, 2 per player, never the
  exact same combo twice (different lines are different combos).
Sizing: ¼ Kelly × uncertainty shrink, scaled by edge between $1 and $5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from combo_edge import money


def required_ev(payout: float, curve: list[list[float]], longshot_payout: float = 50.0, longshot_ev: float = 1.0) -> float:
    if payout > longshot_payout:
        return longshot_ev
    pts = sorted((float(p), float(e)) for p, e in curve)
    if payout <= pts[0][0]:
        return pts[0][1]
    if payout >= pts[-1][0]:
        return pts[-1][1]
    lx = math.log(payout)
    for (p0, e0), (p1, e1) in zip(pts, pts[1:]):
        if p0 <= payout <= p1:
            t = (lx - math.log(p0)) / (math.log(p1) - math.log(p0))
            return e0 + t * (e1 - e0)
    return pts[-1][1]


@dataclass
class Exposure:
    spent_today: float = 0.0
    bets_by_game: dict[str, int] = field(default_factory=dict)
    bets_by_player: dict[str, int] = field(default_factory=dict)
    bought_combos: set[str] = field(default_factory=set)


@dataclass
class GateInput:
    identity: str
    payout: float
    fair_joint: float
    fair_lower: float
    family_ok: bool
    family_note: str
    quote_age_s: float
    combo_ticker: str
    game_id: str
    players: list[str]
    rfq_stake: float


@dataclass
class GateResult:
    passes: bool
    reason: str
    chip: str
    required_ev: float
    ev: float
    ev_lower: float
    stake: float
    cap: str | None = None
    stale: bool = False
    size_mismatch: bool = False


def size_stake(fair: float, fair_lower: float, payout: float, ev: float, req: float, s: Any) -> float:
    """¼ Kelly × uncertainty shrink, then scaled by edge between min and max stake (whole dollars)."""
    bankroll = float(s.get("bankroll"))
    lo, hi = float(s.get("min_stake")), float(s.get("max_stake_per_combo"))
    shrink = max(0.0, min(1.0, fair_lower / fair)) if fair > 0 else 0.0
    kelly = bankroll * float(s.get("kelly_fraction")) * money.kelly_fraction(fair, payout) * shrink
    excess = max(0.0, ev - req)
    scale = min(1.0, excess / max(float(s.get("edge_for_max_stake")), 1e-6))
    edge_scaled = lo + (hi - lo) * scale
    stake = min(kelly, edge_scaled, hi)
    return float(max(lo, math.floor(stake + 1e-9)))


def evaluate(g: GateInput, exp: Exposure, s: Any) -> GateResult:
    req = required_ev(g.payout, s.get("payout_curve"), float(s.get("longshot_payout")), float(s.get("longshot_required_ev")))
    ev = money.ev(g.fair_joint, g.payout)
    evl = money.ev(g.fair_lower, g.payout)

    def no(reason: str, chip: str, cap: str | None = None, stake: float = 0.0, **kw: Any) -> GateResult:
        return GateResult(False, reason, chip, req, ev, evl, stake, cap, **kw)

    if g.identity != "id":
        return no("anonymous quote: never executed (re-quote under your ID first)", "Below gate (anonymous)")
    if ev < req:
        if g.payout > float(s.get("longshot_payout")):
            return no(f"above {s.get('longshot_payout'):g}x needs +{req:.0%} EV (has {ev:+.0%})", f"Below gate (needs +{req:.0%})")
        return no(f"EV {ev:+.0%} below the +{req:.0%} needed at {g.payout:.1f}x", f"Below gate (needs +{req:.0%})")
    if evl <= 0:
        return no(f"EV at the lower bound is {evl:+.0%}", "Below gate (lower bound)")
    if not g.family_ok:
        return no(g.family_note, "Below gate (calibration)")
    stake = size_stake(g.fair_joint, g.fair_lower, g.payout, ev, req, s)
    per_combo = float(s.get("max_stake_per_combo"))
    remaining = float(s.get("daily_spend_cap")) - exp.spent_today
    if g.combo_ticker in exp.bought_combos:
        return no("already bought this exact combo", "Cap reached (same combo)", "same_combo")
    if exp.bets_by_game.get(g.game_id, 0) >= int(s.get("max_bets_per_game")):
        return no("3 bets on this game already", "Cap reached (per game)", "per_game")
    for p in g.players:
        if exp.bets_by_player.get(p, 0) >= int(s.get("max_bets_per_player")):
            return no("2 bets on this player already", "Cap reached (per player)", "per_player")
    bet = min(g.rfq_stake, per_combo)
    if bet > remaining + 1e-9:
        return no(f"daily cap: ${remaining:.2f} left of ${s.get('daily_spend_cap'):g}", "Cap reached (daily $)", "daily", stake=stake)
    if g.quote_age_s > float(s.get("max_quote_age_s")):
        return GateResult(False, f"quote is {g.quote_age_s:.1f}s old (max {s.get('max_quote_age_s'):g}s)", "Stale: refresh", req, ev, evl, stake, None, stale=True)
    if stake < g.rfq_stake - 0.5:
        return GateResult(False, f"sizing wants ${stake:.0f} but the RFQ was for ${g.rfq_stake:.0f}", "Passes gates", req, ev, evl, stake, None, size_mismatch=True)
    return GateResult(True, "passes every gate", "Passes gates", req, ev, evl, bet)
