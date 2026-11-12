"""Same-game 2-leg candidates: the proven families plus exploration strata.

Proven families (guaranteed budget shares, config/core.yaml):
* qb_ptd_team_total — QB 3+/4+ pass TDs × own team total within ±7 of the main line (5+ → exploration)
* qb_pyds_rec_yds  — QB 275–325+ pass yards × own top-2 WR/TE (by rec-yds median) 100–190+ rec yards
* rec_td_team_total — top-2 WR/TE (by anytime-TD price) 2+/3+ TDs × own team total within ±7
* ml_h1_lead        — team ML × same team's 1H lead ladder
Exploration: leg-type pair × relation × tailness × side, including "No" versions of correlated pairs.
Leg liquidity is never used to filter (combos get quoted even without a single-market price).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any

from combo_edge import money
from combo_edge.sports.base import ComboLeg, Game, Leg, legs_key
from combo_edge.sports.nfl import priors
from combo_edge.sports.nfl.fair import FairModel, FairResult
from combo_edge.sports.nfl.players import player_key

FAMILIES = ("qb_ptd_team_total", "qb_pyds_rec_yds", "rec_td_team_total", "ml_h1_lead")
SINGLE_PER_EVENT = {"ml", "spread", "game_total", "h1_winner", "h1_spread", "h1_total", "win_margin"}


@dataclass
class Candidate:
    game: Game
    a: ComboLeg
    b: ComboLeg
    family: str | None
    stratum: str
    exploration: bool
    group: str  # round-robin key (player or team)
    fair: FairResult | None = None
    leg_payout_a: float = 0.0
    leg_payout_b: float = 0.0
    synthetic_legs: int = 0
    pred_id_payout: float = 0.0
    pred_ev: float = 0.0
    pred_ev_lower: float = 0.0
    pred_source: str = "prior"
    db_id: int | None = None
    combo_ticker: str | None = None
    collection: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def leg_product(self) -> float:
        return self.leg_payout_a * self.leg_payout_b

    def selections(self) -> list[dict]:
        return [self.a.selection(), self.b.selection()]

    def key(self, collection: str) -> str:
        return legs_key(collection, self.selections())

    @property
    def players(self) -> list[str]:
        return sorted({player_key(cl.leg) for cl in (self.a, self.b) if cl.leg.player})

    @property
    def label(self) -> str:
        return f"{self.a.wording} + {self.b.wording}"


def leg_payout(cl: ComboLeg) -> tuple[float, bool]:
    """Fee-inclusive payout of buying the leg alone at its ask; synthetic (fair, no vig) if no ask."""
    l = cl.leg
    if cl.side == "yes":
        ask = l.yes_ask if (l.yes_ask and 0 < l.yes_ask < 1) else None
    else:
        ask = (1 - l.yes_bid) if (l.yes_bid and 0 < l.yes_bid < 1) else None
    if ask is not None:
        return money.payout(ask), False
    return money.payout(min(max(cl.p, 0.005), 0.995)), True


def forced(a: ComboLeg, b: ComboLeg) -> bool:
    """True when one leg logically forces (or excludes) the other — Kalshi rejects these."""
    la, lb = a.leg, b.leg
    if la.ticker == lb.ticker:
        return True
    if la.ladder_key == lb.ladder_key:
        return True  # two rungs of one ladder
    if la.event_ticker == lb.event_ticker and la.stat in SINGLE_PER_EVENT:
        return True
    for x, y in ((a, b), (b, a)):
        lx, ly = x.leg, y.leg
        if x.side == "yes" and y.side == "yes":
            if lx.stat in ("pass_td", "td") and ly.stat == "team_total" and lx.team == ly.team and lx.k and ly.line is not None:
                if 6 * lx.k > ly.line:
                    return True
            if lx.stat == "spread" and ly.stat == "ml" and lx.team == ly.team:
                return True
            if lx.stat == "h1_spread" and ly.stat == "h1_winner" and lx.team == ly.team:
                return True
            if lx.stat == "pass_td" and ly.stat == "game_total" and lx.k and ly.line is not None and 6 * lx.k > ly.line:
                return True
    return False


def class_key(a: ComboLeg, b: ComboLeg) -> str:
    """Rejected-combo class: stats, lines and sides with team relation (learned from Kalshi rejections)."""
    def part(cl: ComboLeg) -> str:
        line = "" if cl.leg.line is None else f"{cl.leg.line:g}"
        return f"{cl.leg.stat}[{line}]{cl.side}"
    rel = "same" if (a.leg.team and a.leg.team == b.leg.team) else ("opp" if (a.leg.team and b.leg.team) else "game")
    p1, p2 = sorted([part(a), part(b)])
    return f"{p1}+{p2}|{rel}"


def tail_bucket(p: float) -> str:
    return "tail" if p < 0.12 else ("mid" if p < 0.4 else "fav")


def stratum(a: ComboLeg, b: ComboLeg) -> str:
    rel = priors.relation(a.leg.team, b.leg.team, player_key(a.leg) if a.leg.player else None,
                          player_key(b.leg) if b.leg.player else None, a.leg.stat, b.leg.stat)
    (s1, t1, d1), (s2, t2, d2) = sorted([(a.leg.stat, tail_bucket(a.p), a.side), (b.leg.stat, tail_bucket(b.p), b.side)])
    return f"{s1}×{s2}|{rel}|{t1}-{t2}|{d1}-{d2}"


def generate(game: Game, fm: FairModel, settings: Any, *, rejected: set[str] | None = None, rng: random.Random | None = None,
             max_exploration: int = 400) -> list[Candidate]:
    rejected = rejected or set()
    rng = rng or random.Random(hash(game.id) & 0xFFFF)
    r = fm.roster
    window = float(settings.get("team_total_window"))
    by = {}
    for l in game.legs:
        if l.protected:
            continue
        by.setdefault((l.stat, l.team), []).append(l)
    out: list[Candidate] = []
    seen: set[tuple] = set()

    def add(a: Leg, sa: str, b: Leg, sb: str, family: str | None, group: str, exploration: bool) -> None:
        ca, cb = ComboLeg(a, sa), ComboLeg(b, sb)
        key = tuple(sorted([(a.ticker, sa), (b.ticker, sb)]))
        if key in seen or forced(ca, cb) or class_key(ca, cb) in rejected:
            return
        seen.add(key)
        out.append(Candidate(game, ca, cb, family, stratum(ca, cb), exploration, group))

    def team_total_rungs(team: str) -> list[Leg]:
        main = game.env.main_team_line.get(team) if game.env else None
        legs = by.get(("team_total", team), [])
        if main is None:
            return legs
        return [l for l in legs if l.line is not None and abs(l.line - main) <= window + 1e-9]

    for team in (game.away_code, game.home_code):
        tts = team_total_rungs(team)
        # Family 1: QB pass TDs × own team total
        for l in by.get(("pass_td", team), []):
            if r.roles.get(player_key(l)) != "QB" or not l.k:
                continue
            for t in tts:
                if l.k in (3, 4):
                    add(l, "yes", t, "yes", "qb_ptd_team_total", player_key(l), False)
                elif l.k >= 5:
                    add(l, "yes", t, "yes", "qb_ptd_team_total", player_key(l), True)
                # "No" versions of the correlated pair go to exploration
                if l.k in (3, 4):
                    add(l, "yes", t, "no", None, player_key(l), True)
        # Family 2: QB pass yards × own top-2 receiver yards
        top_yds = set(r.top_yardage.get(team, []))
        qb_yds = [l for l in by.get(("pass_yds", team), []) if r.roles.get(player_key(l)) == "QB" and l.line and 274 <= l.line <= 325]
        rec_yds = [l for l in by.get(("rec_yds", team), []) if player_key(l) in top_yds and l.line and 99 <= l.line <= 190]
        for q in qb_yds:
            for w in rec_yds:
                add(w, "yes", q, "yes", "qb_pyds_rec_yds", player_key(w), False)
        # Family 3: top-2 receivers (anytime-TD price) 2+/3+ TDs × own team total
        top_td = set(r.top_receivers.get(team, []))
        for l in by.get(("td", team), []):
            if player_key(l) not in top_td or not l.k:
                continue
            for t in tts:
                if l.k in (2, 3):
                    add(l, "yes", t, "yes", "rec_td_team_total", player_key(l), False)
        # Family 4: team ML × same team 1H lead ladder
        for ml in by.get(("ml", team), []):
            for h in by.get(("h1_spread", team), []):
                add(h, "yes", ml, "yes", "ml_h1_lead", team, False)
    # Exploration strata: sample pairs across leg types, relations, tails and sides
    legs = [l for l in game.legs if not l.protected or rng.random() < 0.15]  # protected legs: exploration only
    pool = list(itertools.combinations(legs, 2))
    rng.shuffle(pool)
    n_exp = 0
    for a, b in pool:
        if n_exp >= max_exploration:
            break
        sa, sb = rng.choice([("yes", "yes"), ("yes", "yes"), ("yes", "no"), ("no", "yes")])
        before = len(out)
        add(a, sa, b, sb, None, f"x:{a.stat}×{b.stat}", True)
        n_exp += len(out) - before
    return out


def price_all(cands: list[Candidate], fm: FairModel, settings: Any, maker: Any | None = None) -> None:
    """Fill fair value, leg payouts and the predicted ID payout/EV for each candidate."""
    for c in cands:
        c.fair = fm.price(c.a, c.b, c.family)
        c.leg_payout_a, syn_a = leg_payout(c.a)
        c.leg_payout_b, syn_b = leg_payout(c.b)
        c.synthetic_legs = int(syn_a) + int(syn_b)
        if maker is not None:
            pay, src = maker.predict_id_payout(c)
        else:
            pay, src = prior_id_payout(c), "prior"
        c.pred_id_payout, c.pred_source = pay, src
        c.pred_ev = money.ev(c.fair.fair_joint, pay)
        c.pred_ev_lower = money.ev(c.fair.lower(max(pay, c.fair.fair_payout), settings), pay)


def prior_id_payout(c: Candidate) -> float:
    """Before an ID maker model exists: ~0.9 × leg product for proven families, the product for
    separate-game pairs, and a shrunk ratio (0.95) otherwise."""
    if c.family in FAMILIES:
        r = 0.9
    elif c.game is None:
        r = 1.0
    else:
        r = 0.95
    return max(1.01, r * c.leg_product)


def band_mid(settings: Any) -> float:
    lo, hi = settings.get("payout_band")
    return math.sqrt(float(lo) * float(hi))
