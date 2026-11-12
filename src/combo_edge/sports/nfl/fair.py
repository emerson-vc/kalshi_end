"""Fair value of a same-game pair.

1. Leg probabilities: Kalshi ladder probability blended with the simulator's own probability —
   Kalshi dominates on liquid legs, the model gets more weight on thin or extrapolated ones.
   Legs where the two disagree a lot are flagged for a manual check (e.g. in OddsJam).
2. Joint: P(A) × P(B | A), where the conditional is read from the simulator at the legs' own
   thresholds, moved to the blended B marginal in log-odds, then shifted toward nflverse history
   for the family. The t-copula prior is used only when the simulator can't price a leg.
3. Tail guards: conditional capped at the family's strongest anchor (or the history-supported
   level where history shows more), joint capped at ``joint_cap_multiple`` × independence.
4. ``fair_lower`` widens with payout (+10% of fair per doubling above 20x by default).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from combo_edge.sports.base import ComboLeg, Game, Leg
from combo_edge.sports.nfl import calibration, priors
from combo_edge.sports.nfl import simulator as sim
from combo_edge.sports.nfl.ladders import expit, logit
from combo_edge.sports.nfl.players import Roster, build_roster, player_key

FAMILY_LABELS = {
    "qb_ptd_team_total": "QB pass TDs + own team total",
    "qb_pyds_rec_yds": "QB pass yards + own top receiver yards",
    "rec_td_team_total": "Top receiver TDs + own team total",
    "ml_h1_lead": "Team win + same team 1H lead",
}


@dataclass
class FairResult:
    fair_joint: float
    u_base: float
    p_a: float
    p_b: float
    cond: float
    anchor_first: bool
    independent: float
    method: str
    delta: float
    hist_n: float
    capped: list[str] = field(default_factory=list)
    explanation: dict = field(default_factory=dict)

    @property
    def fair_payout(self) -> float:
        return 1.0 / max(self.fair_joint, 1e-9)

    def lower(self, payout_ref: float, s: Any) -> float:
        return lower_bound(self.fair_joint, self.u_base, payout_ref, s)


def lower_bound(fair: float, u_base: float, payout_ref: float, s: Any) -> float:
    pivot = float(s.get("lower_bound_pivot_payout"))
    per = float(s.get("lower_bound_per_doubling"))
    doublings = max(0.0, math.log2(max(payout_ref, 1.0) / pivot))
    u = min(u_base + per * doublings, 0.8)
    return fair * (1.0 - u)


def leg_var(leg: Leg, roster: Roster) -> str | None:
    t = leg.team
    s = leg.stat
    if leg.protected:
        return None
    if s == "team_total":
        return f"pts:{t}"
    if s == "game_total":
        return "pts:total"
    if s in ("spread", "ml"):
        return f"margin:{t}"
    if s in ("h1_spread",) or (s == "h1_winner" and t):
        return f"h1margin:{t}"
    if s == "h1_team_total":
        return f"h1:{t}"
    if s == "h1_total":
        return "h1:total"
    if leg.player:
        k = roster.sim_key.get(player_key(leg))
        if not k:
            return None
        role = roster.roles.get(player_key(leg))
        if s == "pass_td" and role == "QB":
            return f"pass_td:{k}"
        if s == "pass_yds" and role == "QB":
            return f"pass_yds:{k}"
        if s in ("rec_yds", "rush_yds", "receptions", "td", "rush_rec_yds"):
            return f"{s}:{k}"
    return None


def leg_line(leg: Leg) -> float:
    if leg.stat in ("ml", "h1_winner"):
        return 0.0
    return float(leg.line or 0.0)


def market_weight(leg: Leg) -> float:
    if leg.extrapolated or not leg.two_sided:
        return 0.35
    if (leg.spread or 1) <= 0.03 and (leg.volume or 0) >= 200:
        return 0.9
    return 0.7


class FairModel:
    def __init__(self, game: Game, settings: Any, *, positions: dict | None = None, pass_shares: dict | None = None, n_paths: int | None = None):
        self.game = game
        self.s = settings
        self.roster = build_roster(game, positions, pass_shares)
        n = int(n_paths or settings.get("sim_paths") or 20000)
        seed = int(hashlib.sha1(game.id.encode()).hexdigest()[:8], 16)
        self.sim = sim.simulate(self.roster.home, self.roster.away, n=n, seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self._ind_cache: dict[tuple, np.ndarray] = {}
        self.vars: dict[str, str | None] = {}
        for l in game.legs:
            v = leg_var(l, self.roster)
            self.vars[l.ticker] = v
            pm = sim.raw_prob(self.sim, v, leg_line(l)) if v else None
            l.p_model = pm
            pk = l.p_market if l.p_market is not None else 0.5
            if pm is not None:
                w = market_weight(l)
                l.p_fair = float(expit(w * logit(pk) + (1 - w) * logit(min(max(pm, 0.002), 0.998))))
                l.disagreement = abs(logit(pk) - logit(min(max(pm, 0.002), 0.998))) > 0.9
            else:
                l.p_fair = pk
                l.disagreement = False

    def _indicator(self, cl: ComboLeg) -> np.ndarray | None:
        v = self.vars.get(cl.leg.ticker)
        if v is None:
            return None
        key = (cl.leg.ticker, cl.side)
        if key not in self._ind_cache:
            self._ind_cache[key] = sim.event(self.sim, v, leg_line(cl.leg), side_yes=(cl.side == "yes"))
        return self._ind_cache[key]

    def anchor_first(self, a: ComboLeg, b: ComboLeg, family: str | None) -> bool:
        if family == "qb_ptd_team_total":
            return a.leg.stat == "pass_td"
        if family == "rec_td_team_total":
            return a.leg.stat == "td"
        if family == "qb_pyds_rec_yds":
            return a.leg.stat == "rec_yds"
        if family == "ml_h1_lead":
            return a.leg.stat == "h1_spread"
        return a.p <= b.p

    def price(self, a: ComboLeg, b: ComboLeg, family: str | None) -> FairResult:
        g = self.game
        first = self.anchor_first(a, b, family)
        A, B = (a, b) if first else (b, a)
        pA, pB = A.p, B.p
        indep = pA * pB
        capped: list[str] = []
        ia, ib = self._indicator(A), self._indicator(B)
        delta, hist_n, hist_cap = 0.0, 0.0, None
        if ia is not None and ib is not None and ia.sum() >= 30:
            cond, _raw, _pb_sim = sim.conditional(ia, ib, pB)
            method = "simulator"
            if family in calibration.FAMILY_A_B:
                implied = g.env.implied(A.leg.team, g) if g.env else 24.0
                # history cells are for YES/YES selections
                if A.side == "yes" and B.side == "yes":
                    a_key = A.leg.k if family in ("qb_ptd_team_total", "rec_td_team_total") else A.leg.line
                    delta, hist_n, hist_cap = calibration.lookup(family, pA, pB, implied, a_key=a_key, b_line=B.leg.line)
                    cond = float(expit(logit(min(max(cond, 1e-4), 1 - 1e-4)) + delta))
        else:
            rel = priors.relation(a.leg.team, b.leg.team, player_key(a.leg) if a.leg.player else None,
                                  player_key(b.leg) if b.leg.player else None, a.leg.stat, b.leg.stat)
            j, _ = priors.prior_joint(a.leg.p_fair or 0.5, b.leg.p_fair or 0.5, a.side, b.side, rel, family,
                                      joint_cap_multiple=float(self.s.get("joint_cap_multiple")))
            cond = j / max(pA, 1e-9)
            method = "prior"
        cap = priors.ANCHOR_CAP.get(family, priors.ANCHOR_CAP[None]) if family else 0.95
        if hist_cap is not None:
            cap = max(cap, min(hist_cap, 0.995))
        if cond > cap and (method == "prior" or family):
            cond = cap
            capped.append(f"conditional capped at {cap:.2f}")
        joint = pA * cond
        jcap = float(self.s.get("joint_cap_multiple")) * indep
        if joint > jcap:
            joint = jcap
            capped.append(f"joint capped at {self.s.get('joint_cap_multiple')}x independence")
        joint = min(joint, pA, pB)
        joint = max(joint, 1e-6)
        n_ext = int(a.leg.extrapolated) + int(b.leg.extrapolated)
        n_dis = int(a.leg.disagreement) + int(b.leg.disagreement)
        u = float(self.s.get("lower_bound_base")) + 0.10 * n_ext + 0.05 * n_dis
        if method == "prior":
            u += 0.10
        fam_ok, fam_note = calibration.family_passes(family)
        if family and (family not in calibration.FAMILY_A_B):
            u += 0.05
        if not fam_ok:
            u += 0.10
        if family is None:
            u += 0.03
        expl = {
            "A": A.wording, "B": B.wording, "pA": round(pA, 4), "pB": round(pB, 4), "cond": round(joint / pA, 4),
            "joint": round(joint, 5), "independent": round(indep, 5), "lift": round(joint / max(indep, 1e-9), 2),
            "method": method, "history_n": int(hist_n), "history_shift": round(delta, 3), "family_check": fam_note,
            "extrapolated_legs": n_ext, "disagreeing_legs": n_dis, "capped": capped,
            "sentence": sentence(A, B, joint / pA, pB),
            "legs": [
                {"wording": cl.wording, "p_market": cl.leg.p_market, "p_model": cl.leg.p_model, "p_fair": cl.leg.p_fair,
                 "extrapolated": cl.leg.extrapolated, "check_by_hand": cl.leg.disagreement}
                for cl in (a, b)
            ],
        }
        return FairResult(joint, u, a.p, b.p, joint / pA, first, indep, method, delta, hist_n, capped, expl)


def sentence(A: ComboLeg, B: ComboLeg, cond: float, pB: float) -> str:
    return (f"When “{A.wording}” hits, “{B.wording}” also hits about {cond:.0%} of the time, "
            f"versus {pB:.0%} on its own.")


def maker_sentence(ratio: float | None, lift: float) -> str:
    if ratio is None:
        return ""
    if lift >= 1.5 and ratio >= 0.8:
        return "Makers price it close to the product of the legs, as if the two were unrelated."
    if lift >= 1.5:
        return "Makers price in some of the link between the legs, but not all of it."
    return "The legs are only loosely linked, so the product of the legs is a fair baseline."
