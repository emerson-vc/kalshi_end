"""Fair value of a same-game pair.

1. Leg probabilities come from the market: the de-vigged mid ((bid + ask) / 2, normalised over yes/no).
   A "thin" leg (spread > 5¢, spread > 40% of the mid, or no ask) is blended 50/50 with the player's
   history base rate for that rung, adds +0.10 to the lower-bound uncertainty and shows "thin leg".
2. Joint: P(A) × P(B | A). The conditional comes from the nflverse history table (sports/nfl/history.py)
   whenever the conditioning event has n ≥ history_min_n (40) games, moved to the market's B price in
   log-odds; team win uses the history A effect on top of the market's win probability (spread-aware).
   A pair is near-locked when history says one leg hits ≥ 93% of the time the other does: the implying
   leg becomes A and fair = P(A) × P(B|A) with the raw history rate. Only when history is thin does the
   simulator price the pair, capped at the nearest history rate + 0.05.
   Simulator path (legacy description): Each player leg is first recalibrated so the simulator's marginal equals
   the probability the pair is priced with (sim.calibrated_event: the player's distribution shifted
   until P_sim(leg) = P(leg)); team legs use their own threshold, moved to the B marginal in
   log-odds. The conditional is measured in that recalibrated simulation — never on the model's own
   extreme tail when the model and market disagree — then shifted toward nflverse history for the
   family. The t-copula prior is used only when the simulator can't price a leg.
3. Tail guards: the lift P(B|A) / P(B) is capped by nflverse history for that stat pair, rung and
   receiver class (WR1 / WR2 / WR3+ / TE): the empirical lift's upper bound plus a margin, never the
   simulator's own lift when that is higher. Then the conditional is capped at the family's strongest
   anchor (or the history-supported level where history shows more), and the joint at
   ``joint_cap_multiple`` × independence.
4. ``fair_lower`` widens with payout (+10% of fair per doubling above 20x by default). A player leg whose
   raw model probability is more than 2x away from the market's (rarer side) adds +0.30 to the
   uncertainty and blocks auto-buy ("model disagrees with market").
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from combo_edge.sports.base import ComboLeg, Game, Leg
from combo_edge.sports.nfl import calibration, history, priors
from combo_edge.sports.nfl import simulator as sim
from combo_edge.sports.nfl.ladders import expit, logit
from combo_edge.sports.nfl.players import Roster, build_roster, player_key

FAMILY_LABELS = {
    "near_lock_team_total": "Near-lock: player TDs + own team total",
    "qb_tail_team_win": "QB big game + own team wins",
    "same_player_rec_yds": "Receptions + receiving yards (same player)",
    "qb_ptd_team_total": "QB pass TDs + own team total",
    "qb_pyds_rec_yds": "QB pass yards + top-2 receiver yards",
    "qb_ptd_rec_td": "Trial: QB pass TDs + WR1/TE1 2+ TDs",
    "rb_ryds_win": "Trial: RB1 100+ rush yards + team wins",
    "qb_pyds_opp_total": "Trial: QB 300+ yards + opponent total over",
    "ml_h1_lead": "Trial: team win + same team 1H lead",
    "rec_td_team_total": "Top receiver TDs + own team total (retired)",
}
TRIAL_FAMILIES = ("qb_ptd_rec_td", "rb_ryds_win", "qb_pyds_opp_total", "ml_h1_lead")


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
    model_disagrees: list[str] = field(default_factory=list)  # legs whose model and market are > 2x apart
    source: str = ""                                          # "history n=137" | "simulator" | "prior"
    near_lock: dict | None = None                             # {"rate", "n", "ticker", "A", "B"} when near-locked
    thin_legs: list[str] = field(default_factory=list)

    @property
    def fair_payout(self) -> float:
        return 1.0 / max(self.fair_joint, 1e-9)

    @property
    def joint_pA(self) -> float:
        """P of the leg the conditional is taken on (A)."""
        return self.fair_joint / max(self.cond, 1e-9)

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


THIN_U = 0.10
HISTORY_CAP_MARGIN = 0.05
DISAGREE_RATIO = 2.0
DISAGREE_U = 0.30
NEAR_CERTAIN = 0.90  # history cells where one leg nearly implies the other: the conditional carries over, not the lift


def model_market_ratio(leg: Leg) -> float | None:
    """How far the simulator's raw probability is from the market's, as a ratio of the rarer-side
    probabilities (≥ 1), or None if either is missing."""
    if leg.p_model is None or leg.p_market is None:
        return None
    a = min(leg.p_model, 1 - leg.p_model)
    b = min(leg.p_market, 1 - leg.p_market)
    return max(a, b) / max(min(a, b), 1e-4)


def market_mid(leg: Leg, abs_thr: float = 0.05, rel_thr: float = 0.40) -> tuple[float | None, bool]:
    """(de-vigged mid YES probability or None, thin?). Thin: no ask, spread > abs_thr, or spread > rel_thr × mid."""
    ya = leg.yes_ask if (leg.yes_ask is not None and 0 < leg.yes_ask < 1) else None
    if ya is None:
        return None, True
    yb = leg.yes_bid if (leg.yes_bid is not None and 0 < leg.yes_bid < 1) else 0.0
    if yb > ya:
        yb, ya = ya, yb
    mid = (yb + ya) / 2
    nb = leg.no_bid if (leg.no_bid is not None and 0 < leg.no_bid < 1) else None
    na = leg.no_ask if (leg.no_ask is not None and 0 < leg.no_ask < 1) else None
    no_mid = (nb + na) / 2 if (nb is not None and na is not None) else 1 - mid
    p = mid / (mid + no_mid) if (mid + no_mid) > 0 else mid
    spread = ya - yb
    thin = yb <= 0 or spread > abs_thr + 1e-9 or spread > rel_thr * max(mid, 1e-6) + 1e-9
    return min(max(p, 1e-4), 1 - 1e-4), thin


def market_weight(leg: Leg) -> float:
    if leg.extrapolated or not leg.two_sided:
        return 0.35
    if (leg.spread or 1) <= 0.03 and (leg.volume or 0) >= 200:
        return 0.9
    return 0.7


class FairModel:
    def __init__(self, game: Game, settings: Any, *, positions: dict | None = None, pass_shares: dict | None = None, n_paths: int | None = None,
                 legacy: bool = False):
        """``legacy=True`` prices the old way (conditional at the model's own thresholds, no lift caps, no
        disagreement block) — only for before/after comparisons."""
        self.game = game
        self.s = settings
        self.legacy = legacy
        self.roster = build_roster(game, positions, pass_shares)
        n = int(n_paths or settings.get("sim_paths") or 20000)
        seed = int(hashlib.sha1(game.id.encode()).hexdigest()[:8], 16)
        self.sim = sim.simulate(self.roster.home, self.roster.away, n=n, seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self._ind_cache: dict[tuple, np.ndarray] = {}
        self.vars: dict[str, str | None] = {}
        self.hist = None if legacy else history.load()
        abs_thr = float(settings.get("thin_spread_abs") or 0.05)
        rel_thr = float(settings.get("thin_spread_rel") or 0.40)
        hw = float(settings.get("thin_history_weight") if settings.get("thin_history_weight") is not None else 0.5)
        for l in game.legs:
            v = leg_var(l, self.roster)
            self.vars[l.ticker] = v
            pm = sim.raw_prob(self.sim, v, leg_line(l)) if v else None
            l.p_model = pm
            pk = l.p_market if l.p_market is not None else 0.5
            if not legacy:
                mid, thin = market_mid(l, abs_thr, rel_thr)
                p = mid if mid is not None else pk  # no ask: the ladder fit from the neighbouring rungs
                l.thin = bool(thin)
                if thin and self.hist is not None and l.player:
                    role = self.roster.hist_role.get(player_key(l))
                    rk = self.hist.event_key(role, l.stat, l.line) if role and l.stat in history.ROLE_STATS.get(role, ()) else None
                    base = self.hist.player_base(l.player, l.stat, l.line, rk) if rk else None
                    if base is not None:
                        p = (1 - hw) * p + hw * base
                l.p_fair = float(min(max(p, 1e-4), 1 - 1e-4))
                l.disagreement = pm is not None and abs(logit(pk) - logit(min(max(pm, 0.002), 0.998))) > 0.9
                continue
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
            if not self.legacy and sim.is_player_var(v):
                p_yes = cl.leg.p_fair if cl.leg.p_fair is not None else (cl.leg.p_market if cl.leg.p_market is not None else 0.5)
                self._ind_cache[key] = sim.calibrated_event(self.sim, v, leg_line(cl.leg), p_yes, side_yes=(cl.side == "yes"))
            else:
                self._ind_cache[key] = sim.event(self.sim, v, leg_line(cl.leg), side_yes=(cl.side == "yes"))
        return self._ind_cache[key]

    def receiver_class(self, leg: Leg) -> str | None:
        return self.roster.rec_class.get(player_key(leg)) if leg.player else None

    def lift_cap(self, A: ComboLeg, B: ComboLeg) -> tuple[float | None, dict | None, str | None]:
        """History lift cap for a same-team YES/YES pair (see calibration.lift_cap)."""
        if A.side != "yes" or B.side != "yes" or not (A.leg.team and A.leg.team == B.leg.team):
            return None, None, None
        rec = next((x.leg for x in (A, B) if x.leg.stat in calibration.RECEIVER_STATS and x.leg.player), None)
        if rec is not None and self.roster.roles.get(player_key(rec), "").startswith("QB"):
            return None, None, None
        cls = self.receiver_class(rec) if rec is not None else None
        cap, cell = calibration.lift_cap(A.leg.stat, A.leg.line, B.leg.stat, B.leg.line, cls or "ALL")
        if cell is not None and max(cell["joint"] / cell["n_a"], cell["joint"] / cell["n_b"]) >= NEAR_CERTAIN:
            return None, cell, cls
        return cap, cell, cls

    def anchor_first(self, a: ComboLeg, b: ComboLeg, family: str | None) -> bool:
        if family == "qb_ptd_team_total":
            return a.leg.stat == "pass_td"
        if family == "rec_td_team_total":
            return a.leg.stat == "td"
        if family == "qb_pyds_rec_yds":
            return a.leg.stat == "rec_yds"
        if family in ("qb_ptd_rec_td", "qb_tail_team_win", "qb_pyds_opp_total"):
            return a.leg.stat in ("pass_td", "pass_yds")
        if family == "near_lock_team_total":
            return a.leg.stat in ("pass_td", "td")
        if family == "same_player_rec_yds":
            return a.leg.stat == "receptions"
        if family == "rb_ryds_win":
            return a.leg.stat == "rush_yds"
        if family == "ml_h1_lead":
            return a.leg.stat == "h1_spread"
        return a.p <= b.p

    # ------------------------------------------------------------------ history
    def hist_key(self, cl: ComboLeg, other: ComboLeg) -> str | None:
        """History event key for a YES leg, relative to the other leg's team (own / opponent)."""
        if self.hist is None:
            return None
        l = cl.leg
        if l.player:
            role = self.roster.hist_role.get(player_key(l))
            if role is None or l.stat not in history.ROLE_STATS.get(role, ()):
                return None
            return self.hist.event_key(role, l.stat, l.line)
        ref = other.leg.team if other.leg.player else None  # team legs are only keyed against a player leg
        if not (l.team and ref):
            return None
        if l.stat == "team_total":
            return self.hist.event_key("TEAM" if l.team == ref else "OPP", "pts", l.line)
        if l.stat == "ml":
            return "TEAM:win:0" if l.team == ref else "OPP:win:0"
        return None

    def hist_rate(self, a_leg: Leg, b_leg: Leg) -> tuple[float | None, int]:
        """History P(B yes | A yes) (shrunk, not market-adjusted) and n_A — used to gate the families."""
        A, B = ComboLeg(a_leg, "yes"), ComboLeg(b_leg, "yes")
        ka, kb = self.hist_key(A, B), self.hist_key(B, A)
        if not ka or not kb or "OPP:win" in (ka, kb):
            return None, 0
        c = self.hist.conditional(ka, kb)
        return (c["rate"], int(c["n"])) if c else (None, 0)

    def _hist(self, A: ComboLeg, B: ComboLeg, *, adjust: bool = True) -> dict | None:
        """History P(B | A) for A = YES: shrunk rate, the market-adjusted rate, n_A."""
        if A.side != "yes":
            return None
        ka, kb = self.hist_key(A, ComboLeg(B.leg, "yes")), self.hist_key(ComboLeg(B.leg, "yes"), A)
        if not ka or not kb or ka == "OPP:win:0":
            return None
        opp_win = kb == "OPP:win:0"
        c = self.hist.conditional(ka, "TEAM:win:0" if opp_win else kb)
        if c is None:
            return None
        rate = c["rate"]
        p_yes_b = B.leg.p_fair if B.leg.p_fair is not None else 0.5
        if kb.endswith(":win:0"):
            own_win_mkt = (1 - p_yes_b) if opp_win else p_yes_b
            w = self.hist.win_conditional(ka, own_win_mkt) if adjust else None
            mk = w["rate"] if w else rate
            if opp_win:
                rate, mk = 1 - rate, 1 - mk
        else:
            mk = history.adjust_to_market(rate, c["pB"], p_yes_b) if adjust else rate
        if B.side == "no":
            rate, mk = 1 - rate, 1 - mk
        return {"rate": rate, "market": mk, "n": int(c["n"]), "ka": ka, "kb": kb, "raw": c["raw"]}

    def price(self, a: ComboLeg, b: ComboLeg, family: str | None) -> FairResult:
        if self.legacy or self.hist is None:
            return self._price_sim(a, b, family)
        min_n = int(self.s.get("history_min_n") or 40)
        thr = float(self.s.get("near_lock_threshold") or 0.93)
        first = self.anchor_first(a, b, family)
        A, B = (a, b) if first else (b, a)
        h1, h2 = self._hist(A, B), self._hist(B, A)
        near = None
        if A.side == "yes" and B.side == "yes":
            opts = [(h, swap) for h, swap in ((h1, False), (h2, True)) if h and h["n"] >= min_n and h["rate"] >= thr]
            if opts:
                h, swap = max(opts, key=lambda o: o[0]["rate"])
                if swap:
                    A, B, first = B, A, not first
                near = {"rate": h["rate"], "n": h["n"], "ticker": A.leg.ticker, "A": A.wording, "B": B.wording}
                h1 = h
        if not near:
            if not (h1 and h1["n"] >= min_n) and h2 and h2["n"] >= min_n:
                A, B, first, h1 = B, A, not first, h2
        if h1 and h1["n"] >= min_n:
            return self._price_history(a, b, A, B, first, h1, near, family)
        res = self._price_sim(a, b, family)
        # simulator only where history is thin: cap at the nearest history rate + 0.05
        hs = self._hist(A, B) if (A.leg.ticker, A.side) == (res.explanation.get("A_ticker"), res.explanation.get("A_side")) else None
        hs = hs or (self._hist(B, A) if (B.leg.ticker, B.side) == (res.explanation.get("A_ticker"), res.explanation.get("A_side")) else None)
        pa_used = res.joint_pA
        if hs is not None and res.cond > hs["market"] + HISTORY_CAP_MARGIN:
            cond = hs["market"] + HISTORY_CAP_MARGIN
            res.capped.append(f"simulator capped at history {hs['market']:.0%} + 5% (n={hs['n']})")
            res.fair_joint = max(min(pa_used * cond, res.p_a, res.p_b), 1e-6)
            res.cond = res.fair_joint / pa_used
            res.explanation.update(cond=round(res.cond, 4), joint=round(res.fair_joint, 5), capped=res.capped,
                                   lift=round(res.fair_joint / max(res.independent, 1e-9), 2))
        thin = [cl.wording for cl in (a, b) if cl.leg.thin]
        res.u_base += THIN_U * sum(1 for cl in (a, b) if cl.leg.thin and not cl.leg.extrapolated)
        res.thin_legs = thin
        res.source = "simulator" if res.method == "simulator" else res.method
        res.explanation.update(source=res.source, thin_legs=thin, near_lock=None,
                               history_note=(f"history thin (n={hs['n']})" if hs else "no history for this pair"))
        return res

    def _price_history(self, a: ComboLeg, b: ComboLeg, A: ComboLeg, B: ComboLeg, first: bool, h: dict, near: dict | None,
                       family: str | None) -> FairResult:
        pA, pB = A.p, B.p
        indep = pA * pB
        cond = h["rate"] if near else h["market"]
        joint = max(min(pA * cond, pA, pB), 1e-6)
        n_thin = sum(1 for cl in (a, b) if cl.leg.thin or cl.leg.extrapolated)
        n_dis = int(a.leg.disagreement) + int(b.leg.disagreement)
        u = float(self.s.get("lower_bound_base")) + THIN_U * n_thin + 0.05 * n_dis + (0.03 if h["n"] < 100 else 0.0)
        disagree = self._disagreements(a, b)
        if disagree:
            u += DISAGREE_U
        if family is None:
            u += 0.03
        thin = [cl.wording for cl in (a, b) if cl.leg.thin]
        source = f"history n={h['n']}"
        expl = {
            "A": A.wording, "B": B.wording, "A_ticker": A.leg.ticker, "A_side": A.side, "pA": round(pA, 4), "pB": round(pB, 4),
            "cond": round(joint / pA, 4), "joint": round(joint, 5), "independent": round(indep, 5), "lift": round(joint / max(indep, 1e-9), 2),
            "method": "history", "source": source, "history_n": h["n"], "history_rate": round(h["rate"], 4), "history_raw": round(h["raw"], 4),
            "history_keys": [h["ka"], h["kb"]], "history_shift": 0.0, "family_check": "history table", "near_lock": near,
            "thin_legs": thin, "extrapolated_legs": int(a.leg.extrapolated) + int(b.leg.extrapolated), "disagreeing_legs": n_dis,
            "capped": [], "model_disagrees": disagree, "sentence": sentence(A, B, joint / pA, pB),
            "legs": [
                {"wording": cl.wording, "p_market": cl.leg.p_market, "p_model": cl.leg.p_model, "p_fair": cl.leg.p_fair,
                 "extrapolated": cl.leg.extrapolated, "thin": cl.leg.thin, "check_by_hand": cl.leg.disagreement}
                for cl in (a, b)
            ],
        }
        res = FairResult(joint, u, a.p, b.p, joint / pA, first, indep, "history", 0.0, float(h["n"]), [], expl, disagree,
                         source=source, near_lock=near, thin_legs=thin)
        return res

    def _disagreements(self, a: ComboLeg, b: ComboLeg) -> list[str]:
        out = []
        for cl in (a, b):
            r = model_market_ratio(cl.leg) if sim.is_player_var(self.vars.get(cl.leg.ticker)) else None
            if r is not None and r > DISAGREE_RATIO:
                out.append(f"{cl.leg.wording}: model {cl.leg.p_model:.0%} vs market {cl.leg.p_market:.0%}")
        return out

    # ------------------------------------------------------------------ simulator / prior
    def _price_sim(self, a: ComboLeg, b: ComboLeg, family: str | None) -> FairResult:
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
                    delta, hist_n, hist_cap = calibration.lookup(family, pA, pB, implied, a_key=a_key, b_line=B.leg.line,
                                                                 calibrated=not self.legacy)
                    cond = float(expit(logit(min(max(cond, 1e-4), 1 - 1e-4)) + delta))
            if not self.legacy:
                lcap, lcell, lcls = self.lift_cap(A, B)
                if lcap is not None and cond > lcap * pB:
                    cond = lcap * pB
                    capped.append(f"lift capped at {lcap:.2f}x by history ({lcls or 'all receivers'}, rungs "
                                  f"{lcell['a']:g} / {lcell['b']:g}: lift {lcell['lift']:.2f}, {lcell['joint']} joint games)")
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
        disagree: list[str] = []
        if not self.legacy:
            disagree = self._disagreements(a, b)
            if disagree:
                u += DISAGREE_U
        fam_ok, fam_note = calibration.family_passes(family)
        if family and (family not in calibration.FAMILY_A_B):
            u += 0.05
        if not fam_ok:
            u += 0.10
        if family is None:
            u += 0.03
        expl = {
            "A": A.wording, "B": B.wording, "A_ticker": A.leg.ticker, "A_side": A.side, "pA": round(pA, 4), "pB": round(pB, 4), "cond": round(joint / pA, 4),
            "joint": round(joint, 5), "independent": round(indep, 5), "lift": round(joint / max(indep, 1e-9), 2),
            "method": method, "history_n": int(hist_n), "history_shift": round(delta, 3), "family_check": fam_note,
            "extrapolated_legs": n_ext, "disagreeing_legs": n_dis, "capped": capped, "model_disagrees": disagree,
            "sentence": sentence(A, B, joint / pA, pB),
            "legs": [
                {"wording": cl.wording, "p_market": cl.leg.p_market, "p_model": cl.leg.p_model, "p_fair": cl.leg.p_fair,
                 "extrapolated": cl.leg.extrapolated, "check_by_hand": cl.leg.disagreement}
                for cl in (a, b)
            ],
        }
        return FairResult(joint, u, a.p, b.p, joint / pA, first, indep, method, delta, hist_n, capped, expl, disagree,
                          source=method)


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
