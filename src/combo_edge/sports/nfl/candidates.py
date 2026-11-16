"""Same-game 2-leg candidates: core and trial families (both teams, yes/yes) plus exploration.

Core (shares in config/core.yaml):
* near_lock_team_total — QB 3+/4+/5+ pass TDs, and RB1 / WR1 / TE1 2+ TDs, × own team-total rungs where
                         history says the total hits ≥ 80% of the time the player leg does (incl. low rungs)
* qb_tail_team_win     — QB 3+/4+ pass TDs and 275+/300+/325+ pass yards × own team to win
* same_player_rec_yds  — receptions rung × rec-yards rung of the same WR1 / WR2 / TE1 where history
                         P(yards | receptions) ≥ 85%, plus the next yards rung up
* qb_ptd_team_total    — QB 2+–5+ pass TDs × own team-total rungs not already near-locks
* qb_pyds_rec_yds      — QB pass yards × rec yards of the top 2 receivers (rec-yds median) whose main line
                         is ≥ min_receiver_main_line; yardage legs use only their tail rungs
                         (YES priced inside core_yardage_prob_range)
Trial (each gets its own share of new creations; P&L shown separately):
* qb_ptd_rec_td — QB 3+/4+ pass TDs × WR1 / TE1 2+ TDs
* rb_ryds_win   — RB1 100+ rush yards (main rush line ≥ min_rb_main_rush_line) × own team to win
* qb_pyds_opp_total — QB 300+ pass yards × opponent team total over
* ml_h1_lead    — team ML × same team 1H lead ladder
Pairs forced() rejects (Kalshi refuses them) or in a rejected class are skipped; near-locked pairs are not
forced and are kept. "No" versions and everything else are exploration.
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

CORE_FAMILIES = ("near_lock_team_total", "qb_tail_team_win", "same_player_rec_yds", "qb_ptd_team_total", "qb_pyds_rec_yds")
TRIAL_FAMILIES = ("qb_ptd_rec_td", "rb_ryds_win", "qb_pyds_opp_total", "ml_h1_lead")
FAMILIES = CORE_FAMILIES + TRIAL_FAMILIES
QB_PTD_TT_K = (2, 3, 4, 5)
NEAR_LOCK_QB_K = (3, 4, 5)
NEAR_LOCK_SKILL_ROLES = ("RB1", "WR1", "TE1")
QB_TAIL_TD_K = (3, 4)
QB_TAIL_YDS = (274.5, 299.5, 324.5)
QB_PTD_REC_K = (3, 4)
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


def _team_total_window(settings: Any) -> float | None:
    w = settings.get("team_total_window")
    try:
        w = float(w) if w not in (None, "", "none", "off") else None
    except (TypeError, ValueError):
        w = None
    return w if (w is not None and w > 0) else None


def yardage_range(settings: Any) -> tuple[float, float]:
    v = settings.get("core_yardage_prob_range") or [0.0, 1.0]
    return float(v[0]), float(v[1])


def yes_prob(l: Leg) -> float:
    return l.p_fair if l.p_fair is not None else (l.p_market if l.p_market is not None else 0.5)


def generate(game: Game, fm: Any, settings: Any, *, rejected: set[str] | None = None, rng: random.Random | None = None,
             max_exploration: int = 400, trace: dict | None = None) -> list[Candidate]:
    """Core and trial family pairs for both teams, then sampled exploration pairs.

    ``fm`` needs ``roster`` and, for the history-gated families, ``hist_rate(a_leg, b_leg)`` (FairModel).
    ``trace`` (optional) is {"key": sorted((ticker, side), ...)}; every attempt to add that exact pair is
    recorded in trace["attempts"] as (family, result) with result one of added / forced / rejected_class /
    duplicate."""
    rejected = rejected or set()
    rng = rng or random.Random(hash(game.id) & 0xFFFF)
    r = fm.roster
    window = _team_total_window(settings)
    ylo, yhi = yardage_range(settings)
    tail = lambda l: ylo <= yes_prob(l) <= yhi  # noqa: E731
    min_n = int(settings.get("history_min_n") or 40)
    nl_min = float(settings.get("near_lock_team_total_min") or 0.80)
    sp_min = float(settings.get("same_player_min") or 0.85)
    rec_main = float(settings.get("min_receiver_main_line") or 50.5)
    rush_main = float(settings.get("min_rb_main_rush_line") or 60.5)
    hist_rate = getattr(fm, "hist_rate", None)

    def hist_ok(a: Leg, b: Leg, thr: float) -> bool:
        if hist_rate is None:
            return False
        rate, n = hist_rate(a, b)
        return rate is not None and n >= min_n and rate >= thr

    by: dict[tuple, list[Leg]] = {}
    for l in game.legs:
        if l.protected:
            continue
        by.setdefault((l.stat, l.team), []).append(l)
    out: list[Candidate] = []
    seen: set[tuple] = set()
    tkey = tuple(trace["key"]) if trace else None

    def add(a: Leg, sa: str, b: Leg, sb: str, family: str | None, group: str, exploration: bool) -> None:
        ca, cb = ComboLeg(a, sa), ComboLeg(b, sb)
        key = tuple(sorted([(a.ticker, sa), (b.ticker, sb)]))
        result = "added"
        if key in seen:
            result = "duplicate"
        elif forced(ca, cb):
            result = "forced"
        elif class_key(ca, cb) in rejected:
            result = "rejected_class"
        if tkey is not None and key == tkey:
            trace.setdefault("attempts", []).append((family, result))
        if result != "added":
            return
        seen.add(key)
        out.append(Candidate(game, ca, cb, family, stratum(ca, cb), exploration, group))

    for team in (game.away_code, game.home_code):
        opp = game.home_code if team == game.away_code else game.away_code
        tts = team_total_rungs(game, by, team, window)
        opp_tts = by.get(("team_total", opp), [])
        mls = by.get(("ml", team), [])
        role_of = lambda l: r.hist_role.get(player_key(l))  # noqa: E731
        qb_ptd = [l for l in by.get(("pass_td", team), []) if r.roles.get(player_key(l)) == "QB" and l.k]
        qb_yds = [l for l in by.get(("pass_yds", team), []) if r.roles.get(player_key(l)) == "QB" and l.line]
        tds = [l for l in by.get(("td", team), []) if l.k]
        # near_lock_team_total (first, so the QB rungs it covers aren't repeated as qb_ptd_team_total)
        for l in qb_ptd:
            if l.k in NEAR_LOCK_QB_K:
                for t in tts:
                    if hist_ok(l, t, nl_min):
                        add(l, "yes", t, "yes", "near_lock_team_total", player_key(l), False)
        for l in tds:
            if l.k == 2 and role_of(l) in NEAR_LOCK_SKILL_ROLES:
                for t in tts:
                    if hist_ok(l, t, nl_min):
                        add(l, "yes", t, "yes", "near_lock_team_total", player_key(l), False)
        # qb_tail_team_win
        for ml in mls:
            for l in qb_ptd:
                if l.k in QB_TAIL_TD_K:
                    add(l, "yes", ml, "yes", "qb_tail_team_win", player_key(l), False)
            for l in qb_yds:
                if any(abs(l.line - x) <= 5 for x in QB_TAIL_YDS):
                    add(l, "yes", ml, "yes", "qb_tail_team_win", player_key(l), False)
        # same_player_rec_yds
        for l in by.get(("receptions", team), []):
            if role_of(l) not in ("WR1", "WR2", "TE1") or not l.k:
                continue
            yds = sorted((y for y in by.get(("rec_yds", team), []) if player_key(y) == player_key(l) and y.line), key=lambda y: y.line)
            ok = [y for y in yds if hist_ok(l, y, sp_min)]
            nxt = [y for y in yds if ok and y.line > max(o.line for o in ok)][:1]
            for y in ok + nxt:
                add(l, "yes", y, "yes", "same_player_rec_yds", player_key(l), False)
        # qb_ptd_team_total: the rest of the QB TD × own team-total rungs; "No" versions to exploration
        for l in qb_ptd:
            for t in tts:
                if l.k in QB_PTD_TT_K:
                    add(l, "yes", t, "yes", "qb_ptd_team_total", player_key(l), False)
                elif l.k > max(QB_PTD_TT_K):
                    add(l, "yes", t, "yes", None, player_key(l), True)
                if l.k in (3, 4):
                    add(l, "yes", t, "no", None, player_key(l), True)
        # qb_pyds_rec_yds: top 2 receivers by rec-yds median, main line ≥ min_receiver_main_line
        top2 = [k for k in r.top_yardage.get(team, [])[:2] if (r.rec_median.get(k) or 0) >= rec_main]
        rec_tail = [l for l in by.get(("rec_yds", team), []) if player_key(l) in top2 and l.line and tail(l)]
        for q in (q for q in qb_yds if tail(q)):
            for w in rec_tail:
                add(w, "yes", q, "yes", "qb_pyds_rec_yds", player_key(w), False)
        # trial: qb_ptd_rec_td — QB 3+/4+ × WR1 / TE1 2+ TDs
        for l in qb_ptd:
            if l.k in QB_PTD_REC_K:
                for w in tds:
                    if w.k == 2 and role_of(w) in ("WR1", "TE1"):
                        add(l, "yes", w, "yes", "qb_ptd_rec_td", player_key(w), False)
        # trial: rb_ryds_win — RB1 100+ rush yards × own win
        for l in by.get(("rush_yds", team), []):
            if role_of(l) == "RB1" and l.line and abs(l.line - 99.5) < 1 and (r.rush_median.get(player_key(l)) or 0) >= rush_main:
                for ml in mls:
                    add(l, "yes", ml, "yes", "rb_ryds_win", player_key(l), False)
        # trial: qb_pyds_opp_total — QB 300+ × opponent team total over
        for q in qb_yds:
            if abs(q.line - 299.5) < 1:
                for t in opp_tts:
                    add(q, "yes", t, "yes", "qb_pyds_opp_total", player_key(q), False)
        # trial: ml_h1_lead — team ML × same team 1H lead ladder
        for ml in mls:
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


def team_total_rungs(game: Game, by: dict, team: str, window: float | None) -> list[Leg]:
    legs = by.get(("team_total", team), [])
    main = game.env.main_team_line.get(team) if game.env else None
    if window is None or main is None:
        return legs
    return [l for l in legs if l.line is not None and abs(l.line - main) <= window + 1e-9]


def core_family_for(game: Game, roster: Any, a: ComboLeg, b: ComboLeg, settings: Any, fm: Any = None) -> tuple[str | None, list[str]]:
    """Which family rule this exact pair would match, and if none, why not (for the trace tool)."""
    why: list[str] = []
    for x in (a, b):
        if x.leg.protected:
            why.append(f"“{x.leg.wording}” is a protected leg (exploration only, rarely sampled)")
    if a.side != "yes" or b.side != "yes":
        why.append("a No side: No versions are exploration only")
    if why:
        return None, why
    stats = {x.leg.stat: x for x in (a, b)} if a.leg.stat != b.leg.stat else {}
    pair = frozenset((a.leg.stat, b.leg.stat))
    role = lambda cl: roster.hist_role.get(player_key(cl.leg)) if cl.leg.player else None  # noqa: E731
    min_n = int(settings.get("history_min_n") or 40)
    if pair in (frozenset(("pass_td", "team_total")), frozenset(("td", "team_total"))):
        p = stats.get("pass_td") or stats["td"]
        t = stats["team_total"]
        if t.leg.team != p.leg.team:
            return None, ["the team total is the opponent's: no family pairs a player's TDs with the opponent total"]
        if p.leg.stat == "pass_td" and roster.roles.get(player_key(p.leg)) != "QB":
            return None, [f"{p.leg.player} is not the starting QB (the most-traded passer)"]
        if p.leg.stat == "td" and (role(p) not in NEAR_LOCK_SKILL_ROLES or p.leg.k != 2):
            return None, [f"only RB1 / WR1 / TE1 2+ TDs pair with the own team total ({p.leg.player} is {role(p) or 'none of these'})"]
        if fm is not None and hasattr(fm, "hist_rate"):
            rate, n = fm.hist_rate(p.leg, t.leg)
            if p.leg.stat == "td" and not (rate is not None and n >= min_n and rate >= float(settings.get("near_lock_team_total_min") or 0.8)):
                return None, [f"history P(total | TDs) = {rate if rate is None else round(rate, 2)} (n={n}) is below the near-lock gate"]
        return ("near_lock_team_total" if p.leg.stat == "td" else "qb_ptd_team_total"), []
    if pair in (frozenset(("pass_td", "ml")), frozenset(("pass_yds", "ml"))):
        q = stats.get("pass_td") or stats["pass_yds"]
        if stats["ml"].leg.team != q.leg.team:
            return None, ["the win leg is the opponent's (only own team to win is a family)"]
        return "qb_tail_team_win", []
    if pair == frozenset(("receptions", "rec_yds")):
        if player_key(a.leg) != player_key(b.leg):
            return None, ["receptions and yards of different players: exploration only"]
        if role(a) not in ("WR1", "WR2", "TE1"):
            return None, [f"{a.leg.player} is not the team's WR1 / WR2 / TE1"]
        return None, ["history P(yards | receptions) is below the 85% gate and it isn't the next rung up"]
    if pair == frozenset(("pass_yds", "rec_yds")):
        return None, ["receiver is not one of the top 2 by rec-yds median with a main line ≥ min_receiver_main_line, "
                      "or a yardage rung is outside core_yardage_prob_range"]
    if pair == frozenset(("pass_yds", "team_total")):
        return None, ["only QB 300+ × the opponent total over is a (trial) family"]
    if pair == frozenset(("ml", "h1_spread")):
        return "ml_h1_lead", []
    return None, [f"no family pairs {a.leg.stat} with {b.leg.stat} (exploration only)"]


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
        nl = c.fair.near_lock
        if nl:
            alone = c.leg_payout_a if c.a.leg.ticker == nl["ticker"] else c.leg_payout_b
            nl["leg_alone_payout"] = alone
            c.fair.explanation["near_lock"] = nl


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
