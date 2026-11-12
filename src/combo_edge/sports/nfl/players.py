"""Turn a game's legs into simulator specs: roles, usage and medians, all read from Kalshi ladders
(nflverse positions and team pass shares fill gaps)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from combo_edge.sports.base import Game, Leg
from combo_edge.sports.nfl.ladders import median_line
from combo_edge.sports.nfl.nflverse import norm_name
from combo_edge.sports.nfl.simulator import PlayerSpec, TeamSpec, expected_off_tds, solve_score_logit


@dataclass
class Roster:
    home: TeamSpec
    away: TeamSpec
    roles: dict[str, str] = field(default_factory=dict)       # player key → QB | WR | RB
    sim_key: dict[str, str] = field(default_factory=dict)     # player key → key used in the simulator
    top_receivers: dict[str, list[str]] = field(default_factory=dict)  # team → [player keys] by anytime-TD price
    top_yardage: dict[str, list[str]] = field(default_factory=dict)    # team → [player keys] by rec-yds median
    names: dict[str, str] = field(default_factory=dict)


def player_key(l: Leg) -> str:
    return l.player_id or (l.player or "")


def _td_prob(legs: list[Leg]) -> float | None:
    for l in sorted(legs, key=lambda x: x.line or 0):
        if l.stat == "td" and (l.k == 1) and not l.protected and l.p_market is not None:
            return l.p_market
    return None


def build_roster(g: Game, positions: dict[str, str] | None = None, pass_shares: dict[str, float] | None = None) -> Roster:
    positions = positions or {}
    pass_shares = pass_shares or {}
    by_player: dict[str, list[Leg]] = defaultdict(list)
    for l in g.legs:
        if l.player and l.team and "D/ST" not in l.player:
            by_player[player_key(l)].append(l)
    teams: dict[str, TeamSpec] = {}
    roster = Roster(home=None, away=None)  # type: ignore[arg-type]
    for code in (g.home_code, g.away_code):
        implied = g.env.implied(code, g) if g.env else 22.0
        tspec = TeamSpec(code=code, implied=implied, pass_share=pass_shares.get(code, 0.62))
        team_players = {k: v for k, v in by_player.items() if v[0].team == code}
        # QB: has passing ladders; starter = most liquid
        qbs = [(k, v) for k, v in team_players.items() if any(l.stat in ("pass_td", "pass_yds") for l in v)]
        qbs.sort(key=lambda kv: -sum((l.volume or 0) for l in kv[1]))
        exp_off = expected_off_tds(solve_score_logit(implied))
        if qbs:
            k, legs = qbs[0]
            yds = median_line([l for l in legs if l.stat == "pass_yds" and not l.protected])
            ptd = sorted([l for l in legs if l.stat == "pass_td" and not l.protected], key=lambda l: l.k or 0)
            if ptd:
                e_ptd = sum(l.p_market or 0 for l in ptd)
                last = ptd[-1].p_market or 0
                e_ptd += last * last / max(1e-3, 1 - last)  # geometric tail beyond the last rung
                tspec.pass_share = min(max(e_ptd / max(exp_off, 0.5), 0.35), 0.85)
            tspec.players.append(PlayerSpec(k, code, "QB", yds_median=yds or 220.0))
            roster.roles[k] = "QB"
            roster.sim_key[k] = k
            roster.names[k] = legs[0].player or k
            for k2, _ in qbs[1:]:
                roster.roles[k2] = "QB2"
        exp_pass = exp_off * tspec.pass_share
        exp_rush = exp_off * (1 - tspec.pass_share)
        wr, rb = [], []
        for k, legs in team_players.items():
            if roster.roles.get(k, "").startswith("QB"):
                continue
            rec_med = median_line([l for l in legs if l.stat == "rec_yds" and not l.protected])
            rush_med = median_line([l for l in legs if l.stat == "rush_yds" and not l.protected])
            pos = positions.get(norm_name(legs[0].player or ""))
            is_rb = pos == "RB" or (pos is None and rush_med is not None and rush_med >= 20 and (rec_med is None or rush_med > rec_med))
            if pos in ("WR", "TE"):
                is_rb = False
            p_td = _td_prob(legs)
            lam = -math.log(max(1e-6, 1 - p_td)) if p_td is not None else None
            roster.names[k] = legs[0].player or k
            if is_rb:
                rb.append((k, rush_med, rec_med, lam))
            elif rec_med is not None or lam is not None:
                wr.append((k, rec_med, lam))
        wr_shares = [(lam or 0.05 * exp_pass) / max(exp_pass, 0.3) for _, _, lam in wr]
        scale = min(1.0, 0.92 / sum(wr_shares)) if wr_shares and sum(wr_shares) > 0.92 else 1.0
        for (k, med, lam), sh in zip(wr, wr_shares):
            tspec.players.append(PlayerSpec(k, code, "WR", yds_median=med or 25.0, td_share=sh * scale))
            roster.roles[k] = "WR"
            roster.sim_key[k] = k
        for k, rush_med, rec_med, lam in rb:
            lam = lam or 0.25
            tspec.players.append(PlayerSpec(k, code, "RB", yds_median=rush_med or 40.0, rec_median=rec_med or 12.0,
                                            td_share=min(0.9, 0.8 * lam / max(exp_rush, 0.3)), rec_td_share=min(0.3, 0.2 * lam / max(exp_pass, 0.3))))
            roster.roles[k] = "RB"
            roster.sim_key[k] = k
        roster.top_receivers[code] = [k for k, _, lam in sorted(wr, key=lambda x: -(x[2] or 0))][:2]
        roster.top_yardage[code] = [k for k, med, _ in sorted(wr, key=lambda x: -(x[1] or 0))][:2]
        teams[code] = tspec
    roster.home = teams[g.home_code]
    roster.away = teams[g.away_code]
    return roster
