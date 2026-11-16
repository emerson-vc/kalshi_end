"""t-copula priors (df = 4) anchored on the finds, with tail guards.

Used only where the simulator can't price a leg. Guards (from v1's failure):
* the conditional is capped at the family's strongest anchor;
* the joint is capped at a multiple of independence (default 6x).
"""

from __future__ import annotations

from functools import lru_cache

from scipy import stats

DF = 4

# Anchors from the spec (P(B | A) observed / believed for the proven families).
ANCHOR_CAP = {
    "qb_ptd_team_total": 0.92,
    "rec_td_team_total": 0.92,
    "qb_pyds_rec_yds": 0.92,
    "qb_ptd_rec_td": 0.92,
    "near_lock_team_total": 0.995,
    "same_player_rec_yds": 0.995,
    "qb_tail_team_win": 0.92,
    "rb_ryds_win": 0.92,
    "qb_pyds_opp_total": 0.92,
    "ml_h1_lead": 0.88,
    None: 0.90,
}

# Prior correlation by relation between the legs.
RHO = {
    "same_player_tail_own_team": 0.55,
    "qb_yds_own_receiver": 0.60,
    "qb_td_own_receiver": 0.50,
    "ml_h1_lead": 0.65,
    "same_player": 0.45,
    "same_team": 0.25,
    "game_level": 0.25,
    "opponent": -0.10,
    "separate_game": 0.0,
}


@lru_cache(maxsize=20000)
def t_joint_upper(pa: float, pb: float, rho: float) -> float:
    """P(A and B) for upper-tail events with marginals pa, pb under a bivariate t copula."""
    pa = min(max(pa, 1e-4), 1 - 1e-4)
    pb = min(max(pb, 1e-4), 1 - 1e-4)
    if abs(rho) < 1e-6:
        return pa * pb
    # By symmetry, P(X > q(1-pa), Y > q(1-pb)) = CDF(q(pa), q(pb)).
    xa = stats.t.ppf(pa, DF)
    xb = stats.t.ppf(pb, DF)
    mvt = stats.multivariate_t(loc=[0.0, 0.0], shape=[[1.0, rho], [rho, 1.0]], df=DF)
    j = float(mvt.cdf([xa, xb], maxpts=20000, random_state=7))
    return min(max(j, max(0.0, pa + pb - 1)), min(pa, pb))


def prior_joint(pa: float, pb: float, side_a: str, side_b: str, relation: str, family: str | None, *, joint_cap_multiple: float = 6.0) -> tuple[float, dict]:
    """``pa``/``pb`` are the YES probabilities of the two markets; sides select YES or NO."""
    rho = RHO.get(relation, 0.2)
    j_yy = t_joint_upper(round(pa, 3), round(pb, 3), rho)  # joint of the YES events (rounded for the cache)
    if side_a == "yes" and side_b == "yes":
        j = j_yy
    elif side_a == "yes":
        j = pa - j_yy
    elif side_b == "yes":
        j = pb - j_yy
    else:
        j = 1 - pa - pb + j_yy
    qa = pa if side_a == "yes" else 1 - pa
    qb = pb if side_b == "yes" else 1 - pb
    j = max(j, 1e-6)
    cap_cond = ANCHOR_CAP.get(family, ANCHOR_CAP[None])
    j = min(j, cap_cond * min(qa, qb))  # P(other leg | rarer leg) ≤ the family's strongest anchor
    j = min(j, joint_cap_multiple * qa * qb)
    return j, {"rho": rho, "relation": relation, "cap_cond": cap_cond}


def relation(a_team: str | None, b_team: str | None, a_player: str | None, b_player: str | None, a_stat: str, b_stat: str, same_game: bool = True) -> str:
    if not same_game:
        return "separate_game"
    team_level = {"team_total", "ml", "spread", "h1_spread", "h1_team_total", "h1_winner"}
    if a_player and b_player and a_player == b_player:
        return "same_player"
    if {a_stat, b_stat} == {"ml", "h1_spread"} and a_team == b_team:
        return "ml_h1_lead"
    if (a_stat == "pass_yds" and b_stat in ("rec_yds",)) or (b_stat == "pass_yds" and a_stat in ("rec_yds",)):
        return "qb_yds_own_receiver" if a_team == b_team else "opponent"
    if {a_stat, b_stat} in ({"pass_td", "td"}, {"pass_td", "rec_yds"}) and a_player != b_player:
        return "qb_td_own_receiver" if a_team == b_team else "opponent"
    if a_stat in ("game_total", "h1_total") or b_stat in ("game_total", "h1_total"):
        return "game_level"
    if a_team and b_team and a_team != b_team:
        return "opponent"
    if (a_player and b_stat in team_level) or (b_player and a_stat in team_level):
        return "same_player_tail_own_team" if a_team == b_team else "opponent"
    return "same_team"

