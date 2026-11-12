"""Drive-based NFL game simulator that prices legs jointly.

Per simulated game and team:
* drives ~ shared pace (both teams get about the same number);
* a team-strength latent Z (big days and bad days) drives the per-drive scoring rate;
* on big days field goals turn into touchdowns: the TD share of scores rises with Z;
* each game's pass-TD share is drawn from Beta(mean = team pass share, concentration ≈ 12)
  (v1 used a fixed share, which made P(team over | QB 4+ pass TD) ≈ 0.99 instead of ~0.90);
* QB yards move with pass TDs and Z; receiver yards are a noisy share of QB yards;
* pass TDs go to receivers by TD share, rush TDs to backs.

The score rate is solved so the expected team score equals the market's implied team total.
Legs are priced by quantile matching: each leg's indicator is the simulated variable above the
cutoff whose simulated frequency equals the (blended) market probability. That keeps market
marginals and takes the dependence from the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# ---- constants tuned on nflverse 2019-2026 (team sd, P(4+ pass TD), P(o28.5|4+), P(win|4+)) ----
BASE_DRIVES = 11.5
DRIVE_SD = 0.4
SIGMA_Z = 0.12           # team-strength latent sd (log-odds of scoring per drive)
TEAM_Z_CORR = 0.05       # small positive correlation between the teams' latents
TD_SHARE_BASE = 0.55     # log-odds of TD share of scores at Z = 0 (≈ 63% of scores are TDs)
TD_SHARE_SLOPE = 0.4     # big days turn field goals into TDs
PASS_SHARE_KAPPA = 12.0  # Beta concentration for the per-game pass-TD share
DST_TD_RATE = 0.06       # defensive / special-teams TDs per team-game
QB_YDS_PER_PTD = 30.0
QB_YDS_PER_Z = 40.0
QB_YDS_RESID = 50.0
REC_SHARE_SIGMA = 0.42
RUSH_SIGMA = 0.5
GAME_SCRIPT = 0.40       # pass-share log-odds shift per 14 points trailing (comebacks throw more)


@dataclass
class PlayerSpec:
    key: str             # player id (or name)
    team: str
    role: str            # QB | WR | RB
    yds_median: float | None = None   # pass yds (QB), rec yds (WR/TE), rush yds (RB)
    rec_median: float | None = None   # receiving yards median for RBs
    td_share: float = 0.0             # share of team pass TDs (WR/TE) or rush TDs (RB)
    rec_td_share: float = 0.0         # RB receiving TD share
    receptions_per_yd: float = 1 / 11.5


@dataclass
class TeamSpec:
    code: str
    implied: float
    pass_share: float = 0.62
    players: list[PlayerSpec] = field(default_factory=list)


@dataclass
class SimResult:
    vars: dict[str, np.ndarray]
    n: int

    def get(self, name: str) -> np.ndarray | None:
        return self.vars.get(name)


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()


def expected_points(score_logit: float, drives: float = BASE_DRIVES) -> float:
    z = _GH_X * SIGMA_Z
    p_score = _expit(score_logit + z)
    f_td = _expit(TD_SHARE_BASE + TD_SHARE_SLOPE * z / SIGMA_Z * 0.5)
    pts = p_score * (f_td * 6.96 + (1 - f_td) * 3.0)
    return float(drives * np.sum(_GH_W * pts) + DST_TD_RATE * 6.96)


def expected_off_tds(score_logit: float, drives: float = BASE_DRIVES) -> float:
    z = _GH_X * SIGMA_Z
    p_score = _expit(score_logit + z)
    f_td = _expit(TD_SHARE_BASE + TD_SHARE_SLOPE * z / SIGMA_Z * 0.5)
    return float(drives * np.sum(_GH_W * p_score * f_td))


@lru_cache(maxsize=4096)
def solve_score_logit(implied: float) -> float:
    lo, hi = -6.0, 3.0
    target = max(3.0, min(implied, 50.0))
    for _ in range(60):
        mid = (lo + hi) / 2
        if expected_points(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(home: TeamSpec, away: TeamSpec, n: int = 20000, seed: int = 0) -> SimResult:
    rng = np.random.default_rng(seed)
    D_MAX = 16
    pace = rng.normal(BASE_DRIVES, DRIVE_SD, n)
    d_home = np.clip(np.round(pace + rng.choice([-0.5, 0.5], n)), 6, D_MAX).astype(int)
    d_away = np.clip(np.round(pace + rng.choice([-0.5, 0.5], n)), 6, D_MAX).astype(int)
    zc = rng.standard_normal(n)
    z_home = SIGMA_Z * (math.sqrt(TEAM_Z_CORR) * zc + math.sqrt(1 - TEAM_Z_CORR) * rng.standard_normal(n))
    z_away = SIGMA_Z * (math.sqrt(TEAM_Z_CORR) * zc + math.sqrt(1 - TEAM_Z_CORR) * rng.standard_normal(n))
    out: dict[str, np.ndarray] = {}
    for team, z, d in ((home, z_home, d_home), (away, z_away, d_away)):
        _simulate_scoring(rng, team, z, d, D_MAX, out, n)
    hp, ap = out[f"pts:{home.code}"], out[f"pts:{away.code}"]
    for team, z, margin in ((home, z_home, hp - ap), (away, z_away, ap - hp)):
        _simulate_players(rng, team, z, margin, out, n)
    hh, ah = out[f"h1:{home.code}"], out[f"h1:{away.code}"]
    out["pts:total"] = hp + ap
    out["h1:total"] = hh + ah
    out[f"margin:{home.code}"] = hp - ap
    out[f"margin:{away.code}"] = ap - hp
    out[f"h1margin:{home.code}"] = hh - ah
    out[f"h1margin:{away.code}"] = ah - hh
    return SimResult(out, n)


def _simulate_scoring(rng: np.random.Generator, t: TeamSpec, z: np.ndarray, d: np.ndarray, d_max: int, out: dict, n: int) -> None:
    a = solve_score_logit(t.implied)
    p_score = _expit(a + z)
    f_td = _expit(TD_SHARE_BASE + TD_SHARE_SLOPE * z / SIGMA_Z * 0.5)
    p_td = p_score * f_td
    p_fg = p_score * (1 - f_td)
    u = rng.random((n, d_max))
    live = np.arange(d_max)[None, :] < d[:, None]
    is_td = (u < p_td[:, None]) & live
    is_fg = (u >= p_td[:, None]) & (u < (p_td + p_fg)[:, None]) & live
    xp = rng.random((n, d_max))
    extra = np.where(xp < 0.93, 1, np.where(xp < 0.96, 2, 0))
    drive_pts = is_td * (6 + extra) + is_fg * 3
    half = np.ceil(d / 2).astype(int)
    first_half = np.arange(d_max)[None, :] < half[:, None]
    dst = rng.poisson(DST_TD_RATE, n)
    dst_h1 = rng.binomial(dst, 0.5)
    off_tds = is_td.sum(axis=1)
    pts = drive_pts.sum(axis=1) + dst * 7
    out[f"pts:{t.code}"] = pts.astype(float)
    out[f"h1:{t.code}"] = ((drive_pts * first_half).sum(axis=1) + dst_h1 * 7).astype(float)
    out[f"offtd:{t.code}"] = off_tds.astype(float)


def _simulate_players(rng: np.random.Generator, t: TeamSpec, z: np.ndarray, margin: np.ndarray, out: dict, n: int) -> None:
    a = solve_score_logit(t.implied)
    off_tds = out[f"offtd:{t.code}"].astype(int)
    # per-game pass share ~ Beta(mean, kappa); the mean moves toward passing when trailing
    s = min(max(t.pass_share, 0.2), 0.9)
    s_game = _expit(math.log(s / (1 - s)) - GAME_SCRIPT * margin / 14.0)
    share = rng.beta(PASS_SHARE_KAPPA * s_game, PASS_SHARE_KAPPA * (1 - s_game))
    pass_tds = rng.binomial(off_tds, share)
    rush_tds = off_tds - pass_tds
    out[f"passtd_team:{t.code}"] = pass_tds.astype(float)
    exp_ptd = expected_off_tds(a) * s
    qbs = [p for p in t.players if p.role == "QB"]
    qb_yds = None
    if qbs:
        qb = qbs[0]
        med = qb.yds_median or 225.0
        qb_yds = med + QB_YDS_PER_PTD * (pass_tds - exp_ptd) + QB_YDS_PER_Z * (z / SIGMA_Z) + rng.normal(0, QB_YDS_RESID, n)
        qb_yds = np.maximum(qb_yds, -5.0)
        out[f"pass_td:{qb.key}"] = pass_tds.astype(float)
        out[f"pass_yds:{qb.key}"] = qb_yds
    team_pass_yds = qb_yds if qb_yds is not None else (225.0 + QB_YDS_PER_Z * (z / SIGMA_Z) + rng.normal(0, 60, n))
    ref_pass = (qbs[0].yds_median if qbs and qbs[0].yds_median else 225.0)
    # receivers
    rec_tds_left = pass_tds.copy()
    for p in [p for p in t.players if p.role == "WR"]:
        share_mean = min((p.yds_median or 30.0) / ref_pass, 0.6)
        noise = np.exp(rng.normal(-REC_SHARE_SIGMA**2 / 2, REC_SHARE_SIGMA, n))
        tds = rng.binomial(pass_tds, min(max(p.td_share, 0.0), 0.95))
        tds = np.minimum(tds, rec_tds_left)
        rec_tds_left = rec_tds_left - tds
        yds = np.maximum(team_pass_yds, 0) * share_mean * noise + 12.0 * (tds - p.td_share * exp_ptd)
        yds = np.clip(yds, 0, None)
        out[f"rec_yds:{p.key}"] = yds
        out[f"td:{p.key}"] = tds.astype(float)
        out[f"receptions:{p.key}"] = np.round(yds * p.receptions_per_yd * np.exp(rng.normal(0, 0.22, n)))
    for p in [p for p in t.players if p.role == "RB"]:
        med = p.yds_median or 45.0
        ry = med * np.exp(rng.normal(0, RUSH_SIGMA, n) + 0.25 * (z / SIGMA_Z))
        rtd = rng.binomial(rush_tds, min(max(p.td_share, 0.0), 0.95))
        rec_td = rng.binomial(rec_tds_left, min(max(p.rec_td_share, 0.0), 0.5))
        rec_yds = (p.rec_median or 12.0) * np.exp(rng.normal(-0.2, 0.6, n))
        out[f"rush_yds:{p.key}"] = ry
        out[f"rec_yds:{p.key}"] = rec_yds
        out[f"rush_rec_yds:{p.key}"] = ry + rec_yds
        out[f"td:{p.key}"] = (rtd + rec_td).astype(float)
        out[f"receptions:{p.key}"] = np.round(rec_yds * p.receptions_per_yd * np.exp(rng.normal(0, 0.25, n)))
    for qb in qbs[:1]:
        ry = 12.0 * np.exp(rng.normal(0, 0.8, n))
        out[f"rush_yds:{qb.key}"] = ry


DISCRETE_PREFIXES = ("pts:", "h1:", "margin:", "h1margin:", "pass_td:", "td:", "receptions:", "offtd:", "passtd_team:")


def indicator(sim: SimResult, var: str, p: float, rng: np.random.Generator, side_yes: bool = True) -> np.ndarray | None:
    """Boolean array for 'var is in its top-p tail' (quantile matching with random tie-breaking)."""
    x = sim.get(var)
    if x is None:
        return None
    jitter = rng.random(sim.n) if var.startswith(DISCRETE_PREFIXES) else rng.random(sim.n) * 1e-6
    xj = x + jitter
    p = min(max(p, 1e-4), 1 - 1e-4)
    cut = np.quantile(xj, 1 - p)
    ind = xj > cut
    return ind if side_yes else ~ind


def event(sim: SimResult, var: str, line: float, side_yes: bool = True) -> np.ndarray | None:
    """Boolean array for the leg's own threshold event (var > line), or its complement."""
    x = sim.get(var)
    if x is None:
        return None
    ind = x > line
    return ind if side_yes else ~ind


def conditional(ia: np.ndarray, ib: np.ndarray, pB_target: float | None = None) -> tuple[float, float, float]:
    """(P_sim(B|A) adjusted to the target B marginal, raw P_sim(B|A), P_sim(B)).

    The adjustment moves the conditional in log-odds by the gap between the target and simulated
    marginal of B, so independent legs keep P(B|A) = P(B)."""
    n_a = int(ia.sum())
    pb_sim = float(ib.mean())
    raw = float((ia & ib).sum() / n_a) if n_a else pb_sim
    if pB_target is None:
        return raw, raw, pb_sim
    lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
    adj = 1 / (1 + math.exp(-(lg(raw) + lg(pB_target) - lg(pb_sim))))
    return adj, raw, pb_sim


def raw_prob(sim: SimResult, var: str, line: float) -> float | None:
    """Simulator's own probability that var > line (no market matching)."""
    x = sim.get(var)
    if x is None:
        return None
    return float(np.mean(x > line))
