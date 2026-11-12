"""Ladder-consistent leg probabilities and the game environment.

* Two-sided books give a mid (a binary mid is already de-vigged); one-sided books give bounds only.
* Probabilities are made monotone in the threshold (weighted isotonic fit in log-odds).
* Gaps are interpolated and tails extrapolated linearly in log-odds; such legs are flagged
  ``extrapolated`` so their uncertainty widens downstream.
* Mutually exclusive outcomes (ML, 1H winner) are normalised to sum to one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from combo_edge.sports.base import Game, Leg

# log-odds slope per unit of line, used when a ladder has only one usable point
DEFAULT_SLOPE = {
    "pass_td": -1.35, "td": -1.6, "pass_yds": -0.022, "rec_yds": -0.04, "rush_yds": -0.045, "receptions": -0.6,
    "rush_rec_yds": -0.035, "team_total": -0.12, "game_total": -0.075, "spread": -0.11, "h1_spread": -0.14,
    "h1_team_total": -0.17, "h1_total": -0.11, "pass_att": -0.12, "pass_comp": -0.14, "rush_att": -0.2,
}
EPS = 0.002


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _isotonic_decreasing(xs: list[float], ws: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators for a non-increasing fit."""
    blocks = [[x, w, 1] for x, w in zip(xs, ws)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0] - 1e-12:
            a, b = blocks[i], blocks[i + 1]
            w = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / w, w, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float] = []
    for v, _, n in blocks:
        out.extend([v] * n)
    return out


def _weight(l: Leg) -> float:
    spread = l.spread if l.spread is not None else 0.2
    vol = l.volume or 0.0
    return (1.0 / (spread + 0.01)) * (1.0 + math.log1p(vol) / 5.0)


def fit_ladder(legs: list[Leg]) -> None:
    """Set ``p_market`` and ``extrapolated`` on every leg of one threshold ladder ("over" markets)."""
    legs = sorted(legs, key=lambda l: l.line if l.line is not None else 0.0)
    stat = legs[0].stat
    pts = [(l.line, logit(l.mid), _weight(l)) for l in legs if l.two_sided and l.line is not None]  # type: ignore[arg-type]
    if pts:
        xs = [p[0] for p in pts]
        fitted = _isotonic_decreasing([p[1] for p in pts], [p[2] for p in pts])
        # collapse duplicate lines
        line_vals: dict[float, list[float]] = defaultdict(list)
        for x, y in zip(xs, fitted):
            line_vals[x].append(y)
        knots = sorted((x, sum(v) / len(v)) for x, v in line_vals.items())
    else:
        knots = []
    slope = DEFAULT_SLOPE.get(stat, -0.1)
    if len(knots) >= 2:
        lo_slope = _clamp_slope((knots[1][1] - knots[0][1]) / (knots[1][0] - knots[0][0]), slope)
        hi_slope = _clamp_slope((knots[-1][1] - knots[-2][1]) / (knots[-1][0] - knots[-2][0]), slope)
    else:
        lo_slope = hi_slope = slope
    for l in legs:
        x = l.line if l.line is not None else 0.0
        lo_b = l.yes_bid if (l.yes_bid and l.yes_bid > 0) else 0.0
        hi_b = l.yes_ask if (l.yes_ask and 0 < l.yes_ask < 1) else 1.0
        if hi_b < lo_b:
            lo_b, hi_b = hi_b, lo_b
        if not knots:
            if lo_b > 0 and hi_b < 1:
                p, ext = (lo_b + hi_b) / 2, True
            elif hi_b < 1:
                p, ext = hi_b * 0.75, True
            elif lo_b > 0:
                p, ext = lo_b + (1 - lo_b) * 0.25, True
            else:
                p, ext = 0.5, True
        else:
            y, ext = _interp(knots, x, lo_slope, hi_slope)
            p = expit(y)
            if not l.two_sided:
                ext = True
            p = min(max(p, lo_b), hi_b) if hi_b > lo_b else p
        l.p_market = min(max(p, EPS), 1 - EPS)
        l.extrapolated = ext
    # final monotonic guard (non-increasing in line)
    run = 1.0
    for l in legs:
        run = min(run, l.p_market or run)
        l.p_market = run


def _clamp_slope(s: float, default: float) -> float:
    if not math.isfinite(s) or s >= 0:
        return default
    return max(min(s, default * 0.3), default * 3.0)


def _interp(knots: list[tuple[float, float]], x: float, lo_slope: float, hi_slope: float) -> tuple[float, bool]:
    if x <= knots[0][0]:
        return knots[0][1] + lo_slope * (x - knots[0][0]), x < knots[0][0]
    if x >= knots[-1][0]:
        return knots[-1][1] + hi_slope * (x - knots[-1][0]), x > knots[-1][0]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0, False
            t = (x - x0) / (x1 - x0)
            exact = x in (x0, x1)
            return y0 + t * (y1 - y0), not exact
    return knots[-1][1], True


def _normalize_exclusive(legs: list[Leg]) -> None:
    raw = []
    for l in legs:
        if l.two_sided:
            raw.append(l.mid)
        elif l.yes_ask and 0 < l.yes_ask < 1:
            raw.append(l.yes_ask * 0.9)
        elif l.yes_bid:
            raw.append(l.yes_bid)
        else:
            raw.append(None)
    known = [r for r in raw if r is not None]
    if not known:
        for l in legs:
            l.p_market, l.extrapolated = 1.0 / len(legs), True
        return
    fill = max(0.01, (1 - sum(known)) / max(1, raw.count(None)))
    vals = [r if r is not None else fill for r in raw]
    s = sum(vals)
    for l, v, r in zip(legs, vals, raw):
        l.p_market = v / s
        l.extrapolated = r is None or not l.two_sided


def fit_game(g: Game) -> None:
    ladders: dict[tuple, list[Leg]] = defaultdict(list)
    exclusive: dict[str, list[Leg]] = defaultdict(list)
    for l in g.legs:
        if l.stat in ("ml", "h1_winner", "win_margin"):
            exclusive[l.event_ticker].append(l)
        else:
            ladders[l.ladder_key].append(l)
    for legs in ladders.values():
        fit_ladder(legs)
    for legs in exclusive.values():
        _normalize_exclusive(legs)
    g.env = game_env(g)


@dataclass
class GameEnv:
    away_total: float
    home_total: float
    home_win: float
    home_margin: float
    total: float
    main_team_line: dict[str, float] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def implied(self, code: str | None, g: Game) -> float:
        return self.home_total if code == g.home_code else self.away_total


def median_line(legs: list[Leg]) -> float | None:
    """Line where P(over) = 0.5, by log-odds interpolation of the fitted ladder."""
    pts = sorted((l.line, logit(l.p_market)) for l in legs if l.line is not None and l.p_market is not None)
    if len(pts) < 2:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 >= 0 >= y1) and y0 != y1:
            return x0 + (y0 / (y0 - y1)) * (x1 - x0)
    # extrapolate from the end nearest zero
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if abs(pts[0][1]) < abs(pts[-1][1]) else (pts[-2], pts[-1])
    if y1 == y0:
        return None
    return x0 + (y0 / (y0 - y1)) * (x1 - x0)


def game_env(g: Game) -> GameEnv:
    by = defaultdict(list)
    for l in g.legs:
        if not l.protected:
            by[(l.stat, l.team)].append(l)
    sources = []
    tt = {c: median_line(by[("team_total", c)]) for c in (g.away_code, g.home_code)}
    total = median_line(by[("game_total", None)])
    margin_home = None
    sh, sa = median_line(by[("spread", g.home_code)]), median_line(by[("spread", g.away_code)])
    ml_home = next((l.p_market for l in by[("ml", g.home_code)]), None)
    ml_away = next((l.p_market for l in by[("ml", g.away_code)]), None)
    if ml_home is None and ml_away is not None:
        ml_home = 1 - ml_away
    if sh is not None and sa is not None:
        # "home wins by over X" median is the home margin when home is favoured; combine both sides
        margin_home = (sh - sa) / 2  # home-margin median; the away ladder gives its negative
    elif ml_home is not None:
        margin_home = 13.5 * (ml_home - 0.5) * 2.0  # rough: 50/50 ↔ 0, 75% ↔ ~6.75
    if margin_home is None:
        margin_home = 0.0
    if ml_home is None:
        ml_home = min(max(0.5 + margin_home / 27.0, 0.05), 0.95)
    home = tt.get(g.home_code)
    away = tt.get(g.away_code)
    if total is not None:
        h2, a2 = (total + margin_home) / 2, (total - margin_home) / 2
        home = (home + h2) / 2 if home is not None else h2
        away = (away + a2) / 2 if away is not None else a2
        sources.append("game total + spread")
    if home is not None:
        sources.append("team totals")
    home = home if home is not None else 23.0
    away = away if away is not None else 21.0
    main = {}
    for c in (g.away_code, g.home_code):
        legs = by[("team_total", c)]
        if legs:
            main[c] = min(legs, key=lambda l: abs((l.p_market or 0.5) - 0.5)).line  # type: ignore[assignment]
    return GameEnv(away_total=float(away), home_total=float(home), home_win=float(ml_home), home_margin=float(margin_home),
                   total=float(total if total is not None else home + away), main_team_line=main, sources=sources or ["defaults"])
