"""Conditional-rate table from nflverse: P(B | A) for same-team leg pairs, by player role and rung.

Rows are team-games (regular season). Roles per team-game:
* QB  — the passer with the most attempts (15+ attempts) in that game;
* WR1, WR2, TE1 — the team's top WRs / TE by season targets; RB1 — top RB by season carries.
TDs are rush + receiving TDs. Team legs: own points over a rung, opponent points over a rung, own win.

Event keys look like ``QB:pass_td:3.5`` (3+), ``WR1:rec_yds:49.5``, ``TEAM:pts:24.5``, ``OPP:pts:20.5``,
``TEAM:win:0``. For every kept pair the table stores the joint count, each event's count among rows where
both are defined, and that row count, so P(B|A) and P(A|B) both come from one entry.

Team win also gets a logistic fit per A event: logit P(win) = a + b·margin + c·A, where margin is the
team's pregame expected margin (from nflverse ``spread_line``). Pricing applies the A effect ``c`` to the
market's own win probability, so an underdog gets a lower conditional than the league average.

Small samples shrink toward a parent bucket with a Beta prior of strength ``SHRINK_K`` (WR1 / WR2 / TE1 →
all three receivers pooled; otherwise toward B's base rate).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from combo_edge import paths

log = logging.getLogger(__name__)

BASELINE_FILE = Path(__file__).with_name("history_baseline.json")
CACHE_NAME = "history_table.json"
DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_QB_ATT = 15
SHRINK_K = 20.0
RECEIVER_ROLES = ("WR1", "WR2", "TE1")
PLAYER_ROLES = ("QB", "WR1", "WR2", "TE1", "RB1")

RUNGS: dict[str, tuple[float, ...]] = {
    "pass_td": tuple(k - 0.5 for k in range(1, 6)),
    "pass_yds": tuple(x + 0.5 for x in range(174, 375, 25)),
    "td": (0.5, 1.5, 2.5),
    "rec_yds": tuple(x + 0.5 for x in range(9, 150, 10)) + (174.5, 199.5),
    "receptions": tuple(k - 0.5 for k in range(1, 14)),
    "rush_yds": tuple(x + 0.5 for x in range(29, 150, 10)) + (174.5,),
    "pts": tuple(x + 0.5 for x in range(10, 45)),
}
ROLE_STATS = {
    "QB": ("pass_td", "pass_yds"),
    "WR1": ("td", "rec_yds", "receptions"),
    "WR2": ("td", "rec_yds", "receptions"),
    "TE1": ("td", "rec_yds", "receptions"),
    "RB1": ("td", "rush_yds", "rec_yds", "receptions"),
}
COUNT_STATS = ("pass_td", "td", "receptions")
YARD_TOL = {"pass_yds": 12.5, "rec_yds": 5.0, "rush_yds": 5.0, "pts": 0.51}


def key(role: str, stat: str, line: float) -> str:
    return f"{role}:{stat}:{line:g}"


def parse(k: str) -> tuple[str, str, float]:
    r, s, l = k.split(":")
    return r, s, float(l)


# --------------------------------------------------------------------------- build


def _load(seasons: tuple[int, ...]) -> tuple[Any, Any]:
    import pandas as pd

    d = paths.nflverse_dir()
    g = pd.read_csv(d / "games.csv", low_memory=False)
    g = g[(g.game_type == "REG") & g.season.isin(seasons) & g.home_score.notna() & g.spread_line.notna()]
    rows = []
    for r in g.itertuples():
        rows.append((r.game_id, r.season, r.home_team, r.home_score, r.away_score, float(r.spread_line)))
        rows.append((r.game_id, r.season, r.away_team, r.away_score, r.home_score, -float(r.spread_line)))
    tg = pd.DataFrame(rows, columns=["game_id", "season", "team", "pts", "opp_pts", "margin"])
    frames = []
    for y in seasons:
        f = d / f"stats_player_week_{y}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, low_memory=False))
    sp = pd.concat(frames, ignore_index=True)
    if "season_type" in sp.columns:
        sp = sp[sp.season_type == "REG"]
    cols = ["player_id", "player_display_name", "position", "team", "game_id", "season", "attempts", "passing_yards", "passing_tds",
            "carries", "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]
    sp = sp[[c for c in cols if c in sp.columns]].fillna({"attempts": 0, "passing_yards": 0, "passing_tds": 0, "carries": 0, "rushing_yards": 0,
                                                          "rushing_tds": 0, "receptions": 0, "targets": 0, "receiving_yards": 0, "receiving_tds": 0})
    return tg, sp


def _role_frames(sp: Any) -> dict[str, Any]:
    """role → DataFrame(game_id, team, stat columns) with one row per team-game where the role played."""
    out = {}
    qb = sp[(sp.position == "QB") & (sp.attempts >= MIN_QB_ATT)].sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    out["QB"] = qb.assign(pass_td=qb.passing_tds, pass_yds=qb.passing_yards)[["game_id", "team", "pass_td", "pass_yds"]]

    def ranked(pos: str, by: str, n: int) -> list[Any]:
        p = sp[sp.position == pos]
        tot = p.groupby(["season", "team", "player_id"])[by].sum().reset_index()
        tot["rank"] = tot.groupby(["season", "team"])[by].rank(ascending=False, method="first")
        res = []
        for i in range(1, n + 1):
            ids = tot[tot["rank"] == i][["season", "team", "player_id"]]
            res.append(p.merge(ids, on=["season", "team", "player_id"]))
        return res

    def stats(df: Any) -> Any:
        return df.assign(td=df.rushing_tds + df.receiving_tds, rec_yds=df.receiving_yards, rush_yds=df.rushing_yards)[
            ["game_id", "team", "td", "rec_yds", "receptions", "rush_yds"]].drop_duplicates(["game_id", "team"])

    wr1, wr2 = ranked("WR", "targets", 2)
    (te1,) = ranked("TE", "targets", 1)
    (rb1,) = ranked("RB", "carries", 1)
    out["WR1"], out["WR2"], out["TE1"], out["RB1"] = stats(wr1), stats(wr2), stats(te1), stats(rb1)
    return out


def build(seasons: tuple[int, ...] = DEFAULT_SEASONS, *, with_players: bool = True) -> dict:
    """Build the table from the local nflverse CSVs (needs pandas + scikit-learn)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    tg, sp = _load(tuple(seasons))
    roles = _role_frames(sp)
    base = tg.reset_index(drop=True)
    n_rows = len(base)
    cols: list[str] = []
    E: list[Any] = []
    V: list[Any] = []

    def add(k: str, ev: Any, valid: Any) -> None:
        cols.append(k)
        E.append(np.asarray(ev, dtype=bool) & np.asarray(valid, dtype=bool))
        V.append(np.asarray(valid, dtype=bool))

    ones = np.ones(n_rows, dtype=bool)
    for line in RUNGS["pts"]:
        add(key("TEAM", "pts", line), base.pts > line, ones)
        add(key("OPP", "pts", line), base.opp_pts > line, ones)
    win = (base.pts > base.opp_pts).to_numpy()
    add(key("TEAM", "win", 0), win, ones)
    for role, df in roles.items():
        m = base.merge(df, on=["game_id", "team"], how="left")
        for stat in ROLE_STATS[role]:
            vals = m[stat].to_numpy(dtype=float)
            valid = ~np.isnan(vals)
            for line in RUNGS[stat]:
                hits = np.where(valid, vals > line, False)
                if hits.sum() >= 5:
                    add(key(role, stat, line), hits, valid)
    Em = np.array(E, dtype=float)  # events × rows
    Vm = np.array(V, dtype=float)
    both = Em @ Em.T
    nn = Vm @ Vm.T
    na = Em @ Vm.T  # A count among rows where B is defined
    idx = {k: i for i, k in enumerate(cols)}

    def wanted(ka: str, kb: str) -> bool:
        ra, sa, _ = parse(ka)
        rb, sb, _ = parse(kb)
        team = ("TEAM", "OPP")
        if ra in team and rb in team:
            return False
        if ra in team or rb in team:
            return True
        if ra == rb:
            return sa != sb  # same player, different stats
        return "QB" in (ra, rb)

    pairs: dict[str, list[int]] = {}
    for i, ka in enumerate(cols):
        for j in range(i + 1, len(cols)):
            kb = cols[j]
            if not wanted(ka, kb) or nn[i, j] < 100:
                continue
            pairs[f"{ka}|{kb}"] = [int(both[i, j]), int(na[i, j]), int(na[j, i]), int(nn[i, j])]
    events = {k: [int(Em[i].sum()), int(Vm[i].sum())] for k, i in idx.items()}
    # team win: logistic on pregame margin, with and without each A event
    margin = base.margin.to_numpy(dtype=float)
    win_fit: dict[str, list[float]] = {}
    lr0 = LogisticRegression(C=1e4, max_iter=500).fit(margin.reshape(-1, 1), win)
    win_base = [float(lr0.intercept_[0]), float(lr0.coef_[0][0])]
    for k, i in idx.items():
        r, _, _ = parse(k)
        if r in ("TEAM", "OPP"):
            continue
        valid = Vm[i] > 0
        a = Em[i][valid]
        if a.sum() < 15:
            continue
        X = np.column_stack([margin[valid], a])
        lr = LogisticRegression(C=1e4, max_iter=500).fit(X, win[valid])
        win_fit[k] = [float(lr.intercept_[0]), float(lr.coef_[0][0]), float(lr.coef_[0][1]), int(a.sum())]
    out = {"meta": {"seasons": list(seasons), "team_games": n_rows, "built": datetime.now().isoformat(timespec="seconds"),
                    "min_qb_att": MIN_QB_ATT},
           "events": events, "pairs": pairs, "win_fit": win_fit, "win_base": win_base}
    if with_players:
        out["players"] = player_values(sp)
    return out


def player_values(sp: Any, last_seasons: int = 2) -> dict[str, dict[str, list[float]]]:
    """Normalised player name → stat → per-game values over the latest seasons (for player base rates)."""
    from combo_edge.sports.nfl.nflverse import norm_name

    recent = sp[sp.season >= sp.season.max() - (last_seasons - 1)]
    out: dict[str, dict[str, list[float]]] = {}
    for name, g in recent.groupby("player_display_name"):
        d: dict[str, list[float]] = {}
        if (g.attempts >= MIN_QB_ATT).sum() >= 3:
            q = g[g.attempts >= MIN_QB_ATT]
            d["pass_td"] = q.passing_tds.astype(float).tolist()
            d["pass_yds"] = q.passing_yards.astype(float).tolist()
        if len(g) >= 3:
            d["td"] = (g.rushing_tds + g.receiving_tds).astype(float).tolist()
            if g.targets.sum() > 0:
                d["rec_yds"] = g.receiving_yards.astype(float).tolist()
                d["receptions"] = g.receptions.astype(float).tolist()
            if g.carries.sum() > 0:
                d["rush_yds"] = g.rushing_yards.astype(float).tolist()
        if d:
            out[norm_name(str(name))] = d
    return out


def save(table: dict, path: Path | None = None) -> Path:
    path = path or (paths.nflverse_dir() / CACHE_NAME)
    path.write_text(json.dumps(table, separators=(",", ":")))
    _load_for.cache_clear()
    return path


def nflverse_available() -> bool:
    d = paths.nflverse_dir()
    return (d / "games.csv").exists() and any(d.glob("stats_player_week_*.csv"))


def production_seasons() -> tuple[int, ...]:
    from combo_edge.sports.nfl.nflverse import current_season

    cur = current_season()
    return tuple(range(cur - 5, cur + 1))


def load() -> "HistoryTable | None":
    """The cached table in data/ (built from local nflverse CSVs on first use), else the packaged baseline."""
    return _load_for(str(paths.nflverse_dir()))


@lru_cache(maxsize=4)
def _load_for(nflverse_dir: str) -> "HistoryTable | None":
    cache = Path(nflverse_dir) / CACHE_NAME
    if not cache.exists() and nflverse_available():
        try:
            save(build(production_seasons()), cache)
        except Exception as e:  # history is optional: fall back to the baseline
            log.warning("history table build failed: %s", e)
    for f in (cache, BASELINE_FILE):
        if f.exists():
            try:
                return HistoryTable(json.loads(f.read_text()))
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("bad history table %s: %s", f, e)
    return None


# --------------------------------------------------------------------------- lookup


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HistoryTable:
    def __init__(self, raw: dict):
        self.raw = raw
        self.events: dict[str, list[int]] = raw["events"]
        self.pairs: dict[str, list[int]] = raw["pairs"]
        self.win_fit: dict[str, list[float]] = raw.get("win_fit", {})
        self.win_base: list[float] = raw.get("win_base", [0.0, 0.15])
        self.players: dict[str, dict[str, list[float]]] = raw.get("players", {})
        self.meta = raw.get("meta", {})

    # keys -----------------------------------------------------------------
    def event_key(self, role: str | None, stat: str, line: float | None) -> str | None:
        """Nearest history rung for a market leg (count legs need the exact k)."""
        if role is None or line is None:
            return None
        if stat == "team_total":
            stat = "pts"
        rungs = RUNGS.get(stat)
        if not rungs:
            return None
        if stat in COUNT_STATS:
            r = math.floor(line) + 0.5
            k = key(role, stat, r)
            return k if k in self.events else None
        best = min(rungs, key=lambda x: abs(x - line))
        if abs(best - line) > YARD_TOL.get(stat, 5.0):
            return None
        k = key(role, stat, best)
        return k if k in self.events else None

    def pair(self, ka: str, kb: str) -> tuple[int, int, int, int] | None:
        """(joint, n_A, n_B, n) with A/B in the order asked."""
        v = self.pairs.get(f"{ka}|{kb}")
        if v is not None:
            return v[0], v[1], v[2], v[3]
        v = self.pairs.get(f"{kb}|{ka}")
        if v is not None:
            return v[0], v[2], v[1], v[3]
        return None

    def base(self, k: str) -> float | None:
        e = self.events.get(k)
        return (e[0] / e[1]) if e and e[1] else None

    def _parent(self, ka: str, kb: str) -> tuple[int, int] | None:
        """Pooled (joint, n_A) over WR1 / WR2 / TE1 for whichever leg is a receiver role."""
        ra, sa, la = parse(ka)
        rb, sb, lb = parse(kb)
        joint = n_a = 0
        found = False
        if ra in RECEIVER_ROLES and rb != ra:
            for r in RECEIVER_ROLES:
                p = self.pair(key(r, sa, la), kb)
                if p:
                    joint, n_a, found = joint + p[0], n_a + p[1], True
        elif rb in RECEIVER_ROLES and ra != rb:
            for r in RECEIVER_ROLES:
                p = self.pair(ka, key(r, sb, lb))
                if p:
                    joint, n_a, found = joint + p[0], n_a + p[1], True
        elif ra == rb and ra in RECEIVER_ROLES:
            for r in RECEIVER_ROLES:
                p = self.pair(key(r, sa, la), key(r, sb, lb))
                if p:
                    joint, n_a, found = joint + p[0], n_a + p[1], True
        elif ra in ("QB", "RB1") and sa in RUNGS:
            lower = [x for x in RUNGS[sa] if x < la]  # next-lower rung of A: less extreme, more games
            if lower:
                p = self.pair(key(ra, sa, lower[-1]), kb)
                if p:
                    joint, n_a, found = p[0], p[1], True
        return (joint, n_a) if found and n_a else None

    def conditional(self, ka: str, kb: str) -> dict | None:
        """P(B | A): raw, shrunk (Beta prior toward the parent bucket, else B's base rate), n_A, and B's base
        rate among the pair's rows."""
        p = self.pair(ka, kb)
        if p is None:
            return None
        joint, n_a, n_b, n = p
        if n_a == 0 or n == 0:
            return None
        pb = n_b / n
        par = self._parent(ka, kb)
        prior = (par[0] / par[1]) if par else pb
        shrunk = (joint + SHRINK_K * prior) / (n_a + SHRINK_K)
        return {"raw": joint / n_a, "rate": shrunk, "n": n_a, "pB": pb, "pA": n_a / n, "joint": joint, "rows": n, "prior": prior}

    def win_conditional(self, ka: str, p_win_market: float) -> dict | None:
        """P(own win | A) for this game: the history A effect (logistic, controlling for the pregame margin)
        applied to the market's win probability."""
        f = self.win_fit.get(ka)
        if not f:
            return None
        a, b, c, n = f
        return {"rate": _expit(_logit(p_win_market) + c), "effect": c, "n": int(n)}

    def win_at_margin(self, ka: str, margin: float) -> float | None:
        f = self.win_fit.get(ka)
        if not f:
            return None
        return _expit(f[0] + f[1] * margin + f[2])

    def player_base(self, name: str | None, stat: str, line: float | None, role_key: str | None, k: float = 8.0) -> float | None:
        """A player's own history rate for a rung (latest seasons), shrunk toward the role's base rate."""
        from combo_edge.sports.nfl.nflverse import norm_name

        role_rate = self.base(role_key) if role_key else None
        if line is None:
            return role_rate
        vals = self.players.get(norm_name(name or ""), {}).get(stat)
        if not vals:
            return role_rate
        hits = sum(1 for v in vals if v > line)
        prior = role_rate if role_rate is not None else hits / len(vals)
        return (hits + k * prior) / (len(vals) + k)


def adjust_to_market(rate: float, pb_hist: float, pb_market: float) -> float:
    """Move a history conditional to the market's B marginal in log-odds (independent legs stay independent)."""
    return _expit(_logit(rate) + _logit(pb_market) - _logit(pb_hist))
