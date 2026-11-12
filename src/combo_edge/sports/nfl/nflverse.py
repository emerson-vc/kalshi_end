"""nflverse history: download, cache, and the historical conditionals each family is calibrated to.

Sources (public CSVs on GitHub releases):
* schedules/games.csv — final scores plus closing spread and total lines (→ implied team totals);
* stats_player/stats_player_week_<year>.csv — per-game player stats aggregated from play-by-play.

The per-game player stats are nflverse's own aggregation of play-by-play, which is what the
calibration needs (pass TDs, yards, receiving TDs per player-game). 1st-half scores aren't in these
files, so the ML × 1H-lead family has no history check (it relies on the simulator and the anchor).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from combo_edge import paths

log = logging.getLogger(__name__)

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
FIRST_SEASON = 2019
BASELINE_FILE = Path(__file__).with_name("calibration_baseline.json")


def current_season(ts: float | None = None) -> int:
    d = datetime.fromtimestamp(ts) if ts else datetime.now()
    return d.year if d.month >= 8 else d.year - 1


def sync(*, seasons: list[int] | None = None, force: bool = False) -> dict[str, Any]:
    """Download nflverse CSVs into data/nflverse (current season always refreshed)."""
    import httpx

    d = paths.nflverse_dir()
    cur = current_season()
    seasons = seasons or list(range(FIRST_SEASON, cur + 1))
    got: dict[str, Any] = {}
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "combo-edge/2"}) as c:
        targets = [("games.csv", GAMES_URL)] + [(f"stats_player_week_{y}.csv", STATS_URL.format(year=y)) for y in seasons]
        for name, url in targets:
            path = d / name
            if path.exists() and not force and name != "games.csv" and str(cur) not in name:
                got[name] = "cached"
                continue
            r = c.get(url)
            if r.status_code == 200:
                path.write_bytes(r.content)
                got[name] = f"{len(r.content) // 1024} KB"
            else:
                got[name] = f"HTTP {r.status_code}"
    return got


def available() -> bool:
    d = paths.nflverse_dir()
    return (d / "games.csv").exists() and any(d.glob("stats_player_week_*.csv"))


def load_tables() -> tuple[Any, Any]:
    """(team_games, player_games) pandas DataFrames."""
    import pandas as pd

    d = paths.nflverse_dir()
    g = pd.read_csv(d / "games.csv", low_memory=False)
    g = g[g.home_score.notna() & g.spread_line.notna() & g.total_line.notna() & (g.season >= FIRST_SEASON)]
    rows = []
    for r in g.itertuples():
        rows.append((r.game_id, r.season, r.week, r.home_team, r.away_team, r.home_score, r.away_score, (r.total_line + r.spread_line) / 2, r.spread_line, r.total_line))
        rows.append((r.game_id, r.season, r.week, r.away_team, r.home_team, r.away_score, r.home_score, (r.total_line - r.spread_line) / 2, -r.spread_line, r.total_line))
    tg = pd.DataFrame(rows, columns=["game_id", "season", "week", "team", "opp", "pts", "opp_pts", "implied", "spread", "total"])
    frames = [pd.read_csv(f, low_memory=False) for f in sorted(d.glob("stats_player_week_*.csv"))]
    sp = pd.concat(frames, ignore_index=True)
    keep = ["player_id", "player_display_name", "position", "team", "game_id", "season", "week", "attempts", "passing_yards",
            "passing_tds", "carries", "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]
    sp = sp[[c for c in keep if c in sp.columns]].copy()
    return tg, sp


def _bucket(implied: float) -> float | None:
    for lo, hi, c in ((15.0, 22.5, 20.0), (22.5, 27.5, 25.0), (27.5, 35.0, 30.0)):
        if lo <= implied < hi:
            return c
    return None


def history_cells() -> list[dict]:
    """Historical marginals and conditionals per family cell (computed from nflverse)."""
    tg, sp = load_tables()
    tg["bucket"] = tg.implied.map(_bucket)
    tg = tg[tg.bucket.notna()]
    cells: list[dict] = []

    def add(family: str, a_key: float, b_key: float, bucket: float, a: Any, b: Any) -> None:
        n_a = int(a.sum())
        n = len(a)
        if n_a < 15 or n < 100:
            return
        joint = int((a & b).sum())
        cells.append({
            "family": family, "a": a_key, "b": b_key, "bucket": bucket, "n": n, "n_a": n_a, "n_b": int(b.sum()),
            "pA": n_a / n, "pB": float(b.mean()), "cond": (joint + 0.5) / (n_a + 1.0),
        })

    # QB pass TDs × own team total
    qb = sp[sp.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    m = qb.merge(tg, on=["game_id", "team", "season"])
    for bucket, grp in m.groupby("bucket"):
        for k in (3, 4, 5):
            for line in (20.5, 24.5, 27.5, 28.5, 31.5, 34.5, 38.5):
                if 6 * k > line:
                    continue
                add("qb_ptd_team_total", k, line, bucket, grp.passing_tds >= k, grp.pts > line)
    # top-2 WR/TE (by season targets) receiving TDs × own team total
    rec = sp[sp.position.isin(["WR", "TE"])].copy()
    season_tgts = rec.groupby(["season", "team", "player_id"]).targets.sum().reset_index()
    season_tgts["rank"] = season_tgts.groupby(["season", "team"]).targets.rank(ascending=False, method="first")
    top = season_tgts[season_tgts["rank"] <= 2][["season", "team", "player_id"]]
    rec = rec.merge(top, on=["season", "team", "player_id"])
    rm = rec.merge(tg, on=["game_id", "team", "season"])
    for bucket, grp in rm.groupby("bucket"):
        for k in (2, 3):
            for line in (20.5, 24.5, 27.5, 28.5, 31.5, 34.5):
                if 6 * k > line:
                    continue
                add("rec_td_team_total", k, line, bucket, grp.receiving_tds >= k, grp.pts > line)
    # QB yards given a top-2 WR/TE receiving-yards tail (receiver is the rarer "A" leg)
    qb_yds = qb[["game_id", "team", "passing_yards"]].rename(columns={"passing_yards": "qb_yds"})
    ry = rm.merge(qb_yds, on=["game_id", "team"])
    for bucket, grp in ry.groupby("bucket"):
        for r_line in (79.5, 99.5, 124.5, 149.5, 174.5):
            for y_line in (249.5, 274.5, 299.5, 324.5):
                add("qb_pyds_rec_yds", r_line, y_line, bucket, grp.receiving_yards > r_line, grp.qb_yds > y_line)
    return cells


def team_pass_shares(season: int | None = None) -> dict[str, float]:
    """Team pass-TD share over the latest season (for the simulator's Beta mean)."""
    _, sp = load_tables()
    season = season or int(sp.season.max())
    s = sp[sp.season == season]
    agg = s.groupby("team").agg(p=("passing_tds", "sum"), r=("rushing_tds", "sum"))
    out = {}
    for team, row in agg.iterrows():
        tot = row.p + row.r
        if tot >= 5:
            out[str(team)] = float(row.p / tot)
    return out


def player_positions() -> dict[str, str]:
    """Normalised display name → position for the latest seasons (tells WR/TE from RBs)."""
    _, sp = load_tables()
    recent = sp[sp.season >= sp.season.max() - 1]
    out: dict[str, str] = {}
    for name, pos in recent[["player_display_name", "position"]].drop_duplicates().itertuples(index=False):
        out[norm_name(str(name))] = str(pos)
    return out


def norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def save_cells(cells: list[dict], path: Path | None = None) -> Path:
    path = path or (paths.nflverse_dir() / "history_cells.json")
    path.write_text(json.dumps({"cells": cells, "generated": datetime.now().isoformat(timespec="seconds")}, indent=0))
    return path


def load_cells() -> list[dict]:
    """Cells from the latest local sync, else the packaged baseline (computed at build time)."""
    p = paths.nflverse_dir() / "history_cells.json"
    for f in (p, BASELINE_FILE):
        if f.exists():
            try:
                return json.loads(f.read_text())["cells"]
            except (json.JSONDecodeError, KeyError):
                continue
    return []


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))
