"""History-driven conditionals: the nflverse table, near-locks, spread-aware team win, thin legs, the
new families' shortlist order, the too-good guard and the Found page display filter."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest

from combo_edge import paths
from combo_edge.execution.gates import Exposure, GateInput, evaluate
from combo_edge.sports.base import ComboLeg
from combo_edge.sports.nfl import history
from combo_edge.sports.nfl.fair import FairModel, market_mid
from combo_edge.sports.nfl.players import player_key

sys.path.insert(0, str(Path(__file__).parent))
from test_parser_candidates import _games  # noqa: E402

# (A, B, reference rate) from nflverse 2021–2025 regular season
REFERENCE = [
    ("QB:pass_td:2.5", "WR1:td:1.5", 0.20), ("QB:pass_td:2.5", "TE1:td:1.5", 0.13), ("QB:pass_td:2.5", "TEAM:win:0", 0.74),
    ("QB:pass_td:2.5", "TEAM:pts:20.5", 0.99), ("QB:pass_td:2.5", "TEAM:pts:24.5", 0.86), ("QB:pass_td:2.5", "TEAM:pts:27.5", 0.79),
    ("QB:pass_td:2.5", "TEAM:pts:30.5", 0.65), ("QB:pass_td:3.5", "WR1:td:1.5", 0.26), ("QB:pass_td:3.5", "TE1:td:1.5", 0.20),
    ("QB:pass_td:3.5", "TEAM:win:0", 0.85), ("QB:pass_td:3.5", "TEAM:pts:24.5", 1.00), ("QB:pass_td:3.5", "TEAM:pts:27.5", 0.985),
    ("QB:pass_td:3.5", "TEAM:pts:30.5", 0.85), ("QB:pass_yds:299.5", "TEAM:win:0", 0.56), ("QB:pass_yds:274.5", "TEAM:win:0", 0.575),
    ("TE1:td:1.5", "TEAM:pts:14.5", 0.97), ("WR1:td:1.5", "TEAM:pts:14.5", 0.98), ("WR1:td:1.5", "TEAM:pts:24.5", 0.72),
    ("TE1:receptions:5.5", "TE1:rec_yds:39.5", 0.96), ("WR1:receptions:10.5", "WR1:rec_yds:69.5", 0.99),
]
BASES = [("QB:pass_td:2.5", 0.162), ("QB:pass_td:3.5", 0.051), ("WR1:td:1.5", 0.059), ("TE1:td:1.5", 0.034)]


def _check_reference(t: history.HistoryTable) -> None:
    for a, b, want in REFERENCE:
        got = t.conditional(a, b)
        assert got is not None, (a, b)
        assert got["raw"] == pytest.approx(want, abs=0.03), (a, b, got["raw"])
    for k, want in BASES:
        assert t.base(k) == pytest.approx(want, abs=0.03), k


def test_packaged_baseline_reproduces_reference_rates():
    t = history.HistoryTable(__import__("json").loads(history.BASELINE_FILE.read_text()))
    assert t.meta["seasons"] == [2021, 2022, 2023, 2024, 2025]
    _check_reference(t)


def test_build_from_nflverse_reproduces_reference_rates(monkeypatch):
    real = paths.REPO_ROOT / "data" / "nflverse"
    if not ((real / "games.csv").exists() and all((real / f"stats_player_week_{y}.csv").exists() for y in history.DEFAULT_SEASONS)):
        pytest.skip("nflverse 2021–2025 not downloaded (run combo-edge nflverse-sync)")
    monkeypatch.setenv("COMBO_EDGE_DATA_DIR", str(paths.REPO_ROOT / "data"))
    _check_reference(history.HistoryTable(history.build(history.DEFAULT_SEASONS, with_players=False)))


def test_build_on_a_small_fixture(monkeypatch, tmp_path):
    """Synthetic season: 120 games; QB throws 3+ TDs in exactly the games his team scores 28+."""
    d = tmp_path / "data" / "nflverse"
    d.mkdir(parents=True)
    rng = random.Random(1)
    games, stats = [], []
    for i in range(120):
        gid = f"2024_{i:03d}_AAA_BBB"
        hs, as_ = rng.choice([10, 17, 21, 24, 28, 31, 35]), rng.choice([10, 17, 21, 24, 28])
        games.append({"game_id": gid, "season": 2024, "game_type": "REG", "home_team": "AAA", "away_team": "BBB",
                      "home_score": hs, "away_score": as_, "spread_line": 3.0, "total_line": 44.0})
        for team, pts in (("AAA", hs), ("BBB", as_)):
            stats.append({"player_id": f"{team}QB", "player_display_name": f"{team} QB", "position": "QB", "team": team, "game_id": gid,
                          "season": 2024, "season_type": "REG", "attempts": 30, "passing_yards": 250, "passing_tds": 3 if pts >= 28 else 1,
                          "carries": 2, "rushing_yards": 5, "rushing_tds": 0, "receptions": 0, "targets": 0, "receiving_yards": 0, "receiving_tds": 0})
            stats.append({"player_id": f"{team}WR", "player_display_name": f"{team} WR", "position": "WR", "team": team, "game_id": gid,
                          "season": 2024, "season_type": "REG", "attempts": 0, "passing_yards": 0, "passing_tds": 0, "carries": 0,
                          "rushing_yards": 0, "rushing_tds": 0, "receptions": 6, "targets": 9, "receiving_yards": 70, "receiving_tds": 1})
    for name, rows in (("games.csv", games), ("stats_player_week_2024.csv", stats)):
        with open(d / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    monkeypatch.setenv("COMBO_EDGE_DATA_DIR", str(tmp_path / "data"))
    t = history.HistoryTable(history.build((2024,), with_players=False))
    assert t.meta["team_games"] == 240
    c = t.conditional("QB:pass_td:2.5", "TEAM:pts:27.5")
    assert c["raw"] == pytest.approx(1.0) and c["n"] == sum(1 for g in games for p in (g["home_score"], g["away_score"]) if p >= 28)
    assert t.conditional("TEAM:pts:27.5", "QB:pass_td:2.5") is not None  # either order


# ---------------------------------------------------------------------------- pricing

@pytest.fixture
def s():
    from combo_edge.settings import Settings
    from combo_edge.storage.db import DB

    return Settings(DB(":memory:"))


def _price(leg, p: float, spread: float = 0.02) -> None:
    leg.yes_bid, leg.yes_ask = round(p - spread / 2, 4), round(p + spread / 2, 4)
    leg.no_bid, leg.no_ask = round(1 - leg.yes_ask, 4), round(1 - leg.yes_bid, 4)


def _was():
    g = _games()["WAS"]
    leg = lambda pred: next(l for l in g.legs if not l.protected and pred(l))  # noqa: E731
    return g, leg


def test_two_player_td_combo_uses_history(s):
    """Brissett 3+ TD × Flowers 2+ TD (here Daniels × his WR1 Diggs): market 15% and 6% → P(B|A) ≈ 0.20, fair ≈ 33x."""
    g, leg = _was()
    qb3 = leg(lambda l: l.stat == "pass_td" and l.player == "Jayden Daniels" and l.k == 3)
    wr2 = leg(lambda l: l.stat == "td" and l.player == "Stefon Diggs" and l.k == 2)
    _price(qb3, 0.15)
    _price(wr2, 0.06, 0.02)
    fm = FairModel(g, s, n_paths=4000)
    assert fm.roster.hist_role[player_key(wr2)] == "WR1"
    r = fm.price(ComboLeg(qb3, "yes"), ComboLeg(wr2, "yes"), "qb_ptd_rec_td")
    assert r.p_a == pytest.approx(0.15, abs=1e-3) and r.p_b == pytest.approx(0.06, abs=1e-3)
    assert r.source.startswith("history n=") and not r.near_lock
    assert r.cond == pytest.approx(0.20, abs=0.03)  # the simulator said 0.35
    assert r.fair_payout == pytest.approx(33, abs=4)


def test_herbert_4td_over_24_5_is_near_locked(s):
    g, leg = _was()
    qb4 = leg(lambda l: l.stat == "pass_td" and l.player == "Jayden Daniels" and l.k == 4)
    tt = leg(lambda l: l.stat == "team_total" and l.team == "WAS" and l.line == 24.5)
    _price(qb4, 0.05, 0.01)
    _price(tt, 0.41)
    fm = FairModel(g, s, n_paths=4000)
    r = fm.price(ComboLeg(tt, "yes"), ComboLeg(qb4, "yes"), "near_lock_team_total")  # either order in
    assert r.near_lock and r.near_lock["ticker"] == qb4.ticker and r.near_lock["rate"] >= 0.93 and r.near_lock["n"] >= 100
    assert r.fair_payout == pytest.approx(1 / 0.05, rel=0.03)  # ≈ leg A alone
    assert not forced_pair(qb4, tt)


def forced_pair(a, b) -> bool:
    from combo_edge.sports.nfl.candidates import forced

    return forced(ComboLeg(a, "yes"), ComboLeg(b, "yes"))


def test_herbert_4td_team_win_is_spread_adjusted(s):
    g, leg = _was()
    qb4 = leg(lambda l: l.stat == "pass_td" and l.player == "Jayden Daniels" and l.k == 4)
    ml = leg(lambda l: l.stat == "ml" and l.team == "WAS")
    _price(qb4, 0.05, 0.01)
    _price(ml, 0.31)  # a +6.5 underdog
    fm = FairModel(g, s, n_paths=4000)
    r = fm.price(ComboLeg(qb4, "yes"), ComboLeg(ml, "yes"), "qb_tail_team_win")
    assert r.source.startswith("history")
    assert 0.31 < r.cond < 0.85  # above his market win price, below the league-average 0.85
    t = fm.hist
    assert t.win_at_margin("QB:pass_td:3.5", -6.5) < 0.85 < t.win_at_margin("QB:pass_td:3.5", 6.5) + 0.1


def test_thin_leg_is_blended_and_flagged(s):
    g, leg = _was()
    qb4 = leg(lambda l: l.stat == "pass_td" and l.player == "Jayden Daniels" and l.k == 4)
    tt = leg(lambda l: l.stat == "team_total" and l.team == "WAS" and l.line == 24.5)
    qb4.yes_bid, qb4.yes_ask, qb4.no_bid, qb4.no_ask = 0.01, 0.03, 0.97, 0.99  # spread 2¢ = 100% of the 2¢ mid
    _price(tt, 0.41)
    p, thin = market_mid(qb4)
    assert thin and p == pytest.approx(0.02)
    fm = FairModel(g, s, n_paths=4000)
    base = fm.hist.player_base(qb4.player, "pass_td", qb4.line, "QB:pass_td:3.5")  # his own rate, shrunk to the QB base
    assert qb4.thin and qb4.p_fair == pytest.approx(0.5 * 0.02 + 0.5 * base, abs=1e-6)
    r = fm.price(ComboLeg(qb4, "yes"), ComboLeg(tt, "yes"), "near_lock_team_total")
    assert r.thin_legs == [qb4.wording]
    assert r.u_base >= float(s.get("lower_bound_base")) + 0.10
    # a tight leg is not thin, and the probability is the de-vigged mid
    _price(tt, 0.41, 0.02)
    assert market_mid(tt) == (pytest.approx(0.41), False)
    tt.yes_ask = None
    assert market_mid(tt) == (None, True)  # no ask


def _gi(**kw):
    base = dict(identity="id", payout=40.0, fair_joint=1 / 20, fair_lower=1 / 22, family_ok=True, family_note="ok", quote_age_s=0.5,
                combo_ticker="C1", game_id="G1", players=["P1"], rfq_stake=1.0)
    base.update(kw)
    return GateInput(**base)


def test_too_good_to_be_true_guard(settings):
    r = evaluate(_gi(n_makers=9), Exposure(), settings)  # +100% EV, 9 makers
    assert not r.passes and r.chip == "Check by hand"
    assert evaluate(_gi(n_makers=5), Exposure(), settings).passes  # few makers: allowed
    nl = {"rate": 0.99, "n": 137, "ticker": "X"}
    assert evaluate(_gi(n_makers=9, near_lock=nl), Exposure(), settings).passes  # near-lock exemption
    assert not evaluate(_gi(n_makers=9, near_lock=nl, thin_legs=["X"]), Exposure(), settings).passes  # not when a leg is thin
    assert not evaluate(_gi(n_makers=9, near_lock={**nl, "n": 60}), Exposure(), settings).passes  # nor with history n < 100


def test_shortlist_order_near_lock_then_qb_tail(s):
    from types import SimpleNamespace

    from combo_edge.pipeline.scan import shortlist_tier

    def cand(family, near=None, pay=30.0):
        return SimpleNamespace(family=family, fair=SimpleNamespace(near_lock=near), pred_id_payout=pay)

    nl = {"leg_alone_payout": 31.0}
    assert shortlist_tier(cand("near_lock_team_total", nl, pay=50.0), None) == 1  # predicted ID beats leg A alone
    assert shortlist_tier(cand("near_lock_team_total", nl, pay=20.0), 40.0) == 1  # the anonymous quote beats it
    assert shortlist_tier(cand("near_lock_team_total", nl, pay=20.0), 25.0) == 3
    assert shortlist_tier(cand("qb_tail_team_win"), None) == 2
    assert shortlist_tier(cand("qb_pyds_rec_yds"), None) == 3


def test_found_page_shows_every_payout_by_default():
    """The Herbert 4+ × LAC o24.5 ID quote (77.9x) was hidden: the Found filter defaulted to the 8–50x band."""
    from combo_edge.web.app import FOUND_DISPLAY_BAND

    assert FOUND_DISPLAY_BAND[0] <= 3 and FOUND_DISPLAY_BAND[1] >= 1000
    js = (paths.REPO_ROOT / "src" / "combo_edge" / "web" / "static" / "app.js").read_text()
    assert "ce-filters-v2" in js and "this.hi < 1000" in js  # old saved 8–50x filters are dropped; top of the slider = no limit
