"""Parser on recorded API responses, ladders, candidates, coverage and the tail guards."""

from __future__ import annotations

import json
import math
from collections import Counter

import pytest

from combo_edge.paths import MOCK_FIXTURES
from combo_edge.sports.nfl.candidates import FAMILIES, Candidate, forced, generate, price_all
from combo_edge.sports.nfl.fair import FairModel
from combo_edge.sports.nfl.ladders import fit_game
from combo_edge.sports.nfl.parser import build_game, coverage, parse_market


def _games():
    ms = json.loads((MOCK_FIXTURES / "milestones.json").read_text())["milestones"]
    evs = json.loads((MOCK_FIXTURES / "events.json").read_text())["events"]
    out = []
    for m in ms:
        g = build_game(m, [e for e in evs if e["event_ticker"] in m["related_event_tickers"]])
        fit_game(g)
        out.append(g)
    return {g.away_code: g for g in out}


@pytest.fixture(scope="module")
def games():
    return _games()


def test_parser_on_recorded_responses(games):
    g = games["WAS"]
    assert (g.away_code, g.home_code) == ("WAS", "NYG")
    stats = Counter(l.stat for l in g.legs)
    for s in ("ml", "spread", "game_total", "team_total", "pass_td", "pass_yds", "rec_yds", "rush_yds", "receptions", "td", "h1_spread"):
        assert stats[s] > 0, s
    d4 = next(l for l in g.legs if l.ticker == "KXNFLPASSTDS-26NOV12WASNYG-WASJDANIELS5-4")
    assert d4.wording == "Jayden Daniels: 4+ passing TDs" and d4.k == 4 and d4.team == "WAS" and not d4.protected
    tt = next(l for l in g.legs if l.ticker == "KXNFLTEAMTOTAL-26NOV12WASNYG-WAS25")
    assert tt.wording == "WAS Commanders over 24.5 points"
    assert any(l.protected for l in g.legs if l.stat == "pass_td")
    # unknown series is reported, not crashed on
    assert any(u["series"] == "KXNFLLONGREC" for u in g.unparsed)


def test_parser_never_crashes_on_garbage(games):
    g = games["WAS"]
    assert parse_market({"ticker": "X", "status": "active"}, {"event_ticker": "KXNFLWEIRD-26NOV12WASNYG"}, g, {}) is None
    assert parse_market({"ticker": "X", "status": "active", "title": None, "custom_strike": None},
                        {"event_ticker": "KXNFLPASSTDS-26NOV12WASNYG"}, g, {}) is None


def test_ladders_monotone_and_flagged(games):
    for g in games.values():
        by = {}
        for l in g.legs:
            if l.line is not None and l.stat not in ("ml", "h1_winner", "win_margin"):
                by.setdefault(l.ladder_key, []).append(l)
        for legs in by.values():
            ps = [l.p_market for l in sorted(legs, key=lambda x: x.line)]
            assert all(a >= b - 1e-12 for a, b in zip(ps, ps[1:]))
    assert any(l.extrapolated for l in games["WAS"].legs)


def test_coverage_flags_games_with_props_and_no_candidates(games):
    cov = coverage(list(games.values()), {games["WAS"].id: 10})
    red = {r["matchup"] for r in cov["games"] if r["red"]}
    assert "SF @ DAL" in red and "WAS @ NYG" not in red
    assert cov["unparsed"][0]["examples"]


@pytest.fixture(scope="module")
def priced(games):
    from combo_edge.settings import Settings
    from combo_edge.storage.db import DB

    s = Settings(DB(":memory:"))
    out = []
    for g in games.values():
        fm = FairModel(g, s, n_paths=8000)
        cs = generate(g, fm, s)
        price_all(cs, fm, s)
        out.extend(cs)
    return s, out


def test_families_generated_and_no_forced_pairs(priced):
    _, cands = priced
    fams = Counter(c.family for c in cands if not c.exploration)
    for f in FAMILIES:
        assert fams[f] > 0, f
    assert not any(forced(c.a, c.b) for c in cands)
    # 4+ pass TDs never paired with a team total it forces (6k > line)
    for c in cands:
        if c.family == "qb_ptd_team_total":
            assert 6 * c.a.leg.k <= c.b.leg.line


def test_tail_guards_ev_lower_not_monotone_in_payout(priced):
    """EV at the lower bound must not keep rising with payout (v1 aimed at 40x–200x lottery tails)."""
    s, cands = priced
    buckets = {}
    for c in cands:
        if c.b.side != "yes" or c.a.side != "yes":
            continue
        p = c.pred_id_payout
        b = 0 if p < 8 else 1 if p < 20 else 2 if p < 50 else 3 if p < 100 else 4
        buckets.setdefault(b, []).append(c.pred_ev_lower)
    means = [sum(v) / len(v) for k, v in sorted(buckets.items()) if len(v) >= 5]
    assert len(means) >= 3
    assert not all(a < b for a, b in zip(means, means[1:])), means
    for c in cands:
        f = c.fair
        if f.method != "history":  # the 6x guard is for simulator / prior tails; history rates (10+ rec → 90+ yds) can exceed it
            assert f.fair_joint <= 6 * f.independent + 1e-9
        assert f.fair_joint <= min(f.p_a, f.p_b) + 1e-9


def test_lower_bound_widens_for_longshots(priced):
    s, cands = priced
    short = [c for c in cands if 8 <= c.fair.fair_payout < 15]
    long = [c for c in cands if c.fair.fair_payout >= 60]
    if short and long:
        rel = lambda c: c.fair.lower(c.fair.fair_payout, s) / c.fair.fair_joint  # noqa: E731
        assert sum(map(rel, long)) / len(long) < sum(map(rel, short)) / len(short)
