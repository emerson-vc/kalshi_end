"""Simulator calibration (spec 1.3) and fair-model behaviour on recorded games."""

from __future__ import annotations

import json

import numpy as np
import pytest

from combo_edge.paths import MOCK_FIXTURES
from combo_edge.sports.nfl import calibration
from combo_edge.sports.nfl import simulator as sim
from combo_edge.sports.nfl.ladders import expit, fit_game, logit
from combo_edge.sports.nfl.parser import build_game


def _generic(implied: float, n: int = 60000):
    home = sim.TeamSpec("H", implied, 0.62, [sim.PlayerSpec("QB", "H", "QB", yds_median=240.0)])
    away = sim.TeamSpec("A", 22.0, 0.62, [sim.PlayerSpec("QBA", "A", "QB", yds_median=225.0)])
    return sim.simulate(home, away, n=n, seed=5)


@pytest.mark.parametrize("implied", [22.5, 25.0, 27.5])
def test_team_over_given_qb_4td_is_calibrated(implied):
    """Spec 1.3: an implied team total of 22.5–27.5 must give P(o28.5 | QB 4+ pass TD) in [0.82, 0.93]
    for the model used in pricing (simulator with Beta pass share, FG→TD on big days, shifted to history)."""
    r = _generic(implied)
    ptd, pts = r.vars["pass_td:QB"], r.vars["pts:H"]
    a = ptd >= 4
    raw = float((pts[a] > 28.5).mean())
    pA, pB = float(a.mean()), float((pts > 28.5).mean())
    # pricing with market marginals equal to the simulator's: no marginal adjustment, then the shift
    delta, n_hist, _ = calibration.lookup("qb_ptd_team_total", pA, pB, implied, a_key=4, b_line=28.5)
    shifted = float(expit(logit(raw) + delta))
    assert n_hist > 30
    assert 0.82 <= shifted <= 0.93, (implied, raw, shifted)
    # the raw simulator is already far from v1's 0.99
    assert raw < 0.97


def test_beta_pass_share_spreads_pass_tds():
    r = _generic(25.0)
    ptd = r.vars["pass_td:QB"]
    off = r.vars["offtd:H"]
    share = np.where(off > 0, ptd / np.maximum(off, 1), np.nan)
    assert np.nanstd(share) > 0.2  # a fixed share would give binomial-only spread
    assert 0.05 < (ptd >= 4).mean() < 0.11


def test_score_rate_matches_implied_total():
    for implied in (17.5, 24.0, 31.0):
        r = _generic(implied, n=40000)
        assert r.vars["pts:H"].mean() == pytest.approx(implied, abs=0.6)


def test_family_checks_pass_on_baseline_history():
    rep = calibration.family_report()
    assert rep["qb_ptd_team_total"]["status"] == "pass"
    assert rep["qb_pyds_rec_yds"]["status"] == "pass"
    assert rep["rec_td_team_total"]["status"] == "pass"
    assert rep["ml_h1_lead"]["status"] == "no_history"


@pytest.fixture(scope="module")
def was_nyg():
    ms = json.loads((MOCK_FIXTURES / "milestones.json").read_text())["milestones"]
    evs = json.loads((MOCK_FIXTURES / "events.json").read_text())["events"]
    m = next(x for x in ms if x["title"].startswith("WAS"))
    g = build_game(m, [e for e in evs if e["event_ticker"] in m["related_event_tickers"]])
    fit_game(g)
    return g


def test_fair_model_prices_family_pair(was_nyg, settings):
    from combo_edge.sports.base import ComboLeg
    from combo_edge.sports.nfl.fair import FairModel

    fm = FairModel(was_nyg, settings, n_paths=20000)
    qb4 = next(l for l in was_nyg.legs if l.stat == "pass_td" and l.player == "Jayden Daniels" and l.k == 4 and not l.protected)
    tt = next(l for l in was_nyg.legs if l.stat == "team_total" and l.team == "WAS" and l.line == 28.5)
    r = fm.price(ComboLeg(qb4, "yes"), ComboLeg(tt, "yes"), "qb_ptd_team_total")
    assert r.method == "history" and r.source.startswith("history n=")  # nflverse table: QB 4+ → team > 28.5
    # history: P(team > 28.5 | QB 4+) ≈ 0.98 league-wide, moved to WAS's market price for o28.5
    assert 0.70 <= r.cond <= 0.995
    assert r.fair_joint <= min(r.p_a, r.p_b)
    assert "WAS Commanders over 28.5 points" in r.explanation["sentence"]
    # NO side is the complement
    rn = fm.price(ComboLeg(qb4, "yes"), ComboLeg(tt, "no"), "qb_ptd_team_total")
    assert r.fair_joint + rn.fair_joint == pytest.approx(r.p_a, rel=0.08)


def test_lower_bound_widens_with_payout(settings):
    from combo_edge.sports.nfl.fair import lower_bound

    f = 0.05
    lb20 = lower_bound(f, 0.08, 20, settings)
    lb40 = lower_bound(f, 0.08, 40, settings)
    lb80 = lower_bound(f, 0.08, 80, settings)
    assert lb20 == pytest.approx(f * 0.92)
    assert lb40 == pytest.approx(f * 0.82)
    assert lb80 == pytest.approx(f * 0.72)
