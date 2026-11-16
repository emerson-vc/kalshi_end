"""Fair value when the simulator and the market disagree on a leg: market-matched legs, history lift
caps, and the "model disagrees with market" block."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

from combo_edge.execution.gates import Exposure, GateInput, evaluate
from combo_edge.sports.base import ComboLeg
from combo_edge.sports.nfl import calibration
from combo_edge.sports.nfl import simulator as sim
from combo_edge.sports.nfl.candidates import generate
from combo_edge.sports.nfl.fair import FairModel
from combo_edge.sports.nfl.players import player_key

sys.path.insert(0, str(Path(__file__).parent))
from test_parser_candidates import _games  # noqa: E402


@pytest.fixture(scope="module")
def s():
    from combo_edge.settings import Settings
    from combo_edge.storage.db import DB

    return Settings(DB(":memory:"))


def _lambert_smith_case(s, legacy: bool):
    """A WR3+ whose 50+ rec-yds leg the model puts at 2% but the market prices at 13%, with his QB's 300+
    pass yards priced at 9% (the KeAndre Lambert-Smith 50+ × Herbert 300+ card)."""
    g = _games()["WAS"]
    fm = FairModel(g, s, n_paths=20000, legacy=legacy)
    team_keys = {player_key(l) for l in g.legs if l.team == "WAS"}
    wr3 = next(k for k, c in fm.roster.rec_class.items() if c == "WR3+" and k in team_keys)
    la = next(l for l in g.legs if l.stat == "rec_yds" and player_key(l) == wr3 and not l.protected)
    la.line = 49.5
    x = fm.sim.get(fm.vars[la.ticker])
    fm.sim.vars[fm.vars[la.ticker]] = x * (49.5 / float(np.quantile(x, 0.98)))  # model: P(50+) = 2%
    la.p_model = float((fm.sim.get(fm.vars[la.ticker]) > 49.5).mean())
    la.p_market = la.p_fair = 0.13
    qb = next(k for k, r in fm.roster.roles.items() if r == "QB" and k in team_keys)
    lq = next(l for l in g.legs if l.stat == "pass_yds" and player_key(l) == qb and not l.protected)
    lq.line = 299.5
    lq.p_model = float((fm.sim.get(fm.vars[lq.ticker]) > 299.5).mean())
    lq.p_market = lq.p_fair = 0.09
    fm._ind_cache.clear()
    return fm, la, lq


def test_lambert_smith_herbert_conditional_and_fair(s):
    fm, la, lq = _lambert_smith_case(s, legacy=False)
    assert la.p_model == pytest.approx(0.02, abs=0.003)
    r = fm.price(ComboLeg(la, "yes"), ComboLeg(lq, "yes"), "qb_pyds_rec_yds")
    assert r.p_a == pytest.approx(0.13) and r.p_b == pytest.approx(0.09)
    assert r.cond < 0.25
    assert r.fair_payout > 30
    # flagged: widened lower bound and listed on the card
    assert r.model_disagrees and "model 2% vs market 13%" in r.model_disagrees[0]
    assert r.u_base >= 0.30 + float(s.get("lower_bound_base"))


def test_old_path_reproduces_the_inflated_fair(s):
    fm, la, lq = _lambert_smith_case(s, legacy=True)
    r = fm.price(ComboLeg(la, "yes"), ComboLeg(lq, "yes"), "qb_pyds_rec_yds")
    assert r.cond > 0.4 and r.fair_payout < 20  # the bug: conditional read at the model's 2% tail
    assert not r.model_disagrees


def test_recalibration_alone_lowers_the_conditional(s):
    fm, la, lq = _lambert_smith_case(s, legacy=False)
    ia, ib = fm._indicator(ComboLeg(la, "yes")), fm._indicator(ComboLeg(lq, "yes"))
    assert float(np.mean(ia)) == pytest.approx(0.13, abs=0.002)  # the sim's marginal now matches
    assert float(np.mean(ib)) == pytest.approx(0.09, abs=0.002)
    old_ia = sim.event(fm.sim, fm.vars[la.ticker], 49.5)
    assert sim.conditional(ia, ib, 0.09)[0] < sim.conditional(old_ia, ib, 0.09)[0] - 0.1


def test_qb_ptd_team_total_now_priced_from_history(s):
    """QB pass TDs × own team total: the conditional now comes from the nflverse table (n ≥ 40 for 2+…5+)."""
    n_hist = n = 0
    for g in _games().values():
        fm = FairModel(g, s, n_paths=6000)
        for c in generate(g, fm, s):
            if c.family in ("qb_ptd_team_total", "near_lock_team_total") and c.a.leg.stat == "pass_td":
                r = fm.price(c.a, c.b, c.family)
                n += 1
                n_hist += r.source.startswith("history")
                assert 0 < r.cond <= 1
    assert n > 50 and n_hist / n > 0.9


@pytest.mark.parametrize("var_prefix", ["pass_yds:", "pass_td:", "td:"])
def test_calibrated_event_hits_target_marginal(s, var_prefix):
    g = _games()["WAS"]
    fm = FairModel(g, s, n_paths=20000)
    var = next(v for v in fm.sim.vars if v.startswith(var_prefix))
    x = fm.sim.get(var)
    line = float(np.quantile(x, 0.9)) - 0.5 if var_prefix == "pass_yds:" else 1.5
    for p in (0.02, 0.13, 0.4):
        w = sim.calibrated_event(fm.sim, var, line, p)
        assert float(w.mean()) == pytest.approx(p, abs=0.002), (var, p)
        assert w.min() >= 0 and w.max() <= 1
        # monotone in the stat: every path strictly above the boundary is fully in
        assert w[x > np.sort(x)[::-1][int(p * x.size)]].min() == 1.0


def test_discrete_ties_use_propensity(s):
    g = _games()["WAS"]
    fm = FairModel(g, s, n_paths=20000)
    var = next(v for v in fm.sim.vars if v.startswith("pass_td:"))
    x, up = fm.sim.get(var), fm.sim.get(f"up:{var}")
    p_sim = float((x > 2.5).mean())
    w = sim.calibrated_event(fm.sim, var, 2.5, min(0.95, p_sim * 1.5))  # market above the sim: raise his tail
    boundary = (x == 2) & (w > 0)
    assert boundary.any()
    assert (up[boundary] > 0).all()  # only paths with other TDs a higher pass share could claim


def test_lift_cap_lookup_by_receiver_class():
    wr1, _ = calibration.lift_cap("rec_yds", 49.5, "pass_yds", 299.5, "WR1")
    wr3, cell = calibration.lift_cap("pass_yds", 299.5, "rec_yds", 49.5, "WR3+")  # either order
    assert wr1 and wr3 and wr3 > wr1 and cell["cls"] == "WR3+" and cell["cap"] > cell["lift"]
    assert calibration.lift_cap("rec_yds", 49.5, "pass_yds", 299.5, "RB") == (None, None)
    assert calibration.lift_cap("ml", None, "h1_spread", 3.5, None) == (None, None)


def test_near_certain_history_is_not_lift_capped(s):
    g = _games()["WAS"]
    fm = FairModel(g, s, n_paths=4000)
    ptd = next(l for l in g.legs if l.stat == "pass_td" and l.k == 4 and l.team == "WAS" and not l.protected)
    tt = next(l for l in g.legs if l.stat == "team_total" and l.team == "WAS" and l.line == 24.5)
    cap, cell, _ = fm.lift_cap(ComboLeg(ptd, "yes"), ComboLeg(tt, "yes"))
    assert cap is None and cell is not None  # 4+ pass TDs ⇒ 24+ points: the conditional carries over, not the lift


def test_gate_blocks_model_disagreement(settings):
    gi = GateInput(identity="id", payout=20.0, fair_joint=1 / 12, fair_lower=1 / 14, family_ok=True, family_note="ok",
                   quote_age_s=0.5, combo_ticker="C1", game_id="G1", players=["P1"], rfq_stake=1.0,
                   model_disagrees=["X: 50+ receiving yards: model 2% vs market 13%"])
    r = evaluate(gi, Exposure(), settings)
    assert not r.passes and r.chip == "Model disagrees with market"
    gi.model_disagrees = []
    assert evaluate(gi, Exposure(), settings).passes
