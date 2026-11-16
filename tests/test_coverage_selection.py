"""Exhaustive scan: no kickoff-order cut, core families never filtered by the payout band, creation
budget counts only new combos, exploration held to its share, and the "why did it miss this?" trace."""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from combo_edge.execution.executor import Timing
from combo_edge.pipeline.scan import ScanParams, run_scan
from combo_edge.pipeline.selection import apply_candidate_cap, is_core, select
from combo_edge.sports.nfl.candidates import generate, price_all
from combo_edge.sports.nfl.fair import FairModel

sys.path.insert(0, str(Path(__file__).parent))
from test_parser_candidates import _games  # noqa: E402

FAST = Timing(confirm_window_s=0.3, exec_timer_s=0.1, slack_s=0.5)
HERBERT_STYLE = ("KXNFLPASSTDS-26NOV12WASNYG-WASJDANIELS5-4", "KXNFLTEAMTOTAL-26NOV12WASNYG-WAS25")  # QB 4+ TDs × own o24.5


@pytest.fixture(scope="module")
def was():
    from combo_edge.settings import Settings
    from combo_edge.storage.db import DB

    s = Settings(DB(":memory:"))
    g = _games()["WAS"]
    fm = FairModel(g, s, n_paths=3000)
    return s, g, fm


def _week(was, n_games: int = 14):
    """14 copies of a full fixture game with later and later kickoffs (sharing the priced legs)."""
    s, g, fm = was
    out, games = [], []
    for i in range(n_games):
        gi = copy.copy(g)
        gi.id = f"G{i:02d}"
        gi.start_ts = g.start_ts + i * 3 * 3600
        cs = generate(gi, fm, s, rng=random.Random(i))
        price_all(cs, fm, s)
        out.extend(cs)
        games.append(gi)
    return s, games, out


def test_last_game_keeps_all_core_after_cap(was):
    s, games, cands = _week(was)
    last = games[-1].id
    cap = int(s.get("max_candidates_per_run"))
    assert len(cands) > cap  # the cap binds
    # the old kickoff-order cut dropped the late games entirely
    assert not any(c.game.id == last for c in cands[:cap])
    kept, cut = apply_candidate_cap(cands, cap)
    n_core = sum(1 for c in cands if is_core(c))
    assert len(kept) == max(cap, n_core) and len(kept) + len(cut) == len(cands)  # core is never cut, even past the cap
    core_last = [c for c in cands if c.game.id == last and is_core(c)]
    assert core_last and all(c.extra.get("cut_reason") is None for c in core_last)
    per_game = Counter(c.game.id for c in kept)
    fair = cap // len(games)
    # a smaller cap still keeps every game's core
    kept2, _ = apply_candidate_cap(cands, 3000)
    assert Counter(c.game.id for c in kept2 if is_core(c)) == Counter(c.game.id for c in cands if is_core(c))
    for gm in games:
        total = sum(1 for c in cands if c.game.id == gm.id)
        assert per_game[gm.id] >= min(fair, total), gm.id
    assert all(c.extra["cut_reason"] == "candidate_cap" and not is_core(c) for c in cut)


def test_core_never_cut_even_when_cap_is_smaller_than_core(was):
    s, games, cands = _week(was, 6)
    n_core = sum(1 for c in cands if is_core(c))
    kept, cut = apply_candidate_cap(cands, n_core // 2)
    assert sum(1 for c in kept if is_core(c)) == n_core
    assert {c.game.id for c in kept} == {g.id for g in games}


def test_core_count_per_game_in_target_range(was):
    s, g, fm = was
    cs = generate(g, fm, s)
    n_core = sum(1 for c in cs if is_core(c))
    assert 150 <= n_core <= 700, n_core  # a full fixture game (every own team-total rung, tail yardage rungs)


def test_every_family_generated_on_fixture_games(was):
    s, _, _ = was
    fams = Counter()
    for g in _games().values():
        fm = FairModel(g, s, n_paths=3000)
        fams.update(c.family for c in generate(g, fm, s) if is_core(c))
    assert set(fams) == {"near_lock_team_total", "qb_tail_team_win", "same_player_rec_yds", "qb_ptd_team_total", "qb_pyds_rec_yds",
                         "qb_ptd_rec_td", "rb_ryds_win", "qb_pyds_opp_total", "ml_h1_lead"}, fams


def _find(cands, a, b):
    return next(c for c in cands if {c.a.leg.ticker, c.b.leg.ticker} == {a, b} and c.a.side == c.b.side == "yes")


def test_herbert_style_generated_and_selected_at_70x(was):
    s, g, fm = was
    cs = generate(g, fm, s)
    price_all(cs, fm, s)
    c = _find(cs, *HERBERT_STYLE)
    assert c.family == "near_lock_team_total" and not c.exploration  # 4+ TDs cover 24.5 ~100% of the time
    c.pred_id_payout, c.pred_ev, c.pred_ev_lower = 70.0, 0.9, 0.5  # predicted 70x: above the old 8x–50x band
    chosen = select(cs, s, int(s.get("creation_budget_per_run")))  # the default new-combo budget
    assert any(x is c for x in chosen)
    assert c.extra["sel"]["pool"] == "core_new:near_lock_team_total"
    # with a strong edge it is near the top of its family even on a tiny budget
    c.pred_ev_lower = 2.0
    assert any(x is c for x in select(cs, s, 20)) and c.extra["sel"]["rank"] == 1


def test_existing_combos_do_not_use_creation_budget(was):
    s, g, fm = was
    cs = generate(g, fm, s)
    price_all(cs, fm, s)
    core = [c for c in cs if is_core(c)]
    for i, c in enumerate(core[:50]):
        c.combo_ticker = f"EXISTING-{i}"
    budget = 10
    chosen = select(cs, s, budget)
    existing = [c for c in chosen if c.combo_ticker]
    new = [c for c in chosen if not c.combo_ticker]
    assert len([c for c in existing if is_core(c)]) == 50  # every existing core combo re-screened
    assert len(new) <= budget  # only new ones count
    # with no budget at all, the existing ones are still re-screened and nothing new is chosen
    chosen0 = select(cs, s, 0)
    assert len(chosen0) == 50 and all(c.combo_ticker for c in chosen0)


def test_exploration_never_exceeds_its_share(was):
    s, g, fm = was
    cs = generate(g, fm, s)
    price_all(cs, fm, s)
    share = float(s.get("exploration_share"))
    for budget in (20, 100, 1000):
        chosen = select(cs, s, budget)
        n_exp = sum(1 for c in chosen if not is_core(c))
        assert n_exp <= int(round(budget * share)), (budget, n_exp)
    # no core candidates at all: the unused core budget still doesn't flow into exploration
    only_exp = [c for c in cs if not is_core(c)]
    chosen = select(only_exp, s, 100)
    assert len(chosen) == int(round(100 * share))


@pytest.fixture
def small_settings(settings):
    return settings.with_overrides(quote_window_s=0.5, sim_paths=3000, creation_budget_per_run=30, id_shortlist_per_run=4,
                                   max_candidates_per_run=900)


async def test_scan_summary_and_trace(session, exchange, db, small_settings):
    from combo_edge.pipeline.trace import LegSpec, trace_combo

    db.set_state("execution_mode", "paper")
    sm = await run_scan(ScanParams(), session=session, db=db, settings=small_settings, timing=FAST)
    assert "error" not in sm, sm.get("error")
    assert sm["candidate_cap"]["cut"] > 0 and sm["core_coverage"]["total"] > 0
    assert {r["matchup"] for r in sm["per_game"]} == {"WAS @ NYG", "LAR @ ARI", "SF @ DAL"}
    assert sm["created_this_run"] <= 30
    run_id = sm["run_id"]

    # a candidate the cap cut: the trace says so
    row = db.one("SELECT leg_a, side_a, leg_b, side_b FROM candidates WHERE run_id=? AND cut_reason='candidate_cap' LIMIT 1", (run_id,))
    t = await trace_combo(db, LegSpec(row["leg_a"], row["side_a"]), LegSpec(row["leg_b"], row["side_b"]), session=session, settings=small_settings)
    assert t["verdict"] == "cut by the candidate cap"
    assert "ranked" in t["steps"][-1]["detail"]

    # a selected, quoted combo goes further down the pipeline
    row = db.one("SELECT leg_a, side_a, leg_b, side_b FROM candidates WHERE run_id=? AND selected=1 AND combo_ticker IS NOT NULL LIMIT 1", (run_id,))
    t = await trace_combo(db, LegSpec(row["leg_a"], row["side_a"]), LegSpec(row["leg_b"], row["side_b"]), session=session, settings=small_settings)
    stages = [s["stage"] for s in t["steps"]]
    assert "Anonymous quote" in stages and "ID shortlist" in stages

    # a pair no rule generates: the trace names the rule
    t = await trace_combo(db, LegSpec("KXNFLPASSTDS-26NOV12WASNYG-WASJDANIELS5-4", "no"), LegSpec("KXNFLTEAMTOTAL-26NOV12WASNYG-NYG25", "no"),
                          session=session, settings=small_settings)
    if t["verdict"] == "not generated":
        assert t["steps"][0]["detail"]

    # second scan: existing combos are re-quoted without spending the creation budget
    sm2 = await run_scan(ScanParams(), session=session, db=db, settings=small_settings, timing=FAST)
    assert "error" not in sm2, sm2.get("error")
    assert sm2["selection"]["core_existing"] >= sm["selection"]["core_new"] - sm["resolution"].get("rejected", 0) - 1
    assert sm2["created_this_run"] <= 30
    assert sm2["core_coverage"]["covered"] >= sm["core_coverage"]["covered"]
    stored = json.loads(db.one("SELECT summary_json FROM runs WHERE id=?", (sm2["run_id"],))["summary_json"])
    assert stored["selection"]["new_budget"] <= 30
