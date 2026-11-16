"""Execution gates, sizing and the real accept flow (mock exchange only — nothing touches demo/prod)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from combo_edge.execution.executor import DecisionContext, Executor, Timing, set_mode
from combo_edge.execution.gates import Exposure, GateInput, evaluate, required_ev, size_stake
from combo_edge.kalshi.mock.exchange import MockMaker
from combo_edge.pipeline.quotes import QuoteBook, RfqRunner

FAST = Timing(confirm_window_s=0.3, exec_timer_s=0.1, slack_s=0.6)
CURVE = [[3, 0.08], [10, 0.12], [20, 0.18], [35, 0.25], [50, 0.35], [75, 0.45], [100, 0.55], [150, 0.70], [200, 0.85]]


@pytest.mark.parametrize("payout,expected", [(2, 0.08), (3, 0.08), (10, 0.12), (20, 0.18), (35, 0.25), (50, 0.35),
                                             (75, 0.45), (100, 0.55), (150, 0.70), (200, 0.85)])
def test_payout_curve_points(payout, expected):
    assert required_ev(payout, CURVE) == pytest.approx(expected)


def test_payout_curve_interpolates_in_log_payout():
    import math

    mid = math.sqrt(10 * 20)
    assert required_ev(mid, CURVE) == pytest.approx(0.15)
    assert 0.12 < required_ev(14, CURVE) < 0.18


def test_no_cliff_past_50x():
    # continuous at 50x: 50.1x needs barely more than +35%, not +100%
    assert required_ev(50.1, CURVE) == pytest.approx(0.35, abs=0.002)
    assert required_ev(60, CURVE) < 0.45
    # monotone all the way to the end of the curve
    pays = [3, 5, 10, 20, 35, 50, 51, 60, 75, 90, 100, 120, 150, 180, 200]
    reqs = [required_ev(p, CURVE) for p in pays]
    assert all(a <= b for a, b in zip(reqs, reqs[1:]))
    # above the last point (200x): +100%
    assert required_ev(201, CURVE) == pytest.approx(1.0)
    assert required_ev(500, CURVE) == pytest.approx(1.0)


def test_gate_needs_about_45pct_at_75x(settings):
    assert settings.get("payout_curve") == CURVE  # the shipped default
    r = evaluate(gi(payout=75, fair_joint=1.40 / 75, fair_lower=1.3 / 75), Exposure(), settings)  # +40%: below
    assert not r.passes and r.required_ev == pytest.approx(0.45)
    assert evaluate(gi(payout=75, fair_joint=1.50 / 75, fair_lower=1.3 / 75), Exposure(), settings).passes  # +50%


def gi(**kw):
    base = dict(identity="id", payout=20.0, fair_joint=1 / 12, fair_lower=1 / 14, family_ok=True, family_note="ok",
                quote_age_s=0.5, combo_ticker="C1", game_id="G1", players=["P1"], rfq_stake=1.0)
    base.update(kw)
    return GateInput(**base)


def test_gate_passes_and_sizes(settings):
    r = evaluate(gi(), Exposure(), settings)
    assert r.passes and r.stake >= 1


def test_anonymous_quote_never_executes(settings):
    r = evaluate(gi(identity="anonymous"), Exposure(), settings)
    assert not r.passes and "anonymous" in r.reason


def test_below_gate_and_lower_bound(settings):
    assert not evaluate(gi(fair_joint=1 / 18.5, fair_lower=1 / 20), Exposure(), settings).passes  # +8% at 20x < +18%
    r = evaluate(gi(fair_joint=1 / 12, fair_lower=1 / 21), Exposure(), settings)
    assert not r.passes and "lower bound" in r.reason


def test_above_200x_needs_100pct(settings):
    assert not evaluate(gi(payout=250, fair_joint=1.9 / 250, fair_lower=1.5 / 250), Exposure(), settings).passes  # +90%
    assert evaluate(gi(payout=250, fair_joint=2.2 / 250, fair_lower=1.6 / 250), Exposure(), settings).passes  # +120%
    # 60x at +71% used to fail (the old 50x cliff demanded +100%); now it passes (needs ~+39%)
    assert evaluate(gi(payout=60, fair_joint=1 / 35, fair_lower=1 / 40), Exposure(), settings).passes


def test_caps(settings):
    assert evaluate(gi(), Exposure(bought_combos={"C1"}), settings).cap == "same_combo"
    assert evaluate(gi(), Exposure(bets_by_game={"G1": 3}), settings).cap == "per_game"
    assert evaluate(gi(), Exposure(bets_by_player={"P1": 2}), settings).cap == "per_player"
    assert evaluate(gi(), Exposure(spent_today=29.5), settings).cap == "daily"
    assert evaluate(gi(), Exposure(bets_by_game={"G1": 2}, bets_by_player={"P1": 1}), settings).passes
    # different lines on the same player are different combos
    assert evaluate(gi(combo_ticker="C2"), Exposure(bought_combos={"C1"}, bets_by_player={"P1": 1}), settings).passes


def test_family_check_and_stale(settings):
    assert not evaluate(gi(family_ok=False, family_note="history check failed"), Exposure(), settings).passes
    r = evaluate(gi(quote_age_s=3.5), Exposure(), settings)
    assert not r.passes and r.stale


def test_stake_scales_with_edge(settings):
    weak = size_stake(1 / 16, 1 / 17, 20.0, 20 / 16 - 1, 0.18, settings)
    strong = size_stake(1 / 9, 1 / 10, 20.0, 20 / 9 - 1, 0.18, settings)
    assert 1 <= weak <= 3
    assert strong == 5
    assert size_stake(1 / 9, 1 / 10, 20.0, 20 / 9 - 1, 0.18, settings.with_overrides(max_stake_per_combo=3.0)) == 3


def _cand(exchange, ticker):
    fair = SimpleNamespace(fair_joint=0.2, fair_payout=5.0, lower=lambda pay, s: 0.18)
    game = SimpleNamespace(id="G1", start_ts=time.time() + 86400)
    leg = SimpleNamespace(wording="leg", ticker="L")
    return SimpleNamespace(fair=fair, game=game, players=["P1"], family="qb_ptd_team_total", db_id=None,
                           a=SimpleNamespace(wording="A", leg=leg), b=SimpleNamespace(wording="B", leg=leg))


async def _combo(session):
    res = await session.client.create_combo("KXMVECROSSCATEGORY-R", [
        {"market_ticker": "KXNFLPASSTDS-26NOV12WASNYG-WASJDANIELS5-3", "event_ticker": "KXNFLPASSTDS-26NOV12WASNYG", "side": "yes"},
        {"market_ticker": "KXNFLTEAMTOTAL-26NOV12WASNYG-WAS29", "event_ticker": "KXNFLTEAMTOTAL-26NOV12WASNYG", "side": "yes"},
    ])
    return res["market_ticker"]


async def _run_real(session, exchange, db, settings, stake=1.0):
    db.set_state("execution_mode", "real")
    await session.open_ws(wait=2)
    book = QuoteBook()
    book.attach(session.ws)
    s = settings.with_overrides(quote_window_s=0.8, max_quote_age_s=10.0)
    runner = RfqRunner(session, db, s, book, run_id=None, settle_s=0.3)
    ex = Executor(session, db, s, timing=FAST)
    ticker = await _combo(session)
    ctx = DecisionContext(_cand(exchange, ticker), stake, None)
    out = await runner.run_one(ticker, "id", stake=stake, purpose="test", decide=lambda o, r: ex.consider(o, ctx))
    await asyncio.sleep(0.3)
    return out


async def test_real_accept_fills_yes(session, exchange, db, settings):
    exchange.makers = [MockMaker("m1", False, "independent", 0.0, latency=(0.01, 0.05), replace_prob=0)]
    out = await _run_real(session, exchange, db, settings)
    assert out.decision.executed, out.decision.reason
    assert exchange.accepts and exchange.accepts[0]["side"] == "no"
    e = db.one("SELECT * FROM executions")
    assert e["status"] == "filled" and e["book"] == "real"
    assert exchange.fills and exchange.fills[0]["outcome_side"] == "yes"


async def test_maker_no_confirm(session, exchange, db, settings):
    exchange.makers = [MockMaker("m1", False, "independent", 0.0, latency=(0.01, 0.05), replace_prob=0, confirms=False)]
    out = await _run_real(session, exchange, db, settings)
    assert not out.decision.executed
    assert "maker" in out.decision.reason
    assert db.one("SELECT status FROM executions")["status"] == "maker_no_confirm"
    assert not exchange.fills


async def test_wrong_side_counts_never_accepted(session, exchange, db, settings):
    exchange.makers = [MockMaker("m1", False, "independent", 0.0, latency=(0.01, 0.05), replace_prob=0, wrong_side_counts=True)]
    out = await _run_real(session, exchange, db, settings)
    assert not out.decision.executed
    assert exchange.accepts == []


async def test_side_mismatch_after_accept_deletes_and_turns_off(session, exchange, db, settings, monkeypatch):
    exchange.makers = [MockMaker("m1", False, "independent", 0.0, latency=(0.01, 0.05), replace_prob=0)]
    orig = exchange._accept

    def bad_accept(rfq_id, quote_id, side):
        q = exchange.quotes[quote_id]
        q["no_contracts_fp"] = "1.00"  # exchange reports a count that doesn't match buying YES
        return orig(rfq_id, quote_id, side)

    monkeypatch.setattr(exchange, "_accept", bad_accept)
    exchange.confirm_window_s = 1.5  # maker confirms after ~0.3 s, like a real HVM maker inside the 3 s window
    out = await _run_real(session, exchange, db, settings)
    assert not out.decision.executed
    assert db.get_state("execution_mode") == "off"
    assert not exchange.fills  # RFQ deleted before the maker confirmed: killed


def test_real_forbidden_by_env(db):
    env = SimpleNamespace(name="prod", forbid_live=True)
    ok, why = set_mode(db, env, "real")
    assert not ok and "FORBID" in why
    env2 = SimpleNamespace(name="prod", forbid_live=False)
    ok2, why2 = set_mode(db, env2, "real")
    assert not ok2 and "side mapping" in why2  # needs verified side mapping first


async def test_kill_cancels_open_rfqs(session, exchange, db):
    from combo_edge.jobs.killswitch import is_killed, kill

    ticker = await _combo(session)
    rid = await session.client.create_rfq(ticker, target_cost=5, anonymous=True)
    db.insert("rfqs", {"id": rid, "env": "mock", "identity": "anonymous", "combo_ticker": ticker, "status": "open", "created_ts": time.time()})
    res = await kill(db, session, "test")
    assert res["cancelled"] == 1
    assert exchange.rfqs[rid].status == "closed"
    assert is_killed(db) and db.get_state("execution_mode") == "off" and db.get_state("schedulers_paused")
