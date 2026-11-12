"""Quote handling: replacement, REST ordering/statuses, early buffering, identity tagging, side check."""

from __future__ import annotations

import time

import pytest

from combo_edge import money
from combo_edge.kalshi.mock.exchange import MockMaker
from combo_edge.pipeline.quotes import QuoteBook, QuoteObs, RfqRunner, side_check


def _msg(qid, rfq, maker, no_bid, created):
    return {"msg": {"quote_id": qid, "rfq_id": rfq, "quote_creator_id": maker, "yes_bid_dollars": "0.0010",
                    "no_bid_dollars": f"{no_bid:.4f}", "created_ts": created}}


def test_replacement_keeps_newest_by_maker_created_ts():
    b = QuoteBook()
    st = b.register("r1", "id", time.time(), 5.0)
    # the newer quote arrives FIRST locally; maker timestamps decide, not receive time
    b.on_quote_created(_msg("q2", "r1", "m1", 0.980, "2026-11-11T20:00:02Z"), time.time())
    b.on_quote_created(_msg("q1", "r1", "m1", 0.950, "2026-11-11T20:00:01Z"), time.time() + 0.1)
    best = st.best_open()
    assert [q.id for q in best] == ["q2"]


def test_early_quotes_are_buffered():
    b = QuoteBook()
    b.on_quote_created(_msg("q1", "r9", "m1", 0.97, "2026-11-11T20:00:01Z"), time.time())
    st = b.register("r9", "anonymous", time.time(), 5.0)
    assert len(st.best_open()) == 1


def test_best_sorted_by_yes_cost_across_makers():
    b = QuoteBook()
    st = b.register("r1", "id", time.time(), 5.0)
    b.on_quote_created(_msg("a", "r1", "m1", 0.95, "2026-11-11T20:00:01Z"), time.time())
    b.on_quote_created(_msg("b", "r1", "m2", 0.98, "2026-11-11T20:00:01Z"), time.time())
    assert st.best_open()[0].id == "b"
    assert st.best_open()[0].payout == pytest.approx(money.payout(0.02))


def test_rest_drops_cancelled():
    b = QuoteBook()
    st = b.register("r1", "id", time.time(), 5.0)
    b.apply_rest(st, [
        {"id": "x", "creator_id": "m1", "no_bid_dollars": "0.99", "yes_bid_dollars": "0", "created_ts": "2026-11-11T20:00:01Z", "status": "cancelled"},
        {"id": "y", "creator_id": "m1", "no_bid_dollars": "0.97", "yes_bid_dollars": "0", "created_ts": "2026-11-11T20:00:02Z", "status": "open"},
    ], time.time())
    assert [q.id for q in st.best_open()] == ["y"]


async def test_mock_rest_lists_all_statuses_best_first(session, exchange):
    exchange.makers = [MockMaker("m1", True, replace_prob=1.0, latency=(0.01, 0.02)), MockMaker("m2", True, latency=(0.01, 0.02), margin=0.5)]
    res = await session.client.create_combo("KXMVECROSSCATEGORY-R", [
        {"market_ticker": "KXNFLPASSTDS-26NOV12WASNYG-WASJDANIELS5-3", "event_ticker": "KXNFLPASSTDS-26NOV12WASNYG", "side": "yes"},
        {"market_ticker": "KXNFLTEAMTOTAL-26NOV12WASNYG-WAS29", "event_ticker": "KXNFLTEAMTOTAL-26NOV12WASNYG", "side": "yes"}])
    rid = await session.client.create_rfq(res["market_ticker"], target_cost=5, anonymous=False)
    import asyncio

    await asyncio.sleep(0.2)
    qs = await session.client.list_quotes(rid)
    assert any(q["status"] == "cancelled" for q in qs)  # replacement cancelled m1's first quote
    nob = [float(q["no_bid_dollars"]) for q in qs]
    assert nob == sorted(nob, reverse=True)


async def test_ws_blink_falls_back_to_rest(session, exchange, db, settings):
    exchange.makers = [MockMaker("m1", True, latency=(0.15, 0.2), replace_prob=0)]
    await session.open_ws(wait=2)
    book = QuoteBook()
    book.attach(session.ws)
    runner = RfqRunner(session, db, settings, book, run_id=None, window_s=0.6)
    res = await session.client.create_combo("KXMVECROSSCATEGORY-R", [
        {"market_ticker": "KXNFLPASSTDS-26NOV12WASNYG-WASJDANIELS5-3", "event_ticker": "KXNFLPASSTDS-26NOV12WASNYG", "side": "yes"},
        {"market_ticker": "KXNFLTEAMTOTAL-26NOV12WASNYG-WAS29", "event_ticker": "KXNFLTEAMTOTAL-26NOV12WASNYG", "side": "yes"}])
    import asyncio

    async def drop():
        await asyncio.sleep(0.05)
        exchange.drop_ws()

    task = asyncio.create_task(drop())
    out = await runner.run_one(res["market_ticker"], "anonymous", stake=5, purpose="test")
    await task
    assert out.quotes and out.state.used_rest
    row = db.one("SELECT identity, ws_blinked, used_rest, outcome FROM rfqs WHERE id=?", (out.rfq_id,))
    assert row["identity"] == "anonymous" and row["ws_blinked"] == 1 and row["used_rest"] == 1 and row["outcome"] == "quoted"
    assert {q["identity"] for q in db.query("SELECT identity FROM quotes WHERE rfq_id=?", (out.rfq_id,))} == {"anonymous"}


def test_identity_constraint_in_db(db):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        db.insert("rfqs", {"id": "x", "identity": "maybe"})


def test_side_check():
    yes_cost = 0.019
    q = QuoteObs("q", "r", "m", 0.001, 1 - yes_cost, None, money.contracts_for_target(5, yes_cost), 0, 0, "ws")
    assert side_check(q, 5) == "buy_yes_on_no"
    q2 = QuoteObs("q", "r", "m", 0.001, 1 - yes_cost, None, money.contracts_for_target(5, 1 - yes_cost), 0, 0, "ws")
    assert side_check(q2, 5) == "buy_no_on_no"
