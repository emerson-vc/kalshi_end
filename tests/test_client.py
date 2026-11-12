"""Client: allowlist, retries on safe calls only, limiter sizing, websocket reconnect and gaps."""

from __future__ import annotations

import asyncio
import time

import pytest

from combo_edge.kalshi.client import AcceptRefused, EndpointRefused, KalshiError, check_endpoint
from combo_edge.kalshi.limiter import Limiter, TokenBucket, bucket_for


def test_allowlist_refuses_unknown_and_quote_creation():
    check_endpoint("GET", "/communications/quotes")
    with pytest.raises(EndpointRefused):
        check_endpoint("POST", "/communications/quotes")  # quote creation (maker side) is never allowed
    with pytest.raises(EndpointRefused):
        check_endpoint("DELETE", "/communications/rfqs/abc/quotes/def")
    with pytest.raises(EndpointRefused):
        check_endpoint("POST", "/some/unknown/endpoint")


def test_accept_requires_capability():
    from combo_edge.execution.capability import accept_path

    with pytest.raises(AcceptRefused):
        check_endpoint("PUT", accept_path("r1", "q1"))
    with pytest.raises(AcceptRefused):
        check_endpoint("PUT", accept_path("r1", "q1"), capability=object())


def test_capability_cannot_be_forged():
    from combo_edge.execution.capability import AcceptCapability

    with pytest.raises(PermissionError):
        AcceptCapability(object(), "r1", "q1")


def test_capability_bound_to_rfq_and_quote():
    from combo_edge.execution.capability import _mint, accept_path

    cap = _mint("r1", "q1")
    check_endpoint("PUT", accept_path("r1", "q1"), capability=cap)
    with pytest.raises(AcceptRefused):
        check_endpoint("PUT", accept_path("r1", "q2"), capability=cap)


def test_token_bucket_math():
    b = TokenBucket(rate=10, capacity=20)
    t0 = time.monotonic()
    assert b.try_take(20, t0)
    assert not b.try_take(1, t0)
    assert b.wait_time(5, t0) == pytest.approx(0.5, abs=1e-6)
    assert b.try_take(5, t0 + 0.5)


async def test_token_bucket_acquire_waits():
    b = TokenBucket(rate=100, capacity=10)
    await b.acquire(10)
    t = time.monotonic()
    await b.acquire(5)
    assert time.monotonic() - t >= 0.04


def test_limiter_configured_from_limits_and_costs():
    lim = Limiter(fraction=0.6)
    lim.configure(
        {"usage_tier": "advanced", "read": {"refill_rate": 300, "bucket_capacity": 900}, "write": {"refill_rate": 300, "bucket_capacity": 900}},
        {"default_cost": 10, "endpoint_costs": [{"method": "GET", "path": "/trade-api/v2/communications/rfqs/{rfq_id}/quotes/{quote_id}", "cost": 2}]},
    )
    assert lim.read.rate == pytest.approx(180)
    assert lim.write.capacity == pytest.approx(540)
    assert lim.cost_for("GET", "/communications/rfqs/abc/quotes/def") == 2
    assert lim.cost_for("POST", "/communications/rfqs") == 10
    assert bucket_for("POST", "/communications/rfqs") == "write"
    assert bucket_for("DELETE", "/communications/rfqs/x") == "write"
    assert bucket_for("GET", "/communications/quotes") == "read"
    assert bucket_for("POST", "/multivariate_event_collections/X") == "read"


async def test_get_retries_on_5xx(session, exchange):
    exchange.fail_next[r"GET /milestones"] = 2
    ms = await session.client.milestones(type="football_game")
    assert len(ms) == 3


async def test_post_rfq_does_not_retry(session, exchange):
    exchange.fail_next[r"POST /communications/rfqs"] = 1
    with pytest.raises(KalshiError) as e:
        await session.client.create_rfq("KXNFLGAME-26NOV12WASNYG-WAS", target_cost=5, anonymous=True)
    assert e.value.status == 503
    assert sum(1 for m, p in exchange.calls if m == "POST" and p == "/communications/rfqs") == 1


async def test_rfq_limits_409_and_open_limit(session, exchange):
    t = "KXNFLGAME-26NOV12WASNYG-WAS"
    await session.client.create_rfq(t, target_cost=5, anonymous=False)
    with pytest.raises(KalshiError) as e:
        await session.client.create_rfq(t, target_cost=5, anonymous=False)
    assert e.value.status == 409
    exchange.open_rfq_limit = 1
    with pytest.raises(KalshiError) as e2:
        await session.client.create_rfq("KXNFLGAME-26NOV12WASNYG-NYG", target_cost=5, anonymous=False)
    assert e2.value.status == 429 and e2.value.code == "open_rfq_limit_exceeded"


async def test_ws_reconnects_and_records_blinks(session, exchange):
    assert await session.open_ws(wait=2)
    ws = session.ws
    t0 = time.time()
    exchange.drop_ws()
    await asyncio.sleep(0.05)
    assert ws.blinked_between(t0, time.time())
    for _ in range(100):
        if ws.connected.is_set():
            break
        await asyncio.sleep(0.02)
    assert ws.connected.is_set()
    assert ws.reconnects >= 1


async def test_ws_seq_gap_is_a_blink(session, exchange):
    assert await session.open_ws(wait=2)
    conn = exchange.ws_conns[-1]
    conn.push("rfq_deleted", {"id": "x"})
    conn.push("rfq_deleted", {"id": "y"}, skip_seq=True)
    await asyncio.sleep(0.05)
    assert any(d == u for d, u in session.ws.blinks)
