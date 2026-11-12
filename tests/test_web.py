"""Dashboard: every page and partial renders on an empty and a populated DB; actions respond."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(exchange):
    from combo_edge.web.app import create_app

    with TestClient(create_app()) as c:
        yield c


PAGES = ["/", "/bets", "/bets?tab=pnl", "/bets?book=paper", "/insights", "/system", "/p/header", "/p/pipeline", "/p/results", "/p/bets", "/api/state", "/login"]


@pytest.mark.parametrize("path", PAGES)
def test_pages_render_empty(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text[:500]


def test_mode_switching(client, db):
    assert "PAPER" in client.post("/api/mode", data={"mode": "paper"}).text.upper()
    r = client.post("/api/mode", data={"mode": "real"})
    assert "confirmation" in r.text
    r = client.post("/api/mode", data={"mode": "real", "confirm": "yes"})
    assert db.get_state("execution_mode") == "real"  # mock exchange: allowed
    client.post("/api/mode", data={"mode": "paper"})


def test_bankroll_and_settings(client, db):
    client.post("/api/bankroll", data={"bankroll": "1500"})
    from combo_edge.settings import Settings

    assert Settings(db).get("bankroll") == 1500.0
    r = client.post("/api/settings", data={"daily_spend_cap": "25", "payout_band": "[6, 40]"})
    assert "saved" in r.text
    assert Settings(db).get("daily_spend_cap") == 25.0 and Settings(db).get("payout_band") == [6.0, 40.0]
    r = client.post("/api/settings", data={"payout_band": "[50, 10]"})
    assert "weren't saved" in r.text


def test_kill_and_resume(client, db):
    client.post("/api/kill")
    assert db.get_state("execution_mode") == "off"
    assert "Kill switch is on" in client.get("/").text
    client.post("/api/resume")
    assert "Kill switch is on" not in client.get("/").text


async def test_pages_after_scan(exchange, db, settings):
    from combo_edge.execution.executor import Timing
    from combo_edge.kalshi.session import build_session
    from combo_edge.pipeline.scan import ScanParams, run_scan
    from combo_edge.settings import env_config
    from combo_edge.web.app import create_app

    sess = build_session(env_config("mock"), exchange=exchange)
    s = settings.with_overrides(quote_window_s=0.5, sim_paths=3000, creation_budget_per_run=25, id_shortlist_per_run=4)
    db.set_state("execution_mode", "paper")
    await run_scan(ScanParams(), session=sess, db=db, settings=s, timing=Timing(0.3, 0.1, 0.4))
    await sess.close()
    with TestClient(create_app()) as c:
        for path in PAGES:
            r = c.get(path)
            assert r.status_code == 200, (path, r.text[:800])
        html = c.get("/").text
        assert "Pays" in html and "x</span>" in html
