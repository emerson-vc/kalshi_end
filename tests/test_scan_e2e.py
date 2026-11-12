"""Full scan against the mock exchange: identity tagging, budgets, paper buys, cleanup."""

from __future__ import annotations

import pytest

from combo_edge.execution.executor import Timing
from combo_edge.pipeline.scan import ScanParams, run_scan

FAST = Timing(confirm_window_s=0.3, exec_timer_s=0.1, slack_s=0.5)


@pytest.fixture
def fast_settings(settings):
    return settings.with_overrides(quote_window_s=0.6, sim_paths=4000, creation_budget_per_run=40, id_shortlist_per_run=6)


async def test_scan_paper_mode(session, exchange, db, fast_settings):
    db.set_state("execution_mode", "paper")
    summary = await run_scan(ScanParams(), session=session, db=db, settings=fast_settings, timing=FAST)
    assert "error" not in summary, summary.get("error")
    assert summary["games"] == 3 and summary["legs"] > 500
    assert summary["selected"] > 0 and summary["resolved"] > 0
    assert summary["anon_sent"] > 0 and summary["id_sent"] > 0
    # identity: every quote's identity equals how its RFQ was sent
    bad = db.query("SELECT q.id FROM quotes q JOIN rfqs r ON r.id=q.rfq_id WHERE q.identity != r.identity")
    assert bad == []
    # makers that only quote identified RFQs never appear on anonymous quotes
    anon_makers = {r["maker_id"] for r in db.query("SELECT maker_id FROM quotes WHERE identity='anonymous'")}
    assert not ({"mkr-charlie", "mkr-delta", "mkr-echo"} & anon_makers)
    # every executed/booked decision is on an ID quote
    rows = db.query("SELECT r.identity FROM executions e JOIN rfqs r ON r.id=e.rfq_id")
    assert all(r["identity"] == "id" for r in rows)
    assert all(e["book"] == "paper" for e in db.query("SELECT book FROM executions"))
    assert exchange.accepts == []  # paper mode never calls accept
    # nothing left open
    assert db.scalar("SELECT COUNT(*) FROM rfqs WHERE status='open'") == 0
    assert not any(r.status == "open" for r in exchange.rfqs.values())
    # ID budget respected (paper cap)
    assert db.scalar("SELECT COUNT(*) FROM rfqs WHERE identity='id'") <= int(fast_settings.get("id_rfq_daily_cap_paper"))


async def test_id_budget_stops_identified_rfqs(session, exchange, db, fast_settings):
    s = fast_settings.with_overrides(id_rfq_daily_cap_paper=2, id_shortlist_per_run=6)
    summary = await run_scan(ScanParams(), session=session, db=db, settings=s, timing=FAST)
    assert db.scalar("SELECT COUNT(*) FROM rfqs WHERE identity='id'") <= 2
    assert summary["shortlist"] <= 2


async def test_dry_run_books_nothing(session, exchange, db, fast_settings):
    db.set_state("execution_mode", "paper")
    await run_scan(ScanParams(dry_run=True), session=session, db=db, settings=fast_settings, timing=FAST)
    assert db.scalar("SELECT COUNT(*) FROM executions") == 0


async def test_kill_switch_blocks_scan(session, db, fast_settings):
    from combo_edge.jobs.killswitch import apply_kill_state, resume

    apply_kill_state(db, "test")
    with pytest.raises(RuntimeError):
        await run_scan(ScanParams(), session=session, db=db, settings=fast_settings, timing=FAST)
    assert db.get_state("execution_mode") == "off"
    resume(db)
    assert db.get_state("execution_mode") == "off"
