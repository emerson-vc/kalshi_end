"""combo-edge command line. With no command it starts the dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

from combo_edge.settings import Settings, env_config
from combo_edge.storage.db import get_db


def _settings() -> Settings:
    return Settings(get_db())


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


async def _with_session(fn: Any) -> Any:
    from combo_edge.kalshi.session import build_session

    s = build_session(env_config())
    try:
        return await fn(s)
    finally:
        await s.close()


def cmd_doctor(_: argparse.Namespace) -> int:
    from combo_edge import paths
    from combo_edge.execution.executor import side_mapping_status
    from combo_edge.sports.nfl import calibration, nflverse

    env = env_config()
    ok = True
    print(f"Combo Edge doctor — {env.describe()}")
    print(f"  data dir: {paths.data_dir()}")
    print(f"  python:   {sys.version.split()[0]} ({__import__('platform').machine()})")
    if env.forbid_live:
        print("  COMBO_EDGE_FORBID_LIVE=1: Real execution is impossible")

    async def checks(s: Any) -> None:
        nonlocal ok
        try:
            lim = await s.client.get_limits()
            print(f"  ✔ keys valid; API tier {lim.get('usage_tier')} (read {lim['read']['refill_rate']}/s, write {lim['write']['refill_rate']}/s)")
        except Exception as e:
            ok = False
            print(f"  ✘ authenticated call failed: {e}")
            return
        try:
            print(f"  ✔ communications id: {(await s.client.communications_id())[:8]}…")
        except Exception as e:
            print(f"  ! communications id: {e}")
        good = await s.open_ws(wait=10)
        print(f"  {'✔' if good else '✘'} websocket {'connected' if good else 'did not connect'}")
        ok = ok and good
        colls = await s.client.collections()
        nfl = [c["collection_ticker"] for c in colls if any(a["ticker"].startswith("KXNFL") for a in c.get("associated_events", []))]
        print(f"  ✔ open collections with NFL legs: {', '.join(nfl) or 'none'}")
        ms = await s.client.milestones(type="football_game", minimum_start_date=__import__("combo_edge.timeutil", fromlist=["iso"]).iso(time.time()))
        print(f"  ✔ upcoming NFL games listed: {sum(1 for m in ms if (m.get('details') or {}).get('league') == 'NFL')}")

    try:
        asyncio.run(_with_session(checks))
    except RuntimeError as e:
        ok = False
        print(f"  ✘ {e}")
    db = get_db()
    st = side_mapping_status(db, env.name)
    print(f"  side mapping: {'verified' if st['verified'] else 'not verified yet'} {st['counts']}; canary {'passed' if st['canary_passed'] else 'pending'}")
    print(f"  nflverse history: {'synced' if nflverse.available() else 'not synced (packaged baseline in use) — run `combo-edge nflverse-sync`'}")
    for fam, r in calibration.family_report().items():
        print(f"  calibration {fam}: {r['status']}" + (f" (error {r.get('mae_raw')} → {r.get('mae_shifted')})" if r.get("cells") else ""))
    print("OK" if ok else "Problems found (see above)")
    return 0 if ok else 1


def cmd_scan(a: argparse.Namespace) -> int:
    from combo_edge.pipeline.scan import ScanParams, run_scan

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = ScanParams(dry_run=a.dry_run, budget=a.budget, max_new=a.max_new, stake=a.stake, id_shortlist=a.id_shortlist, kind="cli")
    summary = asyncio.run(run_scan(p, settings=_settings()))
    _print({k: v for k, v in summary.items() if k != "coverage"})
    return 1 if summary.get("error") else 0


def cmd_trace(a: argparse.Namespace) -> int:
    from combo_edge.pipeline.trace import parse_leg, trace_combo

    try:
        la, lb = parse_leg(a.leg_a), parse_leg(a.leg_b)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    db, s = get_db(), _settings()

    async def go(sess: Any) -> dict:
        return await trace_combo(db, la, lb, run_id=a.run, session=sess, settings=s, env=env_config().name)

    out = asyncio.run(_with_session(go))
    print(f"Run {out['run_id']}: {out['legs'][0]}  +  {out['legs'][1]}")
    for st in out["steps"]:
        mark = {True: "ok ", False: "NO ", None: " - "}[st["ok"]]
        print(f"  [{mark}] {st['stage']:<20} {st['detail']}")
    print(f"Verdict: {out['verdict']}")
    return 0


def cmd_limits(_: argparse.Namespace) -> int:
    from combo_edge.pipeline import budget

    async def go(s: Any) -> dict:
        return {"limits": await s.client.get_limits(), "endpoint_costs": await s.client.get_endpoint_costs()}

    out = asyncio.run(_with_session(go))
    out["local"] = budget.gauges(_settings(), get_db(), env_config().name)
    _print(out)
    return 0


def cmd_upgrade(_: argparse.Namespace) -> int:
    from combo_edge.kalshi.client import KalshiError

    async def go(s: Any) -> Any:
        return await s.client.upgrade_tier()

    try:
        asyncio.run(_with_session(go))
        print("Upgraded to the Advanced API tier.")
        get_db().set_state("upgrade_tier_offer", False)
        return 0
    except KalshiError as e:
        if e.status == 403:
            print("Kalshi refused (403): none of your last 100 orders was placed through the API yet. "
                  "Make one real bet through combo-edge (the $1 canary counts), then try again.")
        else:
            print(f"Upgrade failed: {e}")
        return 1


def cmd_kill(_: argparse.Namespace) -> int:
    from combo_edge.jobs.killswitch import apply_kill_state, kill

    db = get_db()
    try:
        res = asyncio.run(_with_session(lambda s: kill(db, s, "CLI kill")))
    except Exception as e:
        apply_kill_state(db, "CLI kill")
        res = {"execution": "off", "schedulers": "paused", "cancel_error": str(e)}
    _print(res)
    return 0


def cmd_resume(_: argparse.Namespace) -> int:
    from combo_edge.jobs.killswitch import resume

    resume(get_db())
    print("Kill switch cleared. Execution stays OFF until you turn it on.")
    return 0


def cmd_recorder(a: argparse.Namespace) -> int:
    from combo_edge.jobs.recorder import run_recorder

    logging.basicConfig(level=logging.INFO)
    _print(asyncio.run(_with_session(lambda s: run_recorder(s, get_db(), duration_s=a.duration))))
    return 0


def cmd_requote_due(_: argparse.Namespace) -> int:
    from combo_edge.learning import fade
    from combo_edge.learning.requote import requote_due

    db, s = get_db(), _settings()

    async def go(sess: Any) -> dict:
        r = await requote_due(sess, db, s)
        if fade.due(db, sess.env.name) and not db.get_state("schedulers_paused", False):
            r["fade"] = await fade.fade_check(sess, db, s)
        return r

    _print(asyncio.run(_with_session(go)))
    return 0


def cmd_fade(a: argparse.Namespace) -> int:
    from combo_edge.learning import fade

    _print(asyncio.run(_with_session(lambda s: fade.fade_check(s, get_db(), _settings(), force=a.force))))
    return 0


def cmd_daily_tick(_: argparse.Namespace) -> int:
    from combo_edge.jobs.scheduler import tick

    _print(asyncio.run(tick(get_db(), _settings())))
    return 0


def cmd_launchd(a: argparse.Namespace) -> int:
    from combo_edge.jobs import launchd

    fn = {"install": launchd.install, "uninstall": launchd.uninstall, "status": launchd.status}[a.action]
    r = fn(a.job)
    _print(r) if isinstance(r, dict) else print(r)
    return 0


def cmd_nflverse(_: argparse.Namespace) -> int:
    from combo_edge.sports.nfl import nflverse

    _print(nflverse.sync())
    cells = nflverse.history_cells()
    print(f"saved {len(cells)} history cells to {nflverse.save_cells(cells)}")
    lifts = nflverse.lift_cells()
    print(f"saved {len(lifts)} lift cells to {nflverse.save_lift_cells(lifts)}")
    from combo_edge.sports.nfl import history

    table = history.build(history.production_seasons())
    print(f"saved the history table ({len(table['pairs'])} pairs, seasons {table['meta']['seasons']}) to {history.save(table)}")
    return 0


def cmd_fit(_: argparse.Namespace) -> int:
    from combo_edge.learning.maker_model import MakerModel
    from combo_edge.sports.nfl import calibration

    db = get_db()
    calibration.compute_shifts.cache_clear()
    rep = calibration.family_report()
    for fam, r in rep.items():
        db.insert("calibration", {"family": fam, "ts": time.time(), "data_json": r}, replace=True)
    _print({"calibration": {k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in rep.items()},
            "maker_id": MakerModel(db, "id").fit(), "maker_anonymous": MakerModel(db, "anonymous").fit()})
    return 0


def cmd_smoke(a: argparse.Namespace) -> int:
    from combo_edge.jobs.smoke import smoke

    _print(asyncio.run(smoke(env_config(a.env), n_combos=a.combos, stake=a.stake, max_new=a.max_new)))
    return 0


def cmd_build_css(_: argparse.Namespace) -> int:
    from combo_edge.web.css import build

    print(build())
    return 0


def cmd_wake(a: argparse.Namespace) -> int:
    from combo_edge.jobs import launchd
    from combo_edge.jobs.scheduler import scheduled_hour

    s = _settings()
    if a.action == "disable":
        s.set("wake_enabled", False)
        print("Wake schedule disabled in settings. To remove it from macOS run:\n  " + launchd.wake_cancel_command())
        return 0
    s.set("wake_enabled", True)
    hour = scheduled_hour(get_db(), env_config().name, s)
    mins = int(s.get("wake_minutes_before"))
    t = (hour * 60 - mins) % (24 * 60)
    print("Wake schedule enabled. macOS needs sudo for this, so run it yourself (Mac must be on power):\n  "
          + launchd.wake_command(t // 60, t % 60))
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    from combo_edge.web.server import serve

    return serve(port=a.port, lan=a.lan, open_browser=not a.no_browser)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="combo-edge", description="Kalshi NFL combo edge finder (paper by default).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--lan", action="store_true", help="serve on your home network (needs COMBO_EDGE_LAN_PASSCODE in .env)")
    p.add_argument("--no-browser", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    sc = sub.add_parser("scan", help="run one scan now")
    sc.add_argument("--dry-run", action="store_true")
    sc.add_argument("--budget", type=int)
    sc.add_argument("--max-new", type=int)
    sc.add_argument("--stake", type=float)
    sc.add_argument("--id-shortlist", type=int)
    sc.set_defaults(fn=cmd_scan)
    tr = sub.add_parser("trace", help="why did a scan miss this combo? (TICKER or TICKER:no for each leg)")
    tr.add_argument("leg_a")
    tr.add_argument("leg_b")
    tr.add_argument("--run", type=int, help="run id (default: the latest run)")
    tr.set_defaults(fn=cmd_trace)
    sub.add_parser("doctor", help="check keys, websocket, collections, calibration").set_defaults(fn=cmd_doctor)
    sub.add_parser("limits", help="show API limits and local budgets").set_defaults(fn=cmd_limits)
    sub.add_parser("upgrade-tier", help="request the Advanced API tier").set_defaults(fn=cmd_upgrade)
    sub.add_parser("kill", help="kill switch: execution off, cancel open RFQs, pause schedulers").set_defaults(fn=cmd_kill)
    sub.add_parser("resume", help="clear the kill switch (execution stays off)").set_defaults(fn=cmd_resume)
    r = sub.add_parser("recorder", help="read-only feed recorder")
    r.add_argument("--duration", type=float, default=None)
    r.set_defaults(fn=cmd_recorder)
    sub.add_parser("requote-due", help="re-quote panel combos that are due (never creates, never buys)").set_defaults(fn=cmd_requote_due)
    f = sub.add_parser("fade-check", help="weekly ID-vs-anonymous control quotes")
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fade)
    sub.add_parser("dailyscan-tick", help="run today's automatic scan if it's due").set_defaults(fn=cmd_daily_tick)
    ld = sub.add_parser("launchd", help="manage background agents")
    ld.add_argument("action", choices=["install", "uninstall", "status"])
    ld.add_argument("job", choices=["recorder", "requote", "dailyscan"])
    ld.set_defaults(fn=cmd_launchd)
    sub.add_parser("nflverse-sync", help="download nflverse history and rebuild calibration cells").set_defaults(fn=cmd_nflverse)
    sub.add_parser("fit", help="refit maker models and calibration").set_defaults(fn=cmd_fit)
    sm = sub.add_parser("smoke", help="live smoke test: tiny RFQs both ways, all cancelled, nothing bought")
    sm.add_argument("--env", choices=["demo", "prod", "mock"], default="demo")
    sm.add_argument("--combos", type=int, default=5)
    sm.add_argument("--max-new", type=int, default=10)
    sm.add_argument("--stake", type=float, default=5.0)
    sm.set_defaults(fn=cmd_smoke)
    sub.add_parser("build-css", help="rebuild the Tailwind stylesheet").set_defaults(fn=cmd_build_css)
    w = sub.add_parser("wake", help="optional pmset wake before the daily scan")
    w.add_argument("action", choices=["enable", "disable"])
    w.set_defaults(fn=cmd_wake)
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        return cmd_dashboard(a)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
