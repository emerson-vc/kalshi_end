"""The dashboard (FastAPI + server-rendered Jinja, HTMX + Alpine, server-sent events).

Nothing needs a manual refresh: SSE events tell HTMX which partials to re-fetch (header gauges,
pipeline, results, bets). Scans from other processes are picked up by a DB poller.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from combo_edge import __version__, money, paths
from combo_edge.events import bus, toast
from combo_edge.jobs.killswitch import is_killed
from combo_edge.pipeline import budget
from combo_edge.settings import META_BY_KEY, SETTINGS_META, Settings, core_defaults, env_config
from combo_edge.storage.db import get_db
from combo_edge.timeutil import fmt_ago, fmt_kickoff
from combo_edge.web import charts, views

HERE = Path(__file__).parent
WATCH_DIRS = [paths.PACKAGE_DIR]

FOUND_DISPLAY_BAND = [2, 1000]  # 1000 = no upper limit (app.js)
GLOSSARY = [
    ("Pays", "What the best ID quote pays per $1 staked, fee included, e.g. 49.0x. Same number the Kalshi app shows."),
    ("Fair", "What the combo should pay if priced at our fair probability (1 ÷ fair joint probability)."),
    ("Edge", "Expected profit per $1 at the quoted payout: fair probability × payout − 1. +88% means $0.88 per $1 on average."),
    ("Edge (low)", "Edge using the fair model's lower bound, which widens for long shots. Ranking and gates use this."),
    ("Needs", "The minimum edge required at this payout (the payout curve, smooth in log-payout): +35% at 50x, +45% at 75x, +85% at 200x; above 200x it's +100%."),
    ("Leg product", "Payout of leg A × payout of leg B bought separately — what the combo would pay if the legs were unrelated."),
    ("Ratio", "Combo payout ÷ leg product. Near 1.0 on strongly linked legs means makers priced them as unrelated: the edge."),
    ("ID quote", "A quote on an RFQ sent under your Kalshi ID — the price you'd actually get. Only these are ever bought."),
    ("Anonymous quote", "A quote on an RFQ sent anonymously. Fewer makers, much worse prices; used only to screen."),
    ("Makers", "How many market makers quoted the RFQ."),
    ("Quote age", "Seconds since the quote arrived. Over 3 s it's stale and must be refreshed before buying."),
    ("Take rate", "Your combo fills ÷ identified RFQs the tool sent. Low take rate can make makers treat you like anonymous flow."),
    ("ID asks", "Identified RFQs sent today from every source (scan, refresh, re-checks, re-quote panel). Capped per day."),
    ("Creations", "Combos created this week via the API (Kalshi allows 5,000 per week; 300 are left for your phone)."),
    ("CLV", "Closing line value: the payout you got ÷ the payout near kickoff − 1. Positive means you beat the close."),
    ("Paper / Real", "Paper records what it would buy without trading. Real buys automatically when every gate passes."),
    ("Canary", "The first Real bet is $1, checked against your fills (YES side, right contract count) before bigger bets."),
    ("Fade monitor", "Weekly: control combos quoted both ways. If the ID-vs-anonymous gap shrinks, makers may be fading you."),
]


def code_mtime() -> float:
    m = 0.0
    for d in WATCH_DIRS:
        for p in d.rglob("*"):
            if p.suffix in (".py", ".html", ".js", ".css") and p.is_file():
                m = max(m, p.stat().st_mtime)
    return m


def create_app(*, lan: bool = False, argv: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="Combo Edge", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    st = app.state
    st.started = time.time()
    st.code_mtime = code_mtime()
    st.restart_pending = False
    st.scan_task = None
    st.scan_stop = None
    st.argv = argv or []
    st.lan = lan
    st.passcode = os.environ.get("COMBO_EDGE_LAN_PASSCODE", "")
    st.mock_session = None

    env_name = env_config().name

    def db():  # noqa: ANN202
        return get_db()

    def settings() -> Settings:
        return Settings(db())

    # ---------------------------------------------------------------- helpers
    async def session():  # noqa: ANN202
        from combo_edge.kalshi.session import build_session

        if env_name == "mock":
            if st.mock_session is None:
                st.mock_session = build_session(env_config())
            return st.mock_session, False
        return build_session(env_config()), True

    def restart_needed() -> bool:
        return code_mtime() > st.code_mtime + 0.5

    def header_ctx() -> dict:
        d = db()
        s = settings()
        g = budget.gauges(s, d, env_name)
        from combo_edge.learning.fade import trend

        done = views.latest_run(d, env_name, finished=True)
        return {
            **g, "core_coverage": ((done or {}).get("summary") or {}).get("core_coverage"),
            "killed": is_killed(d), "restart_needed": restart_needed(), "fade_warning": trend(d, env_name)["warning"],
            "scan_running": bool(st.scan_task and not st.scan_task.done()) or bool(__import__("combo_edge.pipeline.scan", fromlist=["scan_running"]).scan_running(d)),
            "forbid_live": env_config().forbid_live,
        }

    def base_ctx(request: Request, page: str) -> dict:
        return {"request": request, "page": page, "h": header_ctx(), "version": __version__,
                "build_time": time.strftime("%b %-d %H:%M", time.localtime(st.code_mtime)), "env_name": env_name}

    templates.env.globals.update(fmt=views.fmt, money=money, fmt_kickoff=fmt_kickoff, fmt_ago=fmt_ago, now=time.time, json=json)

    def render(name: str, ctx: dict) -> HTMLResponse:
        return templates.TemplateResponse(ctx["request"], name, ctx)

    # ---------------------------------------------------------------- LAN passcode
    @app.middleware("http")
    async def guard(request: Request, call_next):  # noqa: ANN001, ANN202
        client = request.client.host if request.client else ""
        local = client in ("127.0.0.1", "::1", "localhost", "testclient")
        if not st.lan or local or request.url.path.startswith("/static") or request.url.path == "/login":
            return await call_next(request)
        tok = request.cookies.get("ce_auth", "")
        good = hmac.new(st.passcode.encode(), b"combo-edge", hashlib.sha256).hexdigest()
        if not hmac.compare_digest(tok, good):
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> HTMLResponse:
        return render("login.html", {"request": request, "error": False})

    @app.post("/login")
    async def login(request: Request, passcode: str = Form("")) -> Any:
        if st.passcode and hmac.compare_digest(passcode, st.passcode):
            r = RedirectResponse("/", status_code=303)
            r.set_cookie("ce_auth", hmac.new(st.passcode.encode(), b"combo-edge", hashlib.sha256).hexdigest(), httponly=True, samesite="strict", max_age=30 * 86400)
            return r
        return render("login.html", {"request": request, "error": True})

    # ---------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    async def found(request: Request) -> HTMLResponse:
        return render("found.html", {**base_ctx(request, "found"), **found_ctx()})

    def found_ctx() -> dict:
        d, s = db(), settings()
        from combo_edge.jobs.scheduler import next_scan_ts

        run = views.latest_run(d, env_name)
        done = views.latest_run(d, env_name, finished=True)
        rows = views.found_rows(d, env_name, done["id"] if done else None)
        # display filter defaults wide open (2x – any): the payout band is not a filter on what you see
        return {"run": run, "done": done, "rows": rows, "next_scan": next_scan_ts(d, env_name, s), "s": s,
                "band": FOUND_DISPLAY_BAND, "families": sorted({r["family_label"] for r in rows}), "games": sorted({r["matchup"] for r in rows}),
                "id_left": budget.id_left(s, d, env_name), "stages": (run or {}).get("progress", {}).get("stages"),
                "id_short": max(0, min(int(s.get("id_shortlist_per_run")), budget.id_left(s, d, env_name) - int(s.get("id_reserve") or 0))),
                "trace_games": views.trace_games(d, env_name)}

    @app.get("/bets", response_class=HTMLResponse)
    async def bets_page(request: Request, book: str = "all", tab: str = "taken") -> HTMLResponse:
        return render("bets.html", {**base_ctx(request, "bets"), **bets_ctx(book, tab)})

    def bets_ctx(book: str, tab: str) -> dict:
        d, s = db(), settings()
        mode = budget.execution_mode(d)
        cur_book = "real" if mode == "real" else "paper"
        ctx: dict[str, Any] = {"book": book, "tab": tab, "s": s}
        ctx["bets"] = views.bets(d, env_name, None if book == "all" else book)
        ctx["exposure"] = views.exposure_table(d, env_name, cur_book)
        ctx["cur_book"] = cur_book
        ctx["log"] = views.decision_log(d, env_name)
        ctx["pnl"] = {b: views.pnl(d, env_name, b) for b in ("real", "paper")}
        ctx["charts"] = {b: charts.line_chart(ctx["pnl"][b]["series"]) for b in ("real", "paper")}
        ctx["upgrade_offer"] = d.get_state("upgrade_tier_offer", False)
        return ctx

    @app.get("/insights", response_class=HTMLResponse)
    async def insights(request: Request) -> HTMLResponse:
        from combo_edge.learning import fade, reports
        from combo_edge.learning.bandit import Bandit
        from combo_edge.sports.nfl import calibration

        d = db()
        mr = {k: views._j(r["metrics_json"]) for k in ("maker_id", "maker_anonymous") for r in d.query("SELECT metrics_json FROM model_runs WHERE kind=? ORDER BY id DESC LIMIT 1", (k,))}
        edge = reports.edge_report(d, env_name)
        tr = reports.take_rate_history(d, env_name)
        ctx = {**base_ctx(request, "insights"), "edge": edge, "gap": reports.id_vs_anon_gap(d, env_name), "fade": fade.trend(d, env_name),
               "take": tr, "maker": mr, "calib": calibration.family_report(), "promoted": Bandit(d).promoted(), "arms": Bandit(d).table()[:12],
               "when": reports.when_to_run(d, env_name), "roster": reports.maker_roster(d, env_name)[:15], "truth": reports.truth_report(d, env_name),
               "hours_chart": charts.bar_chart([(k, v["mean"]) for k, v in sorted(edge.get("hours", {}).items())]),
               "take_chart": charts.bar_chart([(time.strftime("%b %-d", time.localtime(w["week_start"])), w["rate"]) for w in tr])}
        return render("insights.html", ctx)

    @app.get("/system", response_class=HTMLResponse)
    async def system(request: Request) -> HTMLResponse:
        return render("system.html", {**base_ctx(request, "system"), **system_ctx()})

    def system_ctx() -> dict:
        from combo_edge.execution.executor import side_mapping_status
        from combo_edge.jobs import launchd

        d, s = db(), settings()
        rec = d.get_state("recorder_heartbeat") or {}
        last_lim = d.one("SELECT data_json, ts FROM limits_log WHERE env=? ORDER BY id DESC LIMIT 1", (env_name,))
        done = views.latest_run(d, env_name, finished=True)
        sm = side_mapping_status(d, env_name)
        jobs = {j: launchd.status(j) for j in launchd.JOBS}
        env = env_config()
        checks = [
            ("Keys valid", env.name == "mock" or bool(d.get_state(f"auth_ok:{env_name}")), "mock exchange" if env.name == "mock" else ("last authenticated call succeeded" if d.get_state(f"auth_ok:{env_name}") else "run `combo-edge doctor` or a scan")),
            ("Websocket connected (last scan)", bool(done and done["summary"].get("ws_connected")), "connected during the last scan" if done else "no scan yet"),
            ("Side mapping verified", sm["verified"], f"contract-count checks {sm['counts']}"),
            ("Canary passed", sm["canary_passed"] or env.name == "mock", "first Real bet is a $1 canary"),
            ("Recorder running", bool(rec.get("ts") and time.time() - rec["ts"] < 120), f"last heartbeat {fmt_ago(rec.get('ts'))}; NFL same-game seen {rec.get('nfl_same_game', 0)}"),
            ("Re-quote panel running", jobs["requote"].get("installed", False), "launchd agent installed" if jobs["requote"].get("installed") else "not installed"),
            ("Daily scan scheduled", jobs["dailyscan"].get("installed", False) and bool(s.get("daily_scan_enabled")), "launchd agent installed" if jobs["dailyscan"].get("installed") else "not installed"),
            ("No pending cancels", d.scalar("SELECT COUNT(*) FROM pending_cancels", (), 0) == 0, f"{d.scalar('SELECT COUNT(*) FROM pending_cancels', (), 0)} pending"),
        ]
        groups: dict[str, list] = {}
        for m in SETTINGS_META:
            groups.setdefault(m.group, []).append({"meta": m, "value": s.get(m.key), "default": core_defaults().get(m.key),
                                                   "overridden": d.get_setting_override(m.key) is not None})
        return {"checks": checks, "errors": views.recent_errors(d), "limits": views._j(last_lim["data_json"]) if last_lim else None,
                "jobs": jobs, "groups": groups, "glossary": GLOSSARY, "env_desc": env.describe(), "s": s,
                "wake_cmd": launchd.wake_command(10, 0)}

    # ---------------------------------------------------------------- partials
    @app.get("/p/header", response_class=HTMLResponse)
    async def p_header(request: Request) -> HTMLResponse:
        return render("partials/header.html", base_ctx(request, request.query_params.get("page", "")))

    @app.get("/p/pipeline", response_class=HTMLResponse)
    async def p_pipeline(request: Request) -> HTMLResponse:
        return render("partials/pipeline.html", {**base_ctx(request, "found"), **found_ctx()})

    @app.get("/p/results", response_class=HTMLResponse)
    async def p_results(request: Request) -> HTMLResponse:
        return render("partials/results.html", {**base_ctx(request, "found"), **found_ctx()})

    @app.get("/p/bets", response_class=HTMLResponse)
    async def p_bets(request: Request, book: str = "all", tab: str = "taken") -> HTMLResponse:
        return render("partials/bets_body.html", {**base_ctx(request, "bets"), **bets_ctx(book, tab)})

    # ---------------------------------------------------------------- actions
    @app.post("/api/scan")
    async def api_scan(request: Request, stake: float = Form(5.0), budget_n: int = Form(500), id_shortlist: int | None = Form(None), dry_run: str = Form("")) -> Any:
        from combo_edge.pipeline.scan import ScanParams, run_scan, scan_running

        if (st.scan_task and not st.scan_task.done()) or scan_running(db()):
            return msg("A scan is already running.", "warn")
        if is_killed(db()):
            return msg("Kill switch is on — resume it on the System page first.", "error")
        params = ScanParams(dry_run=bool(dry_run), budget=budget_n, stake=stake, id_shortlist=id_shortlist, kind="manual")
        st.scan_stop = asyncio.Event()

        async def go() -> None:
            sess, own = await session()
            try:
                await run_scan(params, session=sess, db=db(), settings=settings(), stop=st.scan_stop)
                db().set_state(f"auth_ok:{env_name}", True)
            except Exception as e:  # surfaced as a toast
                toast(f"Scan failed: {e}", "error")
            finally:
                if own:
                    await sess.close()
                bus.publish("results", {})

        st.scan_task = asyncio.create_task(go())
        return msg("Scan started.", "info")

    @app.post("/api/mode")
    async def api_mode(mode: str = Form(...), confirm: str = Form("")) -> Any:
        from combo_edge.execution.executor import set_mode

        if mode == "real" and confirm != "yes":
            return msg("Switching to Real needs confirmation.", "warn")
        ok, why = set_mode(db(), env_config(), mode)
        bus.publish("header", {})
        return msg(f"Execution: {mode.upper()}" if ok else f"Can't switch: {why}", "success" if ok else "error")

    @app.post("/api/bankroll")
    async def api_bankroll(bankroll: float = Form(...)) -> Any:
        if bankroll <= 0:
            return msg("Bankroll must be positive.", "error")
        settings().set("bankroll", bankroll)
        bus.publish("header", {})
        return msg(f"Bankroll set to {money.fmt_money(bankroll)}", "success")

    @app.post("/api/kill")
    async def api_kill() -> Any:
        from combo_edge.jobs.killswitch import kill

        if st.scan_stop:
            st.scan_stop.set()
        sess, own = await session()
        try:
            res = await kill(db(), sess, "dashboard kill switch")
        finally:
            if own:
                await sess.close()
        bus.publish("header", {})
        return msg(f"Killed: execution off, {res.get('cancelled', 0)} open RFQs cancelled, schedulers paused.", "error")

    @app.post("/api/resume")
    async def api_resume() -> Any:
        from combo_edge.jobs.killswitch import resume

        resume(db())
        bus.publish("header", {})
        return msg("Kill switch cleared. Execution stays OFF until you turn it on.", "success")

    @app.post("/api/refresh/{candidate_id}")
    async def api_refresh(candidate_id: int) -> Any:
        """A new ID RFQ for this combo; the executor decides on it like any ID quote (counts in the ID budget)."""
        return await quote_again(candidate_id, decide=True)

    @app.post("/api/recheck/{execution_id}")
    async def api_recheck(execution_id: int) -> Any:
        row = db().one("SELECT combo_ticker, kickoff_ts FROM executions WHERE id=?", (execution_id,))
        if not row:
            return msg("Unknown bet.", "error")
        return await quote_ticker(row["combo_ticker"], row["kickoff_ts"], purpose="recheck")

    async def quote_again(candidate_id: int, decide: bool) -> Any:
        from combo_edge.execution.executor import DecisionContext, Executor
        from combo_edge.pipeline.quotes import QuoteBook, RfqRunner
        from combo_edge.pipeline.budget import IdBudgetExhausted
        from combo_edge.sports.nfl import calibration

        d, s = db(), settings()
        c = d.one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
        if not c or not c["combo_ticker"]:
            return msg("Unknown combo.", "error")
        cand = _StoredCandidate(d, c)
        sess, own = await session()
        try:
            book = QuoteBook()
            await sess.open_ws(user_filter_self=True)
            if sess.ws:
                book.attach(sess.ws)
            runner = RfqRunner(sess, d, s, book, run_id=None)
            ex = Executor(sess, d, s)
            ok, note = calibration.family_passes(c["family"])
            ctx = DecisionContext(cand, ex.planned_stake(_stored_stake(d, candidate_id, s)), None, family_ok=ok, family_note=note)
            out = await runner.run_one(c["combo_ticker"], "id", stake=ctx.planned_stake, purpose="refresh", candidate_id=candidate_id,
                                       kickoff_ts=cand.game.start_ts, decide=(lambda o, r: ex.consider(o, ctx)) if decide else None)
        except IdBudgetExhausted as e:
            return msg(str(e), "warn")
        finally:
            if not own and sess.ws:
                await sess.ws.stop()
                sess.ws = None
            if own:
                await sess.close()
        bus.publish("results", {})
        bus.publish("header", {})
        if not out.best_payout:
            return msg("No ID quotes this time.", "info")
        d_ = out.decision
        return msg(f"ID quote {out.best_payout:.1f}x from {out.n_makers} makers — {d_.chip if d_ else ''}", "success")

    async def quote_ticker(ticker: str, kickoff: float | None, purpose: str) -> Any:
        from combo_edge.pipeline.quotes import QuoteBook, RfqRunner
        from combo_edge.pipeline.budget import IdBudgetExhausted

        d, s = db(), settings()
        sess, own = await session()
        try:
            book = QuoteBook()
            await sess.open_ws(user_filter_self=True)
            if sess.ws:
                book.attach(sess.ws)
            runner = RfqRunner(sess, d, s, book, run_id=None)
            out = await runner.run_one(ticker, "id", stake=float(s.get("stake")), purpose=purpose, kickoff_ts=kickoff)
            if out.best_payout:
                d.insert("requote_obs", {"combo_ticker": ticker, "identity": "id", "rfq_id": out.rfq_id, "bucket_h": -1,
                                         "hours_to_kickoff": ((kickoff or time.time()) - time.time()) / 3600, "ts": time.time(), "payout": out.best_payout, "n_makers": out.n_makers})
        except IdBudgetExhausted as e:
            return msg(str(e), "warn")
        finally:
            if not own and sess.ws:
                await sess.ws.stop()
                sess.ws = None
            if own:
                await sess.close()
        bus.publish("header", {})
        return msg(f"Re-check: {out.best_payout:.1f}x now" if out.best_payout else "Re-check: no ID quotes.", "info")

    @app.post("/api/settings")
    async def api_settings(request: Request) -> Any:
        form = await request.form()
        s = settings()
        errors = []
        for key, val in form.items():
            if key not in META_BY_KEY:
                continue
            meta = META_BY_KEY[key]
            try:
                if meta.kind in ("curve", "band", "range", "families"):
                    s.set(key, json.loads(str(val)))
                else:
                    s.set(key, val)
            except Exception as e:
                errors.append(f"{meta.label}: {e}")
        for meta in SETTINGS_META:  # unchecked checkboxes
            if meta.kind == "bool" and meta.key not in form and form.get(f"_present_{meta.key}"):
                s.set(meta.key, False)
        bus.publish("header", {})
        return msg("Settings saved." if not errors else "Some settings weren't saved: " + "; ".join(errors), "success" if not errors else "error")

    @app.post("/api/settings/reset/{key}")
    async def api_reset(key: str) -> Any:
        settings().reset(key)
        return msg(f"{META_BY_KEY[key].label} reset to default.", "success")

    @app.post("/api/launchd/{action}/{job}")
    async def api_launchd(action: str, job: str) -> Any:
        from combo_edge.jobs import launchd

        if action not in ("install", "uninstall"):
            return msg("Unknown action.", "error")
        r = (launchd.install if action == "install" else launchd.uninstall)(job)
        return msg(r, "success")

    @app.post("/api/upgrade-tier")
    async def api_upgrade() -> Any:
        from combo_edge.kalshi.client import KalshiError

        sess, own = await session()
        try:
            await sess.client.upgrade_tier()
            db().set_state("upgrade_tier_offer", False)
            return msg("Upgraded to the Advanced API tier.", "success")
        except KalshiError as e:
            if e.status == 403:
                return msg("Kalshi said no (403): none of your last 100 orders came through the API yet. One real bet through combo-edge fixes that.", "warn")
            return msg(f"Upgrade failed: {e}", "error")
        finally:
            if own:
                await sess.close()

    @app.post("/api/restart")
    async def api_restart() -> Any:
        st.restart_pending = True
        return msg("Restarting as soon as no scan is running…", "info")

    @app.get("/p/trace/legs", response_class=HTMLResponse)
    async def p_trace_legs(request: Request) -> HTMLResponse:
        q = request.query_params
        slot = q.get("slot", "a")
        gid = q.get(f"game_{slot}") or ""
        opts = ['<option value="">Leg…</option>']
        for r in views.trace_legs(db(), env_name, gid):
            opts.append(f'<option value="{html.escape(r["ticker"])}">{html.escape(r["wording"] or r["ticker"])}</option>')
        return HTMLResponse("".join(opts))

    @app.post("/api/trace", response_class=HTMLResponse)
    async def api_trace(request: Request) -> HTMLResponse:
        from combo_edge.pipeline.trace import parse_leg, trace_combo

        form = await request.form()
        legs = []
        try:
            for slot in ("a", "b"):
                raw = str(form.get(f"ticker_{slot}") or "").strip() or str(form.get(f"leg_{slot}") or "").strip()
                if not raw:
                    raise ValueError(f"pick or paste leg {slot.upper()}")
                leg = parse_leg(raw)
                if form.get(f"no_{slot}"):
                    leg.side = "no"
                legs.append(leg)
            run_txt = str(form.get("run") or "").strip()
            run_id = int(run_txt) if run_txt else None
        except ValueError as e:
            return render("partials/trace.html", {"request": request, "t": None, "error": str(e)})
        sess, own = await session()
        try:
            t = await trace_combo(db(), legs[0], legs[1], run_id=run_id, session=sess, settings=settings(), env=env_name)
        finally:
            if own:
                await sess.close()
        return render("partials/trace.html", {"request": request, "t": t, "error": None})

    @app.post("/api/fit")
    async def api_fit() -> Any:
        from combo_edge.learning.maker_model import MakerModel

        try:
            m = await asyncio.to_thread(MakerModel(db(), "id").fit)
        except Exception as e:
            return msg(f"Maker model fit failed ({type(e).__name__}); scans use the built-in estimate", "error")
        return msg(f"Maker model: {m.get('rows')} ID quotes, {'fitted' if m.get('fitted') else 'not enough data yet: ' + str(m.get('reason', ''))}", "info")

    @app.get("/api/state")
    async def api_state() -> Any:
        return JSONResponse({"header": header_ctx(), "version": __version__})

    def msg(text: str, level: str) -> HTMLResponse:
        toast(text, level)
        cls = {"success": "ok", "error": "bad", "warn": "warn"}.get(level, "info")
        return HTMLResponse(f'<span class="inline-msg {cls}">{text}</span>')

    # ---------------------------------------------------------------- SSE + pollers
    @app.get("/events")
    async def sse(request: Request) -> StreamingResponse:
        q = bus.subscribe()

        async def gen():  # noqa: ANN202
            try:
                yield "retry: 2000\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev, data = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"event: {ev}\ndata: {data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def poller() -> None:
        last_progress, last_header, last_run = None, None, None
        while True:
            await asyncio.sleep(1.5)
            try:
                d = db()
                r = d.one("SELECT id, status, progress_json FROM runs WHERE env=? ORDER BY id DESC LIMIT 1", (env_name,))
                if r:
                    if r["progress_json"] != last_progress:
                        last_progress = r["progress_json"]
                        bus.publish("pipeline", {})
                    key = (r["id"], r["status"])
                    if key != last_run:
                        if last_run is not None and r["status"] != "running":
                            bus.publish("results", {})
                        last_run = key
                h = json.dumps(header_ctx(), default=str, sort_keys=True)
                if h != last_header:
                    last_header = h
                    bus.publish("header", {})
                if st.restart_pending and not header_ctx()["scan_running"]:
                    bus.publish("toast", {"message": "Restarting…", "level": "info"})
                    await asyncio.sleep(0.5)
                    os.environ["COMBO_EDGE_NO_BROWSER"] = "1"
                    os.execv(sys.executable, [sys.executable, "-m", "combo_edge.cli", *st.argv])
            except Exception:
                pass

    @app.on_event("startup")
    async def _startup() -> None:
        st.poller = asyncio.create_task(poller())
        if env_name != "mock":
            async def sweep() -> None:
                from combo_edge.pipeline.quotes import sweep_pending_cancels

                try:
                    sess, own = await session()
                    await sweep_pending_cancels(sess, db())
                    if own:
                        await sess.close()
                except Exception as e:
                    db().log("warn", "startup", f"startup cancel sweep failed: {e}")
            asyncio.create_task(sweep())

    return app


def _stored_stake(d: Any, candidate_id: int, s: Any) -> float:
    r = d.one("SELECT stake FROM decisions WHERE candidate_id=? AND stake IS NOT NULL ORDER BY id DESC LIMIT 1", (candidate_id,))
    return float(r["stake"]) if r and r["stake"] else float(s.get("min_stake"))


class _StoredCandidate:
    """Enough of a Candidate (rebuilt from the DB) for the executor to decide on a refreshed quote."""

    def __init__(self, d: Any, c: dict):
        from types import SimpleNamespace

        from combo_edge.settings import Settings
        from combo_edge.sports.nfl.fair import lower_bound

        ex = views._j(c["explanation_json"])
        g = d.one("SELECT * FROM games WHERE id=?", (c["game_id"],)) or {"id": c["game_id"], "start_ts": time.time() + 86400}
        u = ex.get("u_base", 0.08)
        fj = c["fair_joint"]
        self.fair = SimpleNamespace(fair_joint=fj, fair_payout=1 / fj, lower=lambda pay, s: lower_bound(fj, u, pay, s),
                                    model_disagrees=list(ex.get("model_disagrees") or []), near_lock=ex.get("near_lock"),
                                    thin_legs=list(ex.get("thin_legs") or []))
        self.game = SimpleNamespace(id=g["id"], start_ts=g["start_ts"])
        legs = d.query("SELECT ticker, player_id FROM legs WHERE ticker IN (?,?)", (c["leg_a"], c["leg_b"]))
        self.players = sorted({l["player_id"] for l in legs if l["player_id"]})
        self.family = c["family"]
        self.db_id = c["id"]
        self.a = SimpleNamespace(wording=ex.get("wording_a", c["leg_a"]), leg=SimpleNamespace(ticker=c["leg_a"]))
        self.b = SimpleNamespace(wording=ex.get("wording_b", c["leg_b"]), leg=SimpleNamespace(ticker=c["leg_b"]))
        self.label = f"{self.a.wording} + {self.b.wording}"
