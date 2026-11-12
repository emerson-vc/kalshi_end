"""Read models for the dashboard pages (pure DB queries, no network)."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from combo_edge import money
from combo_edge.learning.requote import clv_for
from combo_edge.sports.nfl.fair import FAMILY_LABELS, maker_sentence
from combo_edge.storage.db import DB
from combo_edge.timeutil import fmt_kickoff


def _j(s: str | None) -> dict:
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {}


def latest_run(db: DB, env: str, *, finished: bool = False) -> dict | None:
    q = "SELECT * FROM runs WHERE env=? AND status!='refused'" + (" AND status!='running'" if finished else "") + " ORDER BY id DESC LIMIT 1"
    r = db.one(q, (env,))
    if r:
        r["summary"] = _j(r.get("summary_json"))
        r["progress"] = _j(r.get("progress_json"))
    return r


def found_rows(db: DB, env: str, run_id: int | None) -> list[dict]:
    """Every combo the scan quoted. ID-quoted rows first (by edge at the lower bound), then anonymous-only."""
    if run_id is None:
        return []
    cands = db.query(
        "SELECT c.*, g.title game_title, g.start_ts kickoff, g.away_code, g.home_code, g.event_tickers_json FROM candidates c "
        "LEFT JOIN games g ON g.id=c.game_id WHERE c.run_id=? AND c.combo_ticker IS NOT NULL AND c.selected=1", (run_id,))
    if not cands:
        return []
    ids = [c["id"] for c in cands]
    rfqs = db.query(f"SELECT * FROM rfqs WHERE candidate_id IN ({','.join('?' * len(ids))}) AND status!='never_opened' ORDER BY created_ts", ids)
    by_cand: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rfqs:
        by_cand[r["candidate_id"]][r["identity"]] = r  # latest wins
    rfq_ids = [r["id"] for r in rfqs]
    decisions: dict[str, dict] = {}
    best_q: dict[str, dict] = {}
    if rfq_ids:
        ph = ",".join("?" * len(rfq_ids))
        for d in db.query(f"SELECT * FROM decisions WHERE rfq_id IN ({ph}) ORDER BY id", rfq_ids):
            decisions[d["rfq_id"]] = d
        for q in db.query(f"SELECT rfq_id, received_ts, payout FROM quotes WHERE rfq_id IN ({ph}) AND status IN ('open','accepted','executed') ORDER BY payout DESC", rfq_ids):
            best_q.setdefault(q["rfq_id"], q)
    execs = {e["rfq_id"]: e for e in db.query("SELECT rfq_id, status, book FROM executions WHERE env=?", (env,))}
    rows = []
    for c in cands:
        rs = by_cand.get(c["id"], {})
        rid, ran = rs.get("id"), rs.get("anonymous")
        if not rid and not ran:
            continue
        ex = _j(c.get("explanation_json"))
        fair = c["fair_joint"]
        u = ex.get("u_base", 0.08)
        idp = rid["best_payout"] if rid else None
        anp = ran["best_payout"] if ran else None
        pay = idp or anp
        from combo_edge.settings import core_defaults

        cd = core_defaults()
        def lower(p: float) -> float:
            doublings = max(0.0, math.log2(max(p, 1.0) / cd["lower_bound_pivot_payout"]))
            return fair * (1 - min(u + cd["lower_bound_per_doubling"] * doublings, 0.8))
        edge = (fair * pay - 1) if pay else None
        edge_lo = (lower(max(pay, 1 / fair)) * pay - 1) if pay else None
        d = decisions.get(rid["id"]) if rid else None
        dd = _j(d["details_json"]) if d else {}
        e = execs.get(rid["id"]) if rid else None
        chip = dd.get("chip")
        if e and e["status"] in ("filled",):
            chip = "Bought"
        elif e and e["status"] == "paper_filled":
            chip = "Paper bought"
        if rid and not idp:
            chip = chip or "No ID quotes"
        if not rid:
            chip = "Anonymous only"
        bq = best_q.get(rid["id"]) if rid else None
        ratio = (pay / c["leg_product"]) if (pay and c["leg_product"]) else None
        rows.append({
            "candidate_id": c["id"], "combo_ticker": c["combo_ticker"], "leg_a": ex.get("wording_a") or c["leg_a"], "leg_b": ex.get("wording_b") or c["leg_b"],
            "game": c["game_title"], "matchup": f"{c['away_code']} @ {c['home_code']}", "kickoff": c["kickoff"], "kickoff_text": fmt_kickoff(c["kickoff"]) if c["kickoff"] else "",
            "family": c["family"] or "exploration", "family_label": FAMILY_LABELS.get(c["family"] or "", "Exploration"),
            "id_quoted": bool(idp), "id_payout": idp, "anon_payout": anp, "fair_payout": (1 / fair) if fair else None, "edge": edge, "edge_lower": edge_lo,
            "makers": (rid or ran or {}).get("n_makers"), "anon_makers": ran.get("n_makers") if ran else None,
            "quote_ts": bq["received_ts"] if bq else None, "stake": d["stake"] if d else None, "required_ev": d["required_ev"] if d else None,
            "chip": chip or "—", "reason": d["reason"] if d else "", "leg_product": c["leg_product"], "ratio": ratio,
            "explanation": ex, "maker_sentence": maker_sentence(ratio, ex.get("lift", 1.0)), "p_a": c["p_a"], "p_b": c["p_b"],
            "kalshi_url": _game_url(c.get("event_tickers_json")),
        })
    rows.sort(key=lambda r: (not r["id_quoted"], -(r["edge_lower"] if r["edge_lower"] is not None else -9)))
    return rows


def _game_url(evs_json: str | None) -> str:
    try:
        evs = json.loads(evs_json or "[]")
    except json.JSONDecodeError:
        evs = []
    ev = next((t for t in evs if t.startswith("KXNFLGAME-")), None)
    return f"https://kalshi.com/markets/kxnflgame/{ev.lower()}" if ev else "https://kalshi.com/sports/football/nfl"


def bets(db: DB, env: str, book: str | None = None) -> list[dict]:
    q = "SELECT * FROM executions WHERE env=? AND status IN ('filled','paper_filled','pending','maker_no_confirm','side_mismatch')"
    args: list[Any] = [env]
    if book:
        q += " AND book=?"
        args.append(book)
    out = []
    for e in db.query(q + " ORDER BY ts DESC", args):
        legs = _j(e["legs_json"]) if e["legs_json"] else []
        last = db.one("SELECT payout, ts FROM requote_obs WHERE combo_ticker=? AND payout IS NOT NULL ORDER BY ts DESC LIMIT 1", (e["combo_ticker"],))
        clv = clv_for(db, e["combo_ticker"], "id", e["payout"]) if e["payout"] else None
        status = e["result"] or ("open" if e["status"] in ("filled", "paper_filled") else e["status"])
        out.append({**e, "legs": legs, "latest_requote": last["payout"] if last else None, "clv": clv, "status_text": status,
                    "kickoff_text": fmt_kickoff(e["kickoff_ts"]) if e["kickoff_ts"] else ""})
    return out


def exposure_table(db: DB, env: str, book: str) -> dict:
    from combo_edge.execution.executor import exposure

    e = exposure(db, env, book)
    games = {g["id"]: f"{g['away_code']} @ {g['home_code']}" for g in db.query("SELECT id, away_code, home_code FROM games")}
    names = {r["player_id"]: r["player"] for r in db.query("SELECT DISTINCT player_id, player FROM legs WHERE player_id IS NOT NULL")}
    return {"spent_today": e.spent_today,
            "games": sorted(((games.get(k, k), v) for k, v in e.bets_by_game.items()), key=lambda x: -x[1]),
            "players": sorted(((names.get(k, k), v) for k, v in e.bets_by_player.items()), key=lambda x: -x[1])}


def decision_log(db: DB, env: str, limit: int = 200) -> list[dict]:
    rows = db.query("SELECT d.*, c.explanation_json FROM decisions d LEFT JOIN candidates c ON c.id=d.candidate_id WHERE d.env=? ORDER BY d.id DESC LIMIT ?", (env, limit))
    for r in rows:
        ex = _j(r.pop("explanation_json", None))
        r["label"] = f"{ex.get('wording_a', '')} + {ex.get('wording_b', '')}" if ex else r["combo_ticker"]
        r["when"] = datetime.fromtimestamp(r["ts"]).strftime("%a %-I:%M:%S %p")
    return rows


def pnl(db: DB, env: str, book: str) -> dict:
    rows = db.query("SELECT * FROM executions WHERE env=? AND book=? AND status IN ('filled','paper_filled') ORDER BY ts", (env, book))
    settled = [r for r in rows if r["result"] in ("won", "lost")]
    staked = sum(r["stake"] or 0 for r in settled)
    total = sum(r["pnl"] or 0 for r in settled)
    cum, series = 0.0, []
    for r in sorted(settled, key=lambda r: r["settled_ts"] or r["ts"]):
        cum += r["pnl"] or 0
        series.append((r["settled_ts"] or r["ts"], cum))
    by_week: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "staked": 0.0, "n": 0, "clv": []})
    by_type: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "staked": 0.0, "n": 0, "clv": []})
    for r in rows:
        wk = datetime.fromtimestamp(r["ts"]).strftime("%G-W%V")
        fam = FAMILY_LABELS.get(r["family"] or "", "Exploration")
        clv = clv_for(db, r["combo_ticker"], "id", r["payout"]) if r["payout"] else None
        for grp in (by_week[wk], by_type[fam]):
            if clv is not None:
                grp["clv"].append(clv)
            if r["result"] in ("won", "lost"):
                grp["pnl"] += r["pnl"] or 0
                grp["staked"] += r["stake"] or 0
                grp["n"] += 1
    fin = lambda d: [{"key": k, "pnl": v["pnl"], "staked": v["staked"], "n": v["n"], "roi": (v["pnl"] / v["staked"]) if v["staked"] else None,  # noqa: E731
                      "clv": (sum(v["clv"]) / len(v["clv"])) if v["clv"] else None} for k, v in sorted(d.items())]
    wins = sum(1 for r in settled if r["result"] == "won")
    return {"book": book, "n_bets": len(rows), "n_settled": len(settled), "pnl": total, "staked": staked,
            "roi": (total / staked) if staked else None, "win_rate": (wins / len(settled)) if settled else None,
            "expected_win_rate": (sum(r["fair_joint"] or 0 for r in settled) / len(settled)) if settled else None,
            "series": series, "by_week": fin(by_week), "by_type": fin(by_type)}


def recent_errors(db: DB, limit: int = 8) -> list[dict]:
    return db.query("SELECT * FROM events_log WHERE level IN ('error','warn') ORDER BY id DESC LIMIT ?", (limit,))


def fmt(x: Any, kind: str) -> str:
    if x is None:
        return "—"
    if kind == "payout":
        return money.fmt_payout(x)
    if kind == "edge":
        return money.fmt_edge(x)
    if kind == "money":
        return money.fmt_money(x)
    if kind == "pct":
        return f"{x * 100:.0f}%"
    if kind == "ago":
        s = time.time() - x
        return f"{int(s)}s" if s < 120 else f"{int(s // 60)}m"
    return str(x)
