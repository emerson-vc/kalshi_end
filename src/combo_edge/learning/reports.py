"""Insights: edge labels (ID quotes only), timing ("when to press Run"), calibration and take rate.

Edge label = fair_joint × ID payout − 1, recomputable from raw rows whenever the fair model changes.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from combo_edge.storage.db import DB

HOURS_BUCKETS = [(144, "6d+"), (72, "3d"), (24, "1d"), (6, "6h"), (1, "1h"), (-1e9, "<1h")]
PAYOUT_BUCKETS = [(0, 8, "<8x"), (8, 15, "8–15x"), (15, 25, "15–25x"), (25, 50, "25–50x"), (50, 1e9, "50x+")]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def id_edge_rows(db: DB, env: str) -> list[dict]:
    return db.query(
        "SELECT r.created_ts, r.hours_to_kickoff, r.best_payout, r.n_makers, c.fair_joint, c.family, c.stratum, c.features_json "
        "FROM rfqs r JOIN candidates c ON c.id=r.candidate_id WHERE r.env=? AND r.identity='id' AND r.best_payout IS NOT NULL",
        (env,),
    )


def _bucket_hours(h: float | None) -> str:
    if h is None:
        return "?"
    for lo, lbl in HOURS_BUCKETS:
        if h >= lo:
            return lbl
    return "<1h"


def _bucket_payout(p: float) -> str:
    for lo, hi, lbl in PAYOUT_BUCKETS:
        if lo <= p < hi:
            return lbl
    return "?"


def _stats(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "lo": m - 1.96 * se if n > 1 else None, "hi": m + 1.96 * se if n > 1 else None}


def edge_report(db: DB, env: str) -> dict:
    import json

    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in id_edge_rows(db, env):
        e = r["fair_joint"] * r["best_payout"] - 1
        groups["family"][r["family"] or "exploration"].append(e)
        groups["payout"][_bucket_payout(r["best_payout"])].append(e)
        groups["hours"][_bucket_hours(r["hours_to_kickoff"])].append(e)
        groups["dow"][DOW[datetime.fromtimestamp(r["created_ts"]).weekday()]].append(e)
        f = json.loads(r["features_json"] or "{}")
        liq = (f.get("log_vol_a") or 0) + (f.get("log_vol_b") or 0)
        groups["liquidity"]["thin" if liq < 8 else ("medium" if liq < 14 else "liquid")].append(e)
    return {k: {kk: _stats(vv) for kk, vv in v.items()} for k, v in groups.items()}


def when_to_run(db: DB, env: str) -> dict:
    rows = id_edge_rows(db, env)
    if len(rows) < 30:
        return {"text": "Not enough ID quotes yet to say when edge peaks (need 30). Early in the week is the working assumption.", "best_hour": None, "n": len(rows)}
    by_dow: dict[str, list[float]] = defaultdict(list)
    by_hour: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        e = r["fair_joint"] * r["best_payout"] - 1
        d = datetime.fromtimestamp(r["created_ts"])
        by_dow[DOW[d.weekday()]].append(e)
        by_hour[d.hour].append(e)
    dows = sorted(((sum(v) / len(v), k, len(v)) for k, v in by_dow.items() if len(v) >= 5), reverse=True)
    hours = sorted(((sum(v) / len(v), k, len(v)) for k, v in by_hour.items() if len(v) >= 5), reverse=True)
    best_hour = hours[0][1] if hours else None
    if dows:
        top = [d for _, d, _ in dows[:2]]
        text = f"Edge peaks {'–'.join(top)}: {dows[0][0]:+.0%} average on ID quotes (n={dows[0][2]})."
    else:
        text = "Edge by day isn't clear yet."
    return {"text": text, "best_hour": best_hour, "n": len(rows), "by_dow": dows, "by_hour": hours}


def best_scan_hour(db: DB, env: str, default_hour: int) -> int:
    w = when_to_run(db, env)
    return int(w["best_hour"]) if w.get("best_hour") is not None else int(default_hour)


def take_rate_history(db: DB, env: str, weeks: int = 12) -> list[dict]:
    out = []
    now = time.time()
    for w in range(weeks, 0, -1):
        lo, hi = now - w * 7 * 86400, now - (w - 1) * 7 * 86400
        asks = db.scalar("SELECT COUNT(*) FROM rfqs WHERE env=? AND identity='id' AND created_ts>=? AND created_ts<?", (env, lo, hi), 0)
        fills = db.scalar("SELECT COUNT(DISTINCT order_id) FROM fills WHERE env=? AND is_combo=1 AND ts>=? AND ts<?", (env, lo, hi), 0)
        out.append({"week_start": lo, "asks": asks, "fills": fills, "rate": (fills / asks) if asks else None})
    return out


def id_vs_anon_gap(db: DB, env: str) -> dict:
    """Same combo quoted both ways within a run: geometric mean ID/anonymous payout."""
    rows = db.query(
        "SELECT a.best_payout ap, i.best_payout ip FROM rfqs a JOIN rfqs i ON a.candidate_id=i.candidate_id AND a.run_id IS i.run_id "
        "WHERE a.env=? AND a.identity='anonymous' AND i.identity='id' AND a.best_payout>0 AND i.best_payout>0", (env,))
    if not rows:
        return {"n": 0, "gap": None}
    g = math.exp(sum(math.log(r["ip"] / r["ap"]) for r in rows) / len(rows))
    return {"n": len(rows), "gap": g}


def maker_roster(db: DB, env: str) -> list[dict]:
    rows = db.query(
        "SELECT q.maker_id, q.identity, COUNT(DISTINCT q.rfq_id) n FROM quotes q JOIN rfqs r ON r.id=q.rfq_id WHERE r.env=? GROUP BY q.maker_id, q.identity", (env,))
    by: dict[str, dict[str, Any]] = defaultdict(lambda: {"anonymous": 0, "id": 0})
    for r in rows:
        by[r["maker_id"]][r["identity"]] = r["n"]
    return sorted(({"maker": m[:10], **v, "id_only": v["anonymous"] == 0 and v["id"] > 0} for m, v in by.items()), key=lambda x: -(x["id"] + x["anonymous"]))


def truth_report(db: DB, env: str) -> dict:
    """Settled bets: realised win rate vs mean fair joint per family (both books)."""
    rows = db.query("SELECT family, fair_joint, result FROM executions WHERE env=? AND result IN ('won','lost')", (env,))
    by: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        by[r["family"]].append((r["fair_joint"] or 0, 1 if r["result"] == "won" else 0))
    return {f: {"n": len(v), "expected": sum(p for p, _ in v) / len(v), "actual": sum(w for _, w in v) / len(v)} for f, v in by.items()}
