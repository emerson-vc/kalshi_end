"""History calibration: shift each family's simulated conditional toward nflverse.

For every historical cell (family, thresholds, implied-total bucket) we know the historical
marginals pA, pB and the conditional P(B|A). The simulator is run for a generic game at the bucket's
implied total, the conditional is read at the same thresholds (and moved to the historical B
marginal in log-odds, exactly as pricing does with market marginals), and the shift is
    δ = logit(historical conditional) − logit(simulated conditional),
shrunk by sample size n_A / (n_A + 40). At pricing time δ is a kernel-weighted average of nearby
cells (in logit pA, logit pB and implied total). A family "passes" its check when the shifted
conditionals match history within tolerance and the raw simulator wasn't structurally off.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache

from combo_edge.sports.nfl import simulator as sim
from combo_edge.sports.nfl.nflverse import load_cells, logit

SHRINK_K = 40.0
FAMILY_A_B = {
    # family: (A var, A line from cell key, B var, B line from cell key) for the generic game
    "qb_ptd_team_total": ("pass_td:QB", lambda a: a - 0.5, "pts:H", lambda b: b),
    "rec_td_team_total": ("td:WR1", lambda a: a - 0.5, "pts:H", lambda b: b),
    "qb_pyds_rec_yds": ("rec_yds:WR1", lambda a: a, "pass_yds:QB", lambda b: b),
}


def sim_version() -> str:
    names = [n for n in dir(sim) if n.isupper()]
    blob = json.dumps({n: getattr(sim, n) for n in names}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _generic_game(implied: float) -> tuple[sim.TeamSpec, sim.TeamSpec]:
    home = sim.TeamSpec("H", implied, 0.62, [
        sim.PlayerSpec("QB", "H", "QB", yds_median=235.0),
        sim.PlayerSpec("WR1", "H", "WR", yds_median=68.0, td_share=0.26),
        sim.PlayerSpec("WR2", "H", "WR", yds_median=52.0, td_share=0.18),
        sim.PlayerSpec("RB1", "H", "RB", yds_median=60.0, rec_median=15.0, td_share=0.5, rec_td_share=0.08),
    ])
    away = sim.TeamSpec("A", 22.0, 0.62, [sim.PlayerSpec("QBA", "A", "QB", yds_median=225.0)])
    return home, away


@dataclass
class Shift:
    family: str
    a: float
    b: float
    bucket: float
    n_a: int
    pA: float
    pB: float
    hist: float
    simc: float
    delta: float

    @property
    def shifted(self) -> float:
        return 1 / (1 + math.exp(-(logit(self.simc) + self.delta)))


@lru_cache(maxsize=4)
def compute_shifts(version: str, n_paths: int = 40000) -> tuple[Shift, ...]:
    cells = load_cells()
    out: list[Shift] = []
    by_bucket: dict[float, list[dict]] = {}
    for c in cells:
        by_bucket.setdefault(c["bucket"], []).append(c)
    for bucket, cs in by_bucket.items():
        home, away = _generic_game(float(bucket))
        res = sim.simulate(home, away, n=n_paths, seed=int(bucket * 10))
        for c in cs:
            spec = FAMILY_A_B.get(c["family"])
            if spec is None:
                continue
            va, la, vb, lb = spec
            ia = sim.event(res, va, la(c["a"]))
            ib = sim.event(res, vb, lb(c["b"]))
            if ia is None or ib is None or ia.sum() < 30:
                continue
            simc, _, _ = sim.conditional(ia, ib, c["pB"])
            simc = min(max(simc, 1e-3), 1 - 1e-3)
            w = c["n_a"] / (c["n_a"] + SHRINK_K)
            delta = w * (logit(c["cond"]) - logit(simc))
            out.append(Shift(c["family"], c["a"], c["b"], float(bucket), int(c["n_a"]), c["pA"], c["pB"], c["cond"], simc, delta))
    return tuple(out)


def shifts() -> tuple[Shift, ...]:
    return compute_shifts(sim_version())


def lookup(family: str, pA: float, pB: float, implied: float, a_key: float | None = None, b_line: float | None = None) -> tuple[float, float, float | None]:
    """(δ, effective history n, max nearby historical conditional) for a priced pair.

    Cells with the same A threshold are weighted by distance in the B line (2-point scale) and the
    implied total (4-point scale); only without such cells does a generic kernel in
    (logit pA, logit pB, implied) take over. This keeps a near-certain cell (e.g. o24.5 | 4+ TDs)
    from leaking its large log-odds shift into other lines."""
    rows = [s for s in shifts() if s.family == family]
    if not rows:
        return 0.0, 0.0, None
    same_a = [s for s in rows if a_key is not None and abs(s.a - a_key) < 1e-6] if a_key is not None else []
    ws: list[float] = []
    picked: list[Shift] = []
    if same_a and b_line is not None:
        for s in same_a:
            d2 = ((s.b - b_line) / 2.0) ** 2 + ((s.bucket - implied) / 4.0) ** 2
            w = math.exp(-0.5 * d2)
            if w > 1e-3:
                ws.append(w)
                picked.append(s)
    if not ws:
        la, lb = logit(pA), logit(pB)
        for s in rows:
            d2 = ((logit(s.pA) - la) / 0.5) ** 2 + ((logit(s.pB) - lb) / 0.4) ** 2 + ((s.bucket - implied) / 5.0) ** 2
            w = math.exp(-0.5 * d2)
            if w > 1e-3:
                ws.append(w)
                picked.append(s)
    if not ws:
        return 0.0, 0.0, None
    tot = sum(ws)
    delta = sum(w * s.delta for w, s in zip(ws, picked)) / tot
    top = max(ws)
    near = [(s.hist + 1.64 * math.sqrt(s.hist * (1 - s.hist) / max(s.n_a, 1))) for w, s in zip(ws, picked) if w > 0.3 * top]
    n_eff = sum(w * s.n_a for w, s in zip(ws, picked)) / top
    return delta, n_eff, max(near) if near else None


def family_report() -> dict[str, dict]:
    """Per-family calibration status: error before/after the shift, sample sizes."""
    out: dict[str, dict] = {}
    for fam in ("qb_ptd_team_total", "qb_pyds_rec_yds", "rec_td_team_total", "ml_h1_lead"):
        rows = [s for s in shifts() if s.family == fam]
        if not rows:
            out[fam] = {"status": "no_history", "cells": 0, "n": 0, "note": "no nflverse history for this family; simulator + anchor only"}
            continue
        wsum = sum(s.n_a for s in rows)
        mae_raw = sum(s.n_a * abs(s.simc - s.hist) for s in rows) / wsum
        mae_shift = sum(s.n_a * abs(s.shifted - s.hist) for s in rows) / wsum
        max_delta = max(abs(s.delta) for s in rows)
        ok = mae_shift <= 0.04 and max_delta <= 2.0
        out[fam] = {
            "status": "pass" if ok else "fail", "cells": len(rows), "n": wsum, "mae_raw": round(mae_raw, 4),
            "mae_shifted": round(mae_shift, 4), "max_abs_delta": round(max_delta, 3),
            "examples": [
                {"a": s.a, "b": s.b, "implied": s.bucket, "n": s.n_a, "history": round(s.hist, 3), "sim": round(s.simc, 3), "shifted": round(s.shifted, 3)}
                for s in sorted(rows, key=lambda s: -s.n_a)[:6]
            ],
        }
    return out


def family_passes(family: str | None) -> tuple[bool, str]:
    if family is None:
        return True, "exploration (no family check)"
    rep = family_report().get(family) or {"status": "no_history"}
    st = rep["status"]
    if st == "pass":
        return True, "history check passed"
    if st == "no_history":
        return True, "no history (simulator + anchor)"
    return False, f"history check failed (shifted error {rep.get('mae_shifted')})"
