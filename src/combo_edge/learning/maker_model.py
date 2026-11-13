"""Maker models (scikit-learn HistGradientBoosting; no LightGBM on this Mac).

The ID model predicts log(ID payout ÷ leg product) — the ratio — for a candidate. It is retrained
every run from identified quotes only, weights recent data more (21-day half-life), and reports
holdout accuracy and drift. A separate anonymous model is used only for the screen. Learning never
mixes identities.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from typing import Any

import numpy as np

from combo_edge.storage.db import DB

FEATURES = [
    "family_code", "stat_a_code", "stat_b_code", "line_a", "line_b", "p_a", "p_b", "fair_joint", "log_leg_product",
    "implied", "abs_margin", "hours_to_kickoff", "dow", "spread_a", "spread_b", "log_vol_a", "log_vol_b",
    "synthetic_legs", "existed", "feed_demand", "anon_log_ratio", "anon_makers", "same_team", "side_a_yes", "side_b_yes",
]
FAMILY_CODES = {"qb_ptd_team_total": 1, "qb_pyds_rec_yds": 2, "rec_td_team_total": 3, "ml_h1_lead": 4}
STATS = ["ml", "spread", "game_total", "team_total", "pass_td", "pass_yds", "rec_yds", "rush_yds", "receptions", "td",
         "h1_winner", "h1_spread", "h1_team_total", "h1_total", "rush_rec_yds", "win_margin", "pass_att", "pass_comp", "rush_att"]
MIN_ROWS = 30
HALF_LIFE_DAYS = 21.0


def candidate_features(c: Any, *, now: float | None = None) -> dict:
    now = now or time.time()
    g = c.game
    implied = g.env.implied(c.a.leg.team or c.b.leg.team, g) if g.env else None
    return {
        "family_code": FAMILY_CODES.get(c.family or "", 0),
        "stat_a_code": STATS.index(c.a.leg.stat) if c.a.leg.stat in STATS else -1,
        "stat_b_code": STATS.index(c.b.leg.stat) if c.b.leg.stat in STATS else -1,
        "line_a": c.a.leg.line, "line_b": c.b.leg.line, "p_a": c.a.p, "p_b": c.b.p,
        "fair_joint": c.fair.fair_joint if c.fair else None, "log_leg_product": math.log(max(c.leg_product, 1.0001)),
        "implied": implied, "abs_margin": abs(g.env.home_margin) if g.env else None,
        "hours_to_kickoff": (g.start_ts - now) / 3600, "dow": datetime.fromtimestamp(now).weekday(),
        "spread_a": c.a.leg.spread, "spread_b": c.b.leg.spread,
        "log_vol_a": math.log1p(c.a.leg.volume or 0), "log_vol_b": math.log1p(c.b.leg.volume or 0),
        "synthetic_legs": c.synthetic_legs, "existed": None, "feed_demand": None, "anon_log_ratio": None, "anon_makers": None,
        "same_team": int(bool(c.a.leg.team and c.a.leg.team == c.b.leg.team)),
        "side_a_yes": int(c.a.side == "yes"), "side_b_yes": int(c.b.side == "yes"),
    }


COMPLETED_SCREEN = ("quoted", "no_quotes_timeout")


def anon_maker_count(o: Any) -> int | None:
    """Distinct makers that quoted the anonymous screen at any point (same count as rfqs.n_makers), or None
    when the screen didn't run its full window."""
    if o is None or getattr(o, "outcome", None) not in COMPLETED_SCREEN:
        return None
    st = getattr(o, "state", None)
    if st is not None:
        return len({q.maker_id for q in st.quotes.values()})
    return int(o.n_makers)


def quote_features(c: Any, *, feed_demand: int | None = None, anon: Any = None) -> dict:
    """The candidate's features plus the four that are only known once the scan reaches the quote stage:
    whether the combo already existed, its public-feed frequency, and (for an ID quote) the anonymous screen.

    `anon` is the anonymous RfqOutcome for the same combo, or None when no screen ran. A value that is genuinely
    unknown stays None (blank): anon_log_ratio when the screen got no quotes (anon_makers is then 0), both when
    no screen ran or it was aborted.
    """
    f = dict(c.extra.get("features") or candidate_features(c))
    existed = c.extra.get("existed")
    f["existed"] = None if existed is None else int(bool(existed))
    f["feed_demand"] = feed_demand if feed_demand is not None else c.extra.get("feed_demand")
    # only a screen that ran its full window tells us anything; an aborted or failed one stays blank
    if anon is not None and getattr(anon, "outcome", None) in COMPLETED_SCREEN:
        pay = anon.best_payout
        f["anon_log_ratio"] = math.log(pay / max(c.leg_product, 1.0001)) if pay else None
        f["anon_makers"] = anon_maker_count(anon)
    return f


def _vec(f: dict, cols: list[str] = FEATURES) -> list[float]:
    return [np.nan if f.get(k) is None else float(f[k]) for k in cols]


def usable_columns(X: np.ndarray, names: list[str] = FEATURES) -> list[int]:
    """Indices of columns worth training on: not entirely blank, and more than one distinct non-blank value.
    HistGradientBoosting crashes on an all-NaN column, and a constant column carries no information."""
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        vals = col[~np.isnan(col)]
        if vals.size and np.unique(vals).size > 1:
            keep.append(j)
    return keep


class MakerModel:
    def __init__(self, db: DB, identity: str = "id"):
        assert identity in ("id", "anonymous")
        self.db = db
        self.identity = identity
        self.model: Any = None
        self.columns: list[str] = []
        self.metrics: dict = {"rows": 0, "fitted": False, "status": "not_fitted"}

    def training_rows(self) -> list[tuple[dict, float, float]]:
        # features saved on the RFQ at quote time win; older RFQs fall back to the candidate's features
        rows = self.db.query(
            "SELECT r.best_payout, r.created_ts, COALESCE(r.features_json, c.features_json) AS features_json, c.leg_product "
            "FROM rfqs r JOIN candidates c ON c.id=r.candidate_id "
            "WHERE r.identity=? AND r.best_payout IS NOT NULL AND c.leg_product>0 "
            "AND COALESCE(r.features_json, c.features_json) IS NOT NULL",
            (self.identity,),
        )
        out = []
        for r in rows:
            f = json.loads(r["features_json"])
            out.append((f, math.log(r["best_payout"] / r["leg_product"]), r["created_ts"]))
        return out

    def _not_enough(self, reason: str) -> dict:
        self.model, self.columns = None, []
        self.metrics.update(fitted=False, status="not_enough_data", reason=reason)
        return self.metrics

    def fit(self) -> dict:
        rows = self.training_rows()
        self.model, self.columns = None, []
        self.metrics = {"rows": len(rows), "fitted": False, "status": "not_fitted", "identity": self.identity}
        if len(rows) < MIN_ROWS:
            return self._not_enough(f"{len(rows)} quoted rows, needs {MIN_ROWS}")
        rows.sort(key=lambda r: r[2])
        X = np.array([_vec(f) for f, _, _ in rows], dtype=float)
        y = np.array([t for _, t, _ in rows], dtype=float)
        ts = np.array([r[2] for r in rows], dtype=float)
        ok = np.isfinite(y) & np.isfinite(ts)
        X, y, ts = X[ok], y[ok], ts[ok]
        self.metrics["usable_rows"] = int(len(y))
        if len(y) < MIN_ROWS:
            return self._not_enough(f"{len(y)} usable rows, needs {MIN_ROWS}")
        # a regressor's "one outcome class": every target identical, nothing to learn
        if np.unique(y).size < 2:
            return self._not_enough("every target value is the same")
        keep = usable_columns(X)
        self.metrics["dropped_columns"] = [FEATURES[j] for j in range(len(FEATURES)) if j not in keep]
        if not keep:
            return self._not_enough("no feature column has more than one distinct value")
        X = X[:, keep]
        cols = [FEATURES[j] for j in keep]
        from sklearn.ensemble import HistGradientBoostingRegressor

        w = np.power(0.5, (ts.max() - ts) / (HALF_LIFE_DAYS * 86400))
        split = int(len(y) * 0.8)

        def new_model() -> Any:
            return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=8, random_state=0)

        # holdout only when both sides are big enough and the training part still has usable columns
        if len(y) - split >= 8 and len(usable_columns(X[:split], cols)) == len(cols):
            hm = new_model()
            hm.fit(X[:split], y[:split], sample_weight=w[:split])
            pred = hm.predict(X[split:])
            self.metrics["holdout_mae_log_ratio"] = float(np.mean(np.abs(pred - y[split:])))
            self.metrics["baseline_mae"] = float(np.mean(np.abs(np.mean(y[:split]) - y[split:])))
        m = new_model()
        m.fit(X, y, sample_weight=w)
        recent = ts >= ts.max() - 7 * 86400
        resid = y - m.predict(X)
        self.metrics["drift_7d_mean_residual"] = float(np.mean(resid[recent])) if recent.any() else None
        self.metrics.update(fitted=True, status="fitted", columns=cols)
        self.model, self.columns = m, cols
        self.db.insert("model_runs", {"ts": time.time(), "kind": f"maker_{self.identity}", "metrics_json": self.metrics})
        return self.metrics

    def predict_log_ratio(self, f: dict) -> float | None:
        if self.model is None or not self.columns:
            return None
        return float(self.model.predict(np.array([_vec(f, self.columns)]))[0])

    def predict_id_payout(self, c: Any, anon_payout: float | None = None, anon_makers: int | None = None) -> tuple[float, str]:
        from combo_edge.sports.nfl.candidates import prior_id_payout

        f = quote_features(c)
        if anon_payout:
            f["anon_log_ratio"] = math.log(anon_payout / max(c.leg_product, 1.0001))
        if anon_makers is not None:
            f["anon_makers"] = anon_makers
        lr = self.predict_log_ratio(f)
        if lr is None:
            base = prior_id_payout(c)
            # the anonymous quote is a probable lower bound on the ID price
            return (max(base, anon_payout) if anon_payout else base), "prior"
        pay = max(1.01, c.leg_product * math.exp(lr))
        return (max(pay, anon_payout) if anon_payout else pay), "model"
