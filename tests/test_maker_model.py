"""Maker model: training survives blank/constant feature columns, skips cleanly on too little data, and
the four quote-stage features (existed, feed demand, anonymous ratio, anonymous makers) are saved."""

from __future__ import annotations

import math
import random
import time
from types import SimpleNamespace

import pytest

from combo_edge.learning.maker_model import FEATURES, MIN_ROWS, MakerModel, quote_features
from combo_edge.storage.db import dumps

QUOTE_STAGE = ["existed", "feed_demand", "anon_log_ratio", "anon_makers"]


def _features(rng: random.Random, *, blank_quote_stage: bool) -> dict:
    f = {k: rng.uniform(0, 10) for k in FEATURES}
    f.update(family_code=rng.choice([1, 2, 3]), dow=6, synthetic_legs=0)  # dow and synthetic_legs are constant
    if blank_quote_stage:
        f.update({k: None for k in QUOTE_STAGE})
    else:
        f.update(existed=rng.choice([0, 1]), feed_demand=rng.randint(0, 20),
                 anon_log_ratio=rng.uniform(-2.0, -0.5), anon_makers=rng.randint(0, 3))
    return f


def _seed(db, n: int, *, blank_quote_stage: bool = True, on_rfq: bool = False, constant_target: bool = False, seed: int = 1) -> None:
    rng = random.Random(seed)
    now = time.time()
    for i in range(n):
        f = _features(rng, blank_quote_stage=blank_quote_stage)
        leg_product = 20.0
        payout = 15.0 if constant_target else leg_product * math.exp(-0.3 * f["family_code"] + rng.gauss(0, 0.05))
        cid = db.insert("candidates", {"run_id": 1, "game_id": "g", "leg_product": leg_product,
                                       "features_json": None if on_rfq else dumps(f)})
        db.insert("rfqs", {"id": f"r{i}", "env": "mock", "candidate_id": cid, "combo_ticker": f"C{i}", "identity": "id",
                           "created_ts": now - (n - i) * 3600, "best_payout": payout, "status": "closed",
                           "features_json": dumps(f) if on_rfq else None})


def test_all_blank_columns_are_dropped_not_crashing(db):
    # this exact layout (4 always-NaN columns, 30+ rows) used to raise
    # "ValueError: window shape cannot be larger than input array shape"
    _seed(db, 40, blank_quote_stage=True)
    m = MakerModel(db, "id")
    metrics = m.fit()
    assert metrics["fitted"] and metrics["status"] == "fitted"
    for col in QUOTE_STAGE + ["dow", "synthetic_legs"]:
        assert col in metrics["dropped_columns"]
        assert col not in m.columns
    assert m.columns == metrics["columns"] and "family_code" in m.columns
    # predict uses the same kept columns, and tolerates the dropped ones being present or missing
    f = _features(random.Random(9), blank_quote_stage=False)
    a = m.predict_log_ratio(f)
    b = m.predict_log_ratio({k: v for k, v in f.items() if k in m.columns})
    assert a is not None and a == b


def test_single_row_returns_not_enough_data(db):
    _seed(db, 1)
    m = MakerModel(db, "id")
    metrics = m.fit()
    assert metrics == {**metrics, "rows": 1, "fitted": False, "status": "not_enough_data"}
    assert m.model is None and m.columns == []
    assert m.predict_log_ratio(_features(random.Random(2), blank_quote_stage=True)) is None


def test_every_target_identical_returns_not_enough_data(db):
    _seed(db, MIN_ROWS + 5, constant_target=True)
    metrics = MakerModel(db, "id").fit()
    assert metrics["status"] == "not_enough_data" and not metrics["fitted"]


def test_empty_db_returns_not_enough_data(db):
    assert MakerModel(db, "id").fit()["status"] == "not_enough_data"


def test_four_quote_features_filled_in_are_used(db):
    _seed(db, 40, blank_quote_stage=False, on_rfq=True)  # saved on the RFQ, candidate row has none
    m = MakerModel(db, "id")
    metrics = m.fit()
    assert metrics["fitted"]
    for col in QUOTE_STAGE:
        assert col in m.columns, col
        assert col not in metrics["dropped_columns"]


def _cand(**extra):
    return SimpleNamespace(leg_product=20.0, extra={"features": {k: 1.0 for k in FEATURES} | {k: None for k in QUOTE_STAGE}, **extra})


def test_quote_features_fills_the_four_values():
    anon = SimpleNamespace(outcome="quoted", best_payout=5.0, n_makers=2, rfq_id="a1")
    f = quote_features(_cand(existed=True), feed_demand=7, anon=anon)
    assert f["existed"] == 1 and f["feed_demand"] == 7 and f["anon_makers"] == 2
    assert f["anon_log_ratio"] == pytest.approx(math.log(5.0 / 20.0))


def test_quote_features_leaves_genuinely_unknown_blank():
    # anonymous screen ran but nobody quoted: makers is a known 0, the ratio is unknown
    f = quote_features(_cand(existed=False), feed_demand=0, anon=SimpleNamespace(outcome="no_quotes_timeout", best_payout=None, n_makers=0))
    assert f["existed"] == 0 and f["feed_demand"] == 0 and f["anon_makers"] == 0 and f["anon_log_ratio"] is None
    # no screen / aborted screen / never resolved: all unknown
    for anon in (None, SimpleNamespace(outcome="aborted", best_payout=None, n_makers=0)):
        f = quote_features(_cand(), anon=anon)
        assert all(f[k] is None for k in QUOTE_STAGE)


def test_db_migration_adds_features_column(tmp_path):
    import sqlite3

    from combo_edge.storage.db import DB

    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE rfqs (id TEXT PRIMARY KEY, env TEXT, run_id INTEGER, candidate_id INTEGER, combo_ticker TEXT, "
                "identity TEXT NOT NULL CHECK (identity IN ('anonymous','id')), created_ts REAL, status TEXT, feed_demand INTEGER)")
    con.commit()
    con.close()
    db = DB(p)
    assert "features_json" in {r["name"] for r in db.query("PRAGMA table_info(rfqs)")}
    db.close()
