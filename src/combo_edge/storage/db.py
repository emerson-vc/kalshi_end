"""SQLite storage. One file in data/, WAL mode so the dashboard, scans and launchd jobs can share it.

Everything raw is stored (leg snapshots, every RFQ and quote with identity) so that labels
can be recomputed whenever the fair model improves.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from combo_edge import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_ts REAL NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, env TEXT, kind TEXT, started_ts REAL, finished_ts REAL,
  status TEXT, params_json TEXT, progress_json TEXT, summary_json TEXT,
  limits_before_json TEXT, limits_after_json TEXT, error TEXT, pid INTEGER, heartbeat_ts REAL);

CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY, env TEXT, title TEXT, start_ts REAL, away_name TEXT, home_name TEXT,
  away_code TEXT, home_code TEXT, away_team_id TEXT, home_team_id TEXT, season INTEGER, week INTEGER,
  event_tickers_json TEXT, updated_ts REAL);

CREATE TABLE IF NOT EXISTS legs (
  ticker TEXT PRIMARY KEY, env TEXT, game_id TEXT, event_ticker TEXT, series TEXT, stat TEXT,
  team_code TEXT, team_id TEXT, player TEXT, player_id TEXT, line REAL, title TEXT, wording TEXT,
  protected INTEGER, extra_json TEXT, updated_ts REAL);
CREATE INDEX IF NOT EXISTS legs_game ON legs(game_id);

CREATE TABLE IF NOT EXISTS leg_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, ticker TEXT, ts REAL,
  yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL, spread REAL, bid_size REAL, ask_size REAL,
  volume REAL, volume_24h REAL, open_interest REAL, last_price REAL, previous_price REAL,
  move_since_open REAL, game_volume REAL, prob_market REAL, prob_model REAL, prob_fair REAL,
  extrapolated INTEGER, two_sided INTEGER);
CREATE INDEX IF NOT EXISTS legsnap_run ON leg_snapshots(run_id);

CREATE TABLE IF NOT EXISTS combos (
  ticker TEXT PRIMARY KEY, env TEXT, collection TEXT, legs_key TEXT, legs_json TEXT, game_id TEXT,
  source TEXT, created_by_us INTEGER, first_seen_ts REAL, last_seen_ts REAL);
CREATE INDEX IF NOT EXISTS combos_legs ON combos(env, legs_key);

CREATE TABLE IF NOT EXISTS creations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, env TEXT, ts REAL, run_id INTEGER, collection TEXT,
  legs_key TEXT, combo_ticker TEXT, status TEXT, http_status INTEGER, message TEXT);
CREATE INDEX IF NOT EXISTS creations_ts ON creations(env, ts);

CREATE TABLE IF NOT EXISTS rejected_classes (
  class_key TEXT PRIMARY KEY, example_json TEXT, message TEXT, count INTEGER, first_ts REAL, last_ts REAL);

CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, game_id TEXT, legs_key TEXT, family TEXT,
  stratum TEXT, leg_a TEXT, leg_b TEXT, side_a TEXT, side_b TEXT, p_a REAL, p_b REAL,
  fair_joint REAL, fair_lower REAL, leg_payout_a REAL, leg_payout_b REAL, leg_product REAL,
  pred_id_payout REAL, pred_ev REAL, pred_ev_lower REAL, selected INTEGER, exploration INTEGER,
  combo_ticker TEXT, status TEXT, explanation_json TEXT, features_json TEXT);
CREATE INDEX IF NOT EXISTS cand_run ON candidates(run_id);
CREATE INDEX IF NOT EXISTS cand_combo ON candidates(combo_ticker);
CREATE INDEX IF NOT EXISTS cand_legs ON candidates(leg_a, leg_b);

CREATE TABLE IF NOT EXISTS rfqs (
  id TEXT PRIMARY KEY, env TEXT, run_id INTEGER, candidate_id INTEGER, combo_ticker TEXT,
  identity TEXT NOT NULL CHECK (identity IN ('anonymous','id')), purpose TEXT, stake REAL,
  created_ts REAL, hours_to_kickoff REAL, kickoff_ts REAL, status TEXT, outcome TEXT, deleted_ts REAL,
  time_to_first_quote_ms REAL, n_makers INTEGER, best_payout REAL, second_payout REAL,
  best_yes_cost REAL, error TEXT, ws_blinked INTEGER, used_rest INTEGER, feed_demand INTEGER,
  features_json TEXT);
CREATE INDEX IF NOT EXISTS rfqs_created ON rfqs(env, created_ts);
CREATE INDEX IF NOT EXISTS rfqs_combo ON rfqs(combo_ticker);

CREATE TABLE IF NOT EXISTS quotes (
  id TEXT PRIMARY KEY, rfq_id TEXT, identity TEXT NOT NULL CHECK (identity IN ('anonymous','id')),
  maker_id TEXT, yes_bid REAL, no_bid REAL, yes_contracts REAL, no_contracts REAL,
  created_ts REAL, received_ts REAL, latency_ms REAL, source TEXT, status TEXT, yes_cost REAL,
  payout REAL, side_check TEXT, raw_json TEXT);
CREATE INDEX IF NOT EXISTS quotes_rfq ON quotes(rfq_id);

CREATE TABLE IF NOT EXISTS pending_cancels (
  rfq_id TEXT PRIMARY KEY, env TEXT, added_ts REAL, attempts INTEGER DEFAULT 0, last_error TEXT);

CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, ts REAL, env TEXT, mode TEXT, rfq_id TEXT,
  quote_id TEXT, combo_ticker TEXT, candidate_id INTEGER, action TEXT, reason TEXT, payout REAL,
  fair_joint REAL, fair_lower REAL, ev REAL, ev_lower REAL, required_ev REAL, stake REAL,
  quote_age_s REAL, details_json TEXT);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions(ts);

CREATE TABLE IF NOT EXISTS executions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id INTEGER, env TEXT, book TEXT, ts REAL,
  rfq_id TEXT, quote_id TEXT, combo_ticker TEXT, game_id TEXT, players_json TEXT, family TEXT,
  stake REAL, contracts REAL, yes_price REAL, payout REAL, fair_joint REAL, ev REAL, status TEXT,
  order_id TEXT, fee REAL, is_canary INTEGER DEFAULT 0, kickoff_ts REAL, legs_json TEXT,
  result TEXT, pnl REAL, settled_ts REAL, details_json TEXT);
CREATE INDEX IF NOT EXISTS exec_ts ON executions(book, ts);

CREATE TABLE IF NOT EXISTS fills (
  fill_id TEXT PRIMARY KEY, env TEXT, ticker TEXT, order_id TEXT, ts REAL, outcome_side TEXT,
  count REAL, yes_price REAL, no_price REAL, fee REAL, is_taker INTEGER, is_combo INTEGER, raw_json TEXT);
CREATE INDEX IF NOT EXISTS fills_ts ON fills(env, ts);

CREATE TABLE IF NOT EXISTS settlements (
  combo_ticker TEXT PRIMARY KEY, env TEXT, result TEXT, settled_ts REAL, value REAL);

CREATE TABLE IF NOT EXISTS requote_panel (
  combo_ticker TEXT, identity TEXT, env TEXT, game_id TEXT, kickoff_ts REAL, candidate_id INTEGER,
  added_ts REAL, active INTEGER DEFAULT 1, PRIMARY KEY (combo_ticker, identity));

CREATE TABLE IF NOT EXISTS requote_obs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, combo_ticker TEXT, identity TEXT, rfq_id TEXT, bucket_h REAL,
  hours_to_kickoff REAL, ts REAL, payout REAL, n_makers INTEGER, fair_joint REAL);

CREATE TABLE IF NOT EXISTS fade_obs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, env TEXT, ts REAL, combo_ticker TEXT, anon_payout REAL,
  id_payout REAL, leg_product REAL, anon_makers INTEGER, id_makers INTEGER, anon_rfq TEXT, id_rfq TEXT);

CREATE TABLE IF NOT EXISTS feed_demand (
  combo_ticker TEXT, day TEXT, count INTEGER, obscured INTEGER, PRIMARY KEY (combo_ticker, day));

CREATE TABLE IF NOT EXISTS feed_collections (
  collection TEXT, day TEXT, nfl_same_game INTEGER, PRIMARY KEY (collection, day));

CREATE TABLE IF NOT EXISTS bandit_arms (
  arm TEXT PRIMARY KEY, n INTEGER, n_clv INTEGER, sum_r REAL, sum_r2 REAL, promoted INTEGER DEFAULT 0,
  updated_ts REAL);

CREATE TABLE IF NOT EXISTS model_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, metrics_json TEXT);

CREATE TABLE IF NOT EXISTS calibration (
  family TEXT PRIMARY KEY, ts REAL, data_json TEXT);

CREATE TABLE IF NOT EXISTS events_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, source TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events_log(ts);

CREATE TABLE IF NOT EXISTS limits_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, env TEXT, ts REAL, run_id INTEGER, phase TEXT, data_json TEXT);
"""


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def dumps(o: Any) -> str:
    return json.dumps(o, default=_json_default, separators=(",", ":"))


class DB:
    """Thin thread-safe wrapper around sqlite3 with JSON helpers."""

    def __init__(self, path: Path | str | None = None):
        self.path = str(path or paths.db_path())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(SCHEMA)
        self._migrate()

    # columns added after the first release; CREATE TABLE IF NOT EXISTS won't add them to an existing DB
    _ADDED_COLUMNS = (
        ("rfqs", "features_json", "TEXT"),
        # per-candidate trace info ("why did it miss this?")
        ("candidates", "core", "INTEGER"), ("candidates", "rank_global", "INTEGER"), ("candidates", "cut_reason", "TEXT"),
        ("candidates", "prestatus", "TEXT"), ("candidates", "sel_pool", "TEXT"), ("candidates", "sel_rank", "INTEGER"),
        ("candidates", "sel_cutoff", "INTEGER"), ("candidates", "sel_json", "TEXT"), ("candidates", "short_rank", "INTEGER"),
        ("candidates", "short_cutoff", "INTEGER"), ("candidates", "short_reason", "TEXT"), ("candidates", "short_score", "REAL"),
    )

    def _migrate(self) -> None:
        for table, col, typ in self._ADDED_COLUMNS:
            have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # --- raw helpers -------------------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(sql, [tuple(r) for r in rows])
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Iterable[Any] = (), default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]

    def insert(self, table: str, row: dict[str, Any], *, replace: bool = False, ignore: bool = False) -> int:
        cols = list(row)
        verb = "INSERT OR REPLACE" if replace else ("INSERT OR IGNORE" if ignore else "INSERT")
        sql = f"{verb} INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})"
        vals = [dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()]
        with self._lock:
            cur = self._conn.execute(sql, vals)
            return int(cur.lastrowid or 0)

    def update(self, table: str, key_col: str, key: Any, fields: dict[str, Any]) -> None:
        if not fields:
            return
        sets = ",".join(f"{c}=?" for c in fields)
        vals = [dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
        self.execute(f"UPDATE {table} SET {sets} WHERE {key_col}=?", [*vals, key])

    # --- key/value state ---------------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value_json FROM state WHERE key=?", (key,))
        return json.loads(row["value_json"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO state(key,value_json,updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts",
            (key, dumps(value), time.time()),
        )

    def get_setting_override(self, key: str) -> Any:
        row = self.one("SELECT value_json FROM settings WHERE key=?", (key,))
        return json.loads(row["value_json"]) if row else None

    def set_setting_override(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings(key,value_json,updated_ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts",
            (key, dumps(value), time.time()),
        )

    def clear_setting_override(self, key: str) -> None:
        self.execute("DELETE FROM settings WHERE key=?", (key,))

    def log(self, level: str, source: str, message: str) -> None:
        self.execute(
            "INSERT INTO events_log(ts,level,source,message) VALUES(?,?,?,?)",
            (time.time(), level, source, message[:2000]),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_default: DB | None = None
_default_lock = threading.Lock()


def get_db() -> DB:
    global _default
    with _default_lock:
        if _default is None or _default.path != str(paths.db_path()):
            _default = DB()
        return _default


def reset_default_db() -> None:
    global _default
    with _default_lock:
        _default = None
