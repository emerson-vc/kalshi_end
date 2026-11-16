"""Configuration.

* ``.env`` holds the environment choice and secrets (never printed).
* ``config/core.yaml`` holds every default.
* The dashboard stores overrides in the ``settings`` table; ``Settings.get`` resolves override → default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from combo_edge import paths
from combo_edge.storage.db import DB

load_dotenv(paths.REPO_ROOT / ".env", override=False)

ENV_URLS = {
    "prod": ("https://external-api.kalshi.com/trade-api/v2", "wss://external-api-ws.kalshi.com/trade-api/ws/v2"),
    "demo": ("https://external-api.demo.kalshi.co/trade-api/v2", "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"),
    "mock": ("mock://exchange/trade-api/v2", "mock://exchange/trade-api/ws/v2"),
}


@dataclass(frozen=True)
class EnvConfig:
    name: str  # mock | demo | prod
    rest_base: str
    ws_url: str
    key_id: str | None
    key_path: Path | None
    forbid_live: bool

    @property
    def is_live(self) -> bool:
        return self.name in ("demo", "prod")

    @property
    def has_keys(self) -> bool:
        return bool(self.key_id) and bool(self.key_path) and self.key_path.exists()  # type: ignore[union-attr]

    def describe(self) -> str:
        """Safe description: never includes the key id or key contents."""
        return f"{self.name.upper()} ({self.rest_base}); keys {'configured' if self.has_keys else 'missing'}"


def env_config(name: str | None = None) -> EnvConfig:
    name = (name or os.environ.get("KALSHI_ENV") or "mock").lower()
    if name not in ENV_URLS:
        raise ValueError(f"KALSHI_ENV must be mock, demo or prod (got {name!r})")
    rest, ws = ENV_URLS[name]
    up = name.upper()
    key_id = os.environ.get(f"KALSHI_{up}_API_KEY_ID") or None
    kp = os.environ.get(f"KALSHI_{up}_PRIVATE_KEY_PATH")
    if not kp and name in ("prod", "demo"):
        default = Path.home() / ".kalshi" / f"{name}.pem"
        kp = str(default) if default.exists() else None
    forbid = os.environ.get("COMBO_EDGE_FORBID_LIVE", "0").strip() in ("1", "true", "yes")
    return EnvConfig(name, rest, ws, key_id, Path(kp).expanduser() if kp else None, forbid)


@lru_cache(maxsize=1)
def core_defaults() -> dict[str, Any]:
    with open(paths.CORE_CONFIG) as f:
        raw = yaml.safe_load(f)
    flat: dict[str, Any] = {}
    for group, items in raw.items():
        for k, v in items.items():
            flat[k] = v
    return flat


@dataclass(frozen=True)
class SettingMeta:
    key: str
    group: str
    label: str
    help: str
    kind: str = "float"  # float | optfloat | int | bool | text | curve | band | range | families


SETTINGS_META: list[SettingMeta] = [
    # Bankroll and caps
    SettingMeta("bankroll", "Bankroll and caps", "Bankroll", "Your betting bankroll in dollars; sizing uses it for Kelly.", "float"),
    SettingMeta("kelly_fraction", "Bankroll and caps", "Kelly fraction", "Share of full Kelly to bet before the uncertainty shrink.", "float"),
    SettingMeta("min_stake", "Bankroll and caps", "Smallest bet", "Every bet is at least this many dollars.", "float"),
    SettingMeta("max_stake_per_combo", "Bankroll and caps", "Most per combo", "Never stake more than this on one combo.", "float"),
    SettingMeta("daily_spend_cap", "Bankroll and caps", "Daily spend cap", "Total dollars bet per day; buying stops until tomorrow once used.", "float"),
    SettingMeta("max_bets_per_game", "Bankroll and caps", "Bets per game", "At most this many bets on one game.", "int"),
    SettingMeta("max_bets_per_player", "Bankroll and caps", "Bets per player", "At most this many bets involving one player.", "int"),
    SettingMeta("edge_for_max_stake", "Bankroll and caps", "Edge for full stake", "EV above the gate (at the lower bound) that earns the largest bet; weaker edges scale down toward $1.", "float"),
    # Gates
    SettingMeta("payout_curve", "Gates and the payout curve", "Required EV by payout", "Minimum EV after fees at each payout; values in between are interpolated in log-payout, so there are no jumps between points (default: +35% at 50x, +45% at 75x, +85% at 200x).", "curve"),
    SettingMeta("beyond_curve_required_ev", "Gates and the payout curve", "EV above the curve", "EV needed to auto-buy above the curve's last payout (default 200x; 1.0 = +100%).", "float"),
    SettingMeta("max_quote_age_s", "Gates and the payout curve", "Max quote age (s)", "Older ID quotes are re-requested before buying.", "float"),
    SettingMeta("too_good_ev", "Gates and the payout curve", "Too good: EV", "EV above this with many makers is never auto-bought (yellow \"check by hand\").", "float"),
    SettingMeta("too_good_min_makers", "Gates and the payout curve", "Too good: makers", "Makers needed for the too-good-to-be-true block.", "int"),
    SettingMeta("too_good_exempt_min_n", "Gates and the payout curve", "Too good: near-lock n", "Near-locked combos with both legs not thin and at least this many history games are exempt.", "int"),
    SettingMeta("require_history_calibration", "Gates and the payout curve", "Require history check", "If on, families with no nflverse history can't be bought.", "bool"),
    # Budget
    SettingMeta("stake", "Budget", "Screen stake", "Target cost (dollars) used for anonymous screen RFQs.", "float"),
    SettingMeta("creation_budget_per_run", "Budget", "New combos per run", "Most NEW combos created per scan. Combos that already exist (yours or seen in the public feed) are re-quoted without using it.", "int"),
    SettingMeta("max_new_combos_per_run", "Budget", "Hard POST limit per run", "Safety stop on combo creations (POSTs) per scan, enforced at creation time.", "int"),
    SettingMeta("weekly_creation_cap", "Budget", "Weekly creation cap", "Kalshi's limit on combo creations per week.", "int"),
    SettingMeta("weekly_creation_reserve", "Budget", "Reserve for the app", "Creations left unused each week for your phone.", "int"),
    SettingMeta("id_shortlist_per_run", "Budget", "ID shortlist limit", "Most combos per scan that get an RFQ under your ID. Each scan uses the ID asks left today minus the reserve, up to this.", "int"),
    SettingMeta("id_reserve", "Budget", "ID asks kept back", "ID asks per day the scan leaves unused so you can still ask by hand.", "int"),
    SettingMeta("id_random_share", "Budget", "Random ID picks", "Share of the ID shortlist picked at random mid-band so the maker model learns from unbiased data.", "float"),
    SettingMeta("id_asks_per_game_per_run", "Budget", "ID asks per game", "Most ID RFQs per game in one scan (no point asking more than could be bought).", "int"),
    SettingMeta("id_asks_per_player_per_run", "Budget", "ID asks per player", "Most ID RFQs involving one player in one scan.", "int"),
    SettingMeta("id_rfq_daily_cap", "Budget", "ID asks per day (Real)", "Every RFQ under your ID from any source counts.", "int"),
    SettingMeta("id_rfq_daily_cap_paper", "Budget", "ID asks per day (Paper)", "Lower cap in paper mode: makers see you ask and never buy.", "int"),
    SettingMeta("take_rate_warn", "Budget", "Take-rate warning", "Warn when your 7-day take rate falls below this.", "float"),
    SettingMeta("exploration_share", "Budget", "Exploration share", "Exact share of the new-combo budget spent outside the core families; unused core budget never flows into exploration.", "float"),
    SettingMeta("quote_window_s", "Budget", "Quote window (s)", "How long each RFQ waits for maker quotes.", "float"),
    SettingMeta("max_open_rfqs", "Budget", "Open RFQs at once", "Kalshi allows 100; keep headroom for your phone.", "int"),
    SettingMeta("rate_budget_fraction", "Budget", "Rate-limit share", "Fraction of your API token budget the tool uses.", "float"),
    # Search
    SettingMeta("payout_band", "Search", "Payout band", "Range for the random ID picks and exploration sampling only. Not a filter on families, and the Found page shows every payout by default.", "band"),
    SettingMeta("joint_cap_multiple", "Search", "Joint cap", "Fair joint probability never exceeds this multiple of the independent product.", "float"),
    SettingMeta("lower_bound_base", "Search", "Lower-bound base", "Fair lower bound = fair × (1 − this − widening).", "float"),
    SettingMeta("lower_bound_per_doubling", "Search", "Widening per doubling", "Extra lower-bound width per doubling of payout above the pivot.", "float"),
    SettingMeta("lower_bound_pivot_payout", "Search", "Widening pivot", "Payout above which the lower bound starts widening.", "float"),
    SettingMeta("max_candidates_per_run", "Search", "Candidates per run", "Upper limit on pairs kept per scan, applied after ranking all games together by EV at the lower bound. Core-family pairs are never cut; every game keeps a fair share.", "int"),
    SettingMeta("sim_paths", "Search", "Simulated games", "Monte Carlo paths per game for the joint simulator.", "int"),
    SettingMeta("team_total_window", "Search", "Team-total window", "Blank = every team-total rung. A number keeps only rungs within that many points of the main line.", "optfloat"),
    SettingMeta("core_yardage_prob_range", "Search", "Core yardage rungs", "Pass/rec-yards legs in core families use only rungs whose YES price is in this range, e.g. [0.05, 0.35].", "range"),
    SettingMeta("min_receiver_main_line", "Search", "Receiver main line", "QB yards + receiver yards: only receivers whose main rec-yds line is at least this.", "float"),
    SettingMeta("min_rb_main_rush_line", "Search", "RB main rush line", "RB1 100+ rush yards + win (trial): only RBs whose main rush line is at least this.", "float"),
    SettingMeta("history_min_n", "Search", "History minimum n", "Use the nflverse history conditional when the conditioning leg has at least this many games; otherwise the simulator (capped at history + 5%).", "int"),
    SettingMeta("near_lock_threshold", "Search", "Near-lock threshold", "Near-locked when history says one leg hits at least this often when the other does.", "float"),
    SettingMeta("near_lock_team_total_min", "Search", "Near-lock family gate", "Team-total rungs join the near-lock family when history P(total | player leg) is at least this.", "float"),
    SettingMeta("same_player_min", "Search", "Receptions → yards gate", "Same-player receptions + yards rungs need history P(yards | receptions) at least this.", "float"),
    SettingMeta("thin_spread_abs", "Search", "Thin leg: spread", "A leg is thin when its spread is wider than this (dollars).", "float"),
    SettingMeta("thin_spread_rel", "Search", "Thin leg: spread / mid", "... or wider than this share of its mid, or has no ask.", "float"),
    SettingMeta("thin_history_weight", "Search", "Thin leg: history weight", "Thin legs: probability = (1 − w) × mid + w × the player's history base rate.", "float"),
    SettingMeta("families", "Search", "Families", "Core families split the new-combo budget left after exploration and trials; each trial family (\"trial\": true) gets its own share of the whole budget.", "families"),
    # Schedules
    SettingMeta("daily_scan_enabled", "Schedules", "Daily scan", "Run one automatic scan per day once the agent is installed.", "bool"),
    SettingMeta("daily_scan_default_hour", "Schedules", "Daily scan start hour", "Local hour used until the timing data picks a better one.", "int"),
    SettingMeta("requote_id_panel_size", "Schedules", "ID re-quote panel", "Combos re-quoted under your ID at each time-to-kickoff bucket.", "int"),
    SettingMeta("requote_anon_panel_size", "Schedules", "Anonymous re-quote panel", "Combos re-quoted anonymously at each bucket.", "int"),
    SettingMeta("fade_controls", "Schedules", "Fade-check controls", "Control combos quoted both ways once a week.", "int"),
    SettingMeta("wake_enabled", "Schedules", "Wake Mac before scan", "Use pmset to wake the Mac before the daily scan (needs sudo, power).", "bool"),
    SettingMeta("wake_minutes_before", "Schedules", "Wake lead (min)", "Minutes before the daily scan to wake the Mac.", "int"),
]
META_BY_KEY = {m.key: m for m in SETTINGS_META}


class Settings:
    def __init__(self, db: DB, overrides: dict[str, Any] | None = None):
        self.db = db
        self.run_overrides = overrides or {}

    def get(self, key: str) -> Any:
        if key in self.run_overrides and self.run_overrides[key] is not None:
            return self.run_overrides[key]
        v = self.db.get_setting_override(key)
        if v is not None:
            return v
        return core_defaults().get(key)

    def default(self, key: str) -> Any:
        return core_defaults().get(key)

    def f(self, key: str) -> float:
        return float(self.get(key))

    def i(self, key: str) -> int:
        return int(self.get(key))

    def b(self, key: str) -> bool:
        return bool(self.get(key))

    def set(self, key: str, value: Any) -> None:
        if key not in META_BY_KEY:
            raise KeyError(key)
        v = coerce(META_BY_KEY[key].kind, value)
        if v is None:  # blank optional value: back to the default
            self.db.clear_setting_override(key)
            return
        self.db.set_setting_override(key, v)

    def reset(self, key: str) -> None:
        self.db.clear_setting_override(key)

    def with_overrides(self, **kw: Any) -> "Settings":
        return Settings(self.db, {**self.run_overrides, **{k: v for k, v in kw.items() if v is not None}})


def coerce(kind: str, value: Any) -> Any:
    if kind == "float":
        return float(value)
    if kind == "int":
        return int(float(value))
    if kind == "bool":
        if isinstance(value, str):
            return value.lower() in ("1", "true", "on", "yes")
        return bool(value)
    if kind == "optfloat":
        if value is None or str(value).strip().lower() in ("", "none", "null", "off"):
            return None
        return float(value)
    if kind == "range":
        lo, hi = (float(x) for x in value)
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("range must satisfy 0 <= low < high <= 1")
        return [lo, hi]
    if kind == "band":
        lo, hi = value
        lo, hi = float(lo), float(hi)
        if not (1.0 < lo < hi):
            raise ValueError("payout band must satisfy 1 < low < high")
        return [lo, hi]
    if kind == "curve":
        pts = sorted([[float(p), float(e)] for p, e in value])
        if len(pts) < 2 or any(p <= 1 for p, _ in pts):
            raise ValueError("payout curve needs at least two points with payout > 1")
        return pts
    return value
