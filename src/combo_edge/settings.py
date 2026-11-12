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
    kind: str = "float"  # float | int | bool | text | curve | band | families


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
    SettingMeta("payout_curve", "Gates and the payout curve", "Required EV by payout", "Minimum EV after fees at each payout; values in between are interpolated in log-payout.", "curve"),
    SettingMeta("longshot_payout", "Gates and the payout curve", "Longshot payout", "Above this payout a much higher EV is required.", "float"),
    SettingMeta("longshot_required_ev", "Gates and the payout curve", "Longshot required EV", "EV needed to auto-buy above the longshot payout (1.0 = +100%).", "float"),
    SettingMeta("max_quote_age_s", "Gates and the payout curve", "Max quote age (s)", "Older ID quotes are re-requested before buying.", "float"),
    SettingMeta("require_history_calibration", "Gates and the payout curve", "Require history check", "If on, families with no nflverse history can't be bought.", "bool"),
    # Budget
    SettingMeta("stake", "Budget", "Screen stake", "Target cost (dollars) used for anonymous screen RFQs.", "float"),
    SettingMeta("creation_budget_per_run", "Budget", "Combos per run", "Most combos looked up or created per scan.", "int"),
    SettingMeta("max_new_combos_per_run", "Budget", "New combos per run", "Most brand-new combo creations (POSTs) per scan.", "int"),
    SettingMeta("weekly_creation_cap", "Budget", "Weekly creation cap", "Kalshi's limit on combo creations per week.", "int"),
    SettingMeta("weekly_creation_reserve", "Budget", "Reserve for the app", "Creations left unused each week for your phone.", "int"),
    SettingMeta("id_shortlist_per_run", "Budget", "ID shortlist size", "How many combos per scan get an RFQ under your ID.", "int"),
    SettingMeta("id_random_share", "Budget", "Random ID picks", "Share of the ID shortlist picked at random mid-band so the maker model learns from unbiased data.", "float"),
    SettingMeta("id_rfq_daily_cap", "Budget", "ID asks per day (Real)", "Every RFQ under your ID from any source counts.", "int"),
    SettingMeta("id_rfq_daily_cap_paper", "Budget", "ID asks per day (Paper)", "Lower cap in paper mode: makers see you ask and never buy.", "int"),
    SettingMeta("take_rate_warn", "Budget", "Take-rate warning", "Warn when your 7-day take rate falls below this.", "float"),
    SettingMeta("exploration_share", "Budget", "Exploration share", "Share of the creation budget spent outside the proven families.", "float"),
    SettingMeta("quote_window_s", "Budget", "Quote window (s)", "How long each RFQ waits for maker quotes.", "float"),
    SettingMeta("max_open_rfqs", "Budget", "Open RFQs at once", "Kalshi allows 100; keep headroom for your phone.", "int"),
    SettingMeta("rate_budget_fraction", "Budget", "Rate-limit share", "Fraction of your API token budget the tool uses.", "float"),
    # Search
    SettingMeta("payout_band", "Search", "Payout band", "Combos outside this payout range only get exploration budget.", "band"),
    SettingMeta("joint_cap_multiple", "Search", "Joint cap", "Fair joint probability never exceeds this multiple of the independent product.", "float"),
    SettingMeta("lower_bound_base", "Search", "Lower-bound base", "Fair lower bound = fair × (1 − this − widening).", "float"),
    SettingMeta("lower_bound_per_doubling", "Search", "Widening per doubling", "Extra lower-bound width per doubling of payout above the pivot.", "float"),
    SettingMeta("lower_bound_pivot_payout", "Search", "Widening pivot", "Payout above which the lower bound starts widening.", "float"),
    SettingMeta("max_candidates_per_run", "Search", "Candidates per run", "Upper limit on pairs priced per scan.", "int"),
    SettingMeta("sim_paths", "Search", "Simulated games", "Monte Carlo paths per game for the joint simulator.", "int"),
    SettingMeta("team_total_window", "Search", "Team-total window", "Team-total rungs within this many points of the main line.", "float"),
    SettingMeta("families", "Search", "Proven families", "Guaranteed budget shares for the proven families.", "families"),
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
        self.db.set_setting_override(key, coerce(META_BY_KEY[key].kind, value))

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
