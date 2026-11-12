"""Thompson-sampling bandit over strata (exploration pair types and families).

Reward = EV at the ID quote + CLV, never win/loss (far too noisy at 10x–50x payouts).
Until an arm has 10 CLV observations, the model-EV part is shrunk by half so the bandit doesn't
just learn from its own fair model. Arms that keep showing a large fair-vs-quote gap on ID quotes
are promoted into families (reported in the run summary).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from combo_edge.storage.db import DB

PRIOR_N = 3.0
PRIOR_MEAN = 0.0
REWARD_SD = 0.6
PROMOTE_MIN_N = 15
PROMOTE_MIN_MEAN = 0.25


@dataclass
class Arm:
    arm: str
    n: int = 0
    n_clv: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    promoted: int = 0

    @property
    def mean(self) -> float:
        return (self.sum_r + PRIOR_MEAN * PRIOR_N) / (self.n + PRIOR_N)

    @property
    def sd_mean(self) -> float:
        return REWARD_SD / math.sqrt(self.n + PRIOR_N)

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.sd_mean)


class Bandit:
    def __init__(self, db: DB, seed: int | None = None):
        self.db = db
        self.rng = random.Random(seed)
        self.arms: dict[str, Arm] = {}
        for r in db.query("SELECT * FROM bandit_arms"):
            self.arms[r["arm"]] = Arm(r["arm"], r["n"] or 0, r["n_clv"] or 0, r["sum_r"] or 0.0, r["sum_r2"] or 0.0, r["promoted"] or 0)

    def arm(self, name: str) -> Arm:
        if name not in self.arms:
            self.arms[name] = Arm(name)
        return self.arms[name]

    def sample(self, name: str) -> float:
        return self.arm(name).sample(self.rng)

    def observe(self, name: str, ev_at_id_quote: float | None, clv: float | None = None) -> None:
        a = self.arm(name)
        model_part = ev_at_id_quote or 0.0
        if a.n_clv < 10:
            model_part *= 0.5
        r = model_part + (clv or 0.0)
        r = max(min(r, 3.0), -1.0)
        a.n += 1
        a.sum_r += r
        a.sum_r2 += r * r
        if clv is not None:
            a.n_clv += 1
        self._save(a)

    def _save(self, a: Arm) -> None:
        self.db.execute(
            "INSERT INTO bandit_arms(arm,n,n_clv,sum_r,sum_r2,promoted,updated_ts) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(arm) DO UPDATE SET n=excluded.n,n_clv=excluded.n_clv,sum_r=excluded.sum_r,"
            "sum_r2=excluded.sum_r2,promoted=excluded.promoted,updated_ts=excluded.updated_ts",
            (a.arm, a.n, a.n_clv, a.sum_r, a.sum_r2, a.promoted, time.time()),
        )

    def promotions(self, gaps: dict[str, list[float]]) -> list[str]:
        """Promote exploration strata with a persistent large fair-vs-ID-quote gap."""
        new: list[str] = []
        for stratum, xs in gaps.items():
            if len(xs) < PROMOTE_MIN_N:
                continue
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 1.0
            lo = m - 1.64 * sd / math.sqrt(len(xs))
            a = self.arm(stratum)
            if m >= PROMOTE_MIN_MEAN and lo > 0.1 and not a.promoted:
                a.promoted = 1
                self._save(a)
                new.append(stratum)
        return new

    def promoted(self) -> list[str]:
        return [a.arm for a in self.arms.values() if a.promoted]

    def table(self) -> list[dict]:
        return sorted(
            ({"arm": a.arm, "n": a.n, "n_clv": a.n_clv, "mean": round(a.mean, 3), "promoted": bool(a.promoted)} for a in self.arms.values()),
            key=lambda r: -r["mean"],
        )
