"""Token-bucket rate limiting sized from GET /account/limits and /account/endpoint_costs.

Kalshi meters Read and Write buckets separately; RFQ and quote writes bill the shard-1 Write bucket
(which carries the full tier budget). The tool runs at a fraction (default 60%) of each budget.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    rate: float  # tokens per second
    capacity: float
    tokens: float = field(default=-1.0)
    updated: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens < 0:
            self.tokens = self.capacity
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def wait_time(self, cost: float, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        self._refill(now)
        if self.tokens >= cost:
            return 0.0
        return (cost - self.tokens) / self.rate

    def try_take(self, cost: float, now: float | None = None) -> bool:
        if self.wait_time(cost, now) == 0.0:
            self.tokens -= cost
            return True
        return False

    async def acquire(self, cost: float) -> float:
        """Wait until ``cost`` tokens are available, take them, return seconds waited."""
        cost = min(cost, self.capacity)
        waited = 0.0
        async with self._lock:
            while True:
                w = self.wait_time(cost)
                if w <= 0:
                    self.tokens -= cost
                    return waited
                await asyncio.sleep(w)
                waited += w

    def reconfigure(self, rate: float, capacity: float) -> None:
        self.rate = max(rate, 0.1)
        self.capacity = max(capacity, 1.0)
        self.tokens = min(self.tokens, self.capacity)


# Paths billed to Write per the rate-limit docs (RFQ/quote flow → shard-1 Write bucket).
_WRITE_RFQ = re.compile(r"^/communications/rfqs")


def bucket_for(method: str, path: str) -> str:
    m = method.upper()
    if m != "GET" and _WRITE_RFQ.match(path):
        return "write"
    return "read"


class Limiter:
    def __init__(self, read_rate: float = 200, write_rate: float = 100, fraction: float = 0.6):
        self.fraction = fraction
        self.read = TokenBucket(read_rate * fraction, read_rate * fraction)
        self.write = TokenBucket(write_rate * fraction, write_rate * fraction)
        self.default_cost = 10
        self.costs: dict[tuple[str, str], int] = {}
        self.tier = "unknown"

    def configure(self, limits: dict | None, endpoint_costs: dict | None) -> None:
        if limits:
            self.tier = limits.get("usage_tier", self.tier)
            r = limits.get("read") or {}
            w = limits.get("write") or {}
            if r:
                self.read.reconfigure(r["refill_rate"] * self.fraction, r["bucket_capacity"] * self.fraction)
            if w:
                self.write.reconfigure(w["refill_rate"] * self.fraction, w["bucket_capacity"] * self.fraction)
        if endpoint_costs:
            self.default_cost = int(endpoint_costs.get("default_cost", 10))
            for e in endpoint_costs.get("endpoint_costs", []):
                self.costs[(e["method"].upper(), _normalize(e["path"]))] = int(e["cost"])

    def cost_for(self, method: str, path: str) -> int:
        norm = _normalize(path)
        key = (method.upper(), norm)
        if key in self.costs:
            return self.costs[key]
        for (m, p), c in self.costs.items():
            if m == method.upper() and _match(p, norm):
                return c
        return self.default_cost

    async def acquire(self, method: str, path: str) -> float:
        bucket = self.write if bucket_for(method, path) == "write" else self.read
        return await bucket.acquire(self.cost_for(method, path))

    def snapshot(self) -> dict:
        return {
            "tier": self.tier,
            "fraction": self.fraction,
            "read": {"rate": round(self.read.rate, 1), "capacity": round(self.read.capacity, 1), "tokens": round(self.read.tokens, 1)},
            "write": {"rate": round(self.write.rate, 1), "capacity": round(self.write.capacity, 1), "tokens": round(self.write.tokens, 1)},
        }


def _normalize(path: str) -> str:
    p = path.split("?")[0]
    p = re.sub(r"^/trade-api/v2", "", p)
    return p.rstrip("/") or "/"


def _match(pattern: str, path: str) -> bool:
    rx = "^" + re.sub(r"\\\{[^}]+\\\}", r"[^/]+", re.escape(pattern)) + "$"
    return re.match(rx, path) is not None
