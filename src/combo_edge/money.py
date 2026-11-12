"""Payout, fee and ratio math.

Kalshi's taker fee model is 0.07 × C × P × (1 − P). Per the live docs (Fee Rounding), the trade fee
is rounded *up* to $0.000001 and a direct member's balance is aligned to $0.0001; the spec's
"ceil to the cent per order" is the older rule, and it is kept as ``fee_cents_rule`` for reference.

Payouts are always shown fee-inclusive, the way the Kalshi app shows them:
    payout = 1 / (P + 0.07·P·(1 − P))
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Iterable, Sequence

FEE_RATE = 0.07


def fee_per_contract(p: float) -> float:
    return FEE_RATE * p * (1.0 - p)


def payout(p: float) -> float:
    """Fee-inclusive payout multiple for buying YES at price ``p`` (dollars)."""
    if p <= 0 or p >= 1:
        return float("inf") if p <= 0 else 1.0
    return 1.0 / (p + fee_per_contract(p))


def price_for_payout(m: float) -> float:
    """Inverse of :func:`payout` (solves p + 0.07 p (1-p) = 1/m for p in (0,1))."""
    c = 1.0 / m
    # 0.07 p^2 - 1.07 p + c = 0
    a, b = FEE_RATE, -(1.0 + FEE_RATE)
    disc = b * b - 4 * a * c
    return (-b - math.sqrt(disc)) / (2 * a)


def trade_fee(contracts: float, p: float) -> float:
    """Trade fee for one fill per the docs: ceil to $0.000001."""
    raw = Decimal(str(FEE_RATE)) * Decimal(str(contracts)) * Decimal(str(p)) * (1 - Decimal(str(p)))
    return float(raw.quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def fee_cents_rule(contracts: float, p: float) -> float:
    """Spec's legacy formula: ceil(0.07 × C × P × (1−P)) in cents."""
    return math.ceil(round(FEE_RATE * contracts * p * (1 - p) * 100, 9)) / 100.0


def total_cost(contracts: float, p: float) -> float:
    """Debit for buying ``contracts`` at ``p``: principal + fee, aligned up to $0.0001 (direct member)."""
    raw = Decimal(str(contracts)) * Decimal(str(p)) + Decimal(str(trade_fee(contracts, p)))
    return float(raw.quantize(Decimal("0.0001"), rounding=ROUND_CEILING))


def contracts_for_target(target_cost: float, p: float, *, fee_inclusive: bool = True) -> float:
    """Largest contract count (0.01 steps) whose total cost fits inside ``target_cost``."""
    if p <= 0:
        return 0.0
    per = p + (fee_per_contract(p) if fee_inclusive else 0.0)
    c = float((Decimal(str(target_cost)) / Decimal(str(per))).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))
    if not fee_inclusive:
        return c
    while c > 0 and total_cost(c, p) > target_cost + 1e-9:
        c = round(c - 0.01, 2)
    return max(c, 0.0)


def ev(prob: float, payout_mult: float) -> float:
    """Expected value per dollar staked (fee-inclusive payout)."""
    return prob * payout_mult - 1.0


def ratio(combo_payout: float, leg_payouts: Iterable[float]) -> float:
    prod = 1.0
    for x in leg_payouts:
        prod *= x
    return combo_payout / prod


def leg_product(leg_payouts: Iterable[float]) -> float:
    prod = 1.0
    for x in leg_payouts:
        prod *= x
    return prod


def kelly_fraction(prob: float, payout_mult: float) -> float:
    """Full-Kelly fraction for a bet paying ``payout_mult`` per $1 (net odds payout-1)."""
    b = payout_mult - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (prob * payout_mult - 1.0) / b)


# --- price grid ------------------------------------------------------------------------

DEFAULT_COMBO_RANGES = [
    {"start": "0.0000", "end": "0.0100", "step": "0.0001"},
    {"start": "0.0100", "end": "0.9900", "step": "0.0010"},
    {"start": "0.9900", "end": "1.0000", "step": "0.0001"},
]


def tick_for(p: float, ranges: Sequence[dict] | None = None) -> float:
    for r in ranges or DEFAULT_COMBO_RANGES:
        if float(r["start"]) <= p < float(r["end"]) or (p == 1.0 and float(r["end"]) == 1.0):
            return float(r["step"])
    return 0.01


def snap(p: float, ranges: Sequence[dict] | None = None, *, up: bool = False) -> float:
    step = Decimal(str(tick_for(p, ranges)))
    q = (Decimal(str(p)) / step).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)
    return float(q * step)


def fmt_payout(m: float | None) -> str:
    if m is None or not math.isfinite(m):
        return "—"
    return f"{m:.1f}x" if m < 100 else f"{m:.0f}x"


def fmt_edge(e: float | None) -> str:
    if e is None or not math.isfinite(e):
        return "—"
    return f"{e * 100:+.0f}%"


def fmt_money(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"
