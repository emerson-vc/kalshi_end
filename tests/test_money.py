"""Payout, fee and ratio math against the spec's fixtures (section 3)."""

from __future__ import annotations

import math

import pytest

from combo_edge import money

# (combo, leg payouts, combo payout, ratio) — the finds table from the spec
FINDS = [
    ("Stafford 4+ pass TD + LAR o28.5", [5.56, 1.79], 10.00, 1.01),
    ("Darnold 4+ pass TD + SEA o28.5", [15.63, 4.31], 34.60, 0.51),
    ("Stroud 4+ pass TD + HOU o24.5", [13.40, 3.28], 22.83, 0.52),
    ("Lamb 3+ TD + DAL o28.5", [23.42, 1.90], 49.02, 1.10),
    ("Stafford 275+ yds + Nacua 190+ rec yds", [1.73, 18.73], 27.55, 0.85),
    ("Prescott 275+ yds + Lamb 160+ rec yds", [2.09, 13.40], 19.53, 0.70),
    ("TB ML + TB 1H win by 10.5+", [4.51, 11.74], 23.98, 0.45),
    ("UCL 3 legs, separate games (control)", [1.97, 2.09, 1.97], 8.18, 1.01),
]


@pytest.mark.parametrize("name,legs,combo,expected", FINDS)
def test_ratio_reproduces_fixture(name, legs, combo, expected):
    # Inputs are themselves rounded to 2 decimals, so allow half a display unit plus input rounding
    # (Stafford computes to 1.0048 from the displayed payouts; the table shows 1.01).
    assert money.ratio(combo, legs) == pytest.approx(expected, abs=0.006), name


# Leg payouts in the finds table come from whole-cent leg prices: each must be reproduced by
# payout = 1 / (P + 0.07 P (1-P)) at some cent price. 7 of 12 match to the displayed 2 decimals;
# the rest (e.g. 13.40x vs 13.41x at 7c, 18.73x vs 18.75x at 5c) differ only in the app's display
# rounding (it sizes a fixed stake with cent-rounded fees), so the tolerance is half a display unit or 0.13%.
LEG_PAYOUTS = sorted({p for _, legs, _, _ in FINDS[:-1] for p in legs})


@pytest.mark.parametrize("shown", LEG_PAYOUTS)
def test_leg_payouts_reproduce_on_cent_grid(shown):
    best = min((abs(money.payout(c / 100) - shown), c) for c in range(1, 100))
    # half a display unit (0.005), or 0.13% for the larger payouts the app rounds on a staked basis
    tol = max(0.0055, 0.0013 * shown)
    assert best[0] <= tol, f"{shown}x not reproduced (closest {money.payout(best[1] / 100):.3f} at {best[1]}c)"


@pytest.mark.parametrize("shown", [10.00, 34.60, 22.83, 49.02, 27.55, 19.53, 23.98])
def test_combo_payouts_reproduce_on_combo_grid(shown):
    # Combo grid: 0.1c ticks between $0.01 and $0.99. Allow the app's 2-decimal display rounding
    # plus the 0.1c tick granularity (relative error < 0.5%).
    grid = [i / 1000 for i in range(10, 990)]
    best = min(abs(money.payout(p) - shown) / shown for p in grid)
    assert best < 0.005


def test_payout_formula_and_inverse():
    for p in (0.019, 0.054, 0.0939, 0.5, 0.93):
        m = money.payout(p)
        assert m == pytest.approx(1 / (p + 0.07 * p * (1 - p)))
        assert money.price_for_payout(m) == pytest.approx(p, abs=1e-9)


def test_daniels_identity_fixture():
    # ID quote: YES costs 1.9c; anonymous: best NO bid 0.946 so YES costs 5.4c.
    id_payout = money.payout(0.019)
    anon_payout = money.payout(1 - 0.946)
    assert 49.0 <= id_payout <= 49.5  # spec rounds to "49.0x"; the formula gives 49.25x
    assert 17.0 <= anon_payout <= 17.5  # spec says "17.0x"; the formula gives 17.37x
    assert id_payout / anon_payout > 2.5


def test_fee_rules():
    # docs: trade fee rounded up to $0.000001
    assert money.trade_fee(100, 0.5) == pytest.approx(1.75)
    assert money.trade_fee(1, 0.019) == pytest.approx(0.001305)  # 0.0013047 -> ceil 6dp
    # legacy spec rule: ceil to the cent per order
    assert money.fee_cents_rule(100, 0.5) == pytest.approx(1.75)
    assert money.fee_cents_rule(1, 0.019) == pytest.approx(0.01)


def test_contracts_for_target_is_fee_inclusive():
    c = money.contracts_for_target(5.0, 0.019)
    assert money.total_cost(c, 0.019) <= 5.0
    assert money.total_cost(c + 0.01, 0.019) > 5.0
    assert 245 < c < 247
    # principal only
    assert money.contracts_for_target(5.0, 0.05, fee_inclusive=False) == pytest.approx(100.0)


def test_ev_and_kelly():
    assert money.ev(1 / 26, 49.0) == pytest.approx(49 / 26 - 1)
    f = money.kelly_fraction(0.1, 12.0)
    assert f == pytest.approx((0.1 * 12 - 1) / 11)
    assert money.kelly_fraction(0.05, 10.0) == 0.0


def test_price_grid_snapping():
    assert money.snap(0.00537) == pytest.approx(0.0053)
    assert money.snap(0.00537, up=True) == pytest.approx(0.0054)
    assert money.snap(0.0537) == pytest.approx(0.053)
    assert money.tick_for(0.005) == pytest.approx(0.0001)
    assert money.tick_for(0.5) == pytest.approx(0.001)


def test_formatting():
    assert money.fmt_payout(49.25) == "49.2x"
    assert money.fmt_edge(0.88) == "+88%"
    assert money.fmt_edge(-0.12) == "-12%"
    assert money.fmt_payout(math.inf) == "—"
