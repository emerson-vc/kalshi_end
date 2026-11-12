"""Choose which candidates to look up/create and screen.

* Proven families get guaranteed shares of the non-exploration budget (unused share flows on).
* Inside a family: round-robin across players, starting from rungs nearest the middle of the
  payout band (geometric middle of the band).
* Exploration (default 15%) is split across strata by Thompson sampling; promoted strata draw from
  it first.
* Everything except exploration must sit inside the payout band (default 8x–50x).
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict, defaultdict
from typing import Any

from combo_edge.sports.nfl.candidates import FAMILIES, Candidate, band_mid

EXPLORATION_PAYOUT = (3.0, 150.0)


def _round_robin(cands: list[Candidate], mid: float) -> list[Candidate]:
    groups: "OrderedDict[str, list[Candidate]]" = OrderedDict()
    for c in sorted(cands, key=lambda c: abs(math.log(c.pred_id_payout / mid))):
        groups.setdefault(c.group, []).append(c)
    out: list[Candidate] = []
    while groups:
        for g in list(groups):
            out.append(groups[g].pop(0))
            if not groups[g]:
                del groups[g]
    return out


def select(cands: list[Candidate], settings: Any, budget: int, bandit: Any | None = None, rng: random.Random | None = None) -> list[Candidate]:
    rng = rng or random.Random(7)
    lo, hi = (float(x) for x in settings.get("payout_band"))
    mid = band_mid(settings)
    fam_cfg = settings.get("families") or {}
    expl_share = float(settings.get("exploration_share"))
    budget = max(0, int(budget))
    exp_budget = int(round(budget * expl_share))
    main_budget = budget - exp_budget
    in_band = [c for c in cands if not c.exploration and c.family in FAMILIES and lo <= c.pred_id_payout <= hi]
    ordered = {f: _round_robin([c for c in in_band if c.family == f], mid) for f in FAMILIES}
    shares = {f: float((fam_cfg.get(f) or {}).get("share", 0.0)) for f in FAMILIES}
    total_share = sum(shares.values()) or 1.0
    quotas = {f: int(math.floor(main_budget * shares[f] / total_share)) for f in FAMILIES}
    leftover = main_budget - sum(quotas.values())
    chosen: list[Candidate] = []
    taken: dict[str, int] = defaultdict(int)
    for f in FAMILIES:
        take = ordered[f][: quotas[f]]
        chosen.extend(take)
        taken[f] = len(take)
        leftover += quotas[f] - len(take)
    # unused share flows to the next family in order, until nothing is left to place
    progress = True
    while leftover > 0 and progress:
        progress = False
        for f in FAMILIES:
            if leftover <= 0:
                break
            if taken[f] < len(ordered[f]):
                chosen.append(ordered[f][taken[f]])
                taken[f] += 1
                leftover -= 1
                progress = True
    exp_budget += leftover  # nothing more in band: give it to exploration
    picked = {id(c) for c in chosen}
    pool = [
        c for c in cands
        if id(c) not in picked and EXPLORATION_PAYOUT[0] <= c.pred_id_payout <= EXPLORATION_PAYOUT[1]
        and (c.exploration or c.family is None or not (lo <= c.pred_id_payout <= hi))
    ]
    strata: dict[str, list[Candidate]] = defaultdict(list)
    for c in pool:
        strata[c.stratum].append(c)
    for v in strata.values():
        rng.shuffle(v)
    promoted = set(bandit.promoted()) if bandit else set()
    for s in list(promoted):
        while exp_budget > 0 and strata.get(s):
            chosen.append(strata[s].pop())
            exp_budget -= 1
    while exp_budget > 0 and any(strata.values()):
        live = [s for s, v in strata.items() if v]
        if bandit:
            best = max(live, key=lambda s: bandit.sample(s))
        else:
            best = rng.choice(live)
        chosen.append(strata[best].pop())
        exp_budget -= 1
    return chosen
