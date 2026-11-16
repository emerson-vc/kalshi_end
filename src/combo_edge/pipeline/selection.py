"""Choose which candidates to keep, look up/create and screen.

1. ``apply_candidate_cap``: every game is generated and priced first, then all candidates are ranked
   together by predicted EV at the lower bound. Core-family candidates are never cut; the remaining
   room goes first to each game's fair share (cap ÷ games), then to the best of the rest. No game can
   be dropped because it kicks off late.
2. ``select``: the budget counts only NEW combo creations.
   * Every core candidate whose combo already exists (ours, imported or seen in the public feed) is
     re-screened every run, free.
   * Each trial family ("trial": true in config) gets its own share of the budget; the core families
     split what is left after exploration and the trials, by family share. Inside a family new combos
     are spread round-robin across games (games furthest from kickoff first; mispricings are bigger days
     out), best predicted EV at the lower bound first. Unused share goes to the next core combos (then
     trial combos), never to exploration.
   * Exploration gets exactly exploration_share of the budget (Thompson sampling over strata,
     promoted strata first) and never more. The payout band is not a filter for core combos.
Each candidate's rank, pool and cutoff are written to ``c.extra`` for the "why did it miss this?" trace.
"""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict, defaultdict
from typing import Any

from combo_edge.sports.nfl.candidates import FAMILIES, TRIAL_FAMILIES, Candidate

EXPLORATION_PAYOUT = (3.0, 150.0)


def is_core(c: Candidate) -> bool:
    """A family candidate (core or trial): never cut by the cap, existing combos re-quoted every run."""
    return (not c.exploration) and c.family in FAMILIES


def is_trial(family: str | None, settings: Any = None) -> bool:
    if family is None:
        return False
    if settings is not None:
        cfg = (settings.get("families") or {}).get(family) or {}
        if "trial" in cfg:
            return bool(cfg["trial"])
    return family in TRIAL_FAMILIES


def apply_candidate_cap(cands: list[Candidate], cap: int) -> tuple[list[Candidate], list[Candidate]]:
    """Returns (kept, cut), both in global rank order. Sets extra["rank_global"] on every candidate and
    extra["cut_reason"] = "candidate_cap" on the cut ones."""
    order = sorted(cands, key=lambda c: -c.pred_ev_lower)
    for i, c in enumerate(order, 1):
        c.extra["rank_global"] = i
    cap = max(0, int(cap))
    core = [c for c in order if is_core(c)]
    rest = [c for c in order if not is_core(c)]
    kept: set[int] = {id(c) for c in core}
    room = max(0, cap - len(core))
    games = list(OrderedDict.fromkeys(c.game.id for c in order))
    if rest and room and games:
        fair = cap // len(games)
        core_n = Counter(c.game.id for c in core)
        by_game: dict[str, list[Candidate]] = defaultdict(list)
        for c in rest:
            by_game[c.game.id].append(c)
        want = {g: min(max(0, fair - core_n[g]), len(by_game[g])) for g in games}
        total = sum(want.values())
        if total > room:  # not enough room for every fair share: scale them down evenly
            want = {g: int(math.floor(v * room / total)) for g, v in want.items()}
        for g in games:
            for c in by_game[g][: want[g]]:
                kept.add(id(c))
        room -= sum(want.values())
        for c in rest:
            if room <= 0:
                break
            if id(c) not in kept:
                kept.add(id(c))
                room -= 1
    out_kept, out_cut = [], []
    for c in order:
        if id(c) in kept:
            out_kept.append(c)
        else:
            c.extra["cut_reason"] = "candidate_cap"
            out_cut.append(c)
    return out_kept, out_cut


def spread_by_game(cands: list[Candidate]) -> list[Candidate]:
    """Best pred_ev_lower first inside each game, then round-robin across games, games furthest from
    kickoff first."""
    by_game: dict[str, list[Candidate]] = defaultdict(list)
    for c in sorted(cands, key=lambda c: -c.pred_ev_lower):
        by_game[c.game.id].append(c)
    games = sorted(by_game, key=lambda g: -(by_game[g][0].game.start_ts or 0))
    out: list[Candidate] = []
    i = 0
    while len(out) < len(cands):
        for g in games:
            if i < len(by_game[g]):
                out.append(by_game[g][i])
        i += 1
    return out


def select(cands: list[Candidate], settings: Any, budget: int, bandit: Any | None = None, rng: random.Random | None = None) -> list[Candidate]:
    """``budget`` = new combo creations allowed this run. Candidates with ``combo_ticker`` already set
    (found by ``ComboResolver.prelookup``) exist and don't use it."""
    rng = rng or random.Random(7)
    fam_cfg = settings.get("families") or {}
    expl_share = float(settings.get("exploration_share"))
    budget = max(0, int(budget))
    exp_budget = int(round(budget * expl_share))
    main_budget = budget - exp_budget
    chosen: list[Candidate] = []

    # 1. existing core combos: re-screened every run
    core = [c for c in cands if is_core(c)]
    for c in core:
        if c.combo_ticker:
            c.extra["sel"] = {"pool": "core_existing", "rank": None, "cutoff": None}
            chosen.append(c)

    # 2. new core combos by family share, spread across games
    creatable = [c for c in core if not c.combo_ticker and c.extra.get("prestatus") != "no_collection"]
    for c in core:
        if not c.combo_ticker and c.extra.get("prestatus") == "no_collection":
            c.extra["sel"] = {"pool": f"core_new:{c.family}", "rank": None, "cutoff": None, "note": "no combo collection lists both legs"}
    ordered = {f: spread_by_game([c for c in creatable if c.family == f]) for f in FAMILIES}
    for f, lst in ordered.items():
        for i, c in enumerate(lst, 1):
            c.extra["sel"] = {"pool": f"core_new:{f}", "rank": i, "cutoff": 0}
    shares = {f: float((fam_cfg.get(f) or {}).get("share", 0.0)) for f in FAMILIES}
    trial = [f for f in FAMILIES if is_trial(f, settings)]
    core_fams = [f for f in FAMILIES if f not in trial]
    quotas = {f: int(math.floor(budget * shares[f])) for f in trial}
    core_budget = max(0, main_budget - sum(quotas.values()))
    core_share = sum(shares[f] for f in core_fams) or 1.0
    quotas.update({f: int(math.floor(core_budget * shares[f] / core_share)) for f in core_fams})
    taken = {f: min(quotas[f], len(ordered[f])) for f in FAMILIES}
    leftover = main_budget - sum(taken.values())
    # unused share: next core combos round-robin, then trial combos (never exploration)
    for group in (core_fams, trial):
        progress = True
        while leftover > 0 and progress:
            progress = False
            for f in group:
                if leftover > 0 and taken[f] < len(ordered[f]):
                    taken[f] += 1
                    leftover -= 1
                    progress = True
    for f in FAMILIES:
        for c in ordered[f]:
            c.extra["sel"]["cutoff"] = taken[f]
        chosen.extend(ordered[f][: taken[f]])

    # 3. exploration: exactly its share, never more
    pool = [c for c in cands if not is_core(c) and EXPLORATION_PAYOUT[0] <= c.pred_id_payout <= EXPLORATION_PAYOUT[1]]
    in_pool = {id(c) for c in pool}
    for c in cands:
        if not is_core(c):
            c.extra["sel"] = {"pool": "exploration", "rank": None, "cutoff": exp_budget,
                              "note": "not sampled" if id(c) in in_pool else
                              f"predicted {c.pred_id_payout:.1f}x is outside the exploration range {EXPLORATION_PAYOUT[0]:g}x–{EXPLORATION_PAYOUT[1]:g}x"}
    strata: dict[str, list[Candidate]] = defaultdict(list)
    for c in pool:
        strata[c.stratum].append(c)
    for v in strata.values():
        rng.shuffle(v)
    promoted = set(bandit.promoted()) if bandit else set()
    n_exp = 0

    def take_exp(c: Candidate) -> None:
        nonlocal n_exp
        n_exp += 1
        c.extra["sel"].update(rank=n_exp, note="sampled")
        chosen.append(c)

    for s in sorted(promoted):
        while n_exp < exp_budget and strata.get(s):
            take_exp(strata[s].pop())
    while n_exp < exp_budget and any(strata.values()):
        live = [s for s, v in strata.items() if v]
        best = max(live, key=lambda s: bandit.sample(s)) if bandit else rng.choice(live)
        take_exp(strata[best].pop())
    return chosen
