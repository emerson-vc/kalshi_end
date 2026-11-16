# Combo Edge v2

Finds and bets +EV **Kalshi NFL same-game combos** priced through RFQs. Makers underprice the link
between a star's tail stat and his own team's outcome; Combo Edge finds those combos, prices them
against a calibrated simulator, and buys them **only on quotes requested under your ID**.

Paper mode is the default. Real mode buys automatically whatever passes the gates.

## Setup (Apple Silicon Mac)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv python install 3.12
uv sync
cp .env.example .env        # then fill in your key id and key path
uv run combo-edge doctor
```

`.env`:

| Variable | Meaning |
|---|---|
| `KALSHI_ENV` | `mock` (offline), `demo` or `prod` |
| `KALSHI_PROD_API_KEY_ID` / `KALSHI_PROD_PRIVATE_KEY_PATH` | the key id **that belongs to** the private key file (RSA-PSS or Ed25519 both work) |
| `KALSHI_DEMO_API_KEY_ID` / `KALSHI_DEMO_PRIVATE_KEY_PATH` | optional demo keys |
| `COMBO_EDGE_FORBID_LIVE=1` | makes Real execution impossible |
| `COMBO_EDGE_LAN_PASSCODE` | passcode for `--lan` phone access |

Secrets never leave `.env`; nothing prints them.

## First run

1. `uv run combo-edge doctor`: keys, websocket, collections and calibration all green.
2. `uv run combo-edge nflverse-sync`: downloads nflverse history and rebuilds the calibration cells (a baseline is packaged, so this is optional but recommended weekly).
3. `uv run combo-edge`: the dashboard opens at http://localhost:8000.
4. Press **Run scan** in **Paper** mode. Look at the Found table: ID-quoted rows first, with Pays / Fair / Edge, the gate each would need, and a **Why** drawer.
5. After a few paper scans, the System page shows **Side mapping verified** (contract counts on quotes match "accept NO = buy YES").
6. Switch to **Real** (one confirm). The first real bet is a **$1 canary**, checked against your fills. Afterwards the Bets page offers the Advanced API tier upgrade.
7. On System, install the background jobs: **recorder** (24/7, read-only), **requote** (re-quote panel + weekly fade check) and **dailyscan** (one scan a day at the learned best time).

## What a scan does

legs → candidates for **every game** (5 exhaustive core families + exploration), ranked together by EV at the
lower bound; the 6,000 cap never cuts core pairs and every game keeps a fair share → selected: **every existing
core combo is re-quoted free**, plus up to 500 **new** ones (85% core by family share, spread across games,
furthest kickoff first; exactly 15% exploration) → looked up / created (local index → feed index → POST; only
POSTs count; 5,000/week cap minus 300 for your phone) → **anonymous screen** (cheap, never bought; only proof
of quoting and a floor on the ID price) → **ID shortlist** (ID asks left today − 10 kept back, ≤60; ranked by
model edge vs the leg prices; 10% random; ≤6 per game, ≤4 per player) → **ID RFQs**, decided while each RFQ
is still open → bought (Real) / booked (Paper).

Fair value: leg probabilities are the de-vigged mid; thin legs (spread > 5¢ or > 40% of the mid, or no ask)
are blended 50/50 with the player's history rate and flagged. P(B | A) comes from an nflverse history table
(2021–current, by role QB / WR1 / WR2 / TE1 / RB1 and rung; team win is spread-aware) whenever the leg has
40+ history games; the simulator only fills thin spots, capped at history + 5%. **Near-locked** pairs (one leg
hits ≥ 93% of the time the other does, e.g. QB 4+ TDs → team over 24.5) are priced at ≈ leg A alone; the card
shows "B hits X% when A hits" and "combo pays Nx vs leg A alone Mx".

Core families (yes/yes, both teams): near-lock player TDs × own team total; QB 3+/4+ TDs or 275+/300+/325+ yards
× own team wins; same-player receptions × rec yards; QB pass TDs × own team total (the rest); QB pass yards ×
top-2 receivers' yards. Trial families (10% of new creations each, P&L shown separately): QB TDs × WR1/TE1 2+
TDs, RB1 100+ rush yards × win, QB 300+ × opponent total over, team win × 1H lead.

Guards: "too good to be true" (EV > +80% with 8+ makers is never auto-bought unless it's a well-documented
near-lock), model vs market > 2x, the ID budget (100/day Real, 40/day Paper, from every source), take-rate tracking, a weekly fade monitor, conditional caps, a joint cap of 6× independence, and a lower bound that widens with payout. The payout band (8x–50x) is only a display filter.

Gates (all on an ID quote): payout-scaled required EV, smooth in log-payout (3x +8%, 20x +18%, 50x +35%, 75x +45%, 100x +55%, 150x +70%, 200x +85%; above 200x +100%), EV > 0 at the lower bound, the family passes its history check, quote ≤ 3 s old, and caps ($5/combo, $30/day, 3/game, 2/player, never the same combo twice).

**Why did it miss this?** Found → "Why did it miss this?" (or `combo-edge trace LEG_A LEG_B [--run N]`, `TICKER:no`
for a No side) shows where a combo stopped: not generated (which rule), cut by the cap, not selected (rank vs
cutoff), creation problems, the anonymous quote, the ID shortlist (rank vs cutoff), the ID quote and the gate.

## CLI

```
combo-edge                        dashboard (--lan, --port, --no-browser)
combo-edge scan [--dry-run] [--budget N] [--max-new N] [--stake $] [--id-shortlist N]
combo-edge trace LEG_A LEG_B [--run N]   where a combo stopped in a scan
combo-edge doctor | limits | upgrade-tier | fit | nflverse-sync
combo-edge kill | resume          kill switch (also: touch data/KILL)
combo-edge recorder | requote-due | fade-check | dailyscan-tick
combo-edge launchd {install,uninstall,status} {recorder,requote,dailyscan}
combo-edge smoke --env demo|prod  tiny RFQs both ways, all cancelled, nothing bought
combo-edge wake {enable,disable}  prints the pmset command (needs sudo)
combo-edge build-css              rebuild Tailwind (standalone CLI, no Node)
```

## Keeping the Mac awake

launchd jobs only run while the Mac is awake. Options:
- `caffeinate -dimsu` in a terminal while you want it running;
- System Settings → Battery → Options → "Prevent automatic sleeping on power adapter when the display is off";
- the optional wake schedule: `combo-edge wake enable` prints the `sudo pmset repeat wakeorpoweron …` command for 10 minutes before the daily scan (on power only).

## Tests

```bash
uv run pytest
```

Network access is blocked inside tests; everything runs against the in-repo mock exchange
(`src/combo_edge/kalshi/mock/`, with fixtures recorded from Kalshi's public API). `tests/test_safety.py`
fails the build if the accept endpoint appears outside `execution/`, or if any resting-order,
order-group, block-trade, batch, quote-create or confirm endpoint appears anywhere.

## Troubleshooting

- **401 `INCORRECT_API_KEY_SIGNATURE`**: the key id doesn't belong to the private key file. Copy the id shown next to that key on kalshi.com → Account → API keys (or point `KALSHI_PROD_PRIVATE_KEY_PATH` at the key that id belongs to).
- **403 on upgrade-tier**: none of your last 100 orders came through the API yet. One real bet (the canary) fixes it.
- **"Restart needed" banner**: the code changed on disk. Click Restart; it waits for any running scan.
- **ID budget used up**: identified RFQs stop until tomorrow. Raise the cap on System only if your take rate is healthy.
- **Kill switch stuck on**: `combo-edge resume` (execution stays off until you turn it on).
- **No candidates for a game** (red in Coverage): its props aren't listed yet, or a new market type isn't parsed. The coverage report lists unparsed series with examples.

## Layout

```
config/core.yaml            every default (dashboard overrides live in the DB)
src/combo_edge/kalshi/      auth, limiter, client (endpoint allowlist), websocket, mock exchange
src/combo_edge/sports/nfl/  parser, ladders, simulator, nflverse calibration, priors, fair model, candidates
src/combo_edge/pipeline/    selection, resolution, quotes (identity), budgets, scan, portfolio
src/combo_edge/execution/   gates, sizing, executor (the only code that can accept)
src/combo_edge/learning/    maker models, bandit, re-quote panel, fade monitor, reports
src/combo_edge/jobs/        recorder, scheduler, launchd, kill switch, smoke test
src/combo_edge/web/         FastAPI + HTMX + Alpine + SSE dashboard
data/                       SQLite DB, logs, nflverse cache (gitignored)
```
