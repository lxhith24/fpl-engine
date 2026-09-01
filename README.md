# fpl-engine

Fantasy Premier League optimal-XI engine with **opponent-specific matchup modelling**.

Plan: `~/.hermes/plans/2026-09-01_143000-fpl-optimal-xi-engine.md`
Manager under analysis: entry **8905049**.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo skeleton, venv, DuckDB store | **done** |
| 1 | Ingest: FPL API + Understat | **done** |
| 2 | Entity resolution (FPL ↔ Understat name mapping) | **done — 100% coverage** |
| 3 | Features incl. zone-fit + shrunk head-to-head | **done — leakage-guarded** |
| 4 | Minutes model + component xP models | **done — gate passed** |
| 5 | MILP optimizer (aggressive/differential default) | **done** |
| 6 | Reports + deadline-aware cron | not started |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
```

## Verify

```bash
./.venv/bin/python -m pytest -q -m "not live"            # 179 offline tests
PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase1   # live pull -> DuckDB
PYTHONPATH=src ./.venv/bin/python -m fpl.ingest.backfill # full player_gw history
PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase2   # resolution + coverage gate
```

`verify_phase2` exits non-zero if resolution coverage drops below 98%, so it
works as a CI gate. Verified in both directions: deliberately corrupting one
team mapping drops coverage to 94.67% and the gate fails as intended.

Offline tests run against saved payloads in `tests/fixtures/` — no network, so
they stay green when sites change. `-m live` selects the two network-dependent
guards.

## Data sources

| Source | Endpoint | Notes |
|---|---|---|
| FPL API | `/bootstrap-static/`, `/fixtures/`, `/element-summary/{id}/`, `/event/{gw}/live/` | Public, no auth. Ground truth for scoring/availability. |
| Understat | `GET /getLeagueData/{league}/{season}`, `POST /main/getPlayersStats/` | AJAX JSON, gzipped. Team `history` carries xG/xGA/npxG/npxGA/ppda/deep. |

### Understat: the stale-scrape trap

Nearly every public guide and most community libraries tell you to scrape
`playersData = JSON.parse('...')` out of the league page HTML. **That is dead.**
The page now returns an identical ~18.7 KB shell for every season (2024, 2025 and
2026 all return byte-identical length) with no embedded data — a naive scraper
gets zero rows and no error.

The data moved to the two AJAX endpoints above, discovered by reading
`js/league.min.js`. Both are gzipped. `tests/test_understat.py` has a `live`-marked
regression guard that fails if Understat ever reverts, so the workaround can be
removed if it becomes unnecessary.

## Layout

```
src/fpl/
  config.py        paths, endpoints, rate limits
  store.py         DuckDB schema, upsert-by-PK, gzipped raw snapshots
  ingest/
    http.py        retry + per-host throttle + disk cache
    fpl_api.py     bootstrap/fixtures/element-summary/event-live, DGW-BGW detection
    understat.py   AJAX league + player stats
  verify_phase1.py end-to-end live check
```

Every raw pull is snapshotted to `data/raw/<timestamp>-<name>.json.gz` so any
gameweek is reproducible and back-testable from source bytes.

## Notes

- Python 3.14.6 — all deps (pandas, duckdb, pulp, scikit-learn, xgboost, httpx,
  rapidfuzz, pyarrow) have working wheels; verified before pinning.
- FPL element IDs and Understat player IDs are **different namespaces**. Phase 2
  maps between them (`player_map` table); do not assume they interchange.
- FPL returns numeric stats as strings; parsers coerce and tests assert dtypes.

## Entity resolution: two silent bugs worth knowing about

Both were found against live data, produced **no error**, and simply dropped a
player. Both are now locked down by regression tests.

**1. `Ø` is not `O` + a diacritic.** It's a distinct letter, so Unicode NFD
decomposition cannot strip it — `unicodedata.normalize("NFD", "Ødegaard")`
returns `Ødegaard` unchanged. Martin Ødegaard therefore never matched Understat's
"Martin Odegaard". `normalize()` now transliterates non-decomposable letters
(ø, đ, ð, ł, ß, æ, œ, þ, ı, ŋ) *before* NFD folding.

**2. `token_set_ratio` scores a strict subset as a perfect 100.** David *Raya
Martín* generates the surname-fragment variant `"martin"`, which scored 100
against "Martin Odegaard", won the greedy assignment, and stole Ødegaard's slot —
so fixing bug 1 alone did not restore coverage. Scoring now uses
`token_sort_ratio`/`WRatio`, and a bare single-token variant is only ever
compared against the target's **surname**.

Design choices that follow from this:

- **Team is a hard constraint.** Candidates are only ever from the same club, so
  a bad name match can't cross clubs. The 20-team map is hand-verified, not
  fuzzy — 8 of 20 names differ (`Tottenham`→`Spurs`, `Nottingham Forest`→
  `Nott'm Forest`, …).
- **Greedy one-to-one assignment** within each club, highest score first. This is
  what disambiguates genuine collisions like A.Murphy / J.Murphy at Newcastle.
- **Understat's `position` field is ignored.** It's mostly `'S'` (substitute — a
  role marker, not a position) and would inject noise rather than signal.
- Coverage is measured over players with **≥90 minutes**; fringe players with no
  appearances are legitimately absent from Understat and shouldn't count against
  the gate.

Current live result: **225/225 = 100%** coverage, all 364 Understat players
mapped, 0 unmatched. The 6 sub-95 scores are Brazilian mononyms and
transliteration variants (Alisson, Jair, Yarmoliuk/Yarmolyuk) — all manually
audited and correct.

## Phase 3: the matchup layer

`build_features(as_of_event=N)` returns one row per (player, fixture) in GW N.
Everything is point-in-time: only data available before GW N's deadline.

### Zone fit — "this winger vs their weak flank" as a number

Understat exposes shot coordinates via `getMatchData/{id}`, split by side — so
a home shot IS an away-team concession. Two maps are built from the same rows:

- **player attack map** — share of xG by zone (penalties excluded, or every
  penalty taker looks identical)
- **opponent concession map** — xG allowed per match by zone, shrunk toward the
  league mean

`zone_fit = Σ_z share[player,z] × weakness[opponent,z]`, clipped to [0.75, 1.35].

**Bin tuning mattered.** The first cut (X=.70/.84, Y=.35/.65) put **80.7% of all
xG in one zone**, making the dot product nearly constant. Retuned to
X=[.78,.88], Y=[.40,.60] → ~62%. Verified against 548 non-penalty shots.

**It discriminates.** Against Ipswich (weak wide: 1.19–1.32× league average in
flank zones), flank shooters get **1.109** vs central shooters' **1.032**.
Against Arsenal both get **0.944** — correctly flat for a uniformly solid
defence.

### Head-to-head is shrunk, and ablatable

`shrunk_head_to_head()` with k=12: a 4-match record moves the estimate only 25%
from baseline. With no history it returns the baseline exactly, so the feature
is a no-op rather than noise. `enabled=False` neutralises it for the plan's
ablation test.

## Two more silent bugs found in Phase 3

**3. `events.finished` lies.** At GW3 the API reported only GW1 finished, while
GW2 had been played days earlier — `finished_provisional=True`, 90 minutes,
scores recorded, but `finished=False` because bonus points weren't confirmed.
Trusting the flag **halved the training set**. `played_events()` uses
fixture-level evidence instead. (Caveat: bonus in a provisional GW can still
shift a point or two, so the latest GW's labels are near-final, not final.)

**4. `verify_phase1` only sampled 5 players.** It was a smoke test, so
`player_gw` held 10 rows. Every rolling feature came back NaN. `ingest.backfill`
now pulls all played gameweeks via `/event/{gw}/live/` — 622 rows, 364 players
with history.

## The leakage guard (Task 22)

`tests/test_leakage.py` is the most important file here. It builds features
twice from datasets identical up to GW N and absurd (9999) afterward, then
asserts byte-identical output.

**Verified by sabotage.** Changing one character in `_history_before` —
`event < as_of_event` to `<=` — makes a rolling mean jump from **6.54 to 1851**
and 4 tests fail immediately. A guard nobody has seen fail is not a guard.

## Phase 4: models

Two stages, trained on **113,592 historical player-gameweeks** (2022-23 → 2025-26).

**Stage A — minutes.** Three-class classifier (didn't play / cameo / started).
Minutes dominate FPL error: a 12-xP player on the bench scores 1. Held-out
2025-26 results: **Brier skill +0.573** over base rate, and well calibrated —
predicted 0.85 → observed 0.86, predicted 0.92 → observed 0.93.

**Stage B — points given play.** Separate GBM+RF ensembles per position, since
a defender's scoring process (clean sheets, DefCon) differs fundamentally from
a forward's. Emits σ(xP) as well as the mean, which the aggressive optimizer
needs for differential and captaincy risk.

### Cold start, solved

The plan flagged GW3 (~600 rows) as the biggest limitation. Fixed by pulling
four seasons of history. Note: `raw.githubusercontent.com` is unreachable from
this machine (curl returns 000, a connection failure — not HTTP); the
**jsDelivr CDN mirror** serves identical bytes and works.

### The gate result — and why the metric changed

Walk-forward over 2025-26 GW8–17 (train on everything strictly earlier).

**The plan's original criterion (`mae_high`, error on players scoring >2) FAILED
— and it should not have been the criterion.** Conditioning error on the
*outcome* selects rows where noise landed positive, so it structurally rewards
over-prediction: a calibrated model loses to a wild one by construction. This
is proven, not asserted, in `test_mae_high_is_biased_toward_overprediction`.

Top-11 realised points was also rejected: 11 rows out of ~750 swings from 2.18
to 5.45 between folds, and the paired t-test gives **p=0.387 vs form** — pure
noise. Gating on it would accept or reject the model at random.

**Gate: Spearman rank correlation over all players.** Low variance (sd 0.03),
and it separates cleanly:

| | Spearman | vs model | p-value |
|---|---|---|---|
| **model** | **0.7489** | — | — |
| form (recency-weighted) | 0.7321 | +0.0168 | **0.0075** |
| last-5 mean | 0.7297 | +0.0192 | **0.0135** |
| price-ranked | 0.4219 | +0.3270 | **<0.001** |

Beats all three, all statistically significant. **Gate passed.**

Honest reading: the edge over simple form is **real but small** (+0.017
Spearman). The model earns its place, but "recency-weighted form" is a strong
baseline and anyone claiming a large edge over it should be doubted.

Raw xP under-predicts hauls by 3.62 points (squared-error regressors shrink
toward the mean on a right-skewed target); isotonic recalibration fitted on
training folds only reduces this to 3.27 without disturbing ranking.

### Live GW3 output

`python -m fpl.verify_phase4` — top xP: B.Fernandes 6.02, N.Williams 5.65,
Szoboszlai 5.64 (zone_fit **1.275**, the matchup layer boosting him).
Differentials <5% owned with xP>4: Collins, Murillo, Ajer, McBurnie, Dedić.

Two more fixes: the backfill was dropping zero-minute rows — the entire
negative class for the minutes model — and `groupby.apply` returns a DataFrame
for a single group, which broke rolling features (now `transform`).

## Phase 5: optimizer

MILP via PuLP/CBC. Squad and starting XI are solved jointly, so the formation
is chosen rather than assumed.

Constraints: 15 players, £100.0m budget, 2/5/5/3 by position, max 3 per club,
XI of 11 with ≥1 GKP / ≥3 DEF / ≥1 FWD. 12 fuzz seeds assert every solve is
FPL-legal.

### Aggressive is the default (your stated preference)

`objective = xP + γ·σ − β·ownership`

**Presets are expressed in units of spread, not raw coefficients.** The first
version hard-coded γ=0.5, β=0.02 and all three risk modes returned nearly
identical squads. Measured on live GW3 (219 players with xP>2):

| quantity | spread | note |
|---|---|---|
| xP | 4.02 | |
| σ | 0.66 | and corr(xP, σ) = **0.56** |
| ownership | 70.0 | |

γ=0.5 shifted scores by only 0.33 — under a tenth of the xP spread, so it
never reordered anything. Presets now say "tilt by this fraction of the xP
spread" and are rescaled per gameweek.

Worth knowing: **σ correlates 0.56 with xP**, so variance-seeking partly just
re-picks high-xP players. Ownership is the sharper differential lever, which is
why aggressive leans on it harder.

Live GW3 result — the modes now genuinely separate:

| mode | XI xP | σ | mean own% |
|---|---|---|---|
| safe | 57.70 | 32.50 | 18.5 |
| balanced | 57.89 | 32.66 | 18.2 |
| **aggressive** | **57.06** | **33.19** | **15.0** |

Aggressive gives up **0.83 xP** to cut ownership by 3.5 points. That is the
trade being bought, stated explicitly rather than hidden.

### The differential tax was being charged twice

The XI choice originally used the same ownership-penalised score as squad
selection. On live GW3 that benched **Szoboszlai (5.64 xP, 42% owned)** in
favour of **Ajer (4.92 xP, 4.5% owned)** — giving away 0.73 xP for nothing,
since benching a player you already own makes you no more differential.

Fixed with two scores: ownership decides **who you own**; once owned, the XI is
picked on **merit alone**. Formation moved 5-2-3 → 4-3-3 and XI xP rose
57.12 → 57.84. Locked by `test_no_benched_outfielder_beats_a_worse_starter`.

### Your squad vs the optimum

| | XI xP |
|---|---|
| your best legal XI | 42.66 |
| optimum | 57.06 |
| **gap** | **+14.40** |

13 transfers, −48 in hits. As predicted in the plan, this is a **Wildcard
target, not a one-week move** — one transfer captures a small fraction of it.

## Phase 6: reports + cron

### Daily briefing

`python -m fpl report` generates a markdown briefing:

- Optimal XI with formation, captain/vice picks
- Gap analysis vs your current squad
- **Single best transfer** (not Wildcard rebuild)
- Differentials (<5% owned, xP > 4)
- Zone-fit matchup highlights

### Cron job

`fpl-daily-briefing` — runs 09:00 IST daily. Checks if GW deadline is within
24h; if so, refreshes data and outputs the briefing. Otherwise silent.

**Note:** requires `hermes gateway start` to fire. Currently scheduled, not
active until the gateway runs.

### CLI

```bash
python -m fpl refresh     # full data refresh
python -m fpl predict     # generate predictions
python -m fpl optimize    # run optimizer
python -m fpl report      # full briefing
```
