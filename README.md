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
| 3 | Features incl. zone-fit + shrunk head-to-head | not started |
| 4 | Minutes model + component xP models | not started |
| 5 | MILP optimizer (aggressive/differential default) | not started |
| 6 | Reports + deadline-aware cron | not started |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
```

## Verify

```bash
./.venv/bin/python -m pytest -q -m "not live"            # 51 offline tests
PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase1   # live pull -> DuckDB
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
