# fpl-engine

Fantasy Premier League optimal-XI engine with **opponent-specific matchup modelling**.

Plan: `~/.hermes/plans/2026-09-01_143000-fpl-optimal-xi-engine.md`
Manager under analysis: entry **8905049**.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo skeleton, venv, DuckDB store | **done** |
| 1 | Ingest: FPL API + Understat | **done** |
| 2 | Entity resolution (FPL ↔ Understat name mapping) | not started |
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
./.venv/bin/python -m pytest -q -m "not live"   # 29 offline tests
PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase1   # live pull -> DuckDB
```

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
  exists to map between them; do not assume they interchange.
- FPL returns numeric stats as strings; parsers coerce and tests assert dtypes.
