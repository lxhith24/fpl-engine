"""Config: paths, endpoints, and polite-scraping constants."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/fpl/config.py -> repo)
ROOT = Path(__file__).resolve().parents[2]

DATA = Path(os.environ.get("FPL_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
MODELS = DATA / "models"
REPORTS = DATA / "reports"

DB_PATH = Path(os.environ.get("FPL_DB_PATH", DATA / "fpl.duckdb"))

for _d in (RAW, INTERIM, FEATURES, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- endpoints
FPL_BASE = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP = f"{FPL_BASE}/bootstrap-static/"
FPL_FIXTURES = f"{FPL_BASE}/fixtures/"
FPL_ELEMENT_SUMMARY = f"{FPL_BASE}/element-summary/{{element_id}}/"
FPL_EVENT_LIVE = f"{FPL_BASE}/event/{{event_id}}/live/"
FPL_ENTRY = f"{FPL_BASE}/entry/{{entry_id}}/"
FPL_ENTRY_PICKS = f"{FPL_BASE}/entry/{{entry_id}}/event/{{event_id}}/picks/"

# Understat migrated away from HTML-embedded `JSON.parse(...)` blobs to AJAX
# endpoints. Verified live 2026-09-01 against js/league.min.js.
UNDERSTAT_BASE = "https://understat.com"
UNDERSTAT_LEAGUE_DATA = f"{UNDERSTAT_BASE}/getLeagueData/{{league}}/{{season}}"
UNDERSTAT_PLAYERS_STATS = f"{UNDERSTAT_BASE}/main/getPlayersStats/"

# Manager under analysis (see plan section 9.2).
DEFAULT_ENTRY_ID = 8905049

# Current season for Understat (their API uses numeric years, not "2026-27").
CURRENT_SEASON = 2026

# ---------------------------------------------------------------- politeness
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
FPL_MIN_INTERVAL_S = 0.5
UNDERSTAT_MIN_INTERVAL_S = 1.5
FBREF_MIN_INTERVAL_S = 6.0
HTTP_TIMEOUT_S = 30.0
HTTP_RETRIES = 3

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
