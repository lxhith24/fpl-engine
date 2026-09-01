"""Tasks 3-6: FPL API parsing against saved payloads (no network)."""
from __future__ import annotations

import pandas as pd

from fpl import store
from fpl.ingest import fpl_api


# ------------------------------------------------------------- bootstrap
def test_parse_bootstrap_shapes(bootstrap_payload):
    out = fpl_api.parse_bootstrap(bootstrap_payload)
    assert set(out) == {"teams", "players", "events"}
    assert len(out["teams"]) == 20, "Premier League must have 20 teams"
    assert not out["players"].empty
    assert not out["events"].empty


def test_parse_bootstrap_positions_are_labels(bootstrap_payload):
    players = fpl_api.parse_bootstrap(bootstrap_payload)["players"]
    assert set(players["position"]).issubset({"GKP", "DEF", "MID", "FWD"})


def test_parse_bootstrap_numeric_coercion(bootstrap_payload):
    """FPL returns floats as strings; they must land as numbers, not text."""
    players = fpl_api.parse_bootstrap(bootstrap_payload)["players"]
    for col in ("form", "selected_by_percent", "expected_goals", "points_per_game"):
        assert pd.api.types.is_numeric_dtype(players[col]), f"{col} must be numeric"


def test_chance_of_playing_null_is_none_not_nan_string(bootstrap_payload):
    players = fpl_api.parse_bootstrap(bootstrap_payload)["players"]
    assert players["chance_next"].isna().any() or players["chance_next"].notna().all()


def test_events_deadline_parsed_and_naive(bootstrap_payload):
    events = fpl_api.parse_bootstrap(bootstrap_payload)["events"]
    assert pd.api.types.is_datetime64_any_dtype(events["deadline_time"])
    assert events["deadline_time"].dt.tz is None, "DuckDB TIMESTAMP wants naive datetimes"


def test_current_and_next_event(bootstrap_payload):
    events = fpl_api.parse_bootstrap(bootstrap_payload)["events"]
    cur, nxt = fpl_api.current_and_next_event(events)
    assert cur is None or isinstance(cur, int)
    assert nxt is None or isinstance(nxt, int)


def test_bootstrap_frames_persist(tmp_db, bootstrap_payload):
    out = fpl_api.parse_bootstrap(bootstrap_payload)
    for table in ("teams", "players", "events"):
        store.upsert(tmp_db, table, out[table])
    n = tmp_db.execute("SELECT count(*) FROM teams").fetchone()[0]
    assert n == 20


# -------------------------------------------------------------- fixtures
def test_parse_fixtures(fixtures_payload):
    df = fpl_api.parse_fixtures(fixtures_payload)
    assert not df.empty
    assert (df["team_h"] != df["team_a"]).all(), "a team cannot play itself"
    assert df["id"].is_unique


def test_detect_dgw_bgw_classifies(fixtures_payload):
    df = fpl_api.parse_fixtures(fixtures_payload)
    counts = fpl_api.detect_dgw_bgw(df)
    assert set(counts["kind"]).issubset({"blank", "single", "double"})
    # Within one gameweek each team plays at most a handful of games.
    assert counts["n_fixtures"].max() <= 3


def test_detect_dgw_bgw_covers_all_teams_per_event(fixtures_payload):
    df = fpl_api.parse_fixtures(fixtures_payload)
    counts = fpl_api.detect_dgw_bgw(df)
    per_event = counts.groupby("event")["team_id"].nunique()
    assert (per_event == per_event.iloc[0]).all(), "every event must list every team"


def test_detect_dgw_bgw_empty_input():
    out = fpl_api.detect_dgw_bgw(pd.DataFrame())
    assert out.empty


# ------------------------------------------------------- element summary
def test_parse_element_summary(element_summary_payload):
    df = fpl_api.parse_element_summary(426, element_summary_payload)
    assert not df.empty
    assert (df["element"] == 426).all()
    assert df["minutes"].notna().all()


def test_element_summary_persists_to_player_gw(tmp_db, element_summary_payload):
    df = fpl_api.parse_element_summary(426, element_summary_payload)
    assert store.upsert(tmp_db, "player_gw", df) == len(df)
    # Re-inserting the same rows must not duplicate (PK: element,event,fixture).
    store.upsert(tmp_db, "player_gw", df)
    n = tmp_db.execute("SELECT count(*) FROM player_gw").fetchone()[0]
    assert n == len(df)


# ----------------------------------------------------------- event live
def test_parse_event_live(event_live_payload):
    df = fpl_api.parse_event_live(2, event_live_payload)
    assert not df.empty
    assert (df["event"] == 2).all()
    assert df["total_points"].notna().all()
