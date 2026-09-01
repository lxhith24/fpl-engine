"""Task 2: DuckDB store schema, upsert semantics, snapshot round-trip."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from fpl import store


def test_migrate_is_idempotent(tmp_path):
    con = store.connect(tmp_path / "a.duckdb")
    assert store.migrate(con) == store.SCHEMA_VERSION
    store.migrate(con)
    store.migrate(con)
    rows = con.execute("SELECT count(*) FROM schema_meta").fetchone()[0]
    assert rows == 1, "re-migration must not duplicate version rows"
    con.close()


def test_expected_tables_exist(tmp_db):
    names = {r[0] for r in tmp_db.execute("SHOW TABLES").fetchall()}
    for expected in (
        "teams",
        "players",
        "events",
        "fixtures",
        "player_gw",
        "understat_team_match",
        "understat_player_season",
    ):
        assert expected in names


def test_upsert_replaces_by_primary_key(tmp_db):
    now = dt.datetime.now()
    first = pd.DataFrame(
        [{"id": 1, "name": "Arsenal", "short_name": "ARS", "pulled_at": now}]
    )
    assert store.upsert(tmp_db, "teams", first) == 1

    second = pd.DataFrame(
        [{"id": 1, "name": "Arsenal FC", "short_name": "ARS", "pulled_at": now}]
    )
    store.upsert(tmp_db, "teams", second)

    rows = tmp_db.execute("SELECT id, name FROM teams").fetchall()
    assert rows == [(1, "Arsenal FC")], "upsert must replace, not duplicate"


def test_upsert_rejects_unknown_columns(tmp_db):
    bad = pd.DataFrame([{"id": 1, "nonsense_column": 5}])
    with pytest.raises(ValueError, match="unknown columns"):
        store.upsert(tmp_db, "teams", bad)


def test_upsert_empty_frame_is_noop(tmp_db):
    assert store.upsert(tmp_db, "teams", pd.DataFrame()) == 0


def test_snapshot_round_trip():
    payload = {"hello": "world", "n": [1, 2, 3]}
    path = store.snapshot("unit-test", payload)
    try:
        assert path.exists()
        assert store.load_snapshot(path) == payload
    finally:
        path.unlink(missing_ok=True)
