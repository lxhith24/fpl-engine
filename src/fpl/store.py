"""DuckDB store: schema migration + typed frame round-tripping.

Every raw pull also lands on disk as a timestamped JSON snapshot (plan Task 11)
so any gameweek can be re-run and back-tested from source bytes.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from . import config

SCHEMA_VERSION = 1

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        version     INTEGER NOT NULL,
        applied_at  TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS teams (
        id           INTEGER PRIMARY KEY,
        name         VARCHAR NOT NULL,
        short_name   VARCHAR NOT NULL,
        strength     INTEGER,
        strength_attack_home   INTEGER,
        strength_attack_away   INTEGER,
        strength_defence_home  INTEGER,
        strength_defence_away  INTEGER,
        pulled_at    TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS players (
        id            INTEGER PRIMARY KEY,
        web_name      VARCHAR NOT NULL,
        first_name    VARCHAR,
        second_name   VARCHAR,
        team_id       INTEGER NOT NULL,
        position      VARCHAR NOT NULL,
        now_cost      INTEGER NOT NULL,
        status        VARCHAR,
        chance_next   DOUBLE,
        minutes       INTEGER,
        total_points  INTEGER,
        form          DOUBLE,
        points_per_game DOUBLE,
        selected_by_percent DOUBLE,
        expected_goals          DOUBLE,
        expected_assists        DOUBLE,
        expected_goal_involvements DOUBLE,
        expected_goals_conceded DOUBLE,
        defensive_contribution  DOUBLE,
        starts        INTEGER,
        penalties_order INTEGER,
        corners_order   INTEGER,
        news          VARCHAR,
        pulled_at     TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id            INTEGER PRIMARY KEY,
        name          VARCHAR,
        deadline_time TIMESTAMP,
        is_current    BOOLEAN,
        is_next       BOOLEAN,
        finished      BOOLEAN,
        pulled_at     TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fixtures (
        id            INTEGER PRIMARY KEY,
        event         INTEGER,
        kickoff_time  TIMESTAMP,
        team_h        INTEGER NOT NULL,
        team_a        INTEGER NOT NULL,
        team_h_score  INTEGER,
        team_a_score  INTEGER,
        team_h_difficulty INTEGER,
        team_a_difficulty INTEGER,
        finished      BOOLEAN,
        pulled_at     TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_gw (
        element       INTEGER NOT NULL,
        event         INTEGER NOT NULL,
        fixture       INTEGER,
        opponent_team INTEGER,
        was_home      BOOLEAN,
        minutes       INTEGER,
        total_points  INTEGER,
        goals_scored  INTEGER,
        assists       INTEGER,
        clean_sheets  INTEGER,
        goals_conceded INTEGER,
        saves         INTEGER,
        bonus         INTEGER,
        bps           INTEGER,
        expected_goals   DOUBLE,
        expected_assists DOUBLE,
        expected_goals_conceded DOUBLE,
        defensive_contribution  DOUBLE,
        starts        INTEGER,
        value         INTEGER,
        pulled_at     TIMESTAMP NOT NULL,
        PRIMARY KEY (element, event, fixture)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS understat_team_match (
        team_title  VARCHAR NOT NULL,
        match_date  DATE NOT NULL,
        h_a         VARCHAR,
        xG          DOUBLE,
        xGA         DOUBLE,
        npxG        DOUBLE,
        npxGA       DOUBLE,
        scored      INTEGER,
        missed      INTEGER,
        ppda_att    INTEGER,
        ppda_def    INTEGER,
        ppda_allowed_att INTEGER,
        ppda_allowed_def INTEGER,
        deep        INTEGER,
        deep_allowed INTEGER,
        pts         INTEGER,
        season      INTEGER NOT NULL,
        pulled_at   TIMESTAMP NOT NULL,
        PRIMARY KEY (team_title, match_date, season)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS understat_player_season (
        understat_id VARCHAR NOT NULL,
        player_name  VARCHAR NOT NULL,
        team_title   VARCHAR,
        position     VARCHAR,
        games        INTEGER,
        time         INTEGER,
        goals        INTEGER,
        assists      INTEGER,
        shots        INTEGER,
        key_passes   INTEGER,
        xG           DOUBLE,
        xA           DOUBLE,
        npxG         DOUBLE,
        npg          INTEGER,
        xGChain      DOUBLE,
        xGBuildup    DOUBLE,
        season       INTEGER NOT NULL,
        pulled_at    TIMESTAMP NOT NULL,
        PRIMARY KEY (understat_id, season)
    )
    """,
]


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open (and migrate) the DuckDB store."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    migrate(con)
    return con


def migrate(con: duckdb.DuckDBPyConnection) -> int:
    """Apply DDL idempotently and record the schema version."""
    for ddl in _DDL:
        con.execute(ddl)
    current = con.execute("SELECT max(version) FROM schema_meta").fetchone()[0]
    if current is None or current < SCHEMA_VERSION:
        con.execute(
            "INSERT INTO schema_meta (version, applied_at) VALUES (?, ?)",
            [SCHEMA_VERSION, _dt.datetime.now()],
        )
    return SCHEMA_VERSION


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Replace-by-primary-key insert of a DataFrame into `table`.

    DuckDB has no MERGE, so we delete colliding keys then append. Column order
    is taken from the live table so the caller need not match DDL ordering.
    """
    if df.empty:
        return 0

    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    missing = [c for c in df.columns if c not in cols]
    if missing:
        raise ValueError(f"{table}: unknown columns {missing}")

    frame = df.copy()
    for c in cols:
        if c not in frame.columns:
            frame[c] = None
    frame = frame[cols]

    keys = _primary_key(con, table)
    con.register("_incoming", frame)
    try:
        if keys:
            pred = " AND ".join(f"t.{k} = i.{k}" for k in keys)
            con.execute(
                f"DELETE FROM {table} t WHERE EXISTS "
                f"(SELECT 1 FROM _incoming i WHERE {pred})"
            )
        con.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")
    return len(frame)


def _primary_key(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    rows = con.execute(
        """
        SELECT constraint_column_names
        FROM duckdb_constraints()
        WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'
        """,
        [table],
    ).fetchall()
    return list(rows[0][0]) if rows else []


def snapshot(name: str, payload: Any, *, when: _dt.datetime | None = None) -> Path:
    """Persist a raw API payload as timestamped gzipped JSON (reproducibility)."""
    when = when or _dt.datetime.now()
    stamp = when.strftime("%Y%m%d_%H%M%S")
    path = config.RAW / f"{stamp}-{name}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def load_snapshot(path: Path | str) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
