"""Tasks 7-8: Understat AJAX payload parsing.

Includes a regression guard for the stale-scrape trap: the old
`playersData = JSON.parse('...')` HTML embed no longer exists, so any code
reverting to it would silently yield zero rows.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fpl import store
from fpl.ingest import understat


def test_parse_team_matches_shape(understat_payload):
    df = understat.parse_team_matches(understat_payload, season=2026)
    assert not df.empty
    assert df["team_title"].notna().all()


def test_team_matches_carry_matchup_metrics(understat_payload):
    """xG/xGA/npxG/npxGA/ppda/deep are the matchup layer's fuel (plan 2.1)."""
    df = understat.parse_team_matches(understat_payload, season=2026)
    for col in ("xG", "xGA", "npxG", "npxGA", "ppda_att", "ppda_def", "deep", "deep_allowed"):
        assert col in df.columns, f"missing matchup metric {col}"
        assert df[col].notna().any(), f"{col} is entirely null"


def test_ppda_nested_dict_is_flattened(understat_payload):
    """Understat nests ppda as {att, def}; a naive parse stores dicts."""
    df = understat.parse_team_matches(understat_payload, season=2026)
    assert pd.api.types.is_numeric_dtype(df["ppda_att"])
    assert pd.api.types.is_numeric_dtype(df["ppda_def"])


def test_team_match_xg_values_are_plausible(understat_payload):
    df = understat.parse_team_matches(understat_payload, season=2026)
    assert df["xG"].between(0, 8).all(), "per-match xG outside sane bounds"
    assert df["xGA"].between(0, 8).all()


def test_parse_players_numeric_coercion(understat_payload):
    """Understat returns every stat as a string; they must become numbers."""
    df = understat.parse_players(understat_payload["players"], season=2026)
    assert not df.empty
    for col in ("xG", "xA", "npxG", "xGChain", "xGBuildup"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"
    for col in ("games", "time", "goals", "assists", "shots", "key_passes"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"


def test_players_have_identity_columns(understat_payload):
    df = understat.parse_players(understat_payload["players"], season=2026)
    assert df["understat_id"].notna().all()
    assert df["player_name"].notna().all()
    assert df["understat_id"].is_unique, "understat ids must be unique per season"


def test_team_matches_persist(tmp_db, understat_payload):
    df = understat.parse_team_matches(understat_payload, season=2026)
    store.upsert(tmp_db, "understat_team_match", df)
    store.upsert(tmp_db, "understat_team_match", df)  # idempotent
    n = tmp_db.execute("SELECT count(*) FROM understat_team_match").fetchone()[0]
    assert n == len(df)


def test_players_persist(tmp_db, understat_payload):
    df = understat.parse_players(understat_payload["players"], season=2026)
    store.upsert(tmp_db, "understat_player_season", df)
    n = tmp_db.execute("SELECT count(*) FROM understat_player_season").fetchone()[0]
    assert n == len(df)


def test_empty_players_list_is_safe():
    df = understat.parse_players([], season=2026)
    assert df.empty


@pytest.mark.live
def test_league_page_no_longer_embeds_json():
    """Regression guard: documents WHY we use AJAX, not an HTML scrape.

    If this ever fails, Understat reverted to embedding data in the page and
    the ingest strategy can be simplified.
    """
    import httpx

    from fpl import config

    r = httpx.get(
        f"{config.UNDERSTAT_BASE}/league/EPL/2026",
        headers={"User-Agent": config.USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )
    assert "playersData" not in r.text, "Understat re-added HTML-embedded data"
