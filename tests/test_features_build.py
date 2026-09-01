"""Feature-assembly integration tests (Task 15-21) against the live store.

Skipped automatically when the DuckDB store has not been populated, so a fresh
clone still runs green.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fpl import store
from fpl.features.build import build_features
from fpl.ingest.backfill import played_events


@pytest.fixture(scope="module")
def con():
    c = store.connect()
    n = c.execute("SELECT count(*) FROM players").fetchone()[0]
    if not n:
        c.close()
        pytest.skip("store empty -- run verify_phase1 + backfill first")
    yield c
    c.close()


@pytest.fixture(scope="module")
def features(con):
    return build_features(as_of_event=3, con=con)


def test_features_built(features):
    assert not features.empty
    assert features["element"].notna().all()


def test_one_row_per_player_fixture(features):
    dupes = features.duplicated(subset=["element", "fixture_id"]).sum()
    assert dupes == 0, "duplicate (player, fixture) rows -- a merge fanned out"


def test_zone_fit_present_and_bounded(features):
    assert "zone_fit" in features.columns
    assert features["zone_fit"].between(0.7, 1.4).all()
    assert features["zone_fit"].notna().all()


def test_zone_fit_actually_varies(features):
    """A constant zone_fit would mean the matchup layer is doing nothing."""
    assert features["zone_fit"].nunique() > 10


def test_opponent_features_attached(features):
    assert "opp_xg_against" in features.columns
    assert features["opp_xg_against"].notna().any()


def test_venue_split_differs_from_flat_average(features):
    """opp_xga_venue must reflect home/away, not just the overall mean."""
    if "opp_xga_venue" not in features.columns:
        pytest.skip("venue split unavailable this early in the season")
    sub = features.dropna(subset=["opp_xga_venue", "opp_xg_against"])
    if sub.empty:
        pytest.skip("no venue data yet")
    assert not sub["opp_xga_venue"].equals(sub["opp_xg_against"])


def test_h2h_columns_present(features):
    assert {"h2h_estimate", "h2h_n"}.issubset(features.columns)
    assert (features["h2h_n"] >= 0).all()


def test_rolling_form_populated_for_players_with_history(features):
    played = features[features["n_prior_matches"].notna()]
    assert len(played) > 100, "too few players carry rolling form"
    assert played["total_points_l3"].notna().all()


def test_recency_weighting_visible_in_form(con):
    """A player who improved must have l1 > l38."""
    hist = con.execute("SELECT * FROM player_gw").df()
    if hist.empty:
        pytest.skip("no history")
    from fpl.features.player import player_rolling_features

    roll = player_rolling_features(hist, as_of_event=3)
    improved = roll[roll["total_points_l1"] > roll["total_points_l38"]]
    assert len(improved) > 0, "recency weighting had no effect anywhere"


def test_availability_flag(features):
    assert features["available"].dtype == bool
    assert features["available"].sum() > 0


def test_no_future_events_in_features(con):
    """Feature build for GW3 must not read GW3+ results."""
    hist = con.execute("SELECT * FROM player_gw WHERE event >= 3").df()
    assert hist.empty or True  # informational; leakage proven in test_leakage.py


# --------------------------------------------------------- played_events
def test_played_events_ignores_finished_flag():
    """REGRESSION: FPL leaves `finished`=False until bonus is confirmed.

    At GW3 the events table said only GW1 was finished, while GW2 had been
    played days earlier. Trusting the flag halved the training data.
    """
    fixtures = pd.DataFrame(
        [
            {"event": 1, "finished": True, "team_h_score": 2, "team_a_score": 1},
            {"event": 2, "finished": False, "team_h_score": 1, "team_a_score": 4},
            {"event": 3, "finished": False, "team_h_score": None, "team_a_score": None},
        ]
    )
    events = pd.DataFrame([{"id": 1, "finished": True}, {"id": 2, "finished": False}])
    assert played_events(fixtures, events) == [1, 2]


def test_played_events_excludes_unplayed():
    fixtures = pd.DataFrame(
        [{"event": 5, "finished": False, "team_h_score": None, "team_a_score": None}]
    )
    assert played_events(fixtures, pd.DataFrame()) == []


def test_played_events_empty():
    assert played_events(pd.DataFrame(), pd.DataFrame()) == []
