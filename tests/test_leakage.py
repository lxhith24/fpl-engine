"""Task 22 — POINT-IN-TIME LEAKAGE GUARD.

The single most important test in this project.

A model trained on data that did not exist before the deadline will look
excellent in back-testing and fail in production. The bug is invisible in
ordinary output: features look sane, metrics look great, and only live results
reveal it.

Method: build features twice from datasets that are IDENTICAL up to the target
gameweek and differ wildly from the target gameweek onward. Every
point-in-time feature must produce byte-identical output. Any difference means
information flowed backwards in time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.features import player as pf


TARGET_EVENT = 5


def _base_history() -> pd.DataFrame:
    """Plausible history for two players over gameweeks 1-4."""
    rows = []
    for element in (1, 2):
        for event in range(1, TARGET_EVENT):
            rows.append(
                {
                    "element": element,
                    "event": event,
                    "fixture": event * 10 + element,
                    "opponent_team": (event % 3) + 1,
                    "was_home": event % 2 == 0,
                    "minutes": 90,
                    "total_points": 4 + event,
                    "goals_scored": event % 2,
                    "assists": 0,
                    "bps": 20 + event,
                    "expected_goals": 0.3,
                    "expected_assists": 0.2,
                    "expected_goals_conceded": 1.1,
                    "defensive_contribution": 5.0,
                    "saves": 0,
                    "clean_sheets": 0,
                }
            )
    return pd.DataFrame(rows)


def _with_future(history: pd.DataFrame) -> pd.DataFrame:
    """Same history plus absurd values from the target gameweek onward.

    If any feature reads the future, these numbers are impossible to miss.
    """
    future = []
    for element in (1, 2):
        for event in (TARGET_EVENT, TARGET_EVENT + 1):
            future.append(
                {
                    "element": element,
                    "event": event,
                    "fixture": event * 10 + element,
                    "opponent_team": (event % 3) + 1,
                    "was_home": True,
                    "minutes": 90,
                    "total_points": 9999,
                    "goals_scored": 99,
                    "assists": 99,
                    "bps": 9999,
                    "expected_goals": 99.0,
                    "expected_assists": 99.0,
                    "expected_goals_conceded": 99.0,
                    "defensive_contribution": 99.0,
                    "saves": 99,
                    "clean_sheets": 1,
                }
            )
    return pd.concat([history, pd.DataFrame(future)], ignore_index=True)


# ------------------------------------------------------------------ core
def test_rolling_features_ignore_the_future():
    clean = pf.player_rolling_features(_base_history(), TARGET_EVENT)
    poisoned = pf.player_rolling_features(_with_future(_base_history()), TARGET_EVENT)
    pd.testing.assert_frame_equal(clean, poisoned)


def test_rolling_features_have_no_impossible_values():
    poisoned = pf.player_rolling_features(_with_future(_base_history()), TARGET_EVENT)
    numeric = poisoned.select_dtypes(include=[np.number])
    assert (numeric.abs() < 1000).all().all(), "a 9999 sentinel reached the features"


def test_prior_match_count_excludes_target_event():
    out = pf.player_rolling_features(_with_future(_base_history()), TARGET_EVENT)
    assert (out["n_prior_matches"] == TARGET_EVENT - 1).all()


def test_head_to_head_ignores_the_future():
    opponents = pd.DataFrame([{"element": 1, "opponent_id": 1}, {"element": 2, "opponent_id": 2}])
    clean = pf.head_to_head_feature(_base_history(), TARGET_EVENT, opponents)
    poisoned = pf.head_to_head_feature(_with_future(_base_history()), TARGET_EVENT, opponents)
    pd.testing.assert_frame_equal(clean, poisoned)
    assert (poisoned["h2h_estimate"] < 1000).all()


def test_head_to_head_can_be_ablated():
    """The plan's ablation switch must neutralise the feature entirely."""
    opponents = pd.DataFrame([{"element": 1, "opponent_id": 1}])
    off = pf.head_to_head_feature(_base_history(), TARGET_EVENT, opponents, enabled=False)
    assert (off["h2h_n"] == 0).all()


def test_team_form_ignores_matches_after_cutoff():
    cutoff = pd.Timestamp("2026-09-01")
    base = pd.DataFrame(
        [
            {"team_title": "A", "match_date": pd.Timestamp("2026-08-20"), "h_a": "h",
             "xG": 1.0, "xGA": 1.0, "npxG": 1.0, "npxGA": 1.0, "scored": 1, "missed": 1,
             "deep": 5, "deep_allowed": 5, "ppda_att": 100, "ppda_def": 10},
        ]
    )
    future = pd.concat(
        [
            base,
            pd.DataFrame(
                [
                    {"team_title": "A", "match_date": pd.Timestamp("2026-09-05"), "h_a": "h",
                     "xG": 99.0, "xGA": 99.0, "npxG": 99.0, "npxGA": 99.0, "scored": 99,
                     "missed": 99, "deep": 99, "deep_allowed": 99, "ppda_att": 999, "ppda_def": 1},
                ]
            ),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        pf.team_form_features(base, cutoff), pf.team_form_features(future, cutoff)
    )


def test_rest_days_uses_only_past_fixtures():
    fixtures = pd.DataFrame(
        [
            {"id": 1, "event": 4, "team_h": 1, "team_a": 2,
             "kickoff_time": pd.Timestamp("2026-08-25 14:00")},
            {"id": 2, "event": TARGET_EVENT, "team_h": 1, "team_a": 3,
             "kickoff_time": pd.Timestamp("2026-09-01 14:00")},
            {"id": 3, "event": TARGET_EVENT + 1, "team_h": 1, "team_a": 4,
             "kickoff_time": pd.Timestamp("2026-09-08 14:00")},
        ]
    )
    out = pf.rest_days(fixtures, TARGET_EVENT)
    assert out[out["team_id"] == 1]["rest_days"].iloc[0] == 7


# ------------------------------------------------------- behavioural checks
def test_rolling_horizons_differ_when_form_changes():
    """A leakage guard that froze every feature would also pass trivially.

    This test proves the features actually respond to real (past) data, so the
    guard above is meaningful rather than vacuous.
    """
    rows = []
    for event, pts in enumerate([2, 2, 2, 20], start=1):
        rows.append(
            {"element": 1, "event": event, "fixture": event, "opponent_team": 1,
             "was_home": True, "minutes": 90, "total_points": pts, "goals_scored": 0,
             "assists": 0, "bps": 10, "expected_goals": 0.1, "expected_assists": 0.1,
             "expected_goals_conceded": 1.0, "defensive_contribution": 1.0,
             "saves": 0, "clean_sheets": 0}
        )
    out = pf.player_rolling_features(pd.DataFrame(rows), TARGET_EVENT)
    assert out["total_points_l1"].iloc[0] > out["total_points_l38"].iloc[0]


def test_empty_history_is_safe():
    empty = pd.DataFrame(columns=["element", "event", "minutes", "total_points"])
    assert pf.player_rolling_features(empty, TARGET_EVENT).empty


def test_fixture_context_splits_home_and_away():
    fixtures = pd.DataFrame(
        [{"id": 1, "event": TARGET_EVENT, "team_h": 1, "team_a": 2,
          "team_h_difficulty": 2, "team_a_difficulty": 4,
          "kickoff_time": pd.Timestamp("2026-09-01")}]
    )
    out = pf.fixture_context(fixtures, TARGET_EVENT)
    assert len(out) == 2
    assert set(out["is_home"]) == {True, False}
    home = out[out["is_home"]].iloc[0]
    assert home["team_id"] == 1 and home["opponent_id"] == 2


def test_fixture_context_flags_double_gameweek():
    fixtures = pd.DataFrame(
        [
            {"id": 1, "event": TARGET_EVENT, "team_h": 1, "team_a": 2,
             "team_h_difficulty": 2, "team_a_difficulty": 4, "kickoff_time": pd.Timestamp("2026-09-01")},
            {"id": 2, "event": TARGET_EVENT, "team_h": 3, "team_a": 1,
             "team_h_difficulty": 3, "team_a_difficulty": 3, "kickoff_time": pd.Timestamp("2026-09-03")},
        ]
    )
    out = pf.fixture_context(fixtures, TARGET_EVENT)
    assert out[out["team_id"] == 1]["n_fixtures"].iloc[0] == 2


def test_fixture_context_omits_blank_gameweek_teams():
    fixtures = pd.DataFrame(
        [{"id": 1, "event": TARGET_EVENT, "team_h": 1, "team_a": 2,
          "team_h_difficulty": 2, "team_a_difficulty": 4, "kickoff_time": pd.Timestamp("2026-09-01")}]
    )
    out = pf.fixture_context(fixtures, TARGET_EVENT)
    assert 3 not in set(out["team_id"]), "a team with no fixture must not appear"


def test_set_piece_flags(bootstrap_payload):
    from fpl.ingest import fpl_api

    players = fpl_api.parse_bootstrap(bootstrap_payload)["players"]
    out = pf.set_piece_flags(players)
    assert "is_pen_taker" in out.columns
    assert out["is_pen_taker"].dtype == bool
