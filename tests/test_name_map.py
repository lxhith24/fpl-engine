"""Task 12-14: FPL <-> Understat entity resolution.

These tests encode two real bugs found during Phase 2 development. Both were
silent -- they produced no error, just a missing player -- which is exactly the
failure mode this layer exists to prevent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fpl.resolve import name_map, teams as team_map


# ------------------------------------------------------------------ teams
def test_all_current_understat_teams_map():
    titles = list(team_map.UNDERSTAT_TO_FPL)
    assert team_map.unmapped_titles(titles) == []
    assert len(titles) == 20


def test_team_mapping_is_bijective():
    fpl_names = list(team_map.UNDERSTAT_TO_FPL.values())
    assert len(set(fpl_names)) == len(fpl_names), "two clubs map to one FPL name"


def test_known_divergent_team_names():
    """8 of 20 names differ between sources -- the obvious traps."""
    assert team_map.understat_to_fpl_name("Tottenham") == "Spurs"
    assert team_map.understat_to_fpl_name("Manchester United") == "Man Utd"
    assert team_map.understat_to_fpl_name("Nottingham Forest") == "Nott'm Forest"
    assert team_map.understat_to_fpl_name("Newcastle United") == "Newcastle"


def test_build_team_lookup(bootstrap_payload):
    from fpl.ingest import fpl_api

    tms = fpl_api.parse_bootstrap(bootstrap_payload)["teams"]
    lookup = team_map.build_team_lookup(tms)
    assert len(lookup) >= 20
    assert all(isinstance(v, int) for v in lookup.values())


# -------------------------------------------------------------- normalize
def test_normalize_strips_combining_diacritics():
    assert name_map.normalize("Calafiori") == "calafiori"
    assert name_map.normalize("Guimarães") == "guimaraes"
    assert name_map.normalize("Martín") == "martin"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ødegaard", "odegaard"),
        ("Ødegaard", name_map.normalize("Odegaard")),
        ("Łukasz", "lukasz"),
        ("Đorđević", "dordevic"),
    ],
)
def test_normalize_folds_non_decomposable_letters(raw, expected):
    """REGRESSION: Ø is a distinct letter, not O + combining mark.

    NFD cannot decompose it, so "Ødegaard" never folded to "odegaard" and
    Martin Ødegaard (FPL 15) silently failed to resolve against Understat's
    "Martin Odegaard".
    """
    assert name_map.normalize(raw) == expected


def test_normalize_handles_none_and_empty():
    assert name_map.normalize(None) == ""
    assert name_map.normalize("") == ""


# ------------------------------------------------------------------ score
def test_exact_match_scores_100():
    variants = name_map.fpl_name_variants("Bukayo", "Saka", "Saka")
    assert name_map.score_pair(variants, "Bukayo Saka") == 100.0


def test_abbreviated_web_name_matches_full_name():
    """FPL 'B.Fernandes' must resolve to Understat 'Bruno Fernandes'."""
    variants = name_map.fpl_name_variants("Bruno", "Borges Fernandes", "B.Fernandes")
    assert name_map.score_pair(variants, "Bruno Fernandes") >= 90


def test_forename_fragment_does_not_claim_full_name():
    """REGRESSION: token_set_ratio scored subsets as a perfect 100.

    David *Raya Martín* generates the surname-fragment variant "martin", which
    scored 100 against "Martin Odegaard", won the greedy assignment, and left
    Ødegaard unmatched. A bare token must only be compared to the surname.
    """
    raya = name_map.fpl_name_variants("David", "Raya Martín", "Raya")
    assert name_map.score_pair(raya, "David Raya") == 100.0
    assert name_map.score_pair(raya, "Martin Odegaard") < name_map.MATCH_THRESHOLD


def test_unrelated_names_score_below_threshold():
    variants = name_map.fpl_name_variants("Erling", "Haaland", "Haaland")
    assert name_map.score_pair(variants, "Cole Palmer") < name_map.MATCH_THRESHOLD


# ---------------------------------------------------------------- resolve
@pytest.fixture()
def resolved(bootstrap_payload, understat_payload):
    from fpl.ingest import fpl_api, understat as us_ingest

    boot = fpl_api.parse_bootstrap(bootstrap_payload)
    us = us_ingest.parse_players(understat_payload["players"], season=2026)
    return name_map.resolve(boot["players"], us, boot["teams"]), boot, us


def test_resolve_produces_one_to_one_mapping(resolved):
    res, _, _ = resolved
    assert res.mapping["element"].is_unique, "an FPL player matched twice"
    assert res.mapping["understat_id"].is_unique, "an Understat player matched twice"


def test_resolve_never_crosses_teams(resolved):
    res, boot, us = resolved
    lookup = team_map.build_team_lookup(boot["teams"])
    us_team = dict(zip(us["understat_id"], us["team_title"]))
    fpl_team = dict(zip(boot["players"]["id"], boot["players"]["team_id"]))
    for _, row in res.mapping.iterrows():
        expected = lookup.get(us_team[row["understat_id"]])
        assert fpl_team[row["element"]] == expected, "matched across clubs"


def test_resolve_scores_meet_threshold(resolved):
    res, _, _ = resolved
    assert (res.mapping["score"] >= name_map.MATCH_THRESHOLD).all()


def test_no_unmapped_team_titles(resolved):
    res, _, _ = resolved
    assert res.stats["unmapped_team_titles"] == []


def test_overrides_take_precedence(bootstrap_payload, understat_payload):
    from fpl.ingest import fpl_api, understat as us_ingest

    boot = fpl_api.parse_bootstrap(bootstrap_payload)
    us = us_ingest.parse_players(understat_payload["players"], season=2026)

    element = int(boot["players"].iloc[0]["id"])
    us_id = str(us.iloc[0]["understat_id"])
    res = name_map.resolve(
        boot["players"], us, boot["teams"], overrides={us_id: element}
    )
    row = res.mapping[res.mapping["understat_id"] == us_id]
    assert len(row) == 1
    assert int(row.iloc[0]["element"]) == element
    assert row.iloc[0]["method"] == "override"


def test_load_overrides_parses_comments_only_file():
    """The shipped file is all comments -- must yield an empty dict, not crash."""
    assert name_map.load_overrides() == {}


def test_coverage_ignores_non_players(resolved):
    """Fringe players with no minutes are legitimately absent from Understat."""
    res, boot, _ = resolved
    cov = name_map.coverage(res.mapping, boot["players"], min_minutes=90)
    assert 0.0 <= cov["coverage"] <= 1.0
    assert cov["eligible"] <= len(boot["players"])


def test_empty_inputs_are_safe(bootstrap_payload):
    from fpl.ingest import fpl_api

    boot = fpl_api.parse_bootstrap(bootstrap_payload)
    empty = pd.DataFrame(
        columns=["understat_id", "player_name", "team_title", "time"]
    )
    res = name_map.resolve(boot["players"], empty, boot["teams"])
    assert res.mapping.empty
