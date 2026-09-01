"""Team name mapping: Understat <-> FPL.

Teams are a closed set of 20, so this is a hand-verified lookup rather than
fuzzy matching -- getting a team wrong would silently corrupt every player
match inside it. Verified live 2026-09-01: 8 of 20 names differ between the
two sources.
"""
from __future__ import annotations

import pandas as pd

# Understat `team_title` -> FPL `teams.name`
UNDERSTAT_TO_FPL: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry City",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sunderland": "Sunderland",
    "Tottenham": "Spurs",
}

FPL_TO_UNDERSTAT: dict[str, str] = {v: k for k, v in UNDERSTAT_TO_FPL.items()}

# Historical / alternate spellings seen in older Understat seasons.
_EXTRA_ALIASES: dict[str, str] = {
    "Wolverhampton Wanderers": "Wolves",
    "West Ham": "West Ham",
    "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield Utd",
    "Leicester": "Leicester",
    "Southampton": "Southampton",
    "Luton": "Luton",
    "Burnley": "Burnley",
    "Norwich": "Norwich",
    "Watford": "Watford",
    "Leeds United": "Leeds",
}


def understat_to_fpl_name(title: str) -> str | None:
    """Map an Understat team title to its FPL team name."""
    if title in UNDERSTAT_TO_FPL:
        return UNDERSTAT_TO_FPL[title]
    return _EXTRA_ALIASES.get(title)


def build_team_lookup(teams: pd.DataFrame) -> dict[str, int]:
    """Understat team title -> FPL team id, for the teams in this season."""
    by_name = dict(zip(teams["name"], teams["id"]))
    out: dict[str, int] = {}
    for us_title in list(UNDERSTAT_TO_FPL) + list(_EXTRA_ALIASES):
        fpl_name = understat_to_fpl_name(us_title)
        if fpl_name in by_name:
            out[us_title] = int(by_name[fpl_name])
    return out


def unmapped_titles(understat_titles: list[str]) -> list[str]:
    """Understat titles with no FPL counterpart -- must be empty in practice."""
    return [t for t in understat_titles if understat_to_fpl_name(t) is None]
