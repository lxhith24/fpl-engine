"""Understat ingest (Tasks 7-8).

IMPORTANT (verified live 2026-09-01): Understat NO LONGER embeds
`playersData = JSON.parse('...')` in the league page HTML. Every public guide
and most community libraries still describe that scrape and they are stale --
the page now returns an identical ~18.7KB shell for every season and the data
arrives via AJAX:

    GET  /getLeagueData/{league}/{season}   -> {teams, players, dates}
    POST /main/getPlayersStats/             -> {success, players}

Both are gzipped; httpx decompresses transparently. Team `history` rows carry
the matchup fuel: xG, xGA, npxG, npxGA, ppda, deep, deep_allowed.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from .. import config
from .http import get_json, post_json

_HOST = "understat"


def _referer(league: str, season: int) -> dict[str, str]:
    return {"Referer": f"{config.UNDERSTAT_BASE}/league/{league}/{season}"}


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v) -> int | None:
    n = _num(v)
    return int(n) if n is not None else None


def parse_team_matches(payload: dict, season: int) -> pd.DataFrame:
    """Flatten {teams: {id: {title, history: [...]}}} into one row per match."""
    pulled = _dt.datetime.now()
    rows = []
    for team in payload.get("teams", {}).values():
        title = team.get("title")
        for h in team.get("history", []):
            ppda = h.get("ppda") or {}
            ppda_a = h.get("ppda_allowed") or {}
            rows.append(
                {
                    "team_title": title,
                    "match_date": pd.to_datetime(h.get("date"), errors="coerce"),
                    "h_a": h.get("h_a"),
                    "xG": _num(h.get("xG")),
                    "xGA": _num(h.get("xGA")),
                    "npxG": _num(h.get("npxG")),
                    "npxGA": _num(h.get("npxGA")),
                    "scored": _int(h.get("scored")),
                    "missed": _int(h.get("missed")),
                    "ppda_att": _int(ppda.get("att")),
                    "ppda_def": _int(ppda.get("def")),
                    "ppda_allowed_att": _int(ppda_a.get("att")),
                    "ppda_allowed_def": _int(ppda_a.get("def")),
                    "deep": _int(h.get("deep")),
                    "deep_allowed": _int(h.get("deep_allowed")),
                    "pts": _int(h.get("pts")),
                    "season": season,
                    "pulled_at": pulled,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_date"] = df["match_date"].dt.date
    return df


def parse_players(rows: list[dict], season: int) -> pd.DataFrame:
    pulled = _dt.datetime.now()
    out = [
        {
            "understat_id": str(p.get("id")),
            "player_name": p.get("player_name"),
            "team_title": p.get("team_title"),
            "position": p.get("position"),
            "games": _int(p.get("games")),
            "time": _int(p.get("time")),
            "goals": _int(p.get("goals")),
            "assists": _int(p.get("assists")),
            "shots": _int(p.get("shots")),
            "key_passes": _int(p.get("key_passes")),
            "xG": _num(p.get("xG")),
            "xA": _num(p.get("xA")),
            "npxG": _num(p.get("npxG")),
            "npg": _int(p.get("npg")),
            "xGChain": _num(p.get("xGChain")),
            "xGBuildup": _num(p.get("xGBuildup")),
            "season": season,
            "pulled_at": pulled,
        }
        for p in rows
    ]
    return pd.DataFrame(out)


def fetch_league_data(
    league: str = "EPL", season: int = 2026, *, snapshot: bool = True
) -> dict[str, pd.DataFrame]:
    """Teams + players for a season via the AJAX league endpoint."""
    from .. import store

    url = config.UNDERSTAT_LEAGUE_DATA.format(league=league, season=season)
    payload = get_json(
        url,
        host_key=_HOST,
        min_interval=config.UNDERSTAT_MIN_INTERVAL_S,
        headers={"X-Requested-With": "XMLHttpRequest", **_referer(league, season)},
    )
    if snapshot:
        store.snapshot(f"understat-league-{league}-{season}", payload)
    return {
        "team_matches": parse_team_matches(payload, season),
        "players": parse_players(payload.get("players", []), season),
    }


def fetch_players_stats(
    league: str = "EPL", season: int = 2026, *, snapshot: bool = True
) -> pd.DataFrame:
    """Player season aggregates via the POST endpoint (same shape, filterable)."""
    from .. import store

    payload = post_json(
        config.UNDERSTAT_PLAYERS_STATS,
        {"league": league, "season": str(season)},
        host_key=_HOST,
        headers=_referer(league, season),
    )
    if not payload.get("success", False):
        raise RuntimeError("understat getPlayersStats returned success=false")
    if snapshot:
        store.snapshot(f"understat-players-{league}-{season}", payload)
    return parse_players(payload.get("players", []), season)
