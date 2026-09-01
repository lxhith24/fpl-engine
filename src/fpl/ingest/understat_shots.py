"""Understat match-level shot ingest (Task 17 input).

Shots are the raw material for opponent zone-concession maps. The per-match
endpoint is far cheaper than per-player: `getMatchData/{id}` returns every shot
in one request split by side, and a home shot IS an away-team concession.

    GET /getLeagueData/{league}/{season}  -> .dates[] gives match ids
    GET /getMatchData/{match_id}          -> {shots: {h: [...], a: [...]}}

Coordinates are normalized 0-1 (X = toward the attacked goal, Y = lateral).
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from .. import config
from .http import get_json

_HOST = "understat"
_MATCH_URL = f"{config.UNDERSTAT_BASE}/getMatchData/{{match_id}}"


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_dates(payload: dict, season: int) -> pd.DataFrame:
    """Fixture list from the league payload -- match ids + result flags."""
    rows = []
    for d in payload.get("dates", []):
        rows.append(
            {
                "match_id": str(d.get("id")),
                "kickoff": pd.to_datetime(d.get("datetime"), errors="coerce"),
                "home_title": (d.get("h") or {}).get("title"),
                "away_title": (d.get("a") or {}).get("title"),
                "home_goals": _num((d.get("goals") or {}).get("h")),
                "away_goals": _num((d.get("goals") or {}).get("a")),
                "home_xg": _num((d.get("xG") or {}).get("h")),
                "away_xg": _num((d.get("xG") or {}).get("a")),
                "is_result": bool(d.get("isResult")),
                "season": season,
            }
        )
    return pd.DataFrame(rows)


def parse_match_shots(payload: dict, match_id: str, season: int) -> pd.DataFrame:
    """Flatten both sides' shots, tagging attacking and DEFENDING team.

    `defending_title` is what makes this a concession map: it is the team that
    allowed the shot, which is the opponent profile the matchup layer needs.
    """
    pulled = _dt.datetime.now()
    shots = payload.get("shots") or {}
    rows = []
    for side in ("h", "a"):
        for s in shots.get(side, []):
            home, away = s.get("h_team"), s.get("a_team")
            attacking = home if side == "h" else away
            defending = away if side == "h" else home
            rows.append(
                {
                    "shot_id": str(s.get("id")),
                    "match_id": str(match_id),
                    "season": season,
                    "shot_date": pd.to_datetime(s.get("date"), errors="coerce"),
                    "minute": int(_num(s.get("minute")) or 0),
                    "understat_player_id": str(s.get("player_id")),
                    "player_name": s.get("player"),
                    "attacking_title": attacking,
                    "defending_title": defending,
                    "h_a": side,
                    "X": _num(s.get("X")),
                    "Y": _num(s.get("Y")),
                    "xG": _num(s.get("xG")),
                    "result": s.get("result"),
                    "situation": s.get("situation"),
                    "shot_type": s.get("shotType"),
                    "last_action": s.get("lastAction"),
                    "pulled_at": pulled,
                }
            )
    return pd.DataFrame(rows)


def fetch_match_shots(match_id: str, season: int, *, cache: bool = True) -> pd.DataFrame:
    payload = get_json(
        _MATCH_URL.format(match_id=match_id),
        host_key=_HOST,
        min_interval=config.UNDERSTAT_MIN_INTERVAL_S,
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{config.UNDERSTAT_BASE}/match/{match_id}",
        },
        cache=cache,
        max_age_s=30 * 24 * 3600,  # finished matches never change
    )
    return parse_match_shots(payload, match_id, season)


def fetch_season_shots(
    league: str = "EPL",
    season: int = 2026,
    *,
    finished_only: bool = True,
    limit: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Every shot in a season. Cached per match, so re-runs are cheap."""
    league_payload = get_json(
        config.UNDERSTAT_LEAGUE_DATA.format(league=league, season=season),
        host_key=_HOST,
        min_interval=config.UNDERSTAT_MIN_INTERVAL_S,
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{config.UNDERSTAT_BASE}/league/{league}/{season}",
        },
        cache=True,
        max_age_s=1800,
    )
    dates = parse_dates(league_payload, season)
    if finished_only:
        dates = dates[dates["is_result"]]
    if limit:
        dates = dates.head(limit)

    frames = []
    for i, mid in enumerate(dates["match_id"], 1):
        try:
            frames.append(fetch_match_shots(mid, season))
        except Exception as exc:  # noqa: BLE001 - one bad match must not kill the pull
            if progress:
                print(f"  match {mid} failed: {exc}")
            continue
        if progress and i % 10 == 0:
            print(f"  {i}/{len(dates)} matches")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
