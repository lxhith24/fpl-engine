"""Official FPL API ingest (Tasks 3-6, 11).

Pure parse functions take raw payloads so tests run against saved fixtures with
no network. The `fetch_*` wrappers do IO and snapshot to disk.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

import pandas as pd

from .. import config, store
from .http import get_json

_HOST = "fpl"


def _now() -> _dt.datetime:
    return _dt.datetime.now()


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- bootstrap
def parse_bootstrap(payload: dict) -> dict[str, pd.DataFrame]:
    """Split bootstrap-static into teams / players / events frames."""
    pulled = _now()

    teams = pd.DataFrame(
        [
            {
                "id": t["id"],
                "name": t["name"],
                "short_name": t["short_name"],
                "strength": t.get("strength"),
                "strength_attack_home": t.get("strength_attack_home"),
                "strength_attack_away": t.get("strength_attack_away"),
                "strength_defence_home": t.get("strength_defence_home"),
                "strength_defence_away": t.get("strength_defence_away"),
                "pulled_at": pulled,
            }
            for t in payload["teams"]
        ]
    )

    players = pd.DataFrame(
        [
            {
                "id": e["id"],
                "web_name": e["web_name"],
                "first_name": e.get("first_name"),
                "second_name": e.get("second_name"),
                "team_id": e["team"],
                "position": config.POSITIONS[e["element_type"]],
                "now_cost": e["now_cost"],
                "status": e.get("status"),
                "chance_next": _num(e.get("chance_of_playing_next_round")),
                "minutes": e.get("minutes"),
                "total_points": e.get("total_points"),
                "form": _num(e.get("form")),
                "points_per_game": _num(e.get("points_per_game")),
                "selected_by_percent": _num(e.get("selected_by_percent")),
                "expected_goals": _num(e.get("expected_goals")),
                "expected_assists": _num(e.get("expected_assists")),
                "expected_goal_involvements": _num(e.get("expected_goal_involvements")),
                "expected_goals_conceded": _num(e.get("expected_goals_conceded")),
                "defensive_contribution": _num(e.get("defensive_contribution")),
                "starts": e.get("starts"),
                "penalties_order": e.get("penalties_order"),
                "corners_order": e.get("corners_and_indirect_freekicks_order"),
                "news": e.get("news"),
                "pulled_at": pulled,
            }
            for e in payload["elements"]
        ]
    )

    events = pd.DataFrame(
        [
            {
                "id": ev["id"],
                "name": ev.get("name"),
                "deadline_time": pd.to_datetime(ev.get("deadline_time"), utc=True, errors="coerce"),
                "is_current": bool(ev.get("is_current")),
                "is_next": bool(ev.get("is_next")),
                "finished": bool(ev.get("finished")),
                "pulled_at": pulled,
            }
            for ev in payload["events"]
        ]
    )
    # DuckDB TIMESTAMP columns are naive; drop tz after parsing.
    events["deadline_time"] = events["deadline_time"].dt.tz_localize(None)

    return {"teams": teams, "players": players, "events": events}


def fetch_bootstrap(*, snapshot: bool = True) -> dict[str, pd.DataFrame]:
    payload = get_json(config.FPL_BOOTSTRAP, host_key=_HOST)
    if snapshot:
        store.snapshot("bootstrap-static", payload)
    return parse_bootstrap(payload)


# ---------------------------------------------------------------- fixtures
def parse_fixtures(payload: list[dict]) -> pd.DataFrame:
    pulled = _now()
    df = pd.DataFrame(
        [
            {
                "id": f["id"],
                "event": f.get("event"),
                "kickoff_time": pd.to_datetime(f.get("kickoff_time"), utc=True, errors="coerce"),
                "team_h": f["team_h"],
                "team_a": f["team_a"],
                "team_h_score": f.get("team_h_score"),
                "team_a_score": f.get("team_a_score"),
                "team_h_difficulty": f.get("team_h_difficulty"),
                "team_a_difficulty": f.get("team_a_difficulty"),
                "finished": bool(f.get("finished")),
                "pulled_at": pulled,
            }
            for f in payload
        ]
    )
    if not df.empty:
        df["kickoff_time"] = df["kickoff_time"].dt.tz_localize(None)
    return df


def fetch_fixtures(*, future_only: bool = False, snapshot: bool = True) -> pd.DataFrame:
    url = config.FPL_FIXTURES + ("?future=1" if future_only else "")
    payload = get_json(url, host_key=_HOST)
    if snapshot:
        store.snapshot("fixtures" + ("-future" if future_only else ""), payload)
    return parse_fixtures(payload)


def detect_dgw_bgw(fixtures: pd.DataFrame, n_teams: int = 20) -> pd.DataFrame:
    """Per (event, team) fixture counts: 0 => blank, 2+ => double."""
    if fixtures.empty:
        return pd.DataFrame(columns=["event", "team_id", "n_fixtures", "kind"])

    played = fixtures.dropna(subset=["event"]).copy()
    home = played[["event", "team_h"]].rename(columns={"team_h": "team_id"})
    away = played[["event", "team_a"]].rename(columns={"team_a": "team_id"})
    both = pd.concat([home, away], ignore_index=True)
    counts = (
        both.groupby(["event", "team_id"]).size().reset_index(name="n_fixtures")
    )

    # Teams absent from an event have zero fixtures -> blank gameweek.
    events = sorted(counts["event"].unique())
    teams = sorted(set(fixtures["team_h"]) | set(fixtures["team_a"]))
    full = pd.MultiIndex.from_product([events, teams], names=["event", "team_id"])
    counts = (
        counts.set_index(["event", "team_id"])
        .reindex(full, fill_value=0)
        .reset_index()
    )
    counts["kind"] = counts["n_fixtures"].map(
        lambda n: "blank" if n == 0 else ("double" if n >= 2 else "single")
    )
    counts["event"] = counts["event"].astype(int)
    return counts


# --------------------------------------------------------- element summary
def parse_element_summary(element_id: int, payload: dict) -> pd.DataFrame:
    """Per-gameweek history rows for one player."""
    pulled = _now()
    rows = []
    for h in payload.get("history", []):
        rows.append(
            {
                "element": element_id,
                "event": h.get("round"),
                "fixture": h.get("fixture"),
                "opponent_team": h.get("opponent_team"),
                "was_home": bool(h.get("was_home")),
                "minutes": h.get("minutes"),
                "total_points": h.get("total_points"),
                "goals_scored": h.get("goals_scored"),
                "assists": h.get("assists"),
                "clean_sheets": h.get("clean_sheets"),
                "goals_conceded": h.get("goals_conceded"),
                "saves": h.get("saves"),
                "bonus": h.get("bonus"),
                "bps": h.get("bps"),
                "expected_goals": _num(h.get("expected_goals")),
                "expected_assists": _num(h.get("expected_assists")),
                "expected_goals_conceded": _num(h.get("expected_goals_conceded")),
                "defensive_contribution": _num(h.get("defensive_contribution")),
                "starts": h.get("starts"),
                "value": h.get("value"),
                "pulled_at": pulled,
            }
        )
    return pd.DataFrame(rows)


def fetch_element_summary(element_id: int, *, cache: bool = True) -> pd.DataFrame:
    url = config.FPL_ELEMENT_SUMMARY.format(element_id=element_id)
    payload = get_json(url, host_key=_HOST, cache=cache, max_age_s=1800)
    return parse_element_summary(element_id, payload)


# ------------------------------------------------------------- event live
def parse_event_live(event_id: int, payload: dict) -> pd.DataFrame:
    """Actual per-player returns for a finished gameweek (back-test labels)."""
    pulled = _now()
    rows = []
    for el in payload.get("elements", []):
        s = el.get("stats", {})
        rows.append(
            {
                "element": el["id"],
                "event": event_id,
                "fixture": None,
                "opponent_team": None,
                "was_home": None,
                "minutes": s.get("minutes"),
                "total_points": s.get("total_points"),
                "goals_scored": s.get("goals_scored"),
                "assists": s.get("assists"),
                "clean_sheets": s.get("clean_sheets"),
                "goals_conceded": s.get("goals_conceded"),
                "saves": s.get("saves"),
                "bonus": s.get("bonus"),
                "bps": s.get("bps"),
                "expected_goals": _num(s.get("expected_goals")),
                "expected_assists": _num(s.get("expected_assists")),
                "expected_goals_conceded": _num(s.get("expected_goals_conceded")),
                "defensive_contribution": _num(s.get("defensive_contribution")),
                "starts": s.get("starts"),
                "value": s.get("value"),
                "pulled_at": pulled,
            }
        )
    return pd.DataFrame(rows)


def fetch_event_live(event_id: int, *, snapshot: bool = True) -> pd.DataFrame:
    url = config.FPL_EVENT_LIVE.format(event_id=event_id)
    payload = get_json(url, host_key=_HOST)
    if snapshot:
        store.snapshot(f"event-{event_id}-live", payload)
    return parse_event_live(event_id, payload)


# ------------------------------------------------------------------ entry
def fetch_entry(entry_id: int = config.DEFAULT_ENTRY_ID) -> dict:
    return get_json(config.FPL_ENTRY.format(entry_id=entry_id), host_key=_HOST)


def fetch_entry_picks(entry_id: int, event_id: int) -> dict:
    return get_json(
        config.FPL_ENTRY_PICKS.format(entry_id=entry_id, event_id=event_id),
        host_key=_HOST,
    )


def current_and_next_event(events: pd.DataFrame) -> tuple[int | None, int | None]:
    cur = events.loc[events["is_current"], "id"]
    nxt = events.loc[events["is_next"], "id"]
    return (
        int(cur.iloc[0]) if len(cur) else None,
        int(nxt.iloc[0]) if len(nxt) else None,
    )
