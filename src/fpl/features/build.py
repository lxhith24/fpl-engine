"""Assemble the point-in-time feature table for a target gameweek (Task 15-22).

`build_features(as_of_event=N)` returns one row per (player, fixture) in GW N,
carrying player form, team strength, opponent profile, zone fit, shrunk
head-to-head, and fixture context -- using only information available before
GW N's deadline.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .. import store
from ..resolve import teams as team_map
from . import player as pf
from . import zones as zf


def _deadline_for(events: pd.DataFrame, event_id: int) -> pd.Timestamp:
    row = events[events["id"] == event_id]
    if row.empty or pd.isna(row.iloc[0]["deadline_time"]):
        return pd.Timestamp.now()
    return pd.Timestamp(row.iloc[0]["deadline_time"])


def build_features(
    as_of_event: int,
    *,
    con=None,
    season: int = 2026,
    h2h_enabled: bool = True,
    zone_enabled: bool = True,
) -> pd.DataFrame:
    """One row per (player, upcoming fixture) with all matchup features."""
    owns_con = con is None
    con = con or store.connect()
    try:
        players = con.execute("SELECT * FROM players").df()
        teams = con.execute("SELECT * FROM teams").df()
        events = con.execute("SELECT * FROM events").df()
        fixtures = con.execute("SELECT * FROM fixtures").df()
        player_gw = con.execute("SELECT * FROM player_gw").df()
        team_matches = con.execute(
            "SELECT * FROM understat_team_match WHERE season = ?", [season]
        ).df()
        pmap = con.execute(
            "SELECT * FROM player_map WHERE season = ?", [season]
        ).df()
    finally:
        if owns_con:
            con.close()

    deadline = _deadline_for(events, as_of_event)

    # ---- upcoming fixtures (handles DGW/BGW) ----------------------------
    fx = pf.fixture_context(fixtures, as_of_event)
    if fx.empty:
        return pd.DataFrame()

    base = players[["id", "web_name", "team_id", "position", "now_cost", "status",
                    "chance_next", "selected_by_percent", "minutes", "form",
                    "total_points"]].rename(columns={"id": "element"})
    df = base.merge(fx, on="team_id", how="inner")

    # ---- player rolling form (point-in-time) ----------------------------
    roll = pf.player_rolling_features(player_gw, as_of_event)
    if not roll.empty:
        df = df.merge(roll, on="element", how="left")

    # ---- team + opponent profiles ---------------------------------------
    form = pf.team_form_features(team_matches, deadline)
    if not form.empty:
        lookup = team_map.build_team_lookup(teams)
        form["team_id"] = form["team_title"].map(lookup)
        form = form.dropna(subset=["team_id"])
        form["team_id"] = form["team_id"].astype(int)

        own = form.add_prefix("own_").rename(columns={"own_team_id": "team_id"})
        opp = form.add_prefix("opp_").rename(columns={"opp_team_id": "opponent_id"})
        df = df.merge(own.drop(columns=["own_team_title"]), on="team_id", how="left")
        df = df.merge(opp.drop(columns=["opp_team_title"]), on="opponent_id", how="left")

    # ---- venue-aware opponent concession --------------------------------
    # A player at home faces the opponent's AWAY defensive record.
    if {"opp_xga_home", "opp_xga_away"}.issubset(df.columns):
        df["opp_xga_venue"] = np.where(
            df["is_home"], df["opp_xga_away"], df["opp_xga_home"]
        )
        df["opp_xga_venue"] = df["opp_xga_venue"].fillna(df.get("opp_xg_against"))

    # ---- zone fit --------------------------------------------------------
    df["zone_fit"] = 1.0
    if zone_enabled:
        try:
            shots = _load_shots(season)
        except Exception:  # noqa: BLE001 - enrichment must never block a build
            shots = pd.DataFrame()
        if not shots.empty and not pmap.empty:
            shots = shots[shots["shot_date"] < deadline]
            am = zf.player_attack_map(shots)
            con_map = zf.opponent_concession_map(shots)
            fit = zf.zone_fit(am, con_map)
            if not fit.empty:
                inv = {v: k for k, v in team_map.UNDERSTAT_TO_FPL.items()}
                id_to_title = {
                    int(r["id"]): inv.get(r["name"]) for _, r in teams.iterrows()
                }
                df["_opp_title"] = df["opponent_id"].map(id_to_title)
                # zone_fit uses `understat_player_id`; player_map uses
                # `understat_id`. Align before joining.
                fit = fit.rename(
                    columns={
                        "understat_player_id": "understat_id",
                        "defending_title": "_opp_title",
                    }
                )
                df = df.merge(
                    pmap[["element", "understat_id"]], on="element", how="left"
                )
                df["understat_id"] = df["understat_id"].astype("string")
                fit["understat_id"] = fit["understat_id"].astype("string")
                df = df.merge(
                    fit, on=["understat_id", "_opp_title"], how="left",
                    suffixes=("", "_z"),
                )
                if "zone_fit_z" in df.columns:
                    df["zone_fit"] = df.pop("zone_fit_z").fillna(1.0)
                df = df.drop(columns=["_opp_title"], errors="ignore")

    # ---- shrunk head-to-head --------------------------------------------
    h2h = pf.head_to_head_feature(
        player_gw, as_of_event, df[["element", "opponent_id"]], enabled=h2h_enabled
    )
    if not h2h.empty:
        df = df.merge(
            h2h.drop_duplicates(subset=["element", "opponent_id"]),
            on=["element", "opponent_id"],
            how="left",
        )

    # ---- set pieces, rest, availability ---------------------------------
    df = df.merge(pf.set_piece_flags(players), on="element", how="left")
    rest = pf.rest_days(fixtures, as_of_event)
    if not rest.empty:
        df = df.merge(rest, on="team_id", how="left")

    # De-fragment before the final column additions: the merges above leave
    # the frame highly fragmented, which pandas warns about.
    df = df.copy()
    df["available"] = df["status"].isin(["a", "d"])
    # `fillna` rejects a raw ndarray, so build a Series aligned to the index.
    default_chance = pd.Series(
        np.where(df["status"] == "a", 100.0, 0.0), index=df.index
    )
    df["chance_next"] = df["chance_next"].fillna(default_chance)
    df["as_of_event"] = as_of_event
    df["built_at"] = _dt.datetime.now()
    return df


def _load_shots(season: int) -> pd.DataFrame:
    """Cached season shots, pulled on first use."""
    from ..config import INTERIM
    from ..ingest import understat_shots

    path = INTERIM / f"shots_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    shots = understat_shots.fetch_season_shots("EPL", season)
    if not shots.empty:
        shots.to_parquet(path)
    return shots
