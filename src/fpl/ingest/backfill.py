"""Full per-player gameweek history backfill.

`verify_phase1` only sampled 5 players as a smoke test; the feature layer needs
every player's match history. Uses `/event/{gw}/live/` (one request per
gameweek, 626 players each) rather than `/element-summary/{id}/` (626
requests), then backfills opponent/venue from the fixture table.

Run:  PYTHONPATH=src ./.venv/bin/python -m fpl.ingest.backfill
"""
from __future__ import annotations

import sys

import pandas as pd

from .. import store
from . import fpl_api


def _fixture_lookup(fixtures: pd.DataFrame) -> dict[tuple[int, int], tuple[int, int, bool]]:
    """(event, team_id) -> (fixture_id, opponent_id, was_home)."""
    out: dict[tuple[int, int], tuple[int, int, bool]] = {}
    for _, f in fixtures.dropna(subset=["event"]).iterrows():
        ev = int(f["event"])
        out[(ev, int(f["team_h"]))] = (int(f["id"]), int(f["team_a"]), True)
        out[(ev, int(f["team_a"]))] = (int(f["id"]), int(f["team_h"]), False)
    return out


def played_events(fixtures: pd.DataFrame, events: pd.DataFrame) -> list[int]:
    """Gameweeks whose matches have actually been played.

    Deliberately NOT `events.finished`. FPL only sets that flag once bonus
    points and data checks are finalised, which lags the final whistle by
    days. Verified live 2026-09-01: GW2 matches were complete (90 minutes,
    scores recorded, `finished_provisional=True`) yet `finished` was still
    False and `data_checked` False.

    Trusting `finished` at GW3 would have silently halved the training set --
    one gameweek of history instead of two. Fixture-level evidence is used
    instead.

    Caveat: bonus points in a provisional gameweek can still shift by a point
    or two, so labels for the most recent GW are near-final, not final.
    """
    if fixtures.empty:
        return []
    fx = fixtures.dropna(subset=["event"]).copy()
    played = fx[
        fx["finished"].fillna(False)
        | (fx["team_h_score"].notna() & fx["team_a_score"].notna())
    ]
    if played.empty:
        return []
    return sorted(int(e) for e in played["event"].unique())


def backfill(con=None, *, through_event: int | None = None) -> int:
    owns = con is None
    con = con or store.connect()
    try:
        events = con.execute("SELECT * FROM events").df()
        fixtures = con.execute("SELECT * FROM fixtures").df()
        players = con.execute("SELECT id, team_id FROM players").df()

        finished = played_events(fixtures, events)
        if through_event:
            finished = [e for e in finished if e <= through_event]
        if not finished:
            print("No played gameweeks yet.")
            return 0

        team_of = dict(zip(players["id"], players["team_id"]))
        lookup = _fixture_lookup(fixtures)

        total = 0
        for ev in finished:
            df = fpl_api.fetch_event_live(ev, snapshot=False)
            if df.empty:
                continue
            # KEEP zero-minute rows. They are the negative class for the
            # minutes model (Stage A): "was in the squad, did not play".
            # Filtering them out leaves a classifier that has only ever seen
            # players who featured, which cannot learn non-selection at all.
            # Rows are still restricted to players whose team had a fixture,
            # so a blank-gameweek player contributes nothing.
            df = df.copy()

            meta = df["element"].map(
                lambda e: lookup.get((ev, team_of.get(e, -1)), (None, None, None))
            )
            df["fixture"] = [m[0] for m in meta]
            df["opponent_team"] = [m[1] for m in meta]
            df["was_home"] = [m[2] for m in meta]
            df = df.dropna(subset=["fixture"])
            df["fixture"] = df["fixture"].astype(int)
            df["opponent_team"] = df["opponent_team"].astype(int)

            n = store.upsert(con, "player_gw", df)
            total += n
            print(f"  GW{ev}: {n} player-match rows")
        return total
    finally:
        if owns:
            con.close()


if __name__ == "__main__":
    n = backfill()
    print(f"\nBackfilled {n} rows into player_gw.")
    sys.exit(0)
