"""Phase 0-1 verification: pull everything live and load it into DuckDB.

Run:  ./.venv/bin/python -m fpl.verify_phase1
"""
from __future__ import annotations

import sys

from . import config, store
from .ingest import fpl_api, understat


def main() -> int:
    con = store.connect()
    print(f"DB: {config.DB_PATH}")

    # --- FPL bootstrap ---------------------------------------------------
    boot = fpl_api.fetch_bootstrap()
    for table in ("teams", "players", "events"):
        n = store.upsert(con, table, boot[table])
        print(f"  {table:<24} {n:>5} rows")
    cur, nxt = fpl_api.current_and_next_event(boot["events"])
    print(f"  current GW={cur}  next GW={nxt}")

    # --- fixtures --------------------------------------------------------
    fixtures = fpl_api.fetch_fixtures()
    print(f"  {'fixtures':<24} {store.upsert(con, 'fixtures', fixtures):>5} rows")

    dgw = fpl_api.detect_dgw_bgw(fixtures)
    upcoming = dgw[dgw["event"] >= (nxt or 1)]
    doubles = upcoming[upcoming["kind"] == "double"]
    blanks = upcoming[upcoming["kind"] == "blank"]
    print(f"  upcoming doubles={len(doubles)}  blanks={len(blanks)}")

    # --- element summaries (sample) --------------------------------------
    top = (
        boot["players"].sort_values("total_points", ascending=False).head(5)
    )
    total = 0
    for _, row in top.iterrows():
        hist = fpl_api.fetch_element_summary(int(row["id"]))
        total += store.upsert(con, "player_gw", hist)
    print(f"  {'player_gw (5 players)':<24} {total:>5} rows")

    # --- event live ------------------------------------------------------
    if cur:
        live = fpl_api.fetch_event_live(cur)
        print(f"  event {cur} live returns    {len(live):>5} players")

    # --- Understat -------------------------------------------------------
    us = understat.fetch_league_data("EPL", 2026)
    n_tm = store.upsert(con, "understat_team_match", us["team_matches"])
    n_pl = store.upsert(con, "understat_player_season", us["players"])
    print(f"  {'understat_team_match':<24} {n_tm:>5} rows")
    print(f"  {'understat_player_season':<24} {n_pl:>5} rows")

    # --- joined sanity check: opponent xGA leaderboard --------------------
    print("\nWeakest defences so far (mean xGA/match, the matchup layer's core input):")
    rows = con.execute(
        """
        SELECT team_title,
               count(*)            AS matches,
               round(avg(xGA), 2)  AS mean_xGA,
               round(avg(xG), 2)   AS mean_xG
        FROM understat_team_match
        WHERE season = 2026
        GROUP BY team_title
        HAVING count(*) > 0
        ORDER BY mean_xGA DESC
        LIMIT 5
        """
    ).fetchall()
    for t, m, xga, xg in rows:
        print(f"    {t:<20} {m} matches   xGA {xga}   xG {xg}")

    con.close()
    print("\nPhase 0-1 verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
