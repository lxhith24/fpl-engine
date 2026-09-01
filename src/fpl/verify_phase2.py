"""Phase 2 verification + coverage gate (Task 14).

Resolves the full live player set and FAILS (exit 1) if coverage falls below
the plan's 98% threshold. This is the gate that stops a silent join failure
from quietly dropping star players out of every downstream feature.

Run:  PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase2
"""
from __future__ import annotations

import sys

import pandas as pd

from . import store
from .resolve import name_map


def main(min_minutes: int = name_map.COVERAGE_MIN_MINUTES) -> int:
    con = store.connect()
    players = con.execute("SELECT * FROM players").df()
    understat = con.execute(
        "SELECT * FROM understat_player_season WHERE season = 2026"
    ).df()
    teams = con.execute("SELECT * FROM teams").df()

    if players.empty or understat.empty:
        print("ERROR: store is empty -- run `python -m fpl.verify_phase1` first.")
        con.close()
        return 1

    res = name_map.resolve(players, understat, teams)
    cov = name_map.coverage(res.mapping, players, min_minutes=min_minutes)

    print(f"FPL players            {len(players)}")
    print(f"Understat players      {len(understat)}")
    print(f"Mapped pairs           {res.stats['n_mapped']}")
    print(f"Unmapped Understat     {res.stats['n_unmapped_understat']}")
    print(f"Unmapped team titles   {res.stats['unmapped_team_titles'] or 'none'}")
    print()
    print(
        f"Coverage (>={min_minutes} mins)  "
        f"{cov['matched']}/{cov['eligible']} = {cov['coverage']:.4f}  "
        f"(gate {name_map.COVERAGE_TARGET:.0%})"
    )

    low = res.mapping[res.mapping["score"] < 95]
    if not low.empty:
        merged = (
            low.merge(
                players[["id", "web_name", "first_name", "second_name"]],
                left_on="element",
                right_on="id",
            )
            .merge(understat[["understat_id", "player_name"]], on="understat_id")
            .sort_values("score")
        )
        print(f"\nLow-confidence matches ({len(merged)}) -- audit these:")
        for _, r in merged.iterrows():
            print(
                f"  {r['score']:5.1f}  FPL {r['first_name']} {r['second_name']!r}"
                f"  <->  US {r['player_name']!r}"
            )

    if len(cov["missing"]):
        print(f"\nUNMATCHED with >={min_minutes} mins:")
        print(cov["missing"].to_string(index=False))

    stale = res.unmatched_understat
    stale = stale[stale["time"] >= min_minutes] if "time" in stale else pd.DataFrame()
    if len(stale):
        print(f"\nUnmatched Understat players with >={min_minutes} mins:")
        print(stale[["player_name", "team_title", "time"]].to_string(index=False))

    # Persist the mapping so downstream feature code joins on a stored table
    # rather than re-running fuzzy matching (and re-deciding) every time.
    if not res.mapping.empty:
        import datetime as _dt

        to_store = res.mapping.copy()
        to_store["season"] = 2026
        to_store["pulled_at"] = _dt.datetime.now()
        n = store.upsert(con, "player_map", to_store)
        print(f"\nPersisted {n} mappings to player_map.")

    con.close()

    if cov["coverage"] < name_map.COVERAGE_TARGET:
        print(
            f"\nFAIL: coverage {cov['coverage']:.4f} below "
            f"{name_map.COVERAGE_TARGET:.2f} gate."
        )
        return 1
    print("\nPASS: coverage gate met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
