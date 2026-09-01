#!/usr/bin/env python3
"""Daily cron script for FPL briefing (Task 35).

Runs at 09:00 IST daily. Checks if the next GW deadline is within 24h;
if so, refreshes data, generates predictions, and outputs the briefing.

The output goes to stdout (Hermes captures and delivers it).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

# Add src to path for cron execution.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpl import config, store
from fpl.ingest.fpl_api import current_and_next_event, fetch_bootstrap, fetch_fixtures
from fpl.ingest.backfill import backfill
from fpl.report.briefing import build_gw_report

IST = timezone(timedelta(hours=5, minutes=30))


def main() -> int:
    # Check deadline proximity.
    con = store.connect()

    # Quick refresh of events to get latest deadline.
    bootstrap = fetch_bootstrap(snapshot=False)
    players, teams, events = bootstrap["players"], bootstrap["teams"], bootstrap["events"]
    store.upsert(con, "players", players)
    store.upsert(con, "teams", teams)
    store.upsert(con, "events", events)
    fx = fetch_fixtures(snapshot=False)
    store.upsert(con, "fixtures", fx)

    cur, nxt = current_and_next_event(events)
    if nxt is None:
        print("No upcoming gameweek — season may be over.")
        con.close()
        return 0

    deadline_ts = events[events["id"] == nxt]["deadline_time"].iloc[0]
    # deadline_ts may be a pd.Timestamp or a string depending on how recently
    # the data was refreshed; normalise to UTC datetime.
    if hasattr(deadline_ts, "to_pydatetime"):
        deadline = deadline_ts.to_pydatetime()
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
    else:
        deadline = datetime.fromisoformat(str(deadline_ts).replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)

    hours_to_deadline = (deadline - now).total_seconds() / 3600

    if hours_to_deadline > 24:
        print(f"GW{nxt} deadline is {deadline.astimezone(IST).strftime('%a %d %b %H:%M')} IST "
              f"({hours_to_deadline:.0f}h away) — skipping.")
        con.close()
        return 0

    if hours_to_deadline < 0:
        print(f"GW{nxt} deadline has passed — waiting for next GW.")
        con.close()
        return 0

    # Deadline within 24h — generate briefing.
    print(f"⚽ GW{nxt} deadline in {hours_to_deadline:.1f}h — generating briefing...\n")

    # Full data refresh.
    backfill(con)
    con.close()

    # Generate report.
    report, _ = build_gw_report()
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
