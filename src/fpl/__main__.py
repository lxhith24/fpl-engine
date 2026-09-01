"""FPL Engine CLI entry point.

Usage:
    python -m fpl report          # generate GW briefing
    python -m fpl predict         # just predictions, no report
    python -m fpl optimize        # optimize from cached predictions
    python -m fpl refresh         # full data refresh + predictions
"""
from __future__ import annotations

import argparse
import sys

from . import config, store
from .features.build import build_features
from .ingest.backfill import backfill
from .ingest.fpl_api import (
    current_and_next_event,
    fetch_bootstrap,
    fetch_fixtures,
)
from .ingest.understat import fetch_league_data
from .ingest.understat_shots import fetch_season_shots
from .models.train import predict_gameweek
from .optimize.squad import optimize_squad
from .report.briefing import build_gw_report


def cmd_refresh(args: argparse.Namespace) -> int:
    """Pull latest data from all sources."""
    con = store.connect()
    print("Refreshing FPL bootstrap...", flush=True)
    bootstrap = fetch_bootstrap(snapshot=False)
    store.upsert(con, "players", bootstrap["players"])
    store.upsert(con, "teams", bootstrap["teams"])
    store.upsert(con, "events", bootstrap["events"])

    print("Refreshing fixtures...", flush=True)
    fx = fetch_fixtures(snapshot=False)
    store.upsert(con, "fixtures", fx)

    print("Backfilling player gameweeks...", flush=True)
    n = backfill(con)
    print(f"  {n} rows upserted")

    print("Refreshing Understat...", flush=True)
    us = fetch_league_data("EPL", config.CURRENT_SEASON)
    store.upsert(con, "understat_team_match", us["team_matches"])
    store.upsert(con, "understat_player_season", us["players"])

    print("Refreshing shot data...", flush=True)
    shots = fetch_season_shots("EPL", config.CURRENT_SEASON)
    shots.to_parquet(config.INTERIM / f"shots_{config.CURRENT_SEASON}.parquet", index=False)

    con.close()
    print("Done.")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Generate predictions for the upcoming GW."""
    con = store.connect()
    events = con.execute("SELECT * FROM events").df()
    _, nxt = current_and_next_event(events)
    if nxt is None:
        print("No upcoming gameweek.")
        con.close()
        return 1

    print(f"Building features for GW{nxt}...", flush=True)
    feats = build_features(as_of_event=nxt, con=con)
    print(f"Predicting {len(feats)} player-fixtures...", flush=True)
    pred = predict_gameweek(feats)

    out = config.REPORTS / f"gw{nxt}_predictions.csv"
    pred.to_csv(out, index=False)
    print(f"Saved -> {out}")
    con.close()
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Optimize from cached predictions."""
    con = store.connect()
    events = con.execute("SELECT * FROM events").df()
    _, nxt = current_and_next_event(events)
    con.close()
    if nxt is None:
        print("No upcoming gameweek.")
        return 1

    import pandas as pd
    pred_path = config.REPORTS / f"gw{nxt}_predictions.csv"
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}. Run `python -m fpl predict` first.")
        return 1

    pred = pd.read_csv(pred_path)
    s = optimize_squad(pred, risk=args.risk)
    print(s.summary())
    print("\nXI:")
    cols = ["web_name", "position", "now_cost", "xp", "selected_by_percent"]
    print(s.xi[[c for c in cols if c in s.xi.columns]].to_string(index=False))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate the full GW briefing."""
    report, path = build_gw_report(entry_id=args.entry)
    if args.print:
        print(report)
    print(f"\nSaved -> {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="fpl", description="FPL Optimal XI Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("refresh", help="Pull latest data from all sources")
    sub.add_parser("predict", help="Generate predictions for upcoming GW")

    p_opt = sub.add_parser("optimize", help="Optimize squad from cached predictions")
    p_opt.add_argument("--risk", choices=["safe", "balanced", "aggressive"], default="aggressive")

    p_rep = sub.add_parser("report", help="Generate GW briefing")
    p_rep.add_argument("--entry", type=int, default=config.DEFAULT_ENTRY_ID)
    p_rep.add_argument("--print", action="store_true", help="Print report to stdout")

    args = parser.parse_args()

    handlers = {
        "refresh": cmd_refresh,
        "predict": cmd_predict,
        "optimize": cmd_optimize,
        "report": cmd_report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
