"""Phase 4 verification: train, gate, and predict the upcoming gameweek.

Run:  PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase4
"""
from __future__ import annotations

import sys

import pandas as pd

from . import store
from .features.build import build_features
from .ingest.fpl_api import current_and_next_event
from .models.train import predict_gameweek


def main() -> int:
    con = store.connect()
    events = con.execute("SELECT * FROM events").df()
    _, nxt = current_and_next_event(events)
    if nxt is None:
        print("No upcoming gameweek.")
        con.close()
        return 1

    print(f"Building features for GW{nxt}...")
    feats = build_features(as_of_event=nxt, con=con)
    if feats.empty:
        print("No features built.")
        con.close()
        return 1

    print(f"Predicting {len(feats)} player-fixtures...")
    pred = predict_gameweek(feats)

    cols = [
        "web_name", "position", "now_cost", "p_start", "exp_minutes",
        "pts_given_play", "zone_fit", "xp", "xp_sigma", "selected_by_percent",
    ]
    have = [c for c in cols if c in pred.columns]

    print(f"\n=== TOP 20 by xP for GW{nxt} ===")
    print(pred[have].head(20).to_string(index=False))

    print("\n=== By position ===")
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sub = pred[pred["position"] == pos].head(5)
        print(f"\n{pos}:")
        print(sub[have].to_string(index=False))

    # Differentials: high xP, low ownership -- the aggressive objective's fuel.
    if "selected_by_percent" in pred.columns:
        diff = pred[(pred["selected_by_percent"] < 5) & (pred["xp"] > 2)]
        print(f"\n=== DIFFERENTIALS (<5% owned, xP>2): {len(diff)} ===")
        print(diff[have].head(10).to_string(index=False))

    out = store.config.REPORTS / f"gw{nxt}_predictions.csv"
    pred.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
