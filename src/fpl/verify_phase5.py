"""Phase 5 verification: optimize the upcoming gameweek and compare to your squad.

Run:  PYTHONPATH=src ./.venv/bin/python -m fpl.verify_phase5
"""
from __future__ import annotations

import sys

import pandas as pd

from . import config, store
from .features.build import build_features
from .ingest.fpl_api import current_and_next_event, fetch_entry_picks
from .models.train import predict_gameweek
from .optimize.squad import gap_to_current, optimize_squad, optimize_xi

COLS = ["web_name", "position", "now_cost", "xp", "xp_sigma", "selected_by_percent"]


def main(entry_id: int = config.DEFAULT_ENTRY_ID) -> int:
    con = store.connect()
    events = con.execute("SELECT * FROM events").df()
    cur, nxt = current_and_next_event(events)
    if nxt is None:
        print("No upcoming gameweek.")
        con.close()
        return 1

    pred_path = config.REPORTS / f"gw{nxt}_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
    else:
        pred = predict_gameweek(build_features(as_of_event=nxt, con=con))
        pred.to_csv(pred_path, index=False)

    print(f"=== GW{nxt} OPTIMIZATION ({len(pred)} candidates) ===\n")

    results = {}
    print(f"{'mode':<12}{'XI xP':>8}{'sigma':>8}{'own%':>8}{'cost':>7}  formation  captain")
    for mode in ("safe", "balanced", "aggressive"):
        s = optimize_squad(pred, risk=mode)
        results[mode] = s
        print(
            f"{mode:<12}{s.xi['xp'].sum():>8.2f}{s.xi['xp_sigma'].sum():>8.2f}"
            f"{s.xi['selected_by_percent'].mean():>8.1f}{s.total_cost:>7.1f}"
            f"  {s.formation:<10} {s.captain['web_name']}"
        )

    best = results["aggressive"]
    print(f"\n=== AGGRESSIVE SQUAD (your default) ===")
    print(best.summary())
    print("\nSTARTING XI:")
    print(best.xi[COLS].to_string(index=False))
    print("\nBENCH (autosub order):")
    print(best.bench[COLS].to_string(index=False))

    # --- compare against the squad actually owned --------------------------
    try:
        picks = fetch_entry_picks(entry_id, cur or nxt - 1)
        current = [p["element"] for p in picks["picks"]]
    except Exception as exc:  # noqa: BLE001
        print(f"\n(could not fetch entry {entry_id}: {exc})")
        con.close()
        return 0

    owned = pred[pred["element"].isin(current)]
    if not owned.empty:
        current_xi = optimize_xi(owned)
        print(f"\n=== YOUR SQUAD ({entry_id}) — best legal XI ===")
        print(current_xi.summary())
        print(current_xi.xi[COLS].to_string(index=False))

        gap = gap_to_current(best, current, pred)
        print(f"\n=== GAP TO OPTIMUM ===")
        print(
            f"  your best XI  {current_xi.xi['xp'].sum():.2f} xP\n"
            f"  optimum XI    {best.xi['xp'].sum():.2f} xP\n"
            f"  difference    {best.xi['xp'].sum() - current_xi.xi['xp'].sum():+.2f} xP"
        )
        print(
            f"  {gap['n_transfers']} transfers needed "
            f"(kept {gap['kept']}/15) — hit cost -{gap['hit_cost']} pts"
        )
        print(
            "\n  With 0.0 in the bank this is a WILDCARD target, not a one-week move.\n"
            "  A single transfer captures only a fraction of that gap."
        )

    out = config.REPORTS / f"gw{nxt}_squad.csv"
    best.squad.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
