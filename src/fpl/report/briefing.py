"""GW report generation (Task 34).

Produces a markdown briefing with:
- Optimal XI and formation
- Captain/vice picks with reasoning
- Gap analysis vs current squad
- Single best transfer (not Wildcard rebuild)
- Differentials worth watching
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config, store
from ..features.build import build_features
from ..ingest.fpl_api import current_and_next_event, fetch_entry_picks
from ..models.train import predict_gameweek
from ..optimize.squad import Squad, gap_to_current, optimize_squad, optimize_xi


def _fmt_cost(tenths: int) -> str:
    return f"£{tenths / 10:.1f}m"


def best_single_transfer(
    current_elements: list[int],
    predictions: pd.DataFrame,
    bank: float = 0.0,
) -> dict | None:
    """Find the single transfer that maximises xP gain.

    With 0 FTs this costs -4; with 1 FT it's free. Either way, this is what
    a non-Wildcard week should consider.
    """
    pred = predictions.set_index("element")
    current = pred.loc[pred.index.isin(current_elements)].copy()
    pool = pred.loc[~pred.index.isin(current_elements)].copy()

    if current.empty or pool.empty:
        return None

    best = None
    for out_id, out_row in current.iterrows():
        affordable = pool[pool["now_cost"] <= out_row["now_cost"] + bank * 10]
        if affordable.empty:
            continue
        # Same position replacement keeps formation options open.
        same_pos = affordable[affordable["position"] == out_row["position"]]
        candidates = same_pos if not same_pos.empty else affordable

        top = candidates.nlargest(1, "xp").iloc[0]
        gain = top["xp"] - out_row["xp"]
        if gain > 0 and (best is None or gain > best["gain"]):
            best = {
                "out_id": int(out_id),
                "out_name": out_row["web_name"],
                "out_pos": out_row["position"],
                "out_cost": out_row["now_cost"],
                "out_xp": float(out_row["xp"]),
                "in_id": int(top.name),
                "in_name": top["web_name"],
                "in_pos": top["position"],
                "in_cost": top["now_cost"],
                "in_xp": float(top["xp"]),
                "gain": float(gain),
                "net_cost": float((top["now_cost"] - out_row["now_cost"]) / 10),
            }
    return best


def generate_report(
    gw: int,
    predictions: pd.DataFrame,
    optimal: Squad,
    current_squad: Squad | None = None,
    current_elements: list[int] | None = None,
    transfer: dict | None = None,
    entry_id: int | None = None,
) -> str:
    """Render the full markdown briefing."""
    lines = [
        f"# GW{gw} Briefing",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} IST*",
        "",
    ]

    # --- optimal XI ---
    lines.append("## Optimal XI (aggressive mode)")
    lines.append(f"**Formation {optimal.formation}** | XI xP **{optimal.xi_xp:.2f}** | "
                 f"cost {optimal.total_cost:.1f}m | mean ownership {optimal.xi['selected_by_percent'].mean():.1f}%")
    lines.append("")
    lines.append(f"**Captain:** {optimal.captain['web_name']} ({optimal.captain['xp']:.2f} xP)  ")
    lines.append(f"**Vice:** {optimal.vice['web_name']} ({optimal.vice['xp']:.2f} xP)")
    lines.append("")
    lines.append("| Player | Pos | Cost | xP | σ | Own% |")
    lines.append("|--------|-----|------|---:|--:|-----:|")
    for _, r in optimal.xi.iterrows():
        lines.append(
            f"| {r['web_name']} | {r['position']} | {_fmt_cost(r['now_cost'])} | "
            f"{r['xp']:.2f} | {r.get('xp_sigma', 0):.2f} | {r.get('selected_by_percent', 0):.1f} |"
        )
    lines.append("")
    lines.append("**Bench:** " + ", ".join(
        f"{r['web_name']} ({r['xp']:.2f})" for _, r in optimal.bench.iterrows()
    ))
    lines.append("")

    # --- current squad comparison ---
    if current_squad is not None and current_elements is not None:
        gap = gap_to_current(optimal, current_elements, predictions)
        lines.append("## Your Squad")
        lines.append(f"**Best legal XI:** {current_squad.xi_xp:.2f} xP ({current_squad.formation})")
        lines.append(f"**Gap to optimum:** +{optimal.xi_xp - current_squad.xi_xp:.2f} xP")
        lines.append(f"**Transfers needed:** {gap['n_transfers']} (hit cost −{gap['hit_cost']} pts)")
        lines.append("")
        if gap["n_transfers"] > 2:
            lines.append("> With £0.0 in the bank, closing this gap is **Wildcard territory**, not a weekly transfer.")
            lines.append("")

    # --- single best transfer ---
    if transfer:
        lines.append("## Best Single Transfer")
        lines.append(
            f"**OUT:** {transfer['out_name']} ({transfer['out_pos']}, "
            f"{_fmt_cost(transfer['out_cost'])}, {transfer['out_xp']:.2f} xP)"
        )
        lines.append(
            f"**IN:** {transfer['in_name']} ({transfer['in_pos']}, "
            f"{_fmt_cost(transfer['in_cost'])}, {transfer['in_xp']:.2f} xP)"
        )
        lines.append(f"**Gain:** +{transfer['gain']:.2f} xP | net cost {transfer['net_cost']:+.1f}m")
        lines.append("")

    # --- differentials ---
    diff = predictions[
        (predictions["selected_by_percent"] < 5) & (predictions["xp"] > 4)
    ].nlargest(8, "xp")
    if not diff.empty:
        lines.append("## Differentials (<5% owned, xP > 4)")
        lines.append("| Player | Pos | xP | Own% |")
        lines.append("|--------|-----|---:|-----:|")
        for _, r in diff.iterrows():
            lines.append(f"| {r['web_name']} | {r['position']} | {r['xp']:.2f} | {r['selected_by_percent']:.1f} |")
        lines.append("")

    # --- zone fit highlights ---
    if "zone_fit" in predictions.columns:
        zf = predictions[predictions["zone_fit"] != 1.0]
        best_fit = zf.nlargest(3, "zone_fit")
        if not best_fit.empty:
            lines.append("## Matchup Boosts (zone fit)")
            for _, r in best_fit.iterrows():
                lines.append(f"- **{r['web_name']}** {r['zone_fit']:.2f}× vs opponent weak zones")
            lines.append("")

    return "\n".join(lines)


def build_gw_report(
    entry_id: int = config.DEFAULT_ENTRY_ID,
    con=None,
) -> tuple[str, Path]:
    """Full pipeline: predict, optimize, compare, report."""
    owns = con is None
    con = con or store.connect()

    events = con.execute("SELECT * FROM events").df()
    cur, nxt = current_and_next_event(events)
    if nxt is None:
        raise RuntimeError("No upcoming gameweek")

    # Predictions (cached if already run today).
    pred_path = config.REPORTS / f"gw{nxt}_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
    else:
        feats = build_features(as_of_event=nxt, con=con)
        pred = predict_gameweek(feats)
        pred.to_csv(pred_path, index=False)

    optimal = optimize_squad(pred, risk="aggressive")

    # Current squad.
    current_squad = None
    current_elements = None
    transfer = None
    try:
        picks = fetch_entry_picks(entry_id, cur or nxt - 1)
        current_elements = [p["element"] for p in picks["picks"]]
        owned = pred[pred["element"].isin(current_elements)]
        if not owned.empty:
            current_squad = optimize_xi(owned)
            transfer = best_single_transfer(current_elements, pred, bank=0.0)
    except Exception:  # noqa: BLE001
        pass

    report = generate_report(
        gw=nxt,
        predictions=pred,
        optimal=optimal,
        current_squad=current_squad,
        current_elements=current_elements,
        transfer=transfer,
        entry_id=entry_id,
    )

    out_path = config.REPORTS / f"gw{nxt}_report.md"
    out_path.write_text(report)

    if owns:
        con.close()

    return report, out_path
