"""Assemble the current-gameweek picture the dashboard and chatbot share.

`load_snapshot` reads only cached artifacts: the predictions CSV written by
`python -m fpl predict`, the DuckDB store, and one best-effort FPL picks call
for the manager's actual squad. No model training, no optimiser re-runs beyond
the two MILPs that `build_gw_report` would run anyway.

`build_snapshot` is the pure core -- hand it DataFrames and it returns a
`Snapshot` -- so the digest and the dashboard payload are unit-testable without
a database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config, store
from ..ingest.fpl_api import current_and_next_event, fetch_entry_picks
from ..optimize.squad import Squad, gap_to_current, optimize_squad, optimize_xi
from ..report.briefing import best_single_transfer

# Columns handed to the front-end predictions explorer. Everything else in the
# 125-column feature frame is model plumbing the user does not need to see.
_TABLE_COLS = [
    "element", "web_name", "team", "position", "opponent", "is_home",
    "now_cost", "status", "selected_by_percent", "form",
    "p_start", "exp_minutes", "zone_fit", "difficulty", "xp_sigma", "xp",
]


def _f(value, digits: int | None = None):
    """Coerce a numpy/pandas scalar to a plain JSON-safe number."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(num):
        return None
    return round(num, digits) if digits is not None else num


def _player_view(row: pd.Series, teams: dict[int, str]) -> dict:
    """Slim, display-ready dict for one player-fixture row."""
    team_id = row.get("team_id")
    opp_id = row.get("opponent_id")
    cost_tenths = _f(row.get("now_cost"))
    return {
        "element": int(row["element"]) if "element" in row else None,
        "web_name": row.get("web_name"),
        "position": row.get("position"),
        "team": teams.get(int(team_id), str(team_id)) if pd.notna(team_id) else None,
        "opponent": teams.get(int(opp_id), str(opp_id)) if pd.notna(opp_id) else None,
        "is_home": bool(row["is_home"]) if pd.notna(row.get("is_home")) else None,
        "now_cost": round(cost_tenths / 10, 1) if cost_tenths is not None else None,
        "status": row.get("status"),
        "selected_by_percent": _f(row.get("selected_by_percent"), 1),
        "form": _f(row.get("form"), 1),
        "p_start": _f(row.get("p_start"), 2),
        "exp_minutes": _f(row.get("exp_minutes"), 0),
        "zone_fit": _f(row.get("zone_fit"), 3),
        "difficulty": _f(row.get("difficulty"), 0),
        "xp_sigma": _f(row.get("xp_sigma"), 2),
        "xp": _f(row.get("xp"), 2),
    }


def _squad_view(sq: Squad, teams: dict[int, str]) -> dict:
    return {
        "formation": sq.formation,
        "xi_xp": _f(sq.xi_xp, 2),
        "squad_xp": _f(sq.squad_xp, 2),
        "total_cost": _f(sq.total_cost, 1),
        "mean_ownership": _f(sq.xi["selected_by_percent"].mean(), 1)
        if "selected_by_percent" in sq.xi.columns
        else None,
        "captain": {"web_name": sq.captain["web_name"], "xp": _f(sq.captain["xp"], 2)},
        "vice": {"web_name": sq.vice["web_name"], "xp": _f(sq.vice["xp"], 2)},
        "xi": [_player_view(r, teams) for _, r in sq.xi.iterrows()],
        "bench": [_player_view(r, teams) for _, r in sq.bench.iterrows()],
    }


@dataclass
class Snapshot:
    gw: int
    generated_at: str | None
    has_predictions: bool
    entry_id: int
    teams: dict[int, str]
    predictions: pd.DataFrame
    optimal: Squad | None = None
    your_squad: Squad | None = None
    your_elements: list[int] | None = None
    gap: dict | None = None
    transfer: dict | None = None
    report_md: str | None = None
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- outputs
    def to_dashboard_dict(self) -> dict:
        teams = self.teams
        out: dict = {
            "gw": self.gw,
            "generated_at": self.generated_at,
            "has_predictions": self.has_predictions,
            "entry_id": self.entry_id,
            "notes": self.notes,
            "teams": {str(k): v for k, v in teams.items()},
        }
        if not self.has_predictions:
            return out

        out["optimal"] = _squad_view(self.optimal, teams) if self.optimal else None

        if self.your_squad is not None:
            ys = _squad_view(self.your_squad, teams)
            ys.update(
                available=True,
                gap=_f((self.optimal.xi_xp - self.your_squad.xi_xp), 2)
                if self.optimal
                else None,
                n_transfers=self.gap["n_transfers"] if self.gap else None,
                hit_cost=self.gap["hit_cost"] if self.gap else None,
                transfer=self._transfer_view(),
            )
            out["your_squad"] = ys
        else:
            out["your_squad"] = {"available": False, "reason": "; ".join(self.notes) or None}

        pred = self.predictions.copy()
        pred["team"] = pred["team_id"].map(teams).fillna(pred["team_id"].astype(str))
        pred["opponent"] = pred.get("opponent_id", pd.Series(index=pred.index)).map(teams)
        rows = []
        for _, r in pred.sort_values("xp", ascending=False).iterrows():
            v = _player_view(r, teams)
            rows.append({k: v.get(k) for k in _TABLE_COLS})
        out["predictions"] = rows

        out["differentials"] = [
            {
                "web_name": r["web_name"],
                "position": r["position"],
                "team": teams.get(int(r["team_id"]), str(r["team_id"])),
                "xp": _f(r["xp"], 2),
                "selected_by_percent": _f(r["selected_by_percent"], 1),
            }
            for _, r in pred[
                (pred["selected_by_percent"] < 5) & (pred["xp"] > 4)
            ]
            .nlargest(12, "xp")
            .iterrows()
        ]

        if "zone_fit" in pred.columns:
            boosts = pred[pred["zone_fit"] != 1.0].nlargest(8, "zone_fit")
            out["matchup_boosts"] = [
                {
                    "web_name": r["web_name"],
                    "position": r["position"],
                    "team": teams.get(int(r["team_id"]), str(r["team_id"])),
                    "opponent": teams.get(int(r.get("opponent_id")), None)
                    if pd.notna(r.get("opponent_id"))
                    else None,
                    "zone_fit": _f(r["zone_fit"], 2),
                    "xp": _f(r["xp"], 2),
                }
                for _, r in boosts.iterrows()
            ]
        else:
            out["matchup_boosts"] = []

        out["report_md"] = self.report_md
        return out

    def _transfer_view(self) -> dict | None:
        t = self.transfer
        if not t:
            return None
        return {
            "out": {
                "web_name": t["out_name"],
                "position": t["out_pos"],
                "now_cost": _f(t["out_cost"]) / 10,
                "xp": _f(t["out_xp"], 2),
            },
            "in": {
                "web_name": t["in_name"],
                "position": t["in_pos"],
                "now_cost": _f(t["in_cost"]) / 10,
                "xp": _f(t["in_xp"], 2),
            },
            "gain": _f(t["gain"], 2),
            "net_cost": _f(t["net_cost"], 1),
        }

    def to_chat_context(self, pool_min_minutes: float = 10.0) -> str:
        """Compact text digest that becomes the chatbot's grounding context."""
        if not self.has_predictions:
            return (
                f"No predictions are available for GW{self.gw}. "
                "The user needs to run `python -m fpl predict` first."
            )

        teams = self.teams
        L: list[str] = [
            f"GAMEWEEK {self.gw}",
            f"Predictions generated: {self.generated_at or 'unknown'}",
            f"Manager entry id: {self.entry_id}",
            "",
        ]

        if self.optimal:
            o = self.optimal
            L.append("== FROM-SCRATCH OPTIMAL SQUAD (risk=aggressive) ==")
            L.append(
                f"Formation {o.formation} | XI xP {o.xi_xp:.2f} | cost "
                f"{o.total_cost:.1f}m | mean ownership "
                f"{o.xi['selected_by_percent'].mean():.1f}%"
            )
            L.append(
                f"Captain {o.captain['web_name']} ({o.captain['xp']:.2f} xP), "
                f"Vice {o.vice['web_name']} ({o.vice['xp']:.2f} xP)"
            )
            L.append("Starting XI:")
            for _, r in o.xi.iterrows():
                L.append("  " + _digest_line(r, teams))
            L.append("Bench: " + ", ".join(
                f"{r['web_name']} ({r['xp']:.2f})" for _, r in o.bench.iterrows()
            ))
            L.append("")

        L.append("== MANAGER'S ACTUAL SQUAD ==")
        if self.your_squad is not None:
            ys = self.your_squad
            gap = (self.optimal.xi_xp - ys.xi_xp) if self.optimal else None
            L.append(
                f"Best legal XI: {ys.xi_xp:.2f} xP ({ys.formation})"
                + (f" | gap to optimum +{gap:.2f} xP" if gap is not None else "")
            )
            if self.gap:
                L.append(
                    f"Transfers to reach optimum: {self.gap['n_transfers']} "
                    f"(hit cost -{self.gap['hit_cost']} pts)"
                )
            L.append("Starting XI:")
            for _, r in ys.xi.iterrows():
                L.append("  " + _digest_line(r, teams))
            L.append("Bench: " + ", ".join(
                f"{r['web_name']} ({r['xp']:.2f})" for _, r in ys.bench.iterrows()
            ))
            if self.transfer:
                t = self.transfer
                L.append(
                    f"Best single transfer: OUT {t['out_name']} ({t['out_xp']:.2f} xP) "
                    f"-> IN {t['in_name']} ({t['in_xp']:.2f} xP), "
                    f"+{t['gain']:.2f} xP, net cost {t['net_cost']:+.1f}m"
                )
        else:
            L.append("Not available: " + ("; ".join(self.notes) or "unknown reason"))
        L.append("")

        pred = self.predictions
        if "exp_minutes" in pred.columns:
            pool = pred[pred["exp_minutes"].fillna(0) >= pool_min_minutes]
        else:
            pool = pred
        pool = pool.sort_values("xp", ascending=False)
        L.append(
            f"== PROJECTED PLAYER POOL ({len(pool)} players with >= "
            f"{pool_min_minutes:.0f} expected minutes, best xP first) =="
        )
        L.append(
            "name | team | pos | opp(H/A) | price | own% | p_start | xmin | zone_fit | xP | xP_sigma"
        )
        for _, r in pool.iterrows():
            L.append(_digest_line(r, teams, full=True))
        L.append("")
        L.append(
            "Players not in this pool are projected to barely feature; if asked "
            "about one, say it is outside the projected pool for GW"
            f"{self.gw}."
        )
        return "\n".join(L)


def _digest_line(r: pd.Series, teams: dict[int, str], *, full: bool = False) -> str:
    team = teams.get(int(r["team_id"]), str(r["team_id"])) if pd.notna(r.get("team_id")) else "?"
    opp = teams.get(int(r["opponent_id"])) if pd.notna(r.get("opponent_id")) else "?"
    ha = "H" if r.get("is_home") else "A"
    price = f"{float(r['now_cost']) / 10:.1f}" if pd.notna(r.get("now_cost")) else "?"
    own = f"{float(r['selected_by_percent']):.1f}" if pd.notna(r.get("selected_by_percent")) else "?"
    if not full:
        return (
            f"{r['web_name']} ({r['position']}, {team} vs {opp} {ha}) "
            f"{price}m, {own}% own, {float(r['xp']):.2f} xP"
        )
    pstart = f"{float(r['p_start']):.2f}" if pd.notna(r.get("p_start")) else "?"
    xmin = f"{float(r['exp_minutes']):.0f}" if pd.notna(r.get("exp_minutes")) else "?"
    zf = f"{float(r['zone_fit']):.2f}" if pd.notna(r.get("zone_fit")) else "1.00"
    sig = f"{float(r['xp_sigma']):.2f}" if pd.notna(r.get("xp_sigma")) else "?"
    return (
        f"{r['web_name']} | {team} | {r['position']} | {opp}({ha}) | {price} | "
        f"{own} | {pstart} | {xmin} | {zf} | {float(r['xp']):.2f} | {sig}"
    )


# -------------------------------------------------------------------- assembly
def build_snapshot(
    gw: int,
    predictions: pd.DataFrame | None,
    teams: dict[int, str],
    *,
    entry_id: int,
    your_elements: list[int] | None = None,
    report_md: str | None = None,
    generated_at: str | None = None,
    notes: list[str] | None = None,
) -> Snapshot:
    """Pure core: turn raw frames into a `Snapshot`. No I/O."""
    notes = list(notes or [])
    if predictions is None or predictions.empty:
        return Snapshot(
            gw=gw, generated_at=generated_at, has_predictions=False,
            entry_id=entry_id, teams=teams, predictions=pd.DataFrame(), notes=notes,
        )

    pred = predictions.copy()
    optimal = optimize_squad(pred, risk="aggressive")

    your_squad = gap = transfer = None
    if your_elements:
        owned = pred[pred["element"].isin(your_elements)]
        if not owned.empty:
            your_squad = optimize_xi(owned)
            gap = gap_to_current(optimal, your_elements, pred)
            transfer = best_single_transfer(your_elements, pred, bank=0.0)
        else:
            notes.append("none of the manager's picks are in the predictions frame")

    return Snapshot(
        gw=gw,
        generated_at=generated_at,
        has_predictions=True,
        entry_id=entry_id,
        teams=teams,
        predictions=pred,
        optimal=optimal,
        your_squad=your_squad,
        your_elements=your_elements,
        gap=gap,
        transfer=transfer,
        report_md=report_md,
        notes=notes,
    )


def load_snapshot(entry_id: int = config.DEFAULT_ENTRY_ID, con=None) -> Snapshot:
    """Gather the snapshot from disk + DuckDB + one best-effort FPL picks call."""
    owns = con is None
    con = con or store.connect()
    try:
        events = con.execute("SELECT * FROM events").df()
        cur, nxt = current_and_next_event(events)
        gw = nxt if nxt is not None else (cur or 1)

        teams = {
            int(tid): short
            for tid, short in con.execute(
                "SELECT id, short_name FROM teams"
            ).fetchall()
        }

        pred_path = config.REPORTS / f"gw{gw}_predictions.csv"
        notes: list[str] = []
        predictions = None
        generated_at = None
        if pred_path.exists():
            predictions = pd.read_csv(pred_path)
            if "built_at" in predictions.columns and len(predictions):
                generated_at = str(predictions["built_at"].iloc[0])
            else:
                generated_at = datetime.fromtimestamp(
                    pred_path.stat().st_mtime
                ).isoformat(timespec="seconds")
        else:
            notes.append(
                f"no predictions file at {pred_path.name}; run `python -m fpl predict`"
            )

        report_path = config.REPORTS / f"gw{gw}_report.md"
        report_md = report_path.read_text() if report_path.exists() else None

        your_elements = None
        if predictions is not None:
            try:
                picks = fetch_entry_picks(entry_id, cur or gw - 1)
                your_elements = [p["element"] for p in picks["picks"]]
            except Exception as exc:  # noqa: BLE001 -- offline / bad entry id / preseason
                notes.append(f"could not fetch manager picks ({exc.__class__.__name__})")

        return build_snapshot(
            gw=gw,
            predictions=predictions,
            teams=teams,
            entry_id=entry_id,
            your_elements=your_elements,
            report_md=report_md,
            generated_at=generated_at,
            notes=notes,
        )
    finally:
        if owns:
            con.close()
