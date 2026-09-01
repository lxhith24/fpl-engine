"""Squad and starting-XI optimization (Tasks 30-33).

Mixed-integer linear program via PuLP/CBC.

OBJECTIVE (plan section 9.4 -- aggressive is the DEFAULT for this user):

    maximize  sum_i [ xp_i + gamma * sigma_i - beta * ownership_i ] * x_i

`gamma > 0` is variance-SEEKING: it prefers high-ceiling players over safe
ones. `beta` penalises template picks. Both are CLI-tunable.

Stated plainly, because it is a deliberate choice and not a free lunch:
maximizing variance widens BOTH tails. It raises the chance of a top finish
and the chance of a bad one. It does not raise the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

BUDGET = 1000  # tenths of a million (100.0m)
SQUAD_SIZE = 15
SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11

# Bench players only score via autosubs, so they are worth a small non-zero
# weight -- zero would make the solver fill the bench with 3.8m non-players.
BENCH_WEIGHT = 0.1

# Risk presets are expressed in UNITS OF SPREAD, not raw coefficients, and are
# rescaled per-gameweek by `_calibrate_weights`.
#
# Why: raw coefficients are meaningless without knowing the spread of each
# quantity. Measured on live GW3 (219 players with xP > 2):
#   xP     spread 4.02
#   sigma  spread 0.66   and corr(xP, sigma) = 0.56
#   own%   spread 70.0
# So gamma=0.5 shifted scores by only 0.33 -- less than a tenth of the xP
# spread -- and 'safe', 'balanced' and 'aggressive' returned nearly identical
# squads. The presets below say "tilt by this fraction of the xP spread",
# which behaves consistently whatever the gameweek looks like.
#
# NOTE ON SIGMA: because sigma correlates 0.56 with xP, variance-seeking
# partly just re-selects high-xP players. Ownership is the sharper lever for
# differential-chasing, so aggressive leans on it harder.
RISK_PRESETS = {
    "aggressive": {"gamma_frac": 0.25, "beta_frac": 2.5},
    "balanced": {"gamma_frac": 0.0, "beta_frac": 0.0},
    "safe": {"gamma_frac": -0.20, "beta_frac": -1.0},
}


def _calibrate_weights(
    df: pd.DataFrame, gamma_frac: float, beta_frac: float, *, min_xp: float = 2.0
) -> tuple[float, float]:
    """Convert spread-fractions into raw coefficients for this player pool."""
    pool = df[df["xp"] > min_xp]
    if len(pool) < 20:
        pool = df

    xp_spread = float(pool["xp"].max() - pool["xp"].min()) if len(pool) else 1.0
    xp_spread = max(xp_spread, 1e-6)

    sigma_spread = (
        float(pool["xp_sigma"].max() - pool["xp_sigma"].min())
        if "xp_sigma" in pool.columns and pool["xp_sigma"].notna().any()
        else 0.0
    )
    own_spread = (
        float(pool["selected_by_percent"].max() - pool["selected_by_percent"].min())
        if "selected_by_percent" in pool.columns
        else 0.0
    )

    gamma = (gamma_frac * xp_spread / sigma_spread) if sigma_spread > 1e-6 else 0.0
    beta = (beta_frac * xp_spread / own_spread) if own_spread > 1e-6 else 0.0
    return gamma, beta


@dataclass
class Squad:
    squad: pd.DataFrame
    xi: pd.DataFrame
    bench: pd.DataFrame
    captain: pd.Series
    vice: pd.Series
    formation: str
    total_cost: float
    xi_xp: float
    squad_xp: float
    objective: float
    status: str
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Formation {self.formation}   cost {self.total_cost:.1f}m   "
            f"XI xP {self.xi_xp:.2f}",
            f"Captain: {self.captain['web_name']} ({self.captain['xp']:.2f} xP)"
            f"   Vice: {self.vice['web_name']} ({self.vice['xp']:.2f} xP)",
        ]
        return "\n".join(lines)


def _score(
    df: pd.DataFrame, gamma: float, beta: float, *, ownership_col: str = "selected_by_percent"
) -> pd.Series:
    """Risk-adjusted objective coefficient per player."""
    score = df["xp"].astype(float).copy()
    if gamma and "xp_sigma" in df.columns:
        score = score + gamma * df["xp_sigma"].fillna(0.0).astype(float)
    if beta and ownership_col in df.columns:
        score = score - beta * df[ownership_col].fillna(0.0).astype(float)
    return score


def optimize_squad(
    players: pd.DataFrame,
    *,
    budget: int = BUDGET,
    risk: str = "aggressive",
    gamma: float | None = None,
    beta: float | None = None,
    max_per_club: int = MAX_PER_CLUB,
    locked: list[int] | None = None,
    banned: list[int] | None = None,
    min_start_prob: float = 0.0,
) -> Squad:
    """Pick the best legal 15 and the XI to start from it.

    `players` needs: element, web_name, position, team_id, now_cost, xp
    (optionally xp_sigma, selected_by_percent, p_start).
    """
    preset = RISK_PRESETS.get(risk, RISK_PRESETS["aggressive"])

    df = players.copy()
    for col in ("element", "position", "team_id", "now_cost", "xp"):
        if col not in df.columns:
            raise ValueError(f"players missing required column: {col}")

    if banned:
        df = df[~df["element"].isin(banned)]
    if min_start_prob > 0 and "p_start" in df.columns:
        keep = (df["p_start"] >= min_start_prob) | df["element"].isin(locked or [])
        filtered = df[keep]
        # A hard filter can leave too few players per position to field a
        # legal 15 (5 DEF, 5 MID, 3 FWD, 2 GKP, max 3 per club). Rather than
        # hand CBC an infeasible problem, relax the threshold per position:
        # keep the highest-p_start players needed to satisfy the quota.
        parts = []
        for pos, quota in SQUAD_QUOTA.items():
            pos_keep = filtered[filtered["position"] == pos]
            if len(pos_keep) < quota + 2:  # small buffer for club limits
                pos_all = df[df["position"] == pos].nlargest(
                    quota + 4, "p_start"
                )
                pos_keep = pd.concat([pos_keep, pos_all]).drop_duplicates("element")
            parts.append(pos_keep)
        df = pd.concat(parts, ignore_index=True)

    # One row per player: a double gameweek contributes summed xP.
    if df.duplicated(subset=["element"]).any():
        agg = {c: "first" for c in df.columns if c not in ("xp", "xp_sigma")}
        agg["xp"] = "sum"
        if "xp_sigma" in df.columns:
            agg["xp_sigma"] = "sum"
        df = df.groupby("element", as_index=False).agg(agg)

    df = df.reset_index(drop=True)

    # Calibrate risk weights to THIS gameweek's spreads (see RISK_PRESETS).
    g_auto, b_auto = _calibrate_weights(df, preset["gamma_frac"], preset["beta_frac"])
    g = g_auto if gamma is None else gamma
    b = b_auto if beta is None else beta

    # TWO SCORES, DELIBERATELY.
    #
    # Ownership decides WHO YOU OWN relative to the field -- that is a squad
    # selection question. Once a player is in your 15, whether to start him is
    # a pure points question; charging the differential tax again at XI level
    # produces absurdities.
    #
    # Observed live on GW3: with a single score, Szoboszlai (5.64 xP, 42%
    # owned) scored 6.176 while Ajer (4.92 xP, 4.5% owned) scored 6.277, so
    # the solver benched a player who was 0.73 xP better. Benching someone you
    # already own does not make you more differential in any useful sense; it
    # just scores fewer points.
    df["_squad_score"] = _score(df, g, b)          # ownership counts here
    df["_xi_score"] = _score(df, g, 0.0)           # and not here

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    pick = pulp.LpVariable.dicts("pick", df.index, cat="Binary")
    start = pulp.LpVariable.dicts("start", df.index, cat="Binary")

    # Bench contributes at reduced weight; a starter gets full weight.
    prob += pulp.lpSum(
        BENCH_WEIGHT * df.at[i, "_squad_score"] * pick[i]
        + (1 - BENCH_WEIGHT) * df.at[i, "_xi_score"] * start[i]
        for i in df.index
    )

    prob += pulp.lpSum(pick[i] for i in df.index) == SQUAD_SIZE
    prob += pulp.lpSum(df.at[i, "now_cost"] * pick[i] for i in df.index) <= budget

    for pos, n in SQUAD_QUOTA.items():
        idx = df.index[df["position"] == pos]
        prob += pulp.lpSum(pick[i] for i in idx) == n

    for team in df["team_id"].unique():
        idx = df.index[df["team_id"] == team]
        prob += pulp.lpSum(pick[i] for i in idx) <= max_per_club

    # Starting XI nested inside the squad.
    prob += pulp.lpSum(start[i] for i in df.index) == XI_SIZE
    for i in df.index:
        prob += start[i] <= pick[i]
    for pos in SQUAD_QUOTA:
        idx = df.index[df["position"] == pos]
        prob += pulp.lpSum(start[i] for i in idx) >= XI_MIN[pos]
        prob += pulp.lpSum(start[i] for i in idx) <= XI_MAX[pos]

    for el in locked or []:
        idx = df.index[df["element"] == el]
        if len(idx):
            prob += pick[idx[0]] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"solver returned {status} -- constraints may be infeasible")

    df["_pick"] = [pulp.value(pick[i]) > 0.5 for i in df.index]
    df["_start"] = [pulp.value(start[i]) > 0.5 for i in df.index]

    squad = df[df["_pick"]].copy()
    xi = squad[squad["_start"]].sort_values("xp", ascending=False)
    bench = squad[~squad["_start"]].sort_values("xp", ascending=False)

    # Captain doubles points, so pick on raw xP -- never on the risk-adjusted
    # score, which would hand the armband to a volatile bench-warmer.
    ranked = xi.sort_values("xp", ascending=False)
    captain, vice = ranked.iloc[0], ranked.iloc[1]

    counts = xi["position"].value_counts()
    formation = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"

    return Squad(
        squad=squad.drop(columns=["_pick","_start","_squad_score","_xi_score"], errors="ignore"),
        xi=xi.drop(columns=["_pick","_start","_squad_score","_xi_score"], errors="ignore"),
        bench=bench.drop(columns=["_pick","_start","_squad_score","_xi_score"], errors="ignore"),
        captain=captain,
        vice=vice,
        formation=formation,
        total_cost=float(squad["now_cost"].sum()) / 10,
        xi_xp=float(xi["xp"].sum()),
        squad_xp=float(squad["xp"].sum()),
        objective=float(pulp.value(prob.objective)),
        status=status,
        meta={"risk": risk, "gamma": g, "beta": b, "n_candidates": len(df)},
    )


def optimize_xi(squad: pd.DataFrame, *, gamma: float = 0.0, beta: float = 0.0) -> Squad:
    """Best XI from an EXISTING 15 (no transfers)."""
    df = squad.copy().reset_index(drop=True)
    df["_score"] = _score(df, gamma, beta)

    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    start = pulp.LpVariable.dicts("start", df.index, cat="Binary")
    prob += pulp.lpSum(df.at[i, "_score"] * start[i] for i in df.index)
    prob += pulp.lpSum(start[i] for i in df.index) == XI_SIZE
    for pos in SQUAD_QUOTA:
        idx = df.index[df["position"] == pos]
        prob += pulp.lpSum(start[i] for i in idx) >= XI_MIN[pos]
        prob += pulp.lpSum(start[i] for i in idx) <= XI_MAX[pos]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["_start"] = [pulp.value(start[i]) > 0.5 for i in df.index]

    xi = df[df["_start"]].sort_values("xp", ascending=False)
    bench = df[~df["_start"]].sort_values("xp", ascending=False)
    ranked = xi.sort_values("xp", ascending=False)
    counts = xi["position"].value_counts()

    return Squad(
        squad=df.drop(columns=["_start"]),
        xi=xi.drop(columns=["_start"]),
        bench=bench.drop(columns=["_start"]),
        captain=ranked.iloc[0],
        vice=ranked.iloc[1],
        formation=f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}",
        total_cost=float(df["now_cost"].sum()) / 10,
        xi_xp=float(xi["xp"].sum()),
        squad_xp=float(df["xp"].sum()),
        objective=float(pulp.value(prob.objective)),
        status=pulp.LpStatus[prob.status],
        meta={"mode": "xi_only"},
    )


def gap_to_current(
    optimal: Squad, current_elements: list[int], players: pd.DataFrame
) -> dict:
    """What separates the optimum from the squad actually owned (Task 33).

    The from-scratch optimum is usually unreachable by transfers -- with 0.0 in
    the bank it is a Wildcard target, not an instruction. This quantifies that
    honestly instead of implying a one-week move.
    """
    optimal_ids = set(optimal.squad["element"])
    current_ids = set(current_elements)

    lookup = players.drop_duplicates(subset=["element"]).set_index("element")
    out_ids = current_ids - optimal_ids
    in_ids = optimal_ids - current_ids

    def _rows(ids):
        keep = [i for i in ids if i in lookup.index]
        cols = [c for c in ("web_name", "position", "now_cost", "xp") if c in lookup.columns]
        return lookup.loc[keep, cols].sort_values("xp", ascending=False) if keep else pd.DataFrame()

    n_transfers = len(in_ids)
    return {
        "n_transfers": n_transfers,
        "hit_cost": max(0, n_transfers - 1) * 4,
        "transfers_out": _rows(out_ids),
        "transfers_in": _rows(in_ids),
        "kept": len(current_ids & optimal_ids),
    }
