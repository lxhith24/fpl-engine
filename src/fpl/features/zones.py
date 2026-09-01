"""Shot-zone maps and player/opponent zone fit (Tasks 17-18).

This is the mechanism behind "this left-winger faces their weak right-back".

Understat shot coordinates are normalized 0-1 in the ATTACKING direction:
  X = 0 own goal line ... 1 opponent goal line
  Y = 0 one touchline ... 1 the other

Two maps are built:
  * player attack map   -- where a player takes his shots (share of xG by zone)
  * opponent concession -- where a defence allows shots (xG per match by zone)

The fit multiplier is the dot product of the player's attacking distribution
with the opponent's zone weakness relative to league average. A player who
shoots from exactly where an opponent leaks gets a multiplier > 1.

CRITICAL ORIENTATION NOTE: Y is recorded from the SHOOTER's perspective, so a
left-sided attacker's shots and the defending side's left channel are the same
Y band. No mirroring is applied -- both maps are built from the same shot rows,
one grouped by attacker and one by defender, so they are automatically in the
same frame of reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .shrinkage import K_ZONE, shrink

# Lateral bands (Y) and distance bands (X), tuned empirically against 548
# non-penalty shots (see `scripts/` note in README). The first cut
# (X=.70/.84, Y=.35/.65) put 80.7% of all xG into `close_central`, which makes
# the fit dot-product nearly a constant. These edges bring that to ~62%.
#
# The residual central dominance is real football, not a binning artifact:
# 36% of xG comes from X>=0.94 alone, and 58% of in-box xG sits in
# Y 0.44-0.56. The discriminating signal is therefore the FLANK shares, which
# is exactly where a winger and a striker differ.
X_EDGES = [0.0, 0.78, 0.88, 1.0]
X_LABELS = ["far", "mid", "close"]

Y_EDGES = [0.0, 0.40, 0.60, 1.0]
Y_LABELS = ["left", "central", "right"]

ZONES = [f"{x}_{y}" for x in X_LABELS for y in Y_LABELS]


def assign_zones(shots: pd.DataFrame) -> pd.DataFrame:
    """Label each shot with its `<distance>_<lateral>` zone."""
    if shots.empty:
        return shots.assign(zone=pd.Series(dtype="object"))
    out = shots.copy()
    out["x_band"] = pd.cut(
        out["X"], bins=X_EDGES, labels=X_LABELS, include_lowest=True
    )
    out["y_band"] = pd.cut(
        out["Y"], bins=Y_EDGES, labels=Y_LABELS, include_lowest=True
    )
    out["zone"] = (
        out["x_band"].astype(str) + "_" + out["y_band"].astype(str)
    )
    return out.drop(columns=["x_band", "y_band"])


def _pivot(df: pd.DataFrame, index: str, value: str) -> pd.DataFrame:
    """Zone-columned table, guaranteeing every zone exists."""
    if df.empty:
        return pd.DataFrame(columns=ZONES)
    tbl = df.pivot_table(
        index=index, columns="zone", values=value, aggfunc="sum", fill_value=0.0
    )
    for z in ZONES:
        if z not in tbl.columns:
            tbl[z] = 0.0
    return tbl[ZONES]


def player_attack_map(
    shots: pd.DataFrame, *, exclude_penalties: bool = True
) -> pd.DataFrame:
    """Per-player share of xG by zone (rows sum to 1).

    Penalties are excluded by default: every penalty is the same shot from the
    same spot, so including them makes every penalty taker look identical and
    swamps the open-play signal that actually varies by opponent.
    """
    if shots.empty:
        return pd.DataFrame(columns=ZONES)
    df = shots
    if exclude_penalties and "situation" in df.columns:
        df = df[df["situation"] != "Penalty"]
    df = assign_zones(df)
    totals = _pivot(df, "understat_player_id", "xG")
    row_sum = totals.sum(axis=1)
    shares = totals.div(row_sum.where(row_sum > 0, np.nan), axis=0).fillna(0.0)
    shares["n_shots"] = df.groupby("understat_player_id").size().reindex(shares.index, fill_value=0)
    shares["total_xg"] = row_sum
    return shares


def opponent_concession_map(
    shots: pd.DataFrame, *, exclude_penalties: bool = True
) -> pd.DataFrame:
    """Per-defending-team xG conceded per match, by zone.

    A shot's `defending_title` is the team that allowed it -- this is the
    opponent profile the matchup layer consumes.
    """
    if shots.empty:
        return pd.DataFrame(columns=ZONES)
    df = shots
    if exclude_penalties and "situation" in df.columns:
        df = df[df["situation"] != "Penalty"]
    df = assign_zones(df)

    conceded = _pivot(df, "defending_title", "xG")
    matches = (
        df.groupby("defending_title")["match_id"].nunique().reindex(conceded.index)
    )
    per_match = conceded.div(matches.where(matches > 0, np.nan), axis=0).fillna(0.0)
    per_match["n_matches"] = matches.fillna(0).astype(int)
    return per_match


def league_zone_baseline(concession: pd.DataFrame) -> pd.Series:
    """League-average xG conceded per match, by zone."""
    if concession.empty:
        return pd.Series(0.0, index=ZONES)
    return concession[ZONES].mean()


def shrunk_concession(
    concession: pd.DataFrame, k: float = K_ZONE
) -> pd.DataFrame:
    """Shrink each team's zone concession toward the league average.

    At GW3 a team has ~2 matches of shots. Without shrinkage a single freak
    game makes a defence look catastrophic in one zone; k=25 keeps a 2-match
    sample at ~7% weight on its own observation.
    """
    if concession.empty:
        return concession
    baseline = league_zone_baseline(concession)
    n = concession["n_matches"].to_numpy()[:, None]
    obs = concession[ZONES].to_numpy()
    pri = baseline.to_numpy()[None, :]
    out = (n * obs + k * pri) / (n + k)
    result = pd.DataFrame(out, index=concession.index, columns=ZONES)
    result["n_matches"] = concession["n_matches"]
    return result


def zone_fit(
    attack_map: pd.DataFrame,
    concession: pd.DataFrame,
    *,
    clip: tuple[float, float] = (0.75, 1.35),
) -> pd.DataFrame:
    """Per (player, opponent) multiplier from zone overlap.

    For each opponent, `weakness[z] = conceded[z] / league_avg[z]`. The
    multiplier is the player's xG-share-weighted average weakness:

        fit = sum_z  share[player, z] * weakness[opponent, z]

    A value of 1.0 means the opponent is exactly league-average where this
    player shoots. Clipped, because an unclipped multiplier compounds with the
    model's other terms and a 2-match sample should never swing xP by 2x.
    """
    if attack_map.empty or concession.empty:
        return pd.DataFrame(columns=["understat_player_id", "defending_title", "zone_fit"])

    shrunk = shrunk_concession(concession)
    baseline = league_zone_baseline(concession)
    safe_base = baseline.where(baseline > 1e-9, np.nan)
    weakness = shrunk[ZONES].div(safe_base, axis=1).fillna(1.0)

    shares = attack_map[ZONES]
    # (players x zones) @ (zones x teams) -> players x teams
    fit = shares.to_numpy() @ weakness.to_numpy().T
    frame = pd.DataFrame(fit, index=attack_map.index, columns=weakness.index)

    # Players with no shots have an all-zero share row -> fit 0; make it neutral.
    no_data = shares.sum(axis=1) <= 1e-9
    frame.loc[no_data, :] = 1.0

    long = (
        frame.stack()
        .rename("zone_fit")
        .reset_index()
        .rename(columns={"level_0": "understat_player_id", "level_1": "defending_title"})
    )
    long["zone_fit"] = long["zone_fit"].clip(*clip)
    return long


def describe_weakness(concession: pd.DataFrame, team: str) -> pd.Series:
    """Human-readable zone weakness for one team -- used in report rationales."""
    shrunk = shrunk_concession(concession)
    baseline = league_zone_baseline(concession)
    if team not in shrunk.index:
        return pd.Series(dtype="float64")
    safe_base = baseline.where(baseline > 1e-9, np.nan)
    return (shrunk.loc[team, ZONES] / safe_base).fillna(1.0).sort_values(ascending=False)
