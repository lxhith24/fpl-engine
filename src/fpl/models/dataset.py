"""Training matrix from historical gameweeks (Tasks 23-26).

Builds one row per (player, gameweek) with features computed ONLY from prior
gameweeks, and the label taken from that gameweek. This is the same
point-in-time contract enforced by tests/test_leakage.py: features at index t
use data strictly before t.

Implemented with vectorised expanding/rolling shifts rather than a Python loop
over gameweeks -- 113k rows x 4 seasons is too slow otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.shrinkage import RECENCY_LAMBDA

# Rolling windows (in matches) used as features.
WINDOWS = (3, 5, 10)

_ROLL_COLS = (
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "defensive_contribution",
    "saves",
    "clean_sheets",
    "starts",
)


def _ewm_shifted(g: pd.DataFrame, col: str, lam: float) -> pd.Series:
    """Recency-weighted mean of all PRIOR matches (shifted by one)."""
    return (
        g[col]
        .shift(1)
        .ewm(alpha=1 - lam, adjust=True, min_periods=1)
        .mean()
    )


def build_training_matrix(
    history: pd.DataFrame,
    *,
    lam: float = RECENCY_LAMBDA,
    windows: tuple[int, ...] = WINDOWS,
) -> pd.DataFrame:
    """Point-in-time features + labels for every historical player-gameweek."""
    if history.empty:
        return pd.DataFrame()

    df = history.sort_values(["season", "element", "event"]).copy()
    key = ["season", "element"]
    grouped = df.groupby(key, sort=False)

    out = df[
        [
            "season", "element", "event", "position", "team", "opponent_team",
            "was_home", "minutes", "total_points", "starts", "value",
        ]
    ].copy()

    # ---- labels ---------------------------------------------------------
    out["y_points"] = df["total_points"]
    out["y_minutes"] = df["minutes"]
    # 3-class minutes outcome: 0 = did not play, 1 = cameo, 2 = started
    out["y_minutes_class"] = np.select(
        [df["minutes"] == 0, df["minutes"] < 60],
        [0, 1],
        default=2,
    )

    # ---- recency-weighted prior form ------------------------------------
    # `transform` (not `apply`) is required: with a single group, `apply`
    # returns a DataFrame rather than a Series and the assignment raises.
    # It is also substantially faster on 113k rows.
    for col in _ROLL_COLS:
        if col in df.columns:
            out[f"{col}_ewm"] = grouped[col].transform(
                lambda s: s.shift(1).ewm(alpha=1 - lam, adjust=True, min_periods=1).mean()
            )

    # ---- fixed-window prior means ---------------------------------------
    for w in windows:
        for col in ("total_points", "minutes", "expected_goals", "expected_assists"):
            if col in df.columns:
                out[f"{col}_r{w}"] = grouped[col].transform(
                    lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean()
                )

    # ---- experience / availability proxies -------------------------------
    out["n_prior"] = grouped.cumcount()
    out["prior_minutes_total"] = grouped["minutes"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    out["prior_starts_total"] = grouped["starts"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    out["start_rate"] = out["prior_starts_total"] / out["n_prior"].replace(0, np.nan)
    out["played_last"] = (grouped["minutes"].shift(1) > 0).astype(float)
    out["started_last"] = (grouped["minutes"].shift(1) >= 60).astype(float)

    # ---- opponent strength (prior-only, per season) ----------------------
    out = _attach_opponent_strength(out, df)

    out["is_home"] = df["was_home"].astype(int)
    return out


def _attach_opponent_strength(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Opponent goals-conceded/scored per match, using only earlier gameweeks.

    Uses an expanding mean over prior gameweeks within the same season, so a
    GW10 row sees the opponent's GW1-9 record and nothing later.

    ID NAMESPACE HAZARD: in these CSVs `team` is a NAME ("Arsenal") while
    `opponent_team` is an integer team ID. Joining them directly raises (or
    worse, silently produces all-NaN). Both are resolved to a common
    per-season team-id space first: the ids are 1..20 assigned alphabetically
    by team name within each season, which is FPL's own convention.
    """
    if "team" not in df.columns or "opponent_team" not in df.columns:
        out["opp_conceded_prior"] = np.nan
        out["opp_scored_prior"] = np.nan
        return out

    work = df[["season", "team", "event", "total_points", "goals_scored"]].copy()
    work["goals_conceded"] = (
        df["goals_conceded"] if "goals_conceded" in df.columns else 0.0
    )

    # Map team NAME -> per-season integer id (alphabetical, 1-based).
    name_to_id: dict[tuple[str, str], int] = {}
    for season, grp in work.groupby("season"):
        for idx, name in enumerate(sorted(grp["team"].dropna().unique()), start=1):
            name_to_id[(season, name)] = idx

    work["team_id"] = [
        name_to_id.get((s, t), np.nan) for s, t in zip(work["season"], work["team"])
    ]

    team_gw = (
        work.dropna(subset=["team_id"])
        .groupby(["season", "team_id", "event"])
        .agg(
            team_conceded=("goals_conceded", "mean"),
            team_scored=("goals_scored", "sum"),
        )
        .reset_index()
        .sort_values(["season", "team_id", "event"])
    )
    g = team_gw.groupby(["season", "team_id"], sort=False)
    team_gw["opp_conceded_prior"] = g["team_conceded"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_gw["opp_scored_prior"] = g["team_scored"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    lookup = team_gw[
        ["season", "team_id", "event", "opp_conceded_prior", "opp_scored_prior"]
    ].rename(columns={"team_id": "opponent_team"})

    merged = out.copy()
    merged["opponent_team"] = pd.to_numeric(merged["opponent_team"], errors="coerce")
    lookup["opponent_team"] = pd.to_numeric(lookup["opponent_team"], errors="coerce")
    merged = merged.merge(lookup, on=["season", "opponent_team", "event"], how="left")

    coverage = merged["opp_conceded_prior"].notna().mean()
    if coverage < 0.5:
        # Loud rather than silent: an all-NaN opponent feature would quietly
        # remove the entire opponent dimension from the model.
        import warnings

        warnings.warn(
            f"opponent strength join covered only {coverage:.1%} of rows -- "
            "team id mapping may be wrong for some seasons",
            stacklevel=2,
        )
    return merged


def train_test_split_by_time(
    matrix: pd.DataFrame, *, holdout_season: str | None = None, holdout_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split -- NEVER random.

    A random split leaks the future into training: the same player's GW20 row
    can train a model evaluated on his GW10 row.
    """
    if matrix.empty:
        return matrix, matrix

    if holdout_season and holdout_season in set(matrix["season"]):
        return (
            matrix[matrix["season"] != holdout_season],
            matrix[matrix["season"] == holdout_season],
        )

    order = matrix.sort_values(["season", "event"])
    cut = int(len(order) * (1 - holdout_frac))
    return order.iloc[:cut], order.iloc[cut:]


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers and labels."""
    exclude = {
        "season", "element", "event", "position", "team", "opponent_team",
        "was_home", "minutes", "total_points", "starts", "value",
        "y_points", "y_minutes", "y_minutes_class",
    }
    return [
        c for c in matrix.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(matrix[c])
    ]
