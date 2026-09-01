"""Point-in-time feature assembly (Tasks 15-16, 20-22).

THE CARDINAL RULE OF THIS MODULE: features for gameweek T may only use data
that existed BEFORE gameweek T's deadline. Violating it produces a model that
looks brilliant in back-testing and loses money in production, because it was
trained on the future.

Every public function here takes an explicit `as_of_event` and filters to
`event < as_of_event`. `tests/test_leakage.py` asserts this by construction:
it plants an impossible value in the target gameweek and fails if any feature
moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .shrinkage import RECENCY_LAMBDA, K_H2H, shrunk_head_to_head, weighted_mean

# Rolling horizons from the plan (1/3/5/10/38-match means).
HORIZONS = (1, 3, 5, 10, 38)

_ROLLING_COLS = (
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
)


def _history_before(player_gw: pd.DataFrame, as_of_event: int) -> pd.DataFrame:
    """Strictly-prior gameweeks. The single choke point for leakage."""
    if player_gw.empty:
        return player_gw
    return player_gw[player_gw["event"] < as_of_event]


def player_rolling_features(
    player_gw: pd.DataFrame,
    as_of_event: int,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    lam: float = RECENCY_LAMBDA,
    cols: tuple[str, ...] = _ROLLING_COLS,
) -> pd.DataFrame:
    """Recency-weighted rolling means per player over several horizons.

    Uses only gameweeks strictly before `as_of_event`.
    """
    hist = _history_before(player_gw, as_of_event)
    if hist.empty:
        return pd.DataFrame(columns=["element"])

    hist = hist.sort_values(["element", "event"])
    available = [c for c in cols if c in hist.columns]
    frames = []

    for h in horizons:
        agg = (
            hist.groupby("element")
            .tail(h)
            .groupby("element")[available]
            .agg(lambda s: weighted_mean(s, lam))
        )
        agg.columns = [f"{c}_l{h}" for c in agg.columns]
        frames.append(agg)

    out = pd.concat(frames, axis=1)
    out["n_prior_matches"] = hist.groupby("element").size()
    out["minutes_prior_total"] = hist.groupby("element")["minutes"].sum()
    out["starts_prior"] = (
        hist.assign(_s=(hist["minutes"] >= 60).astype(int)).groupby("element")["_s"].sum()
    )
    return out.reset_index()


def team_form_features(
    team_matches: pd.DataFrame,
    as_of_date: pd.Timestamp,
    *,
    horizon: int = 6,
    lam: float = RECENCY_LAMBDA,
) -> pd.DataFrame:
    """Attack/defence profile per team from Understat, before `as_of_date`.

    Understat rows are dated rather than gameweek-numbered, so the cutoff is a
    timestamp -- but the rule is identical: nothing on or after the deadline.
    """
    if team_matches.empty:
        return pd.DataFrame(columns=["team_title"])

    tm = team_matches.copy()
    tm["match_date"] = pd.to_datetime(tm["match_date"])
    tm = tm[tm["match_date"] < pd.Timestamp(as_of_date)]
    if tm.empty:
        return pd.DataFrame(columns=["team_title"])

    tm = tm.sort_values(["team_title", "match_date"])
    recent = tm.groupby("team_title").tail(horizon)

    def _wm(s: pd.Series) -> float:
        return weighted_mean(s, lam)

    agg = recent.groupby("team_title").agg(
        xg_for=("xG", _wm),
        xg_against=("xGA", _wm),
        npxg_for=("npxG", _wm),
        npxg_against=("npxGA", _wm),
        scored=("scored", _wm),
        conceded=("missed", _wm),
        deep=("deep", _wm),
        deep_allowed=("deep_allowed", _wm),
        n_matches=("xG", "size"),
    )

    # PPDA = passes allowed per defensive action; lower means heavier pressing.
    if {"ppda_att", "ppda_def"}.issubset(recent.columns):
        ppda = recent.assign(
            _ppda=recent["ppda_att"] / recent["ppda_def"].replace(0, np.nan)
        )
        agg["ppda"] = ppda.groupby("team_title")["_ppda"].mean()

    # Home/away splits -- venue matters more than most single features.
    for side, label in (("h", "home"), ("a", "away")):
        sub = recent[recent["h_a"] == side]
        if not sub.empty:
            agg[f"xga_{label}"] = sub.groupby("team_title")["xGA"].mean()
            agg[f"xg_{label}"] = sub.groupby("team_title")["xG"].mean()

    return agg.reset_index()


def fixture_context(
    fixtures: pd.DataFrame,
    as_of_event: int,
    *,
    horizon: int = 1,
) -> pd.DataFrame:
    """Upcoming opponent / venue / difficulty per team for the target GW(s).

    Handles doubles and blanks: a team with two fixtures gets two rows, a team
    with none gets no row (and therefore contributes zero xP).
    """
    if fixtures.empty:
        return pd.DataFrame(
            columns=["team_id", "event", "opponent_id", "is_home", "difficulty", "fixture_id"]
        )

    window = fixtures[
        (fixtures["event"] >= as_of_event) & (fixtures["event"] < as_of_event + horizon)
    ]
    rows = []
    for _, f in window.iterrows():
        rows.append(
            {
                "team_id": f["team_h"], "event": f["event"], "opponent_id": f["team_a"],
                "is_home": True, "difficulty": f.get("team_h_difficulty"),
                "fixture_id": f["id"], "kickoff_time": f.get("kickoff_time"),
            }
        )
        rows.append(
            {
                "team_id": f["team_a"], "event": f["event"], "opponent_id": f["team_h"],
                "is_home": False, "difficulty": f.get("team_a_difficulty"),
                "fixture_id": f["id"], "kickoff_time": f.get("kickoff_time"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    counts = out.groupby(["team_id", "event"]).size().rename("n_fixtures")
    return out.merge(counts, on=["team_id", "event"], how="left")


def rest_days(
    fixtures: pd.DataFrame, as_of_event: int, *, target_event: int | None = None
) -> pd.DataFrame:
    """Days since each team's previous fixture -- congestion / rotation proxy."""
    target_event = target_event or as_of_event
    if fixtures.empty or "kickoff_time" not in fixtures.columns:
        return pd.DataFrame(columns=["team_id", "rest_days"])

    fx = fixtures.dropna(subset=["kickoff_time"]).copy()
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"])

    past = fx[fx["event"] < as_of_event]
    upcoming = fx[fx["event"] == target_event]
    if past.empty or upcoming.empty:
        return pd.DataFrame(columns=["team_id", "rest_days"])

    last: dict[int, pd.Timestamp] = {}
    for _, f in past.iterrows():
        for t in (f["team_h"], f["team_a"]):
            prev = last.get(t)
            if prev is None or f["kickoff_time"] > prev:
                last[t] = f["kickoff_time"]

    rows = []
    for _, f in upcoming.iterrows():
        for t in (f["team_h"], f["team_a"]):
            if t in last:
                rows.append(
                    {"team_id": t, "rest_days": (f["kickoff_time"] - last[t]).days}
                )
    return pd.DataFrame(rows).drop_duplicates(subset=["team_id"])


def head_to_head_feature(
    player_gw: pd.DataFrame,
    as_of_event: int,
    opponents: pd.DataFrame,
    *,
    baseline_col: str = "total_points",
    k: float = K_H2H,
    enabled: bool = True,
) -> pd.DataFrame:
    """Shrunk player-vs-opponent history (plan 2.3).

    `enabled=False` returns pure baselines, which is how the plan's ablation
    test measures whether this feature earns its place at all.

    `opponents` must have columns [element, opponent_id].
    """
    hist = _history_before(player_gw, as_of_event)
    if hist.empty or opponents.empty:
        return opponents.assign(h2h_estimate=np.nan, h2h_n=0)

    baselines = hist.groupby("element")[baseline_col].mean()
    out = []
    for _, row in opponents.iterrows():
        el, opp = row["element"], row["opponent_id"]
        base = float(baselines.get(el, hist[baseline_col].mean()))
        if not enabled:
            out.append({**row.to_dict(), "h2h_estimate": base, "h2h_n": 0})
            continue
        past = hist[(hist["element"] == el) & (hist["opponent_team"] == opp)]
        est, n = shrunk_head_to_head(
            past.sort_values("event")[baseline_col], baseline=base, k=k
        )
        out.append({**row.to_dict(), "h2h_estimate": est, "h2h_n": n})
    return pd.DataFrame(out)


def set_piece_flags(players: pd.DataFrame) -> pd.DataFrame:
    """Penalty / corner duty from the FPL API (Task 21).

    Order 1 is the designated first taker; a penalty taker is worth roughly
    +0.3 xG per match over a non-taker in the same role.
    """
    cols = ["id", "penalties_order", "corners_order"]
    have = [c for c in cols if c in players.columns]
    out = players[have].copy().rename(columns={"id": "element"})
    out["is_pen_taker"] = (out.get("penalties_order") == 1).fillna(False)
    out["takes_corners"] = out.get("corners_order").notna() if "corners_order" in out else False
    return out
