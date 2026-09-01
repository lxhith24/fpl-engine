"""Production xP prediction for an upcoming gameweek (Task 26).

Bridges the historical training matrix and the live Phase 3 feature table.
They share most column names by construction; anything missing is filled with
NaN, which the gradient-boosted models handle natively.

Applies the matchup multipliers the models cannot see -- `zone_fit` from the
opponent's zone-concession profile, and availability from FPL's own status
flags -- on top of the learned base rate.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .dataset import build_training_matrix, feature_columns
from .minutes import train_minutes_model
from .points import combine_expected_points, train_points_model

MODEL_DIR = config.MODELS
MINUTES_PATH = MODEL_DIR / "minutes.pkl"
POINTS_PATH = MODEL_DIR / "points.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"


def train_and_save(matrix: pd.DataFrame | None = None, *, verbose: bool = True) -> dict:
    """Fit both stages on all available history and persist them."""
    import json

    from ..ingest.history import load_history

    if matrix is None:
        matrix = build_training_matrix(load_history())
    feats = feature_columns(matrix)

    if verbose:
        print(f"training on {len(matrix):,} rows, {len(feats)} features")

    mm = train_minutes_model(matrix, feats)
    pm = train_points_model(matrix, feats)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MINUTES_PATH, "wb") as fh:
        pickle.dump(mm, fh)
    with open(POINTS_PATH, "wb") as fh:
        pickle.dump(pm, fh)
    FEATURES_PATH.write_text(json.dumps(feats))

    return {
        "n_rows": len(matrix),
        "n_features": len(feats),
        "positions_modelled": sorted(pm.by_position),
    }


def load_models() -> tuple[object, object, list[str]]:
    import json

    if not (MINUTES_PATH.exists() and POINTS_PATH.exists()):
        raise FileNotFoundError(
            "models not trained -- run `python -m fpl.models.train` first"
        )
    with open(MINUTES_PATH, "rb") as fh:
        mm = pickle.load(fh)
    with open(POINTS_PATH, "rb") as fh:
        pm = pickle.load(fh)
    feats = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.exists() else []
    return mm, pm, feats


# Live feature name -> training feature name.
_ALIASES = {
    "total_points_l3": "total_points_r3",
    "total_points_l5": "total_points_r5",
    "total_points_l10": "total_points_r10",
    "minutes_l3": "minutes_r3",
    "minutes_l5": "minutes_r5",
    "minutes_l10": "minutes_r10",
    "expected_goals_l3": "expected_goals_r3",
    "expected_assists_l3": "expected_assists_r3",
    "total_points_l38": "total_points_ewm",
    "minutes_l38": "minutes_ewm",
    "expected_goals_l38": "expected_goals_ewm",
    "expected_assists_l38": "expected_assists_ewm",
    "defensive_contribution_l38": "defensive_contribution_ewm",
    "bps_l38": "bps_ewm",
    "saves_l38": "saves_ewm",
    "clean_sheets_l38": "clean_sheets_ewm",
    "goals_scored_l38": "goals_scored_ewm",
    "assists_l38": "assists_ewm",
    "n_prior_matches": "n_prior",
    "minutes_prior_total": "prior_minutes_total",
    "starts_prior": "prior_starts_total",
    "opp_conceded": "opp_conceded_prior",
    "opp_scored": "opp_scored_prior",
}


def align_features(live: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Rename live columns into the training vocabulary and fill gaps."""
    df = live.copy()
    for live_name, train_name in _ALIASES.items():
        if live_name in df.columns and train_name not in df.columns:
            df[train_name] = df[live_name]

    if "is_home" in df.columns:
        df["is_home"] = df["is_home"].astype(int)
    if "start_rate" not in df.columns and {"starts_prior", "n_prior_matches"}.issubset(live.columns):
        df["start_rate"] = live["starts_prior"] / live["n_prior_matches"].replace(0, np.nan)
    if "played_last" not in df.columns and "minutes_l1" in live.columns:
        df["played_last"] = (live["minutes_l1"] > 0).astype(float)
    if "started_last" not in df.columns and "minutes_l1" in live.columns:
        df["started_last"] = (live["minutes_l1"] >= 60).astype(float)
    if "opp_conceded_prior" not in df.columns and "opp_xg_against" in live.columns:
        df["opp_conceded_prior"] = live["opp_xg_against"]
    if "opp_scored_prior" not in df.columns and "opp_xg_for" in live.columns:
        df["opp_scored_prior"] = live["opp_xg_for"]

    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    return df


def predict_gameweek(
    features: pd.DataFrame,
    *,
    apply_zone_fit: bool = True,
    apply_availability: bool = True,
) -> pd.DataFrame:
    """Attach xP and sigma to a live Phase 3 feature table."""
    mm, pm, feats = load_models()
    df = align_features(features, feats)

    proba = mm.predict_proba(df)
    pts, sig = pm.predict(df)
    xp, xp_sigma = combine_expected_points(
        proba[:, 0], proba[:, 1], proba[:, 2], pts, sig
    )

    out = features.copy()
    out["p_dnp"] = proba[:, 0]
    out["p_cameo"] = proba[:, 1]
    out["p_start"] = proba[:, 2]
    out["exp_minutes"] = mm.expected_minutes(df)
    out["pts_given_play"] = pts
    out["xp_raw"] = xp
    out["xp_sigma"] = xp_sigma

    xp_adj = xp.copy()

    # Zone fit is a matchup multiplier the model cannot see: it is computed
    # from shot coordinates, not from the tabular features.
    if apply_zone_fit and "zone_fit" in out.columns:
        xp_adj = xp_adj * out["zone_fit"].fillna(1.0).to_numpy()

    # FPL's own availability flags override the model. A player flagged 25%
    # fit is a fact about the world, not something to infer from form.
    if apply_availability:
        if "chance_next" in out.columns:
            xp_adj = xp_adj * (out["chance_next"].fillna(100.0).to_numpy() / 100.0)
        if "available" in out.columns:
            xp_adj = np.where(out["available"].to_numpy(), xp_adj, 0.0)

    out["xp"] = xp_adj
    return out.sort_values("xp", ascending=False)


if __name__ == "__main__":
    info = train_and_save()
    print(info)
