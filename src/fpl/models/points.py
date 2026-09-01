"""Stage B: points model, per position (Tasks 25-26).

Trained separately per position because the point-generating process is
genuinely different: a defender earns from clean sheets and defensive
contributions, a forward from goals. One pooled model has to spend capacity
learning "what position is this" before it can learn anything useful.

Predicts points-per-appearance CONDITIONAL on playing, which is then combined
with Stage A's minutes distribution:

    xP = P(cameo) * pts_cameo + P(start) * pts_start

Also emits sigma(xP) -- the spread across ensemble members plus residual
scatter. Captaincy and differentials are risk decisions, so the aggressive
optimizer needs the variance, not just the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# Appearance points are deterministic; the model only needs to learn the rest.
APPEARANCE_CAMEO = 1.0
APPEARANCE_START = 2.0


@dataclass
class PositionModel:
    ensemble: list
    features: list[str]
    residual_sigma: float
    n_train: int


@dataclass
class PointsModel:
    by_position: dict[str, PositionModel] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def _frame(self, X: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        frame = X[[c for c in feats if c in X.columns]].copy()
        for c in feats:
            if c not in frame.columns:
                frame[c] = np.nan
        return frame[feats]

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mean points given start, sigma) per row, dispatched by position."""
        mean = np.full(len(X), np.nan)
        sigma = np.full(len(X), np.nan)
        pos = X["position"].to_numpy() if "position" in X.columns else np.array([""] * len(X))

        for p, pm in self.by_position.items():
            mask = pos == p
            if not mask.any():
                continue
            frame = self._frame(X[mask], pm.features)
            preds = np.column_stack([est.predict(frame) for est in pm.ensemble])
            mean[mask] = preds.mean(axis=1)
            spread = preds.std(axis=1)
            sigma[mask] = np.sqrt(spread**2 + pm.residual_sigma**2)

        # Positions never seen in training fall back to the global mean.
        if np.isnan(mean).any():
            fallback = np.nanmean(mean) if not np.isnan(mean).all() else 2.0
            mean = np.where(np.isnan(mean), fallback, mean)
            sigma = np.where(np.isnan(sigma), np.nanmax(sigma) if not np.isnan(sigma).all() else 2.0, sigma)
        return mean, sigma


def train_points_model(
    train: pd.DataFrame,
    features: list[str],
    *,
    min_minutes: int = 1,
    random_state: int = 0,
) -> PointsModel:
    """Fit one ensemble per position on appearances only.

    Rows where the player did not feature are excluded: this model answers
    "how many points GIVEN he plays". Non-selection is Stage A's job, and
    mixing the two makes both worse.
    """
    model = PointsModel(features=list(features))

    for pos in POSITIONS:
        sub = train[(train["position"] == pos) & (train["y_minutes"] >= min_minutes)]
        if len(sub) < 200:
            continue

        X = sub[features]
        y = sub["y_points"].astype(float)

        gbm = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
        )
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        )
        gbm.fit(X, y)
        rf.fit(X.fillna(X.median(numeric_only=True)), y)

        resid = y - gbm.predict(X)
        model.by_position[pos] = PositionModel(
            ensemble=[gbm, _RFWrapper(rf, X.median(numeric_only=True))],
            features=list(features),
            residual_sigma=float(resid.std()),
            n_train=len(sub),
        )
    return model


class _RFWrapper:
    """RandomForest cannot handle NaN; impute with training medians."""

    def __init__(self, rf, medians):
        self.rf = rf
        self.medians = medians

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf.predict(X.fillna(self.medians).fillna(0.0))


def combine_expected_points(
    p_dnp: np.ndarray,
    p_cameo: np.ndarray,
    p_start: np.ndarray,
    pts_given_play: np.ndarray,
    sigma_given_play: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold the minutes distribution into expected points and its spread.

    A cameo earns roughly the appearance point plus a scaled share of the
    attacking return; a start earns the full modelled amount.
    """
    cameo_pts = APPEARANCE_CAMEO + 0.35 * np.maximum(pts_given_play - APPEARANCE_START, 0)
    start_pts = pts_given_play

    xp = p_cameo * cameo_pts + p_start * start_pts  # p_dnp contributes 0

    # Total variance = within-outcome variance + between-outcome variance.
    mean_sq = p_cameo * cameo_pts**2 + p_start * start_pts**2
    between = np.maximum(mean_sq - xp**2, 0.0)
    within = (p_cameo + p_start) * sigma_given_play**2
    return xp, np.sqrt(between + within)


def evaluate_points_model(
    model: PointsModel, test: pd.DataFrame, *, min_minutes: int = 1
) -> dict:
    """Overall and high-return accuracy.

    High-return (>2 points) MAE is reported separately because that subgroup
    drives rank movement -- the plan's baseline gate is defined on it.
    """
    sub = test[test["y_minutes"] >= min_minutes]
    if sub.empty:
        return {}
    mean, _ = model.predict(sub)
    y = sub["y_points"].astype(float).to_numpy()

    high = y > 2
    return {
        "n": int(len(sub)),
        "mae": float(mean_absolute_error(y, mean)),
        "rmse": float(np.sqrt(mean_squared_error(y, mean))),
        "mae_high_return": float(mean_absolute_error(y[high], mean[high])) if high.any() else float("nan"),
        "n_high_return": int(high.sum()),
    }
