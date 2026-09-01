"""Stage A: expected-minutes model (Task 23).

Minutes dominate FPL forecast error. A 12-xP player on the bench scores 1, so
getting P(start) wrong costs far more than getting xG slightly wrong. This is
therefore the most important model in the engine, and it is a CLASSIFIER over
three outcomes rather than a regression on minutes:

    0 = did not play
    1 = cameo (<60 minutes, 1 appearance point)
    2 = started (>=60 minutes, 2 appearance points)

Output is a probability distribution, never a point estimate: expected minutes
alone hides the difference between "certain 60" and "coin flip between 0 and
90", which is exactly the risk the aggressive optimizer needs to see.

Calibration matters more than accuracy here. If the model says 70% start, that
should happen ~70% of the time, because downstream xP multiplies by it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

CLASSES = (0, 1, 2)
CLASS_NAMES = {0: "dnp", 1: "cameo", 2: "start"}

# Expected minutes conditional on each outcome, from the historical means.
MINUTES_GIVEN = {0: 0.0, 1: 28.0, 2: 84.0}


@dataclass
class MinutesModel:
    model: object
    features: list[str]
    metrics: dict

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(dnp), P(cameo), P(start) per row, columns ordered by CLASSES."""
        cols = [c for c in self.features if c in X.columns]
        missing = [c for c in self.features if c not in X.columns]
        frame = X[cols].copy()
        for c in missing:
            frame[c] = np.nan
        frame = frame[self.features]
        proba = self.model.predict_proba(frame)

        # Align to CLASSES even if a class was absent from training.
        classes = list(getattr(self.model, "classes_", CLASSES))
        out = np.zeros((len(frame), len(CLASSES)))
        for i, cls in enumerate(CLASSES):
            if cls in classes:
                out[:, i] = proba[:, classes.index(cls)]
        return out

    def expected_minutes(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(X)
        return p @ np.array([MINUTES_GIVEN[c] for c in CLASSES])

    def p_start(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, CLASSES.index(2)]


def train_minutes_model(
    train: pd.DataFrame,
    features: list[str],
    *,
    calibrate: bool = True,
    random_state: int = 0,
) -> MinutesModel:
    """Fit and (by default) calibrate the three-class minutes classifier."""
    X = train[features]
    y = train["y_minutes_class"].astype(int)

    base = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
    )

    if calibrate and y.nunique() > 1:
        # `cv=3` here is an internal fit-time split of the TRAINING data only;
        # the model is still evaluated on a chronologically later holdout.
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base

    model.fit(X, y)
    return MinutesModel(model=model, features=list(features), metrics={})


def evaluate_minutes_model(model: MinutesModel, test: pd.DataFrame) -> dict:
    """Calibration-focused metrics on a held-out, chronologically later set."""
    y = test["y_minutes_class"].astype(int).to_numpy()
    proba = model.predict_proba(test)

    p_start = proba[:, CLASSES.index(2)]
    y_start = (y == 2).astype(int)

    metrics = {
        "n": int(len(test)),
        "brier_start": float(brier_score_loss(y_start, p_start)),
        "base_rate_start": float(y_start.mean()),
        "brier_baseline": float(
            brier_score_loss(y_start, np.full_like(p_start, y_start.mean()))
        ),
        "accuracy": float((proba.argmax(axis=1) == y).mean()),
    }
    try:
        metrics["log_loss"] = float(log_loss(y, proba, labels=list(CLASSES)))
    except ValueError:
        metrics["log_loss"] = float("nan")

    metrics["brier_skill"] = 1.0 - metrics["brier_start"] / max(
        metrics["brier_baseline"], 1e-9
    )
    metrics["calibration"] = calibration_table(y_start, p_start)
    return metrics


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency -- the check that actually matters.

    A model can be accurate and still badly calibrated; since xP multiplies by
    P(start), a systematically overconfident P(start) inflates every forecast.
    """
    df = pd.DataFrame({"p": p, "y": y_true})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    out["gap"] = out["observed"] - out["predicted"]
    return out
