"""Phase 4 model tests: minutes, points, combination, and backtest gating."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.backtest.walk_forward import baseline_predictions, gate_report, score_predictions, significance
from fpl.models import minutes as mn
from fpl.models import points as pts
from fpl.models.dataset import build_training_matrix, feature_columns, train_test_split_by_time


# --------------------------------------------------------------- dataset
def _history(n_players: int = 6, n_events: int = 12) -> pd.DataFrame:
    rows = []
    for el in range(1, n_players + 1):
        for ev in range(1, n_events + 1):
            played = (el + ev) % 4 != 0
            rows.append(
                {
                    "season": "2024-25", "element": el, "event": ev,
                    "position": ["GKP", "DEF", "MID", "FWD"][el % 4],
                    "team": f"T{el % 3}", "opponent_team": (ev % 5) + 1,
                    "was_home": ev % 2 == 0,
                    "minutes": 90 if played else 0,
                    "starts": 1 if played else 0,
                    "total_points": 6 if played else 0,
                    "goals_scored": 1 if played and ev % 3 == 0 else 0,
                    "assists": 0, "clean_sheets": 0, "goals_conceded": 1,
                    "saves": 0, "bonus": 0, "bps": 20 if played else 0,
                    "value": 50 + el, "expected_goals": 0.3, "expected_assists": 0.2,
                    "expected_goals_conceded": 1.1, "defensive_contribution": 4.0,
                }
            )
    return pd.DataFrame(rows)


def test_training_matrix_builds():
    m = build_training_matrix(_history())
    assert not m.empty
    assert {"y_points", "y_minutes", "y_minutes_class"}.issubset(m.columns)


def test_minutes_class_encoding():
    m = build_training_matrix(_history())
    assert set(m["y_minutes_class"]).issubset({0, 1, 2})
    played = m[m["y_minutes"] >= 60]
    assert (played["y_minutes_class"] == 2).all()
    dnp = m[m["y_minutes"] == 0]
    assert (dnp["y_minutes_class"] == 0).all()


def test_first_appearance_has_no_prior_form():
    """A player's debut row must have NaN rolling features, not zeros."""
    m = build_training_matrix(_history())
    first = m.sort_values(["element", "event"]).groupby("element").head(1)
    assert first["total_points_ewm"].isna().all()
    assert (first["n_prior"] == 0).all()


def test_rolling_features_shifted_not_leaking():
    """A row's own points must never appear in its own rolling mean."""
    hist = _history(n_players=1, n_events=6).copy()
    hist.loc[hist["event"] == 6, "total_points"] = 9999
    m = build_training_matrix(hist)
    row = m[m["event"] == 6].iloc[0]
    assert row["total_points_ewm"] < 100, "current gameweek leaked into its own feature"


def test_chronological_split_never_random():
    m = build_training_matrix(_history())
    tr, te = train_test_split_by_time(m, holdout_frac=0.3)
    assert tr["event"].max() <= te["event"].min() + 1


def test_split_by_season():
    m = build_training_matrix(_history())
    m2 = m.copy()
    m2["season"] = "2025-26"
    both = pd.concat([m, m2], ignore_index=True)
    tr, te = train_test_split_by_time(both, holdout_season="2025-26")
    assert set(te["season"]) == {"2025-26"}
    assert "2025-26" not in set(tr["season"])


def test_feature_columns_exclude_labels():
    m = build_training_matrix(_history())
    fc = feature_columns(m)
    for leak in ("y_points", "y_minutes", "y_minutes_class", "total_points", "minutes"):
        assert leak not in fc, f"{leak} must never be a model input"


# --------------------------------------------------------------- minutes
@pytest.fixture(scope="module")
def trained():
    m = build_training_matrix(_history(n_players=40, n_events=20))
    fc = feature_columns(m)
    tr, te = train_test_split_by_time(m, holdout_frac=0.25)
    model = mn.train_minutes_model(tr, fc, calibrate=False)
    return model, tr, te, fc


def test_minutes_proba_is_a_distribution(trained):
    model, _, te, _ = trained
    p = model.predict_proba(te)
    assert p.shape == (len(te), 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all()


def test_expected_minutes_in_range(trained):
    model, _, te, _ = trained
    em = model.expected_minutes(te)
    assert (em >= 0).all() and (em <= 90).all()


def test_missing_feature_columns_are_tolerated(trained):
    """Live data may lack a training column; it must not crash."""
    model, _, te, _ = trained
    partial = te.drop(columns=[c for c in model.features[:3] if c in te.columns])
    p = model.predict_proba(partial)
    assert p.shape == (len(partial), 3)


def test_calibration_table_shape():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.75, 0.95])
    tbl = mn.calibration_table(y, p, bins=5)
    assert {"n", "predicted", "observed", "gap"}.issubset(tbl.columns)
    assert tbl["n"].sum() == len(y)


# ---------------------------------------------------------------- points
def test_points_model_trains_per_position():
    m = build_training_matrix(_history(n_players=80, n_events=20))
    fc = feature_columns(m)
    model = pts.train_points_model(m, fc)
    assert len(model.by_position) >= 1
    for pm in model.by_position.values():
        assert len(pm.ensemble) == 2, "ensemble should hold GBM + RF"


def test_points_predict_returns_mean_and_sigma():
    m = build_training_matrix(_history(n_players=80, n_events=20))
    fc = feature_columns(m)
    model = pts.train_points_model(m, fc)
    mean, sigma = model.predict(m)
    assert len(mean) == len(m)
    assert (sigma >= 0).all(), "sigma must be non-negative"


# ----------------------------------------------------------- combination
def test_combine_zero_start_probability_gives_zero_xp():
    xp, sig = pts.combine_expected_points(
        np.array([1.0]), np.array([0.0]), np.array([0.0]),
        np.array([10.0]), np.array([2.0]),
    )
    assert xp[0] == pytest.approx(0.0), "a certain non-starter must score 0 xP"


def test_combine_certain_start_returns_full_points():
    xp, _ = pts.combine_expected_points(
        np.array([0.0]), np.array([0.0]), np.array([1.0]),
        np.array([8.0]), np.array([1.0]),
    )
    assert xp[0] == pytest.approx(8.0)


def test_combine_is_monotonic_in_start_probability():
    out = [
        pts.combine_expected_points(
            np.array([1 - p]), np.array([0.0]), np.array([p]),
            np.array([8.0]), np.array([1.0]),
        )[0][0]
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert out == sorted(out)


def test_rotation_risk_raises_sigma():
    """A coin-flip starter must carry more variance than a certain one."""
    _, sig_certain = pts.combine_expected_points(
        np.array([0.0]), np.array([0.0]), np.array([1.0]),
        np.array([8.0]), np.array([1.0]),
    )
    _, sig_risky = pts.combine_expected_points(
        np.array([0.5]), np.array([0.0]), np.array([0.5]),
        np.array([8.0]), np.array([1.0]),
    )
    assert sig_risky[0] > sig_certain[0]


# --------------------------------------------------------------- backtest
def test_baselines_produced():
    test = pd.DataFrame(
        {"total_points_ewm": [3.0, 4.0], "total_points_r5": [2.0, 5.0],
         "value": [50.0, 100.0], "y_points": [4.0, 6.0]}
    )
    b = baseline_predictions(test)
    assert set(b) == {"form", "last5", "price"}
    assert all(len(v) == 2 for v in b.values())


def test_score_predictions_reports_high_return_subset():
    y = np.array([0.0, 1.0, 6.0, 12.0])
    pred = np.array([1.0, 1.0, 5.0, 10.0])
    s = score_predictions(y, pred)
    assert s["n_high"] == 2
    assert s["mae_high"] == pytest.approx(1.5)


def test_gate_requires_beating_every_baseline():
    """Gate must FAIL when any baseline out-ranks the model."""
    folds = pd.DataFrame(
        {"model_mae": [1.0], "model_mae_high": [2.0], "model_spearman": [0.70],
         "form_spearman": [0.60], "last5_spearman": [0.60],
         "price_spearman": [0.75]}  # price ranks better -> must fail
    )
    g = gate_report(folds)
    assert g["beats_form"] and g["beats_last5"]
    assert not g["beats_price"]
    assert not g["passed"], "gate must fail when any baseline wins"


def test_gate_passes_when_model_wins_all():
    folds = pd.DataFrame(
        {"model_mae": [0.8], "model_mae_high": [2.0], "model_spearman": [0.80],
         "form_spearman": [0.70], "last5_spearman": [0.65], "price_spearman": [0.40]}
    )
    assert gate_report(folds)["passed"]


def test_gate_reports_significance():
    folds = pd.DataFrame(
        {"model_spearman": [0.75, 0.76, 0.74, 0.75],
         "form_spearman": [0.73, 0.73, 0.72, 0.73],
         "last5_spearman": [0.72, 0.73, 0.72, 0.72],
         "price_spearman": [0.42, 0.43, 0.41, 0.42]}
    )
    sig = significance(folds)
    assert set(sig["baseline"]) == {"form", "last5", "price"}
    assert (sig["mean_diff"] > 0).all()
    assert "p_value" in sig.columns


def test_gate_empty_folds():
    assert gate_report(pd.DataFrame())["passed"] is False


# ------------------------------------------------- metric-bias regression
def test_mae_high_is_biased_toward_overprediction():
    """REGRESSION / DOCUMENTATION: why the plan's `mae_high` is not the gate.

    Conditioning error on the OUTCOME (y > 2) selects rows where noise landed
    positive. A calibrated predictor of the conditional mean therefore loses
    to an inflated one on that subset -- by construction, not by merit.

    Constructed here: `calibrated` predicts the true mean; `inflated` adds a
    constant. The inflated model is objectively worse overall yet wins on
    mae_high.
    """
    rng = np.random.default_rng(0)
    truth = rng.poisson(2.0, 4000).astype(float)
    calibrated = np.full_like(truth, 2.0)
    inflated = np.full_like(truth, 4.5)

    cal = score_predictions(truth, calibrated)
    inf = score_predictions(truth, inflated)

    assert cal["mae"] < inf["mae"], "calibrated must win on overall MAE"
    assert inf["mae_high"] < cal["mae_high"], (
        "the bias this test documents has disappeared -- re-examine the gate"
    )


def test_spearman_rewards_ranking_not_scale():
    """Rank correlation is invariant to the shrinkage that hurts mae_high."""
    y = np.arange(50, dtype=float)
    shrunk = y * 0.3          # correct order, wrong scale
    noisy = y + np.random.default_rng(1).normal(0, 20, 50)

    s_shrunk = score_predictions(y, shrunk)["spearman"]
    s_noisy = score_predictions(y, noisy)["spearman"]
    assert s_shrunk > s_noisy
    assert s_shrunk == pytest.approx(1.0, abs=1e-9)


def test_gate_uses_spearman_not_mae_high():
    """The gate must pass a model that ranks better but has worse mae_high."""
    folds = pd.DataFrame(
        {
            "model_mae": [1.0], "model_mae_high": [3.7], "model_spearman": [0.75],
            "model_top11_actual": [3.9], "model_top30_actual": [4.0],
            "form_mae_high": [3.4], "form_spearman": [0.73],
            "form_top11_actual": [4.2], "form_top30_actual": [3.9],
            "last5_mae_high": [3.5], "last5_spearman": [0.73],
            "last5_top11_actual": [4.1], "last5_top30_actual": [3.7],
            "price_mae_high": [4.7], "price_spearman": [0.42],
            "price_top11_actual": [3.7], "price_top30_actual": [3.1],
        }
    )
    g = gate_report(folds)
    assert g["passed"], "model ranks better than every baseline; gate should pass"
    assert g["model_mae_high_informational"] > g["form_mae_high"]
