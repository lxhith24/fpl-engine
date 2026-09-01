"""Empirical-Bayes shrinkage and recency weighting (plan 2.3).

This is the single most important statistical primitive in the engine. At GW3
almost every per-player, per-opponent or per-zone rate is computed from a
handful of events, and the naive estimate is mostly noise. Shrinkage pulls
small-sample estimates toward a prior; the weight on the observed data grows
with n.

    shrunk = (n * observed + k * prior) / (n + k)

`k` is the "pseudo-count": the number of observations at which the estimate is
weighted 50/50 between data and prior. With k=12 and n=4, the observation moves
the estimate only 25% of the way from the prior -- which is exactly the desired
behaviour for a 4-match head-to-head record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default pseudo-counts, per the plan.
K_H2H = 12.0
K_ZONE = 25.0
K_RATE = 8.0

# Exponential recency decay applied to match-level histories.
RECENCY_LAMBDA = 0.85


def shrink(
    observed: float | np.ndarray | pd.Series,
    n: float | np.ndarray | pd.Series,
    prior: float | np.ndarray | pd.Series,
    k: float = K_RATE,
) -> float | np.ndarray | pd.Series:
    """Blend an observed rate toward a prior by sample size.

    With n=0 the result is exactly the prior; as n -> inf it approaches the
    observation. Never divides by zero because k > 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    obs = np.nan_to_num(np.asarray(observed, dtype=float), nan=0.0)
    n_arr = np.asarray(n, dtype=float)
    pri = np.asarray(prior, dtype=float)
    out = (n_arr * obs + k * pri) / (n_arr + k)
    if isinstance(observed, pd.Series):
        return pd.Series(out, index=observed.index)
    if np.isscalar(observed) and np.isscalar(n) and np.isscalar(prior):
        return float(out)
    return out


def recency_weights(n: int, lam: float = RECENCY_LAMBDA) -> np.ndarray:
    """Weights for `n` observations ordered OLDEST -> NEWEST.

    The most recent observation gets weight 1.0, the one before it `lam`, and
    so on. Returned unnormalized; `weighted_mean` handles normalization.
    """
    if n <= 0:
        return np.array([])
    if not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")
    ages = np.arange(n - 1, -1, -1, dtype=float)
    return lam**ages


def weighted_mean(
    values: pd.Series | np.ndarray | list, lam: float = RECENCY_LAMBDA
) -> float:
    """Recency-weighted mean of a series ordered OLDEST -> NEWEST."""
    arr = pd.Series(values, dtype="float64").to_numpy()
    mask = ~np.isnan(arr)
    arr = arr[mask]
    if arr.size == 0:
        return float("nan")
    w = recency_weights(arr.size, lam)
    return float(np.sum(arr * w) / np.sum(w))


def shrunk_head_to_head(
    h2h_values: pd.Series | np.ndarray | list,
    baseline: float,
    k: float = K_H2H,
    lam: float = RECENCY_LAMBDA,
) -> tuple[float, int]:
    """Recency-weighted, shrunk head-to-head estimate.

    Returns (estimate, n_observations). With no history the estimate is exactly
    the baseline, so this feature is a no-op rather than a source of noise for
    the many player/opponent pairs with no shared history.
    """
    arr = pd.Series(h2h_values, dtype="float64").dropna()
    n = int(arr.size)
    if n == 0:
        return float(baseline), 0
    obs = weighted_mean(arr, lam)
    return float(shrink(obs, n, baseline, k)), n


def credibility(n: float | np.ndarray, k: float = K_RATE) -> float | np.ndarray:
    """Weight placed on the observation, n/(n+k). Useful for reporting.

    Exposed so the gameweek report can say "this h2h read carries 25% weight"
    instead of presenting a shrunk number as if it were a raw one.
    """
    n_arr = np.asarray(n, dtype=float)
    out = n_arr / (n_arr + k)
    return float(out) if np.isscalar(n) else out
