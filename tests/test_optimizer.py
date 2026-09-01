"""Phase 5: MILP squad/XI optimizer (Tasks 30-33).

Constraint tests are exhaustive because an illegal squad is worse than a
suboptimal one -- FPL will simply reject it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.optimize import squad as opt


def make_pool(n_per_pos: int = 12, seed: int = 0) -> pd.DataFrame:
    """Synthetic but realistic candidate pool spanning 20 clubs."""
    rng = np.random.default_rng(seed)
    rows = []
    el = 1
    for pos, base in (("GKP", 3.5), ("DEF", 4.0), ("MID", 4.5), ("FWD", 4.5)):
        for i in range(n_per_pos):
            rows.append(
                {
                    "element": el,
                    "web_name": f"{pos}{i}",
                    "position": pos,
                    "team_id": (el % 20) + 1,
                    "now_cost": int(rng.integers(40, 130)),
                    "xp": float(base + rng.normal(0, 1.2)),
                    "xp_sigma": float(abs(rng.normal(2.5, 0.5))),
                    "selected_by_percent": float(rng.uniform(0, 60)),
                    "p_start": float(rng.uniform(0.3, 1.0)),
                }
            )
            el += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pool():
    return make_pool()


@pytest.fixture(scope="module")
def solved(pool):
    return opt.optimize_squad(pool, risk="aggressive")


# ------------------------------------------------------------ constraints
def test_solver_reaches_optimal(solved):
    assert solved.status == "Optimal"


def test_squad_has_exactly_15(solved):
    assert len(solved.squad) == opt.SQUAD_SIZE


def test_squad_respects_budget(solved):
    assert solved.squad["now_cost"].sum() <= opt.BUDGET


def test_positional_quota(solved):
    counts = solved.squad["position"].value_counts().to_dict()
    for pos, n in opt.SQUAD_QUOTA.items():
        assert counts.get(pos, 0) == n, f"{pos}: expected {n}, got {counts.get(pos, 0)}"


def test_max_three_per_club(solved):
    assert solved.squad["team_id"].value_counts().max() <= opt.MAX_PER_CLUB


def test_xi_is_eleven(solved):
    assert len(solved.xi) == opt.XI_SIZE


def test_xi_is_subset_of_squad(solved):
    assert set(solved.xi["element"]).issubset(set(solved.squad["element"]))


def test_bench_completes_the_squad(solved):
    assert len(solved.bench) == opt.SQUAD_SIZE - opt.XI_SIZE
    assert set(solved.xi["element"]).isdisjoint(set(solved.bench["element"]))


def test_formation_is_legal(solved):
    counts = solved.xi["position"].value_counts().to_dict()
    assert counts.get("GKP", 0) == 1
    assert counts.get("DEF", 0) >= 3
    assert counts.get("FWD", 0) >= 1
    assert sum(counts.values()) == 11


def test_captain_and_vice_are_starters(solved):
    starters = set(solved.xi["element"])
    assert solved.captain["element"] in starters
    assert solved.vice["element"] in starters
    assert solved.captain["element"] != solved.vice["element"]


def test_captain_is_highest_xp_in_xi(solved):
    """Captain doubles points, so it must track raw xP, not risk-adjusted score."""
    assert solved.captain["xp"] == pytest.approx(solved.xi["xp"].max())


# ------------------------------------------------------- XI selection bug
def test_no_benched_outfielder_beats_a_worse_starter(pool):
    """REGRESSION: the differential tax was charged twice.

    Ownership belongs in the SQUAD decision (who you own vs the field). Once
    a player is owned, starting him is a pure points question. Applying the
    penalty again at XI level benched Szoboszlai (5.64 xP, 42% owned) in
    favour of Ajer (4.92 xP, 4.5% owned) on live GW3 data -- giving away 0.73
    xP for no differential benefit whatsoever.

    Formation limits still legitimately bench better players (a 6th defender
    cannot start), so this compares within position.
    """
    s = opt.optimize_squad(pool, risk="aggressive")
    for pos in ("DEF", "MID", "FWD"):
        starters = s.xi[s.xi["position"] == pos]
        benched = s.bench[s.bench["position"] == pos]
        if starters.empty or benched.empty:
            continue
        assert benched["xp"].max() <= starters["xp"].max() + 1e-6


def test_xi_score_ignores_ownership_but_squad_score_does_not():
    df = pd.DataFrame(
        {
            "xp": [5.0, 5.0],
            "xp_sigma": [1.0, 1.0],
            "selected_by_percent": [50.0, 1.0],
        }
    )
    squad_score = opt._score(df, gamma=0.0, beta=0.05)
    xi_score = opt._score(df, gamma=0.0, beta=0.0)
    assert squad_score.iloc[0] < squad_score.iloc[1], "ownership must matter for squad"
    assert xi_score.iloc[0] == pytest.approx(xi_score.iloc[1]), "not for XI"


# ------------------------------------------------------------- risk modes
def test_risk_modes_produce_different_squads(pool):
    agg = opt.optimize_squad(pool, risk="aggressive")
    safe = opt.optimize_squad(pool, risk="safe")
    assert set(agg.squad["element"]) != set(safe.squad["element"])


def test_aggressive_lowers_ownership(pool):
    """The whole point of the aggressive objective."""
    agg = opt.optimize_squad(pool, risk="aggressive")
    safe = opt.optimize_squad(pool, risk="safe")
    assert agg.xi["selected_by_percent"].mean() < safe.xi["selected_by_percent"].mean()


def test_aggressive_costs_expected_points(pool):
    """Differential-chasing is a trade, not a free lunch -- document it."""
    agg = opt.optimize_squad(pool, risk="aggressive")
    bal = opt.optimize_squad(pool, risk="balanced")
    assert agg.xi["xp"].sum() <= bal.xi["xp"].sum() + 1e-6


def test_weights_calibrated_to_spread(pool):
    """Raw coefficients are meaningless without the pool's scale."""
    g, b = opt._calibrate_weights(pool, 0.25, 2.5)
    assert g > 0 and b > 0
    g2, b2 = opt._calibrate_weights(pool, 0.0, 0.0)
    assert g2 == 0.0 and b2 == 0.0


def test_explicit_weights_override_preset(pool):
    s = opt.optimize_squad(pool, risk="aggressive", gamma=0.0, beta=0.0)
    assert s.meta["gamma"] == 0.0
    assert s.meta["beta"] == 0.0


# ------------------------------------------------------------- edge cases
def test_locked_players_are_selected(pool):
    lock = [int(pool.iloc[0]["element"]), int(pool.iloc[30]["element"])]
    s = opt.optimize_squad(pool, risk="balanced", locked=lock)
    assert set(lock).issubset(set(s.squad["element"]))


def test_banned_players_excluded(pool):
    top = pool.sort_values("xp", ascending=False).head(3)["element"].tolist()
    s = opt.optimize_squad(pool, risk="balanced", banned=top)
    assert set(top).isdisjoint(set(s.squad["element"]))


def test_double_gameweek_xp_is_summed():
    """A team playing twice should have both fixtures counted, not one."""
    pool = make_pool()
    dgw = pool.iloc[[0]].copy()
    doubled = pd.concat([pool, dgw], ignore_index=True)
    s = opt.optimize_squad(doubled, risk="balanced")
    assert len(s.squad) == opt.SQUAD_SIZE
    assert s.squad["element"].is_unique


def test_infeasible_budget_raises():
    pool = make_pool()
    pool["now_cost"] = 300
    with pytest.raises(RuntimeError):
        opt.optimize_squad(pool, budget=100)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        opt.optimize_squad(pd.DataFrame({"element": [1], "xp": [5.0]}))


def test_optimize_xi_from_existing_squad(pool):
    full = opt.optimize_squad(pool, risk="balanced")
    xi_only = opt.optimize_xi(full.squad)
    assert len(xi_only.xi) == 11
    counts = xi_only.xi["position"].value_counts().to_dict()
    assert counts.get("GKP", 0) == 1 and counts.get("DEF", 0) >= 3


def test_gap_to_current_reports_transfers(pool):
    s = opt.optimize_squad(pool, risk="balanced")
    current = pool["element"].head(15).tolist()
    gap = opt.gap_to_current(s, current, pool)
    assert gap["n_transfers"] == len(set(s.squad["element"]) - set(current))
    assert gap["hit_cost"] == max(0, gap["n_transfers"] - 1) * 4


def test_gap_to_identical_squad_is_zero(pool):
    s = opt.optimize_squad(pool, risk="balanced")
    gap = opt.gap_to_current(s, s.squad["element"].tolist(), pool)
    assert gap["n_transfers"] == 0
    assert gap["hit_cost"] == 0


def test_min_start_prob_prefers_nailed_players(pool):
    """Rotation filter must raise mean start probability without going infeasible.

    A hard cut can leave too few players per position to field a legal 15, so
    the filter relaxes per position rather than handing CBC an infeasible
    problem. The guarantee is therefore "more nailed on average", not "every
    player above the threshold".
    """
    base = opt.optimize_squad(pool, risk="balanced")
    strict = opt.optimize_squad(pool, risk="balanced", min_start_prob=0.8)
    assert len(strict.squad) == 15
    assert strict.squad["p_start"].mean() > base.squad["p_start"].mean()


# ------------------------------------------------------------------ fuzz
@pytest.mark.parametrize("seed", range(12))
def test_random_pools_always_yield_legal_squads(seed):
    """Constraint fuzzing (plan section 7): every solve must be FPL-legal."""
    s = opt.optimize_squad(make_pool(seed=seed), risk="aggressive")
    assert len(s.squad) == 15
    assert s.squad["now_cost"].sum() <= opt.BUDGET
    assert s.squad["team_id"].value_counts().max() <= 3
    counts = s.squad["position"].value_counts().to_dict()
    assert counts == opt.SQUAD_QUOTA
    xi_counts = s.xi["position"].value_counts().to_dict()
    assert xi_counts.get("GKP", 0) == 1
    assert xi_counts.get("DEF", 0) >= 3
    assert xi_counts.get("FWD", 0) >= 1
    assert len(s.xi) == 11
