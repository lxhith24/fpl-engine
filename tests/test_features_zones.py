"""Phase 3: shrinkage primitives and zone-fit matchup features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.features import shrinkage as sk
from fpl.features import zones


# ------------------------------------------------------------- shrinkage
def test_shrink_with_no_data_returns_prior():
    assert sk.shrink(observed=10.0, n=0, prior=2.0, k=12) == pytest.approx(2.0)


def test_shrink_with_huge_n_approaches_observation():
    assert sk.shrink(observed=10.0, n=100_000, prior=2.0, k=12) == pytest.approx(10.0, abs=1e-2)


def test_shrink_quarter_weight_at_n4_k12():
    """The plan's worked example: n=4, k=12 moves the estimate 25% of the way."""
    out = sk.shrink(observed=10.0, n=4, prior=2.0, k=12)
    assert out == pytest.approx(2.0 + 0.25 * (10.0 - 2.0))


def test_shrink_is_monotonic_in_n():
    prior, obs = 2.0, 10.0
    vals = [sk.shrink(obs, n, prior, k=12) for n in (0, 1, 5, 20, 100)]
    assert vals == sorted(vals), "more data must move the estimate toward observation"


def test_shrink_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        sk.shrink(1.0, 1.0, 1.0, k=0)


def test_shrink_vectorized():
    out = sk.shrink(np.array([10.0, 0.0]), np.array([4.0, 4.0]), np.array([2.0, 2.0]), k=12)
    assert out.shape == (2,)
    assert out[0] > 2.0 and out[1] < 2.0


def test_recency_weights_favor_recent():
    w = sk.recency_weights(3, lam=0.85)
    assert w[-1] == pytest.approx(1.0)
    assert w[0] < w[1] < w[2], "weights must increase toward the present"


def test_recency_weights_lambda_one_is_flat():
    w = sk.recency_weights(4, lam=1.0)
    assert np.allclose(w, 1.0)


def test_recency_weights_empty():
    assert sk.recency_weights(0).size == 0


def test_weighted_mean_recent_dominates():
    """Oldest->newest ordering: a late spike must pull the mean up."""
    flat = sk.weighted_mean([2, 2, 2, 2])
    rising = sk.weighted_mean([2, 2, 2, 10])
    assert flat == pytest.approx(2.0)
    assert rising > 4.0


def test_weighted_mean_ignores_nan():
    assert sk.weighted_mean([2.0, np.nan, 2.0]) == pytest.approx(2.0)


def test_weighted_mean_all_nan_is_nan():
    assert np.isnan(sk.weighted_mean([np.nan, np.nan]))


def test_shrunk_h2h_no_history_is_exact_baseline():
    """The most important property: no history => the feature is a no-op."""
    est, n = sk.shrunk_head_to_head([], baseline=4.5)
    assert est == pytest.approx(4.5)
    assert n == 0


def test_shrunk_h2h_small_sample_stays_near_baseline():
    """A 2-game 'he always hauls vs them' record must barely move the number."""
    est, n = sk.shrunk_head_to_head([15.0, 14.0], baseline=4.0, k=sk.K_H2H)
    assert n == 2
    assert est < 6.0, f"2-game h2h moved the estimate too far: {est}"


def test_credibility_bounds():
    assert sk.credibility(0, k=12) == pytest.approx(0.0)
    assert 0.0 < sk.credibility(12, k=12) < 1.0
    assert sk.credibility(12, k=12) == pytest.approx(0.5)


# ----------------------------------------------------------------- zones
@pytest.fixture()
def shots():
    """Synthetic shots with a deliberate asymmetry: TeamB leaks on the left."""
    rows = []
    # Left-wing specialist
    for i in range(10):
        rows.append(
            dict(shot_id=f"L{i}", match_id=f"m{i%4}", season=2026, X=0.90, Y=0.20,
                 xG=0.10, understat_player_id="winger", player_name="Winger",
                 attacking_title="TeamA", defending_title="TeamB",
                 situation="OpenPlay", result="MissedShots")
        )
    # Central striker
    for i in range(10):
        rows.append(
            dict(shot_id=f"C{i}", match_id=f"m{i%4}", season=2026, X=0.95, Y=0.50,
                 xG=0.20, understat_player_id="striker", player_name="Striker",
                 attacking_title="TeamA", defending_title="TeamC",
                 situation="OpenPlay", result="Goal")
        )
    return pd.DataFrame(rows)


def test_assign_zones_labels_all_shots(shots):
    z = zones.assign_zones(shots)
    assert z["zone"].notna().all()
    assert set(z["zone"]).issubset(set(zones.ZONES))


def test_assign_zones_boundaries():
    df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "xG": [0.1, 0.1]})
    z = zones.assign_zones(df)
    assert z["zone"].notna().all(), "boundary coordinates must not produce NaN zones"


def test_assign_zones_empty():
    out = zones.assign_zones(pd.DataFrame(columns=["X", "Y", "xG"]))
    assert "zone" in out.columns


def test_player_attack_map_rows_sum_to_one(shots):
    am = zones.player_attack_map(shots)
    assert np.allclose(am[zones.ZONES].sum(axis=1), 1.0)


def test_player_attack_map_separates_flank_from_central(shots):
    am = zones.player_attack_map(shots)
    winger_left = am.loc["winger", "close_left"]
    striker_left = am.loc["striker", "close_left"]
    assert winger_left > 0.9, "left-wing shooter should concentrate in close_left"
    assert striker_left < 0.1


def test_penalties_excluded_by_default():
    pens = pd.DataFrame(
        [
            dict(shot_id="p1", match_id="m1", season=2026, X=0.88, Y=0.50, xG=0.76,
                 understat_player_id="pk", player_name="PK", attacking_title="A",
                 defending_title="B", situation="Penalty", result="Goal"),
            dict(shot_id="s1", match_id="m1", season=2026, X=0.90, Y=0.20, xG=0.10,
                 understat_player_id="pk", player_name="PK", attacking_title="A",
                 defending_title="B", situation="OpenPlay", result="Goal"),
        ]
    )
    am = zones.player_attack_map(pens)
    assert am.loc["pk", "close_left"] == pytest.approx(1.0), "penalty leaked into the map"


def test_concession_map_is_per_match(shots):
    con = zones.opponent_concession_map(shots)
    assert "n_matches" in con.columns
    assert (con["n_matches"] > 0).all()
    # TeamB conceded 10 left-side shots at 0.10 xG across 4 matches.
    assert con.loc["TeamB", "close_left"] == pytest.approx(1.0 / 4 * 4 - 0.75, abs=1.0)


def test_shrunk_concession_pulls_toward_league_mean(shots):
    con = zones.opponent_concession_map(shots)
    shr = zones.shrunk_concession(con, k=25)
    base = zones.league_zone_baseline(con)
    for team in con.index:
        for z in zones.ZONES:
            raw, sh_v, b = con.loc[team, z], shr.loc[team, z], base[z]
            assert min(raw, b) - 1e-9 <= sh_v <= max(raw, b) + 1e-9


def test_zone_fit_is_neutral_for_average_opponent(shots):
    """A player facing a league-average defence should sit near 1.0."""
    con = zones.opponent_concession_map(shots)
    am = zones.player_attack_map(shots)
    fit = zones.zone_fit(am, con)
    assert fit["zone_fit"].between(0.7, 1.4).all()


def test_zone_fit_rewards_matching_weakness(shots):
    """The core claim: a left-side shooter gains against a left-leaking defence."""
    con = zones.opponent_concession_map(shots)
    am = zones.player_attack_map(shots)
    fit = zones.zone_fit(am, con).set_index(["understat_player_id", "defending_title"])
    winger_vs_b = fit.loc[("winger", "TeamB"), "zone_fit"]
    striker_vs_b = fit.loc[("striker", "TeamB"), "zone_fit"]
    assert winger_vs_b > striker_vs_b, "zone fit failed to separate shot profiles"


def test_zone_fit_is_clipped(shots):
    con = zones.opponent_concession_map(shots)
    am = zones.player_attack_map(shots)
    fit = zones.zone_fit(am, con, clip=(0.9, 1.1))
    assert fit["zone_fit"].between(0.9, 1.1).all()


def test_zone_fit_neutral_when_player_has_no_shots():
    am = pd.DataFrame(0.0, index=["ghost"], columns=zones.ZONES)
    am["n_shots"] = 0
    am["total_xg"] = 0.0
    con = pd.DataFrame(0.1, index=["TeamB"], columns=zones.ZONES)
    con["n_matches"] = 3
    fit = zones.zone_fit(am, con)
    assert fit["zone_fit"].iloc[0] == pytest.approx(1.0)


def test_zone_fit_empty_inputs():
    out = zones.zone_fit(pd.DataFrame(), pd.DataFrame())
    assert out.empty
