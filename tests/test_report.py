"""Phase 6 tests: report generation and CLI."""
from __future__ import annotations

import pandas as pd
import pytest

from fpl.optimize.squad import Squad
from fpl.report.briefing import best_single_transfer, generate_report


@pytest.fixture
def mock_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "element": [1, 2, 3, 4, 5, 6],
            "web_name": ["Player1", "Player2", "Player3", "Player4", "Player5", "Player6"],
            "position": ["GKP", "DEF", "DEF", "MID", "MID", "FWD"],
            "team_id": [1, 1, 2, 2, 3, 3],
            "now_cost": [50, 55, 60, 80, 90, 100],
            "xp": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "xp_sigma": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "selected_by_percent": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            "zone_fit": [1.0, 1.1, 0.9, 1.2, 0.95, 1.0],
        }
    )


@pytest.fixture
def mock_squad(mock_predictions) -> Squad:
    """Minimal valid Squad for report tests."""
    xi = mock_predictions.iloc[:5].copy()
    bench = mock_predictions.iloc[5:].copy()
    return Squad(
        squad=mock_predictions,
        xi=xi,
        bench=bench,
        captain=xi.iloc[-1],
        vice=xi.iloc[-2],
        formation="3-2-0",
        total_cost=43.5,
        xi_xp=22.5,
        squad_xp=28.5,
        objective=25.0,
        status="Optimal",
    )


def test_best_single_transfer_finds_upgrade(mock_predictions):
    current = [1, 2]  # Player1 (3.5 xP), Player2 (4.0 xP)
    result = best_single_transfer(current, mock_predictions, bank=5.0)

    assert result is not None
    assert result["gain"] > 0
    assert result["out_id"] in current
    assert result["in_id"] not in current


def test_best_single_transfer_respects_budget(mock_predictions):
    current = [1]  # Player1, cost 50
    # With 0 bank, can only afford players <= 50
    result = best_single_transfer(current, mock_predictions, bank=0.0)

    if result:
        assert result["in_cost"] <= 50


def test_best_single_transfer_returns_none_when_no_upgrade():
    # Current player is already the best
    preds = pd.DataFrame(
        {
            "element": [1, 2],
            "web_name": ["Best", "Worst"],
            "position": ["MID", "MID"],
            "now_cost": [100, 50],
            "xp": [10.0, 2.0],
        }
    )
    result = best_single_transfer([1], preds, bank=0.0)
    assert result is None


def test_generate_report_includes_all_sections(mock_squad, mock_predictions):
    report = generate_report(
        gw=3,
        predictions=mock_predictions,
        optimal=mock_squad,
    )

    assert "# GW3 Briefing" in report
    assert "## Optimal XI" in report
    assert "Captain:" in report
    assert "Vice:" in report
    assert "Bench:" in report


def test_generate_report_includes_current_squad_comparison(mock_squad, mock_predictions):
    current_elements = [1, 2, 3]
    report = generate_report(
        gw=3,
        predictions=mock_predictions,
        optimal=mock_squad,
        current_squad=mock_squad,
        current_elements=current_elements,
    )

    assert "## Your Squad" in report
    assert "Gap to optimum" in report


def test_generate_report_includes_transfer_recommendation(mock_squad, mock_predictions):
    transfer = {
        "out_name": "PlayerOut",
        "out_pos": "MID",
        "out_cost": 80,
        "out_xp": 4.0,
        "in_name": "PlayerIn",
        "in_pos": "MID",
        "in_cost": 85,
        "in_xp": 6.0,
        "gain": 2.0,
        "net_cost": 0.5,
    }
    report = generate_report(
        gw=3,
        predictions=mock_predictions,
        optimal=mock_squad,
        transfer=transfer,
    )

    assert "## Best Single Transfer" in report
    assert "PlayerOut" in report
    assert "PlayerIn" in report
    assert "+2.00 xP" in report


def test_generate_report_includes_differentials(mock_squad, mock_predictions):
    # Add a differential player
    preds = mock_predictions.copy()
    preds.loc[0, "selected_by_percent"] = 2.0
    preds.loc[0, "xp"] = 5.0

    report = generate_report(gw=3, predictions=preds, optimal=mock_squad)
    assert "## Differentials" in report


def test_generate_report_includes_zone_fit(mock_squad, mock_predictions):
    # Ensure zone_fit column triggers section
    preds = mock_predictions.copy()
    preds.loc[0, "zone_fit"] = 1.35

    report = generate_report(gw=3, predictions=preds, optimal=mock_squad)
    assert "## Matchup Boosts" in report
