"""Phase 7 tests: dashboard snapshot assembly + FastAPI wiring."""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fpl import config
from fpl.web.chat import ChatEngine, MissingCredentials
from fpl.web.snapshot import build_snapshot


@pytest.fixture
def pool() -> pd.DataFrame:
    """A pool big enough for the squad MILP to field a legal 15."""
    rows = []
    for i in range(30):
        pos = ["GKP", "DEF", "MID", "FWD"][i % 4]
        rows.append(
            {
                "element": i + 1,
                "web_name": f"P{i + 1}",
                "position": pos,
                "team_id": (i % 6) + 1,
                "opponent_id": ((i + 3) % 6) + 1,
                "is_home": bool(i % 2),
                "now_cost": 45 + (i % 5) * 5,
                "status": "a",
                "form": 3.0 + (i % 4),
                "selected_by_percent": float(i % 20),
                "p_start": 0.6 + (i % 4) * 0.1,
                "exp_minutes": 70 + (i % 3) * 5,
                "zone_fit": 1.0 + (0.1 if i % 7 == 0 else 0.0),
                "difficulty": (i % 5) + 1,
                "xp_sigma": 1.0 + (i % 5) * 0.2,
                "xp": 2.0 + (i % 10) * 0.6,
            }
        )
    return pd.DataFrame(rows)


TEAMS = {i: f"T{i}" for i in range(1, 7)}


def test_build_snapshot_no_predictions():
    snap = build_snapshot(4, None, TEAMS, entry_id=999)
    assert snap.has_predictions is False
    d = snap.to_dashboard_dict()
    assert d["gw"] == 4 and d["has_predictions"] is False
    assert "No predictions" in snap.to_chat_context()


def test_build_snapshot_dashboard_shape(pool):
    snap = build_snapshot(
        4, pool, TEAMS, entry_id=999, your_elements=list(range(1, 16))
    )
    d = snap.to_dashboard_dict()

    assert d["has_predictions"] is True
    assert d["optimal"]["formation"].count("-") == 2
    assert len(d["optimal"]["xi"]) == 11
    assert d["optimal"]["captain"]["web_name"]
    assert d["your_squad"]["available"] is True
    assert d["your_squad"]["gap"] is not None
    assert len(d["predictions"]) == len(pool)
    # predictions come back sorted by xP descending
    xps = [r["xp"] for r in d["predictions"] if r["xp"] is not None]
    assert xps == sorted(xps, reverse=True)
    # price is rendered in millions, not tenths
    assert all(r["now_cost"] < 15 for r in d["predictions"])


def test_chat_context_mentions_key_facts(pool):
    snap = build_snapshot(
        7, pool, TEAMS, entry_id=8905049, your_elements=list(range(1, 16))
    )
    ctx = snap.to_chat_context()
    assert "GAMEWEEK 7" in ctx
    assert "OPTIMAL SQUAD" in ctx
    assert "MANAGER'S ACTUAL SQUAD" in ctx
    assert "PROJECTED PLAYER POOL" in ctx
    assert snap.optimal.captain["web_name"] in ctx


def test_chat_engine_requires_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingCredentials):
        ChatEngine()._get_client()


@pytest.mark.skipif(
    not config.DB_PATH.exists(), reason="no local DuckDB store to snapshot"
)
def test_snapshot_endpoint_smoke():
    from fastapi.testclient import TestClient

    from fpl.web.app import app

    client = TestClient(app)
    assert client.get("/").status_code == 200

    r = client.get("/api/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert "gw" in body and "has_predictions" in body
    assert "chat_configured" in body


@pytest.mark.skipif(
    bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
    reason="credentials present — would make a real API call",
)
def test_chat_endpoint_reports_missing_credentials():
    from fastapi.testclient import TestClient

    from fpl.web.app import app

    client = TestClient(app)
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "credential" in r.text.lower() or "anthropic_api_key" in r.text.lower()
