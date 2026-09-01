"""Shared pytest fixtures: saved API payloads, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="session")
def bootstrap_payload():
    return _load("bootstrap.json")


@pytest.fixture(scope="session")
def fixtures_payload():
    return _load("fixtures.json")


@pytest.fixture(scope="session")
def element_summary_payload():
    return _load("element_summary.json")


@pytest.fixture(scope="session")
def event_live_payload():
    return _load("event_live.json")


@pytest.fixture(scope="session")
def understat_payload():
    return _load("understat_league.json")


@pytest.fixture()
def tmp_db(tmp_path):
    from fpl import store

    con = store.connect(tmp_path / "test.duckdb")
    yield con
    con.close()
