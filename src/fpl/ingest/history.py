"""Historical multi-season FPL data (cold-start fix).

At GW3 the current season has ~600 usable rows -- far too few to fit anything.
The plan flagged this as the biggest near-term limitation. The vaastav
Fantasy-Premier-League dataset provides per-gameweek player rows going back
years, with the same column vocabulary as the live API (including
`defensive_contribution` and expected-goals columns).

NOTE ON ACCESS: raw.githubusercontent.com is unreachable from this machine
(curl returns 000, a DNS/connection failure, not an HTTP error). The jsDelivr
CDN mirror serves identical bytes and works, so that is the fetch path.

    https://cdn.jsdelivr.net/gh/vaastav/Fantasy-Premier-League@master/data/<season>/gws/merged_gw.csv
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from .. import config

CDN = "https://cdn.jsdelivr.net/gh/vaastav/Fantasy-Premier-League@master/data"
SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
HISTORY_DIR = config.RAW / "history"

# Columns we consume; anything else in the CSV is ignored.
_KEEP = [
    "element", "name", "position", "team", "opponent_team", "was_home",
    "GW", "kickoff_time", "minutes", "starts", "total_points",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
    "bonus", "bps", "yellow_cards", "red_cards", "value",
    "expected_goals", "expected_assists", "expected_goals_conceded",
    "defensive_contribution", "influence", "creativity", "threat", "ict_index",
]


def download_season(season: str, *, force: bool = False) -> Path | None:
    """Fetch one season's merged gameweek CSV to data/raw/history/."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{season}_merged_gw.csv"
    if path.exists() and not force and path.stat().st_size > 1000:
        return path

    url = f"{CDN}/{season}/gws/merged_gw.csv"
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "120", "-o", str(path), "-w", "%{http_code}", url],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != "200" or path.stat().st_size < 1000:
        path.unlink(missing_ok=True)
        return None
    return path


def load_season(season: str) -> pd.DataFrame:
    """One season as a normalized frame, or empty if unavailable."""
    path = download_season(season)
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    have = [c for c in _KEEP if c in df.columns]
    df = df[have].copy()
    df["season"] = season

    df = df.rename(columns={"GW": "event"})
    for col in ("minutes", "total_points", "starts", "bps", "bonus"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in (
        "expected_goals", "expected_assists", "expected_goals_conceded",
        "defensive_contribution",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(bool)
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
        df["kickoff_time"] = df["kickoff_time"].dt.tz_localize(None)

    # Normalize position labels to the FPL vocabulary.
    df["position"] = (
        df["position"].astype(str).str.upper().replace({"GK": "GKP", "AM": "MID"})
    )
    return df[df["position"].isin(["GKP", "DEF", "MID", "FWD"])]


def load_history(seasons: tuple[str, ...] = SEASONS, *, progress: bool = False) -> pd.DataFrame:
    """All available seasons stacked, oldest first."""
    frames = []
    for s in seasons:
        df = load_season(s)
        if df.empty:
            if progress:
                print(f"  {s}: unavailable")
            continue
        if progress:
            print(f"  {s}: {len(df):,} rows, {df['event'].nunique()} GWs")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["season", "event"])


def season_sort_key(season: str) -> int:
    return int(season.split("-")[0])
