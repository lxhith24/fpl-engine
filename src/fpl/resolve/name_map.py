"""FPL <-> Understat player name resolution (Task 12-14).

Why this module is load-bearing: FPL element ids and Understat player ids are
DIFFERENT NAMESPACES (Bruno Fernandes is FPL 426, Understat 1228). Every
matchup feature joins across them, and a silent join failure drops a star
player from the model without raising anything.

Strategy, in order of reliability:
  1. Team is a HARD constraint -- candidates are only ever the same club.
     Teams are a closed, hand-verified set (see resolve/teams.py).
  2. Score each pair with rapidfuzz over several name variants, because FPL's
     `web_name` is abbreviated ("B.Fernandes") while Understat uses full names
     ("Bruno Fernandes"), and FPL's `second_name` is often the full legal name
     ("Borges Fernandes" -> "Bruno Borges Fernandes").
  3. Resolve within each team by GREEDY ONE-TO-ONE assignment on descending
     score, so two players cannot claim the same counterpart. This is what
     disambiguates real collisions like A.Murphy / J.Murphy at Newcastle.
  4. Manual overrides win over everything.

Deliberately NOT used as a signal: Understat's `position` field. It is mostly
'S' (substitute -- a role marker, not a position), so it carries almost no
positional information and would inject noise.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from . import teams as team_map

# Score below which a pair is never accepted, even as the best remaining option.
MATCH_THRESHOLD = 72.0
# Minutes floor for the coverage gate (plan Task 14).
COVERAGE_MIN_MINUTES = 90
COVERAGE_TARGET = 0.98

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# Letters that NFD CANNOT decompose, because they are distinct letters in their
# own alphabets rather than base+combining-mark pairs. Unicode normalization
# silently leaves these intact, so "Ødegaard" never folds to "odegaard" and the
# match against Understat's "Martin Odegaard" fails at ~55 instead of scoring
# 100. Found live: Martin Ødegaard (FPL 15) went unmatched.
_LETTER_FOLD = str.maketrans(
    {
        "ø": "o", "Ø": "o",
        "đ": "d", "Đ": "d",
        "ð": "d", "Ð": "d",
        "ł": "l", "Ł": "l",
        "ß": "ss",
        "æ": "ae", "Æ": "ae",
        "œ": "oe", "Œ": "oe",
        "þ": "th", "Þ": "th",
        "ı": "i", "İ": "i",
        "ŋ": "n", "Ŋ": "n",
        "ħ": "h", "Ħ": "h",
        "ĸ": "k",
    }
)


def normalize(name: str | None) -> str:
    """Casefold, strip diacritics and punctuation, collapse whitespace.

    90 of 626 FPL players carry diacritics; Understat is inconsistent about
    them, so both sides are folded to ASCII before comparison.

    Two-stage folding is required: NFD handles base+combining-mark characters
    (é -> e), but non-decomposable letters like ø/đ/ł must be transliterated
    explicitly first -- see `_LETTER_FOLD`.
    """
    if not name:
        return ""
    folded = str(name).translate(_LETTER_FOLD)
    decomposed = unicodedata.normalize("NFD", folded)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = _PUNCT.sub(" ", stripped)
    return _WS.sub(" ", cleaned).strip().casefold()


def fpl_name_variants(first: str | None, second: str | None, web: str | None) -> list[str]:
    """Candidate spellings for one FPL player.

    FPL abbreviates in `web_name` in several incompatible ways -- 'B.Fernandes'
    (initial of first name), 'Tóth.A' (surname first), 'O.Dango' (initial of
    the SURNAME with the first name after it). Rather than parse those rules,
    generate every plausible variant and take the best score.
    """
    first_n, second_n, web_n = normalize(first), normalize(second), normalize(web)
    variants = {
        f"{first_n} {second_n}".strip(),
        second_n,
        web_n,
        f"{first_n} {web_n}".strip(),
    }
    # Last token of a multi-part legal surname: "borges fernandes" -> "fernandes"
    if second_n:
        parts = second_n.split()
        if len(parts) > 1:
            variants.add(parts[-1])
            variants.add(f"{first_n} {parts[-1]}".strip())
    return [v for v in variants if v]


def score_pair(fpl_variants: list[str], understat_name: str) -> float:
    """Best fuzzy score of any FPL variant against the Understat name.

    Deliberately does NOT use `token_set_ratio`. That metric scores a strict
    subset as a perfect 100, which caused a real mis-match: David *Raya
    Martín*'s surname-fragment variant "martin" scored 100 against Understat's
    "Martin Odegaard", won the greedy tie, and left Ødegaard unmatched.

    Single-token variants are additionally constrained to compare against the
    target's surname only, so a lone forename can never claim a full name.
    """
    target = normalize(understat_name)
    if not target:
        return 0.0

    target_tokens = target.split()
    target_surname = target_tokens[-1] if target_tokens else target

    best = 0.0
    for v in fpl_variants:
        if v == target:
            return 100.0
        if not v:
            continue
        if len(v.split()) == 1 and len(target_tokens) > 1:
            # Compare a bare single token against the surname, never the whole
            # name -- prevents "martin" from matching "martin odegaard".
            best = max(best, fuzz.ratio(v, target_surname))
            continue
        best = max(best, fuzz.token_sort_ratio(v, target), fuzz.WRatio(v, target))
    return float(best)


@dataclass
class MatchResult:
    mapping: pd.DataFrame           # element <-> understat_id pairs
    unmatched_fpl: pd.DataFrame     # FPL players with no counterpart
    unmatched_understat: pd.DataFrame
    stats: dict = field(default_factory=dict)


def load_overrides(path: Path | str | None = None) -> dict[str, int]:
    """understat_id -> fpl element id, from manual_overrides.yaml.

    Parsed with a tiny reader so PyYAML is not a hard dependency for a file
    that is a flat `key: value` map by construction.
    """
    p = Path(path) if path else Path(__file__).with_name("manual_overrides.yaml")
    if not p.exists():
        return {}
    out: dict[str, int] = {}
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip().strip("'\""), val.strip()
        if key and val:
            try:
                out[key] = int(val)
            except ValueError:
                continue
    return out


def resolve(
    players: pd.DataFrame,
    understat: pd.DataFrame,
    fpl_teams: pd.DataFrame,
    *,
    threshold: float = MATCH_THRESHOLD,
    overrides: dict[str, int] | None = None,
) -> MatchResult:
    """Match FPL players to Understat players, one-to-one within each club."""
    overrides = overrides if overrides is not None else load_overrides()
    team_lookup = team_map.build_team_lookup(fpl_teams)

    us = understat.copy()
    us["fpl_team_id"] = us["team_title"].map(team_lookup)

    rows: list[dict] = []
    used_us: set[str] = set()
    used_fpl: set[int] = set()

    # ---- manual overrides first -----------------------------------------
    fpl_by_id = players.set_index("id")
    for us_id, element in overrides.items():
        if element in fpl_by_id.index and us_id in set(us["understat_id"]):
            rows.append(
                {
                    "element": int(element),
                    "understat_id": us_id,
                    "score": 100.0,
                    "method": "override",
                }
            )
            used_us.add(us_id)
            used_fpl.add(int(element))

    # ---- team-constrained greedy one-to-one ------------------------------
    for team_id, us_group in us.groupby("fpl_team_id", dropna=True):
        fpl_group = players[players["team_id"] == team_id]
        if fpl_group.empty:
            continue

        candidates: list[tuple[float, int, str]] = []
        for _, fp in fpl_group.iterrows():
            if int(fp["id"]) in used_fpl:
                continue
            variants = fpl_name_variants(
                fp.get("first_name"), fp.get("second_name"), fp.get("web_name")
            )
            for _, up in us_group.iterrows():
                if up["understat_id"] in used_us:
                    continue
                s = score_pair(variants, up["player_name"])
                if s >= threshold:
                    candidates.append((s, int(fp["id"]), up["understat_id"]))

        # Highest-confidence pairs claim their partners first.
        for score, element, us_id in sorted(candidates, key=lambda c: -c[0]):
            if element in used_fpl or us_id in used_us:
                continue
            rows.append(
                {
                    "element": element,
                    "understat_id": us_id,
                    "score": score,
                    "method": "fuzzy",
                }
            )
            used_fpl.add(element)
            used_us.add(us_id)

    mapping = pd.DataFrame(rows, columns=["element", "understat_id", "score", "method"])

    unmatched_fpl = players[~players["id"].isin(used_fpl)].copy()
    unmatched_us = us[~us["understat_id"].isin(used_us)].copy()

    stats = {
        "n_mapped": len(mapping),
        "n_fpl": len(players),
        "n_understat": len(us),
        "n_unmapped_understat": len(unmatched_us),
        "unmapped_team_titles": team_map.unmapped_titles(
            sorted(us["team_title"].dropna().unique())
        ),
    }
    return MatchResult(mapping, unmatched_fpl, unmatched_us, stats)


def coverage(
    mapping: pd.DataFrame,
    players: pd.DataFrame,
    *,
    min_minutes: int = COVERAGE_MIN_MINUTES,
) -> dict:
    """Fraction of meaningful-minutes FPL players that resolved.

    Fringe players with no appearances are legitimately absent from Understat,
    so coverage is only measured over players who actually played.
    """
    eligible = players[players["minutes"] >= min_minutes]
    matched = eligible[eligible["id"].isin(set(mapping["element"]))]
    n_elig = len(eligible)
    return {
        "min_minutes": min_minutes,
        "eligible": n_elig,
        "matched": len(matched),
        "coverage": (len(matched) / n_elig) if n_elig else 1.0,
        "missing": eligible[~eligible["id"].isin(set(mapping["element"]))][
            ["id", "web_name", "team_id", "minutes"]
        ],
    }
