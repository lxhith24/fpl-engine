"""Shared HTTP client: retries, per-host rate limiting, on-disk caching.

Understat serves gzip and 403s generic clients; httpx handles decompression
transparently provided we send browser-ish headers.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .. import config

_LAST_CALL: dict[str, float] = {}


def _throttle(host_key: str, min_interval: float) -> None:
    last = _LAST_CALL.get(host_key)
    if last is not None:
        wait = min_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _LAST_CALL[host_key] = time.monotonic()


def _cache_path(url: str, body: str | None = None) -> Path:
    digest = hashlib.sha256(f"{url}|{body or ''}".encode()).hexdigest()[:20]
    return config.INTERIM / "http_cache" / f"{digest}.json"


def get_json(
    url: str,
    *,
    host_key: str = "default",
    min_interval: float = config.FPL_MIN_INTERVAL_S,
    headers: dict[str, str] | None = None,
    cache: bool = False,
    max_age_s: float = 3600.0,
) -> Any:
    """GET JSON with retry/backoff, throttling and optional disk cache."""
    if cache:
        cp = _cache_path(url)
        if cp.exists() and (time.time() - cp.stat().st_mtime) < max_age_s:
            return json.loads(cp.read_text())

    hdrs = {
        "User-Agent": config.USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(config.HTTP_RETRIES):
        _throttle(host_key, min_interval)
        try:
            with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as c:
                resp = c.get(url, headers=hdrs)
            resp.raise_for_status()
            data = resp.json()
            if cache:
                cp = _cache_path(url)
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(data))
            return data
        except Exception as exc:  # noqa: BLE001 - retried below
            last_exc = exc
            if attempt < config.HTTP_RETRIES - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {config.HTTP_RETRIES} attempts: {url}") from last_exc


def post_json(
    url: str,
    data: dict[str, str],
    *,
    host_key: str = "default",
    min_interval: float = config.UNDERSTAT_MIN_INTERVAL_S,
    headers: dict[str, str] | None = None,
) -> Any:
    """POST form-encoded data, expect JSON back."""
    hdrs = {
        "User-Agent": config.USER_AGENT,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(config.HTTP_RETRIES):
        _throttle(host_key, min_interval)
        try:
            with httpx.Client(timeout=config.HTTP_TIMEOUT_S, follow_redirects=True) as c:
                resp = c.post(url, data=data, headers=hdrs)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - retried below
            last_exc = exc
            if attempt < config.HTTP_RETRIES - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"POST failed after {config.HTTP_RETRIES} attempts: {url}") from last_exc
