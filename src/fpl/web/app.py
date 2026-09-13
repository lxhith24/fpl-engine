"""FastAPI app: dashboard JSON + a streaming chat endpoint, served on localhost.

Run with `python -m fpl web` (see `fpl.__main__`). Single user, no auth --
bind only to a loopback address.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import config
from .chat import ChatEngine, MissingCredentials
from .snapshot import Snapshot, load_snapshot

STATIC = Path(__file__).parent / "static"
_SNAPSHOT_TTL_S = 60.0

app = FastAPI(title="fpl-engine dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_engine = ChatEngine()
_cache: dict[int, tuple[float, Snapshot]] = {}
_lock = threading.Lock()


def _snapshot(entry_id: int, *, refresh: bool = False) -> Snapshot:
    """Cached snapshot per entry id. The optimiser MILPs cost ~1-2s, so repeated
    dashboard polls should not rebuild every time."""
    with _lock:
        hit = _cache.get(entry_id)
        if hit and not refresh and (time.time() - hit[0]) < _SNAPSHOT_TTL_S:
            return hit[1]
        snap = load_snapshot(entry_id=entry_id)
        _cache[entry_id] = (time.time(), snap)
        return snap


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    entry: int = config.DEFAULT_ENTRY_ID


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "chat_configured": _chat_configured()}


def _chat_configured() -> bool:
    try:
        _engine._get_client()
        return True
    except MissingCredentials:
        return False


@app.get("/api/snapshot")
def api_snapshot(
    entry: int = config.DEFAULT_ENTRY_ID,
    refresh: bool = Query(False),
) -> JSONResponse:
    snap = _snapshot(entry, refresh=refresh)
    payload = snap.to_dashboard_dict()
    payload["chat_configured"] = _chat_configured()
    return JSONResponse(payload)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> StreamingResponse:
    snap = _snapshot(req.entry)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    def gen():
        try:
            for delta in _engine.stream(messages, snap):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except MissingCredentials as exc:
            yield _sse({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- surface API errors to the panel
            yield _sse({"error": f"{exc.__class__.__name__}: {exc}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
