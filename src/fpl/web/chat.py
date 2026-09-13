"""Claude chat engine, grounded in one `Snapshot`.

Read-only by design: no tools, no ability to run the pipeline or mutate the
store. The model gets the gameweek digest in its system prompt and answers
from it. Streaming, so the panel fills token-by-token.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

from .snapshot import Snapshot

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """\
You are the analyst built into an FPL (Fantasy Premier League) optimal-XI \
engine. A manager is looking at the dashboard for the upcoming gameweek and \
asking you about it.

Ground rules:
- Answer ONLY from the GAMEWEEK DATA below. It is the output of this engine's \
own minutes + points models and a MILP squad optimiser. Do not pull in outside \
knowledge about form, injuries, or fixtures that is not in the data.
- "xP" is the model's expected points for this gameweek; "xP_sigma" is its \
standard deviation (ceiling/volatility). "zone_fit" > 1.00 means the player \
attacks where this opponent is weak. "p_start" is the modelled probability of \
starting; "own%" is FPL ownership.
- The optimal squad uses risk=aggressive: it seeks variance and fades \
template picks, so it will look contrarian. Say so when it is relevant.
- The from-scratch optimum is usually NOT reachable by transfers this week -- \
it is a Wildcard target. The "best single transfer" line is the actionable one.
- Be concise and specific. Quote the numbers. When the data does not settle \
the question, say what is missing rather than guessing.
- You cannot run the pipeline, change settings, or make transfers. If asked, \
explain the CLI command the user would run (e.g. `python -m fpl optimize \
--risk safe`).
"""


class MissingCredentials(RuntimeError):
    """No Anthropic API key / profile available."""


class ChatEngine:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:  # pragma: no cover - install guard
                raise MissingCredentials(
                    "The `anthropic` package is not installed. "
                    "Run: pip install -e '.[web]'"
                ) from exc
            if not (
                os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            ):
                raise MissingCredentials(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY in the "
                    "environment before starting `python -m fpl web`."
                )
            self._client = anthropic.Anthropic()
        return self._client

    def system_prompt(self, snapshot: Snapshot) -> str:
        return SYSTEM_PROMPT + "\n\n=== GAMEWEEK DATA ===\n" + snapshot.to_chat_context()

    def stream(
        self, messages: list[dict], snapshot: Snapshot, *, max_tokens: int = 4096
    ) -> Iterator[str]:
        """Yield answer text deltas. Raises MissingCredentials if unconfigured."""
        client = self._get_client()
        with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt(snapshot),
            thinking={"type": "adaptive"},
            messages=messages,
        ) as stream:
            yield from stream.text_stream
