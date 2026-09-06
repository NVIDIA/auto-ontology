# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ask_question`` — the text-to-SQL agent as a single MCP tool.

Hand-written rather than generated, because ``POST /api/chat/completions``
answers with ``text/event-stream``: the agent emits a step event per graph node
and only then a result. ``FastMCP.from_openapi`` produces request/response
tools and has nothing useful to say about a stream.

Streaming is not merely an obstacle here. A run is many sequential model calls
and routinely takes tens of seconds, which is long enough that a silent tool
looks hung, so the step events are forwarded as MCP progress notifications and
the caller sees the same reasoning trace the web UI shows.

The response the UI receives is markdown with fenced SQL, shaped for a renderer.
This returns the parts separately instead, so a calling agent can use the SQL
and the rows without parsing prose.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from gsf_mcp.config import Settings

logger = logging.getLogger(__name__)

# An agent's context is finite and a question can legitimately match thousands
# of rows. Return a readable prefix and say how much was withheld, rather than
# flooding the caller or silently truncating.
MAX_ROWS = 100

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class DataAnswer(BaseModel):
    """Structured result of one ``ask_question`` call."""

    answer: str = Field(description="Natural-language answer to the question.")
    sql: str = Field(default="", description="The SQL that produced the rows.")
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description=f"Result rows, at most {MAX_ROWS} of them.",
    )
    row_count: int = Field(
        default=0, description="Rows returned by the query before truncation."
    )
    truncated: bool = Field(
        default=False, description=f"True when more than {MAX_ROWS} rows matched."
    )
    reasoning: str = Field(
        default="",
        description="What the agent did, one line per step it took.",
    )


def _parse_rows(raw: Any) -> list[dict[str, Any]]:
    """Normalise the agent's result payload into a list of row dicts.

    The backend sends either a one-item list holding a JSON-records string or
    the records themselves, depending on which path produced the answer.
    """
    if raw is None:
        return []

    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        try:
            raw = json.loads(raw[0])
        except json.JSONDecodeError:
            logger.warning("Result payload was a string but not JSON; dropping rows")
            return []

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _build_answer(payload: dict[str, Any]) -> DataAnswer:
    # A successful run reports rows under `sql_response_from_db`. `result` is
    # only present in the failure shape `_extract_answer` falls back to, where
    # it is None — so reading it first would discard every real answer.
    rows = _parse_rows(payload.get("sql_response_from_db") or payload.get("result"))
    return DataAnswer(
        answer=str(payload.get("response") or ""),
        sql=str(payload.get("sql_code") or ""),
        rows=rows[:MAX_ROWS],
        row_count=len(rows),
        truncated=len(rows) > MAX_ROWS,
        reasoning=str(payload.get("thoughts") or ""),
    )


def _decode_event(line: str) -> dict[str, Any] | None:
    """Return the JSON object on an SSE ``data:`` line, if there is one.

    Heartbeats arrive as ``: ping`` comments and blank lines separate events;
    both are skipped. A malformed payload is dropped rather than failing the
    run — losing one progress update is better than losing the answer.
    """
    if not line.startswith(_SSE_DATA_PREFIX):
        return None
    payload = line[len(_SSE_DATA_PREFIX) :].strip()
    if not payload or payload == _SSE_DONE:
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable SSE payload: %.200s", payload)
        return None
    return event if isinstance(event, dict) else None


async def _raise_for_status(response: httpx.Response) -> None:
    """Turn a non-200 chat response into a ToolError the caller can act on."""
    if response.status_code == 200:
        return

    await response.aread()
    try:
        detail = response.json().get("detail") or response.json().get("error")
    except (json.JSONDecodeError, ValueError):
        detail = response.text.strip()[:300]

    if response.status_code in (401, 403):
        raise ToolError(
            "GSF rejected the credentials for this question. Sign in again, "
            "and check that the account is allowed to use chat. "
            f"({detail})"
        )
    if response.status_code == 409:
        # Either the semantic layer was never built, or this conversation
        # already has a turn in flight. The backend's detail distinguishes them.
        raise ToolError(f"GSF cannot answer right now: {detail}")
    raise ToolError(f"GSF returned HTTP {response.status_code}: {detail}")


def register(mcp: FastMCP, settings: Settings, client: httpx.AsyncClient) -> None:
    """Attach ``ask_question`` to *mcp*."""

    @mcp.tool(
        name="ask_question",
        description=(
            "Ask a natural-language question about the data connected to this "
            "GSF deployment. GSF resolves the question against its semantic "
            "layer, writes SQL, runs it, and returns the answer together with "
            "the SQL it used.\n\n"
            "This is the primary tool and the reason GSF exists. It is also "
            "slow — many sequential model calls, typically tens of seconds — "
            "so use check_answerable first when you are unsure the question is "
            "in scope, and prefer one well-formed question over several "
            "narrow ones."
        ),
    )
    async def ask_question(
        question: str,
        ctx: Context,
        conversation_id: str | None = None,
    ) -> DataAnswer:
        """Run one text-to-SQL turn and return its structured result.

        The API also accepts ``target_db`` to pin retrieval to one database.
        It is not exposed here: it exists for benchmarking, the web UI never
        sends it, and letting the deployment choose is what the semantic layer
        is for.

        Args:
            question: The question, in plain language.
            conversation_id: Continue an existing thread, so follow-up
                questions resolve against earlier turns. Requires the caller
                to hold conversation-write permission, and rejects a second
                question while the first is still running.
        """
        body: dict[str, Any] = {"question": question}
        if conversation_id:
            body["conversation_id"] = conversation_id

        answer: DataAnswer | None = None
        steps = 0

        try:
            async with client.stream(
                "POST",
                "/api/chat/completions",
                json=body,
                headers={"Accept": "text/event-stream"},
                timeout=settings.chat_timeout_s,
            ) as response:
                await _raise_for_status(response)

                async for line in response.aiter_lines():
                    event = _decode_event(line)
                    if event is None:
                        continue

                    kind = event.get("type")
                    if kind == "step":
                        steps += 1
                        label = event.get("label") or event.get("node") or "working"
                        thought = event.get("thought")
                        # No total: the graph's path depends on the question,
                        # so the step count is not known ahead of time.
                        await ctx.report_progress(
                            progress=steps,
                            message=f"{label}: {thought}" if thought else str(label),
                        )
                    elif kind == "result":
                        answer = _build_answer(event.get("answer") or {})
                    elif kind == "error":
                        raise ToolError(
                            f"GSF agent failed: {event.get('message') or 'unknown'}"
                        )
        except httpx.TimeoutException as exc:
            raise ToolError(
                f"GSF did not answer within {settings.chat_timeout_s:.0f}s. "
                "Raise GSF_MCP_CHAT_TIMEOUT_S, or ask a narrower question."
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(
                f"Could not reach GSF at {settings.api_url}: {exc}"
            ) from exc

        if answer is None:
            # The stream ended with no result and no error. This is what a
            # cancelled run looks like from the outside.
            raise ToolError(
                "GSF ended the run without producing an answer. It may have "
                "been cancelled, or the question may not map to the data."
            )
        return answer


__all__ = ["DataAnswer", "MAX_ROWS", "register"]
