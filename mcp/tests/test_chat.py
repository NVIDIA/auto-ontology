# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ask_question`` turns the agent's SSE stream into a structured answer."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from gsf_mcp import chat
from gsf_mcp.config import DEFAULT_SPEC_PATH, Settings

_ANSWER = {
    "response": "There are 42 active customers.",
    "sql_code": "SELECT count(*) FROM customers WHERE active",
    "sql_response_from_db": ['[{"count": 42}]'],
    "thoughts": "- Retrieval: found the customers table",
}

# Captured verbatim from a live run, so the field names here are observed
# rather than assumed. Reading rows from `result` instead looks correct against
# a hand-written fixture and silently returns nothing against the real backend.
_LIVE_ANSWER = {
    "response": (
        "This counts the total number of customers by counting the customer "
        "identifiers in the Customers table."
    ),
    "sql_code": 'SELECT COUNT("CustomerID") AS customer_count FROM "SALES"."CUSTOMERS"',
    "sql_columns": [],
    "custom_analyses_used": [],
    "sql_response_from_db": ['[{"CUSTOMER_COUNT":663}]'],
    "thoughts": "- Constructing SQL: Count all rows in the Customers table.",
}


def _settings(chat_timeout_s: float = 900.0) -> Settings:
    return Settings(
        api_url="http://gsf.test",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=30.0,
        chat_timeout_s=chat_timeout_s,
    )


def _sse(*events: dict[str, Any], done: bool = True) -> str:
    """Render events the way the backend streams them, heartbeats included."""
    body = ": ping\n\n"
    for event in events:
        body += f"data: {json.dumps(event)}\n\n"
    if done:
        body += "data: [DONE]\n\n"
    return body


def _server(
    handler: Callable[[httpx.Request], httpx.Response],
    chat_timeout_s: float = 900.0,
) -> FastMCP:
    settings = _settings(chat_timeout_s)
    client = httpx.AsyncClient(
        base_url=settings.api_url, transport=httpx.MockTransport(handler)
    )
    mcp: FastMCP = FastMCP(name="test")
    chat.register(mcp, settings, client)
    return mcp


def _stream(body: str, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    return handler


def _call(mcp: FastMCP, arguments: dict[str, Any], progress: Any = None) -> Any:
    async def run() -> Any:
        async with Client(mcp, progress_handler=progress) as client:
            return await client.call_tool("ask_question", arguments)

    return asyncio.run(run())


def test_returns_answer_sql_and_rows() -> None:
    mcp = _server(_stream(_sse({"type": "result", "answer": _ANSWER})))

    result = _call(mcp, {"question": "How many active customers?"})

    assert result.data.answer == "There are 42 active customers."
    assert result.data.sql == "SELECT count(*) FROM customers WHERE active"
    assert result.data.rows == [{"count": 42}]
    assert result.data.row_count == 1
    assert result.data.truncated is False
    assert "found the customers table" in result.data.reasoning


def test_forwards_agent_steps_as_progress() -> None:
    # A run takes tens of seconds. Without these the caller cannot tell a slow
    # answer from a hung one.
    seen: list[str] = []

    async def on_progress(
        progress: float, total: float | None, message: str | None
    ) -> None:
        seen.append(message or "")

    mcp = _server(
        _stream(
            _sse(
                {"type": "step", "node": "retrieve", "label": "Retrieving context"},
                {
                    "type": "step",
                    "node": "generate",
                    "label": "Writing SQL",
                    "thought": "joining orders",
                },
                {"type": "result", "answer": _ANSWER},
            )
        )
    )

    _call(mcp, {"question": "q"}, progress=on_progress)

    assert seen == ["Retrieving context", "Writing SQL: joining orders"]


def test_returns_the_rows_from_a_real_backend_payload() -> None:
    # Guards the field name itself: the answer prose does not state the number,
    # so if rows are dropped the tool returns a description of a count with no
    # count in it.
    mcp = _server(_stream(_sse({"type": "result", "answer": _LIVE_ANSWER})))

    result = _call(mcp, {"question": "How many customers are there?"})

    assert result.data.rows == [{"CUSTOMER_COUNT": 663}]
    assert result.data.row_count == 1


def test_falls_back_to_the_failure_shapes_result_field() -> None:
    # `_extract_answer`'s fallback reports `result` instead, so both are read.
    answer = {"response": "SQL can't be constructed.", "result": [{"count": 7}]}
    mcp = _server(_stream(_sse({"type": "result", "answer": answer})))

    assert _call(mcp, {"question": "q"}).data.rows == [{"count": 7}]


def test_accepts_rows_sent_as_plain_records() -> None:
    # The backend sends a one-item list holding JSON in the common path, but
    # records directly on others; both have to land as rows.
    answer = dict(_ANSWER, sql_response_from_db=[{"count": 1}, {"count": 2}])
    mcp = _server(_stream(_sse({"type": "result", "answer": answer})))

    result = _call(mcp, {"question": "q"})

    assert result.data.rows == [{"count": 1}, {"count": 2}]


def test_reports_no_rows_when_the_query_returned_none() -> None:
    answer = dict(_ANSWER, sql_response_from_db=None)
    mcp = _server(_stream(_sse({"type": "result", "answer": answer})))

    result = _call(mcp, {"question": "q"})

    assert result.data.rows == []
    assert result.data.row_count == 0


def test_truncates_large_result_sets_and_says_so() -> None:
    rows = [{"n": n} for n in range(chat.MAX_ROWS + 25)]
    answer = dict(_ANSWER, sql_response_from_db=[json.dumps(rows)])
    mcp = _server(_stream(_sse({"type": "result", "answer": answer})))

    result = _call(mcp, {"question": "q"})

    assert len(result.data.rows) == chat.MAX_ROWS
    assert result.data.row_count == chat.MAX_ROWS + 25
    assert result.data.truncated is True


def test_survives_heartbeats_and_unparseable_lines() -> None:
    body = (
        ": ping\n\n"
        "data: {not json\n\n"
        "\n"
        f"data: {json.dumps({'type': 'result', 'answer': _ANSWER})}\n\n"
        "data: [DONE]\n\n"
    )
    mcp = _server(_stream(body))

    result = _call(mcp, {"question": "q"})

    assert result.data.answer == "There are 42 active customers."


def test_surfaces_an_agent_error_event() -> None:
    mcp = _server(
        _stream(_sse({"type": "error", "message": "no tables matched the question"}))
    )

    with pytest.raises(ToolError, match="no tables matched the question"):
        _call(mcp, {"question": "q"})


def test_reports_a_stream_that_ends_without_an_answer() -> None:
    mcp = _server(_stream(_sse({"type": "step", "node": "retrieve"})))

    with pytest.raises(ToolError, match="without producing an answer"):
        _call(mcp, {"question": "q"})


def test_explains_a_rejected_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid token"})

    with pytest.raises(ToolError, match="rejected the credentials"):
        _call(_server(handler), {"question": "q"})


def test_passes_through_a_conflict_detail() -> None:
    # 409 means either an uncompiled semantic layer or a turn already running;
    # only the backend's detail distinguishes them, so it must survive.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "semantic layer not compiled"})

    with pytest.raises(ToolError, match="semantic layer not compiled"):
        _call(_server(handler), {"question": "q"})


def test_reports_an_unreachable_deployment() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ToolError, match="Could not reach GSF at http://gsf.test"):
        _call(_server(handler), {"question": "q"})


def test_names_the_knob_when_a_run_times_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(ToolError, match="GSF_MCP_CHAT_TIMEOUT_S"):
        _call(_server(handler, chat_timeout_s=5.0), {"question": "q"})


def test_sends_only_the_question_when_nothing_else_is_given() -> None:
    # conversation_id changes backend behaviour and permissions; sending it as
    # a null is not the same as omitting it.
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse({"type": "result", "answer": _ANSWER}).encode(),
            headers={"content-type": "text/event-stream"},
        )

    _call(_server(handler), {"question": "how many?"})

    assert captured == {"question": "how many?"}


def test_forwards_the_conversation_when_given() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse({"type": "result", "answer": _ANSWER}).encode(),
            headers={"content-type": "text/event-stream"},
        )

    _call(
        _server(handler),
        {
            "question": "how many?",
            "conversation_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
        },
    )

    assert captured["conversation_id"] == "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


def test_forwards_evidence_separately_from_question() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse({"type": "result", "answer": _ANSWER}).encode(),
            headers={"content-type": "text/event-stream"},
        )

    _call(
        _server(handler),
        {
            "question": "how many?",
            "evidence": "active refers to customers.status",
        },
    )

    assert captured == {
        "question": "how many?",
        "evidence": "active refers to customers.status",
    }


def test_the_database_cannot_be_pinned_by_a_caller() -> None:
    # target_db exists on the API for benchmarking. Exposing it would invite an
    # agent to route around the semantic layer, so the tool has no such
    # parameter and a caller passing one is rejected rather than silently
    # ignored.
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("the request should never be sent")

    with pytest.raises(ToolError):
        _call(_server(handler), {"question": "how many?", "target_db": "sales"})
