# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("REASONING_API_KEY", "test-key")
os.environ.setdefault("REASONING_ENDPOINT", "https://example.test/v1")
os.environ.setdefault("REASONING_MODEL", "openai/test-model")

from gsf.utils import llm_invoke  # noqa: E402

llm_invoke._API_KEY = "test-key"
llm_invoke._BASE_URL = "https://example.test/v1"
llm_invoke._MODEL_NAME = "openai/test-model"

from gsf.retrieval.text_to_sql import main  # noqa: E402


class _FailAfterGenerationApp:
    """Graph double that fails after one node has produced SQL.

    ``stream_agent_response`` asks for ``stream_mode=["updates", "custom"]``,
    which makes LangGraph yield ``(mode, chunk)`` tuples rather than bare
    node updates — a node's start announcement on ``custom``, then its state
    update on ``updates``. Mirrored here so the double matches the real call.
    """

    def stream(
        self,
        state: dict[str, Any],
        stream_mode: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[tuple[str, Any]]:
        yield ("custom", {"type": "step_start", "node": "validate_intent"})
        yield (
            "updates",
            {
                "validate_intent": {
                    "path_state": {
                        "sql_generation_result": SimpleNamespace(
                            sql_code="SELECT 1",
                            response="ok",
                            thought="generated before downstream failure",
                        )
                    }
                }
            },
        )
        raise KeyError("intent_valid")


def test_stream_error_preserves_partial_generated_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "app", _FailAfterGenerationApp())

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1] == {
        "type": "error",
        "message": "Agent failed after validate_intent: 'intent_valid'",
        "node": "validate_intent",
        "error_type": "KeyError",
        "partial_answer": {
            "sql_code": "SELECT 1",
            "response": "ok",
            "thought": "generated before downstream failure",
        },
    }


def test_nonstreaming_error_exposes_partial_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "app", _FailAfterGenerationApp())

    with pytest.raises(main.AgentRunError) as caught:
        main.get_agent_response({"question": "q"})

    assert caught.value.node == "validate_intent"
    assert caught.value.partial_answer["sql_code"] == "SELECT 1"


class _FailInsideFirstNodeApp:
    """Graph double whose first node raises before it ever returns."""

    def stream(
        self,
        state: dict[str, Any],
        stream_mode: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[tuple[str, Any]]:
        yield ("custom", {"type": "step_start", "node": "retrieve_candidates"})
        raise TimeoutError("retriever unavailable")


def test_error_inside_a_node_names_that_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node that raises produces no update, so completions alone can't name
    it — the failure would be attributed to whichever node last succeeded, or
    to ``graph_start`` when it is the first one. The start announcement is
    what makes the report point at the node that was actually running.
    """
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "app", _FailInsideFirstNodeApp())

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1]["node"] == "retrieve_candidates"
    assert events[-1]["error_type"] == "TimeoutError"
    assert "graph_start" not in events[-1]["message"]


def test_error_after_a_node_completes_names_the_running_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the next node has announced itself, it — not the one that just
    finished — is what the failure is attributed to."""

    class _FailInSecondNode:
        def stream(
            self,
            state: dict[str, Any],
            stream_mode: list[str] | None = None,
            config: dict[str, Any] | None = None,
        ) -> Iterator[tuple[str, Any]]:
            yield ("custom", {"type": "step_start", "node": "validate_sql_query"})
            yield ("updates", {"validate_sql_query": {"path_state": {}}})
            yield ("custom", {"type": "step_start", "node": "execute_sql_query"})
            raise RuntimeError("connector died")

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "app", _FailInSecondNode())

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1]["node"] == "execute_sql_query"
