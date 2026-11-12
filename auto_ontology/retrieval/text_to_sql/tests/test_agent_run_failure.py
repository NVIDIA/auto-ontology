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

from auto_ontology.utils import llm_invoke  # noqa: E402

llm_invoke._API_KEY = "test-key"
llm_invoke._BASE_URL = "https://example.test/v1"
llm_invoke._MODEL_NAME = "openai/test-model"

from auto_ontology.retrieval.text_to_sql import main  # noqa: E402


def _fail_after_generation(_state: Any, _steps: Any) -> Iterator[tuple[str, str, Any]]:
    """Pipeline double that fails after the agent has produced SQL."""
    yield ("start", "react_sql_agent", None)
    yield (
        "end",
        "react_sql_agent",
        {
            "path_state": {
                "sql_generation_result": SimpleNamespace(
                    sql_code="SELECT 1",
                    response="ok",
                    thought="generated before downstream failure",
                )
            }
        },
    )
    raise KeyError("valid_sql")


def test_stream_error_preserves_partial_generated_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "run_steps", _fail_after_generation)

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1] == {
        "type": "error",
        "message": "Agent failed after react_sql_agent: 'valid_sql'",
        "node": "react_sql_agent",
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
    monkeypatch.setattr(main, "run_steps", _fail_after_generation)

    with pytest.raises(main.AgentRunError) as caught:
        main.get_agent_response({"question": "q"})

    assert caught.value.node == "react_sql_agent"
    assert caught.value.partial_answer["sql_code"] == "SELECT 1"


def _fail_inside_first_node(_state: Any, _steps: Any) -> Iterator[tuple[str, str, Any]]:
    """Pipeline double whose first node raises before it ever returns."""
    yield ("start", "react_sql_agent", None)
    raise TimeoutError("retriever unavailable")


def test_error_inside_a_step_names_that_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node that raises produces no update, so completions alone can't name
    it — the failure would be attributed to whichever node last succeeded, or
    to ``start`` when it is the first one. The start step is what makes the
    report point at the node that was actually running.
    """
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "run_steps", _fail_inside_first_node)

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1]["node"] == "react_sql_agent"
    assert events[-1]["error_type"] == "TimeoutError"
    assert "after start" not in events[-1]["message"]


def test_error_after_a_node_completes_names_the_running_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the next node has announced itself, it — not the one that just
    finished — is what the failure is attributed to."""

    def fail_in_second_node(_state: Any, _steps: Any) -> Iterator[tuple[str, str, Any]]:
        yield ("start", "react_sql_agent", None)
        yield ("end", "react_sql_agent", {"path_state": {}})
        yield ("start", "format_and_respond", None)
        raise RuntimeError("connector died")

    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "run_steps", fail_in_second_node)

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1]["node"] == "format_and_respond"
