# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run reports where its time went, so a slow question can be explained."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Any

import pytest

from auto_ontology.utils import llm_invoke


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: None)
    monkeypatch.setattr(
        llm_invoke, "get_non_reasoning_llm_client", lambda **_kwargs: None
    )
    return importlib.import_module("auto_ontology.retrieval.text_to_sql.main")


def _clock(monkeypatch: pytest.MonkeyPatch, main_module: ModuleType) -> list[float]:
    """Replace the timer's clock with one the test advances by hand."""
    now = [100.0]
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: now[0])
    return now


def test_time_between_completions_belongs_to_the_later_node(
    main_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _clock(monkeypatch, main_module)
    timings = main_module._NodeTimings()

    for node_name, seconds in [
        ("_entry_router", 0.25),
        ("question_intent", 2.0),
        ("construct_sql_from_candidates", 18.0),
        ("validate_sql_query", 3.0),
        ("reconstruct_sql", 9.0),
        ("validate_sql_query", 4.5),
    ]:
        now[0] += seconds
        timings.node_finished(node_name)

    assert timings.summary() == (
        "total 36.8s | construct_sql_from_candidates 18.0s | reconstruct_sql 9.0s"
        " | validate_sql_query 7.5s (x2) | question_intent 2.0s"
    )


def test_a_streamed_run_logs_its_timing(
    main_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = _clock(monkeypatch, main_module)

    class _App:
        def stream(self, state: dict, **_kwargs: Any) -> Any:
            for node_name, seconds in [
                ("question_intent", 1.5),
                ("format_and_respond", 0.5),
            ]:
                now[0] += seconds
                yield (
                    "updates",
                    {node_name: {"path_state": {"final_response": {"response": "ok"}}}},
                )

    monkeypatch.setattr(main_module, "app", _App())
    monkeypatch.setattr(main_module, "_build_state", lambda payload: {"path_state": {}})

    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        events = list(main_module.stream_agent_response({"question": "q"}))

    assert events[-1] == {"type": "result", "answer": {"response": "ok"}}
    assert (
        "Text-to-SQL timing: total 2.0s | question_intent 1.5s | format_and_respond 0.5s"
        in caplog.text
    )
