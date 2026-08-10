# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prewarmed chat worker's request loop."""

from typing import Any

from pytest import MonkeyPatch

from gsf.server.chat import worker as worker_module


class _FakeQueue:
    """Queue stub: yields queued messages, then ends the loop like a closed pipe."""

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items = list(items or [])
        self.puts: list[Any] = []

    def get(self) -> Any:
        if self._items:
            return self._items.pop(0)
        # The real loop returns when the pipe closes; this ends the test cleanly.
        raise EOFError

    def put(self, item: Any) -> None:
        self.puts.append(item)


def _run_loop(monkeypatch: MonkeyPatch, connector_batches: list[list[str]]) -> dict:
    """Drive one ASK through ``_worker_loop`` and return the agent payload it built.

    ``connector_batches`` is what successive ``get_connectors()`` calls return, so a
    worker that booted before Neo4j was reachable can be simulated with ``[[], [...]]``.
    """
    # Importing the agent module builds LLM clients at import time and lets an
    # EnvironmentError escape when no key is configured. Stub the factories before that
    # import is triggered below; the clients are never used, since stream_agent_response
    # is replaced too. (Patching the env instead is not enough — model config resolution
    # is cached, so an earlier import in the same session wins.)
    monkeypatch.setattr("gsf.utils.llm_invoke.get_llm_client", lambda *a, **k: None)
    monkeypatch.setattr(
        "gsf.utils.llm_invoke.get_non_reasoning_llm_client", lambda *a, **k: None
    )

    remaining = list(connector_batches)

    def fake_get_connectors() -> list[str]:
        return remaining.pop(0) if remaining else []

    captured: dict = {}

    def fake_stream(payload: dict) -> Any:
        captured.update(payload)
        return iter(())

    monkeypatch.setattr("gsf.connectors.get_connectors", fake_get_connectors)
    monkeypatch.setattr("gsf.utils.get_data_objects_retriever", lambda: "data")
    monkeypatch.setattr("gsf.utils.get_semantic_objects_retriever", lambda: "semantic")
    monkeypatch.setattr("gsf.server.chat.settings_dal.fetch_acronyms", lambda: [])
    monkeypatch.setattr("gsf.server.chat.settings_dal.fetch_custom_prompts", lambda: "")
    monkeypatch.setattr(
        "gsf.retrieval.text_to_sql.main.stream_agent_response", fake_stream
    )

    in_q = _FakeQueue([(worker_module._MSG_ASK, ("q", None, None, None))])
    worker_module._worker_loop(in_q, _FakeQueue())
    return captured


def test_worker_reresolves_connectors_before_building_the_payload(
    monkeypatch: MonkeyPatch,
) -> None:
    """A worker that booted with no connections must heal on the next question.

    Regression: the payload was built from a reference captured *before* the lazy
    re-resolve, so rebinding the local did not reach it and every question kept
    failing with "missing required 'connectors'" until the pod restarted.
    """
    payload = _run_loop(monkeypatch, connector_batches=[[], ["healed-connector"]])

    assert payload["connectors"] == ["healed-connector"]


def test_worker_reuses_the_boot_connectors_when_they_are_present(
    monkeypatch: MonkeyPatch,
) -> None:
    """The healthy path must not re-resolve — that would query Neo4j per question."""
    payload = _run_loop(monkeypatch, connector_batches=[["boot-connector"]])

    assert payload["connectors"] == ["boot-connector"]
    assert payload["question"] == "q"
