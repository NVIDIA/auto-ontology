# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prewarmed chat worker's request loop."""

from typing import Any

from pytest import MonkeyPatch

from auto_ontology.server.chat import worker as worker_module


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


def _run_loop(
    monkeypatch: MonkeyPatch,
    connector_batches: list[list[str]],
    *,
    evidence: str | None = None,
    out_q: _FakeQueue | None = None,
    prewarm: Any = lambda *retrievers: None,
) -> dict:
    """Drive one ASK through ``_worker_loop`` and return the agent payload it built.

    ``connector_batches`` is what successive ``get_connectors()`` calls return, so a
    worker that booted before the store was reachable can be simulated with ``[[], [...]]``.
    """
    # Importing the agent module builds LLM clients at import time and lets an
    # EnvironmentError escape when no key is configured. Stub the factories before that
    # import is triggered below; the clients are never used, since stream_agent_response
    # is replaced too. (Patching the env instead is not enough — model config resolution
    # is cached, so an earlier import in the same session wins.)
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_llm_client", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_non_reasoning_llm_client",
        lambda *a, **k: None,
    )

    remaining = list(connector_batches)

    def fake_get_connectors() -> list[str]:
        return remaining.pop(0) if remaining else []

    captured: dict = {}

    def fake_stream(payload: dict) -> Any:
        captured.update(payload)
        return iter(())

    monkeypatch.setattr("auto_ontology.connectors.get_connectors", fake_get_connectors)
    monkeypatch.setattr(
        "auto_ontology.utils.get_data_objects_retriever", lambda: "data"
    )
    monkeypatch.setattr(
        "auto_ontology.utils.get_semantic_objects_retriever", lambda: "semantic"
    )
    monkeypatch.setattr(
        "auto_ontology.server.chat.settings_dal.fetch_acronyms", lambda: []
    )
    monkeypatch.setattr(
        "auto_ontology.server.chat.settings_dal.fetch_custom_prompts", lambda: ""
    )
    monkeypatch.setattr(
        "auto_ontology.retrieval.text_to_sql.main.stream_agent_response", fake_stream
    )
    # Warming opens real connections; these tests are about the request loop.
    monkeypatch.setattr(worker_module, "_prewarm", prewarm)

    in_q = _FakeQueue([(worker_module._MSG_ASK, ("q", None, None, None, [], evidence))])
    worker_module._worker_loop(in_q, out_q or _FakeQueue())
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
    """The healthy path must not re-resolve — that would query the store per question."""
    payload = _run_loop(monkeypatch, connector_batches=[["boot-connector"]])

    assert payload["connectors"] == ["boot-connector"]
    assert payload["question"] == "q"
    assert payload["evidence"] == ""


def test_worker_forwards_evidence_without_changing_question(
    monkeypatch: MonkeyPatch,
) -> None:
    payload = _run_loop(
        monkeypatch,
        connector_batches=[["boot-connector"]],
        evidence="status means accounts.status",
    )

    assert payload["question"] == "q"
    assert payload["processing_question"] == "q"
    assert payload["evidence"] == "status means accounts.status"


class _Vdb:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.warmed = 0

    def warm(self) -> bool:
        self.warmed += 1
        if self.error is not None:
            raise self.error
        return True


class _Retriever:
    def __init__(self, vdb: _Vdb) -> None:
        self.vdb = vdb


def _stub_prewarm_targets(
    monkeypatch: MonkeyPatch, *, store_error: Exception | None = None
) -> list[str]:
    calls: list[str] = []

    class _Store:
        def query_read(self, statement: str) -> list:
            calls.append(statement)
            if store_error is not None:
                raise store_error
            return []

    def empty_catalog() -> dict:
        calls.append("schema map")
        raise KeyError("database_name")

    monkeypatch.setattr("auto_ontology.dal.session.store", lambda: _Store())
    monkeypatch.setattr(
        "auto_ontology.retrieval.data_access.graph_schemas.get_catalog_schemas",
        empty_catalog,
    )
    return calls


def test_prewarm_opens_every_store_and_survives_a_failing_one(
    monkeypatch: MonkeyPatch,
) -> None:
    """One store being down must not leave the others cold, or stop the worker."""
    calls = _stub_prewarm_targets(monkeypatch)
    broken = _Vdb(RuntimeError("connection refused"))
    healthy = _Vdb()

    worker_module._prewarm(_Retriever(broken), _Retriever(healthy))

    assert calls == ["SELECT 1", "schema map"]
    assert (broken.warmed, healthy.warmed) == (1, 1)


def test_prewarm_stops_when_the_database_is_unreachable(
    monkeypatch: MonkeyPatch,
) -> None:
    """The vector stores share that database and connect with no timeout."""
    calls = _stub_prewarm_targets(
        monkeypatch, store_error=RuntimeError("connection timeout expired")
    )
    vdb = _Vdb()

    worker_module._prewarm(_Retriever(vdb))

    assert calls == ["SELECT 1"]
    assert vdb.warmed == 0


def test_worker_reports_ready_before_it_warms(monkeypatch: MonkeyPatch) -> None:
    """The pool waits on READY with no liveness check; warming must not precede it."""
    out_q = _FakeQueue()
    ready_when_warming: list[bool] = []

    def record(*_retrievers: object) -> None:
        ready_when_warming.append((worker_module._TAG_READY, None) in out_q.puts)

    _run_loop(
        monkeypatch,
        connector_batches=[["boot-connector"]],
        out_q=out_q,
        prewarm=record,
    )

    assert ready_when_warming == [True]
