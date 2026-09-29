# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``_pump``'s handling of the live ``sql`` event.

The event is what lets the UI show the query while it runs; the pump also
remembers the last one so the failure paths — which produce an answer with no
``sql_code`` at all — still persist the query the user watched.
"""

import threading
import uuid
from typing import Any, Iterator

import pytest

from auto_ontology.server.chat import router


@pytest.fixture(name="persisted")
def persisted_fixture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture assistant-turn writes instead of hitting Postgres."""

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        router,
        "persist_assistant_result",
        lambda **kwargs: calls.append(kwargs),
    )
    # The chart step needs an executed result and an LLM; neither is under
    # test here, and `_release` would hand a fake worker back to the pool.
    monkeypatch.setattr(router, "_build_charts_event", lambda *_args: None)
    monkeypatch.setattr(router, "_release", lambda _slot: None)
    return calls


def _run_pump(events: list[dict[str, Any]]) -> router._Slot:
    """Drain ``events`` through ``_pump`` and return the slot it filled."""

    class _FakeWorker:
        def events(self) -> Iterator[dict[str, Any]]:
            return iter(events)

    slot = router._Slot(
        key="user:conv",
        worker=_FakeWorker(),  # type: ignore[arg-type]
        user_id="user",
        conversation_id=uuid.uuid4(),
        analytics_id="analytics",
        question="how many orders?",
        client_alive=threading.Event(),
        cancelled=threading.Event(),
        released=threading.Event(),
    )
    router._pump(slot)
    return slot


def test_sql_event_reaches_the_client_buffer(
    persisted: list[dict[str, Any]],
) -> None:
    """It is replayed like any other event, so a reattached tab sees it too."""

    slot = _run_pump(
        [
            {"type": "sql", "node": "construct_sql_from_candidates", "sql": "SELECT 1"},
            {"type": "result", "answer": {"response": "1", "sql_code": "SELECT 1"}},
        ]
    )

    assert {
        "type": "sql",
        "node": "construct_sql_from_candidates",
        "sql": "SELECT 1",
    } in slot.buffer


def test_error_persists_the_last_streamed_sql(
    persisted: list[dict[str, Any]],
) -> None:
    """An ``error`` event carries no SQL of its own — without the fallback the
    query the user just watched fail would vanish on reload."""

    _run_pump(
        [
            {"type": "sql", "node": "construct_sql_from_candidates", "sql": "SELECT a"},
            {"type": "sql", "node": "reconstruct_sql", "sql": "SELECT b"},
            {"type": "error", "message": "Agent failed: boom"},
        ]
    )

    assert persisted[0]["sql_code"] == "SELECT b"
    assert persisted[0]["response"] == "Agent failed: boom"


def test_answer_without_sql_falls_back_to_the_last_attempt(
    persisted: list[dict[str, Any]],
) -> None:
    """``unconstructable_sql_response`` answers prose only, after several
    attempts the user saw."""

    _run_pump(
        [
            {"type": "sql", "node": "construct_sql_from_candidates", "sql": "SELECT a"},
            {"type": "result", "answer": {"response": "SQL can't be constructed."}},
        ]
    )

    assert persisted[0]["sql_code"] == "SELECT a"


def test_answer_sql_wins_over_the_streamed_one(
    persisted: list[dict[str, Any]],
) -> None:
    """A successful run persists exactly what the answer reports."""

    _run_pump(
        [
            {"type": "sql", "node": "reconstruct_sql", "sql": "SELECT stale"},
            {
                "type": "result",
                "answer": {"response": "ok", "sql_code": "SELECT final"},
            },
        ]
    )

    assert persisted[0]["sql_code"] == "SELECT final"


def test_no_sql_event_leaves_the_turn_unchanged(
    persisted: list[dict[str, Any]],
) -> None:
    """A run that never produced SQL persists none — not an empty string."""

    _run_pump([{"type": "error", "message": "boom"}])

    assert persisted[0]["sql_code"] is None
