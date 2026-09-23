# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `prediction` API parameter forces the branch instead of classifying."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import MonkeyPatch

from auto_ontology.retrieval.text_to_sql.agents import prediction_classification
from auto_ontology.retrieval.text_to_sql.agents.prediction_classification import (
    PredictionClassificationAgent,
)


def _state(override: Any) -> dict:
    state: dict = {"initial_question": "how many orders next month?", "llm": object()}
    if override is not _MISSING:
        state["prediction_override"] = override
    return state


_MISSING = object()


@pytest.fixture
def _no_llm(monkeypatch: MonkeyPatch) -> list:
    """Record any classifier call so tests can assert it was skipped."""
    calls: list = []

    def fake_invoke(*args: object, **kwargs: object) -> None:
        calls.append(args)
        return None

    monkeypatch.setattr(
        prediction_classification, "invoke_with_structured_output", fake_invoke
    )
    return calls


def test_true_forces_prediction_without_classifying(_no_llm: list) -> None:
    result = PredictionClassificationAgent().execute(_state(True))

    assert result == {"decision": "prediction"}
    assert _no_llm == [], "classifier must not run when the caller forced the branch"


def test_false_forces_sql_without_classifying(_no_llm: list) -> None:
    result = PredictionClassificationAgent().execute(_state(False))

    assert result == {"decision": "sql"}
    assert _no_llm == []


def test_none_falls_back_to_classification(_no_llm: list) -> None:
    """An explicit null behaves exactly like today: classify the question."""
    PredictionClassificationAgent().execute(_state(None))

    assert len(_no_llm) == 1


def test_absent_falls_back_to_classification(_no_llm: list) -> None:
    """A caller that omits the field entirely also classifies."""
    PredictionClassificationAgent().execute(_state(_MISSING))

    assert len(_no_llm) == 1


def test_chat_request_defaults_to_none() -> None:
    from auto_ontology.server.chat.helpers import ChatRequest

    request = ChatRequest(question="hi")
    assert request.prediction is None
    assert request.evidence is None
    assert ChatRequest(question="hi", prediction=True).prediction is True
    assert ChatRequest(question="hi", prediction=False).prediction is False
    assert (
        ChatRequest(question="hi", evidence="authoritative").evidence == "authoritative"
    )


def test_chat_request_validates_conversation_uuid() -> None:
    from pydantic import ValidationError

    from auto_ontology.server.chat.helpers import ChatRequest

    request = ChatRequest(
        question="hi",
        conversation_id="d61d8aa3-d496-4fa7-97ce-4f831c162e7f",
    )
    assert str(request.conversation_id) == "d61d8aa3-d496-4fa7-97ce-4f831c162e7f"

    with pytest.raises(ValidationError):
        ChatRequest(question="hi", conversation_id="not-a-uuid")


def test_worker_submit_passes_prediction_through() -> None:
    """The worker protocol carries the flag alongside the question.

    The payload is positional, so its shape is a contract between ``submit`` and the
    worker loop's unpacking — asserting the whole tuple catches a field being added on
    one side only.
    """
    from auto_ontology.server.chat.worker import PrewarmedWorker

    sent: list = []
    worker = PrewarmedWorker.__new__(PrewarmedWorker)
    worker._in_q = type("Q", (), {"put": lambda _self, item: sent.append(item)})()

    worker.submit("q", prediction=True)
    worker.submit("q2")

    assert sent[0][1] == ("q", True, None, None, [], None)
    assert sent[1][1] == ("q2", None, None, None, [], None)


def test_worker_submit_accepts_every_field_by_keyword() -> None:
    """The router passes optional fields by keyword; passing one positionally would collide
    with ``prediction`` and raise 'got multiple values for argument'."""
    from auto_ontology.server.chat.worker import PrewarmedWorker

    sent: list = []
    worker = PrewarmedWorker.__new__(PrewarmedWorker)
    worker._in_q = type("Q", (), {"put": lambda _self, item: sent.append(item)})()

    worker.submit(
        "q",
        prediction=False,
        target_db="analytics",
        subject_token="jwt",
        evidence="authoritative",
    )

    assert sent[0][1] == (
        "q",
        False,
        "analytics",
        "jwt",
        [],
        "authoritative",
    )


def test_worker_payload_unpacks_as_the_loop_expects() -> None:
    """The enqueued tuple must match the worker loop's unpacking arity."""
    from auto_ontology.server.chat.worker import PrewarmedWorker

    sent: list = []
    worker = PrewarmedWorker.__new__(PrewarmedWorker)
    worker._in_q = type("Q", (), {"put": lambda _self, item: sent.append(item)})()

    worker.submit("q", prediction=True, target_db="db", subject_token="tok")

    question, prediction, target_db, subject_token, history, evidence = sent[0][1]
    assert (question, prediction, target_db, subject_token) == ("q", True, "db", "tok")
    assert history == []
    assert evidence is None
