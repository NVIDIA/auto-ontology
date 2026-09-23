# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace
from types import ModuleType
from typing import cast

import pytest

from auto_ontology.retrieval.text_to_sql.state import TextToSQLPayload


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_llm_client", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_non_reasoning_llm_client",
        lambda *a, **k: None,
    )
    return importlib.import_module("auto_ontology.retrieval.text_to_sql.main")


@pytest.mark.parametrize("payload_evidence", [None, ""])
def test_build_state_treats_absent_or_empty_evidence_as_none(
    main_module: ModuleType,
    payload_evidence: str | None,
) -> None:
    payload = {
        "question": "How many accounts?",
        "data_retriever": object(),
        "connectors": [SimpleNamespace(dialect="sqlite", database_name="db_test")],
    }
    if payload_evidence is not None:
        payload["evidence"] = payload_evidence

    state = main_module._build_state(cast(TextToSQLPayload, payload))

    assert state["evidence"] == ""
    assert state["initial_question"] == "How many accounts?"
    assert state["messages"][1].content == "How many accounts?"


def test_build_state_keeps_evidence_separate_from_every_question_value(
    main_module: ModuleType,
) -> None:
    state = main_module._build_state(
        cast(
            TextToSQLPayload,
            {
                "question": "How many accounts?",
                "processing_question": "How many active accounts?",
                "evidence": "active refers to accounts.status_code",
                "data_retriever": object(),
                "connectors": [
                    SimpleNamespace(dialect="sqlite", database_name="db_test")
                ],
            },
        )
    )

    assert state["evidence"] == "active refers to accounts.status_code"
    assert state["initial_question"] == "How many accounts?"
    assert state["path_state"]["processing_question"] == "How many active accounts?"
    assert state["messages"][1].content == "How many active accounts?"


@pytest.mark.parametrize(
    ("payload_value", "expected"),
    [(None, False), (True, True), (False, False)],
)
def test_build_state_propagates_shorten_answer(
    main_module: ModuleType,
    payload_value: bool | None,
    expected: bool,
) -> None:
    payload = {
        "question": "How many accounts?",
        "data_retriever": object(),
        "connectors": [SimpleNamespace(dialect="sqlite", database_name="db_test")],
    }
    if payload_value is not None:
        payload["shorten_answer"] = payload_value

    state = main_module._build_state(cast(TextToSQLPayload, payload))

    assert state["shorten_answer"] is expected
