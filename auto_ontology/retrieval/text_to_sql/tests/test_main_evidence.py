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
    reasoning_llm = object()
    non_reasoning_llm = object()
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_llm_client",
        lambda *a, **k: reasoning_llm,
    )
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_non_reasoning_llm_client",
        lambda *a, **k: non_reasoning_llm,
    )
    module = importlib.import_module("auto_ontology.retrieval.text_to_sql.main")
    monkeypatch.setattr(module, "llm_client", non_reasoning_llm)
    monkeypatch.setattr(module, "reasoning_llm_client", reasoning_llm)
    return module


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
    assert state["messages"][0].content == "How many accounts?"


def test_build_state_uses_non_reasoning_by_default_and_preserves_reasoning_for_sql(
    main_module: ModuleType,
) -> None:
    state = main_module._build_state(
        cast(
            TextToSQLPayload,
            {
                "question": "How many accounts?",
                "data_retriever": object(),
                "connectors": [
                    SimpleNamespace(dialect="sqlite", database_name="db_test")
                ],
            },
        )
    )

    assert state["llm"] is main_module.llm_client
    assert state["reasoning_llm"] is main_module.reasoning_llm_client


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
    assert [message.content for message in state["messages"]] == [
        "How many active accounts?"
    ]


@pytest.mark.parametrize(
    ("payload_value", "expected"),
    [(None, False), (True, True), (False, False)],
)
def test_build_state_propagates_numeric_type_alignment(
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
        payload["numeric_type_alignment"] = payload_value

    state = main_module._build_state(cast(TextToSQLPayload, payload))

    assert state["numeric_type_alignment"] is expected


def _state_with(main_module: ModuleType, **extra: object) -> dict:
    payload = {
        "question": "How many accounts?",
        "data_retriever": object(),
        "connectors": [SimpleNamespace(dialect="sqlite", database_name="db_test")],
        **extra,
    }
    return main_module._build_state(cast(TextToSQLPayload, payload))


def test_build_state_joins_custom_prompts_and_acronyms(
    main_module: ModuleType,
) -> None:
    state = _state_with(
        main_module,
        custom_prompts="Amounts are in USD.",
        acronyms=[{"name": "ARR", "description": "Annual recurring revenue"}],
    )

    assert state["custom_prompts"] == (
        "ORGANIZATION INSTRUCTIONS\n"
        "Amounts are in USD.\n\n"
        "GLOSSARY\n"
        "Definitions the user's organization has registered. Use them to resolve "
        "abbreviations, shortcuts, and internal jargon in the question.\n"
        "- ARR: Annual recurring revenue"
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"custom_prompts": "Amounts are in USD."},
        {"acronyms": [{"name": "ARR", "description": "Annual recurring revenue"}]},
    ],
)
def test_build_state_keeps_either_one_alone(
    main_module: ModuleType, extra: dict
) -> None:
    section = _state_with(main_module, **extra)["custom_prompts"]

    assert ("ORGANIZATION INSTRUCTIONS" in section) == ("custom_prompts" in extra)
    assert ("GLOSSARY" in section) == ("acronyms" in extra)
    assert ("USD" in section) == ("custom_prompts" in extra)
    assert ("ARR" in section) == ("acronyms" in extra)


@pytest.mark.parametrize(
    "extra",
    [{}, {"custom_prompts": "  ", "acronyms": [{"name": "", "description": "x"}]}],
)
def test_build_state_has_no_section_without_either(
    main_module: ModuleType, extra: dict
) -> None:
    assert _state_with(main_module, **extra)["custom_prompts"] == ""


@pytest.mark.parametrize(
    ("extra", "expected"), [({}, False), ({"calculation_only": True}, True)]
)
def test_build_state_propagates_calculation_only(
    main_module: ModuleType, extra: dict, expected: bool
) -> None:
    assert _state_with(main_module, **extra)["calculation_only"] is expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [({}, True), ({"information": True}, True), ({"information": False}, False)],
)
def test_build_state_propagates_information(
    main_module: ModuleType, extra: dict, expected: bool
) -> None:
    assert _state_with(main_module, **extra)["information"] is expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [({}, False), ({"conversation": True}, True), ({"conversation": False}, False)],
)
def test_build_state_propagates_conversation(
    main_module: ModuleType, extra: dict, expected: bool
) -> None:
    assert _state_with(main_module, **extra)["conversation"] is expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [({}, True), ({"calculation": True}, True), ({"calculation": False}, False)],
)
def test_build_state_propagates_calculation(
    main_module: ModuleType, extra: dict, expected: bool
) -> None:
    assert _state_with(main_module, **extra)["calculation"] is expected


def test_connectors_are_required_only_when_there_is_querying_to_do(
    main_module: ModuleType,
) -> None:
    # A database that holds transcripts and no tables has no connector, and
    # refusing it would be refusing the case conversation search exists for.
    with pytest.raises(ValueError, match="missing required 'connectors'"):
        _state_with(main_module, connectors=[])

    state = _state_with(
        main_module, connectors=[], calculation=False, conversation=True
    )

    assert state["calculation"] is False


def test_an_episode_allowed_to_do_nothing_is_refused(main_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="nothing it may do"):
        _state_with(
            main_module, calculation=False, information=False, conversation=False
        )
