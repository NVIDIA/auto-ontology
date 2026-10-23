# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest

from auto_ontology.retrieval.text_to_sql.agents import column_ambiguity_resolver
from auto_ontology.retrieval.text_to_sql.agents.column_ambiguity_resolver import (
    ColumnAmbiguityResolverAgent,
)
from auto_ontology.retrieval.text_to_sql.models import (
    ColumnAmbiguityResolutionModel,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState


def _hit(
    hit_id: str,
    *,
    entity_type: str = "field",
    original_text: str = "location",
    meaning: str = "record location",
    score: float = 0.2,
) -> dict[str, Any]:
    return {
        "id": hit_id,
        "text": f"semantic text for {hit_id}",
        "score": score,
        "query_entity_spec": {
            "type": entity_type,
            "original_text": original_text,
            "meaning": meaning,
        },
    }


def _state(
    hits: list[dict],
    *,
    evidence: str = "",
    subject: str = "record",
) -> AgentState:
    return cast(
        AgentState,
        {
            "initial_question": "Which location belongs to the requested record?",
            "evidence": evidence,
            "llm": object(),
            "path_state": {
                "normalized_question": "Which location belongs to the record?",
                "subject": subject,
                "target_db": "db-a",
                "retrieved_subject_term": {"id": "subject-term"},
                "retrieved_column_attributes": hits,
            },
        },
    )


@pytest.fixture(autouse=True)
def _semantic_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "fetch_attr_column_contexts",
        lambda attr_ids, *, database_name: {
            attr_id: {
                "attr_name": attr_id,
                "attr_description": f"description for {attr_id}",
                "col_name": attr_id,
                "table_id": "subject-table" if "direct" in attr_id else "related-table",
                "table_name": "records" if "direct" in attr_id else "record_details",
                "schema_name": "main",
                "database_name": database_name,
                "term_name": "Record",
                "datatype": "text",
                "is_primary_key": False,
            }
            for attr_id in attr_ids
        },
    )
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "fetch_term_synonyms",
        lambda _attr_ids: {"Record": ["entry"]},
    )
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "fetch_column_attribute_columns_map",
        lambda attr_ids: {
            attr_id: {
                "primary_column": {
                    "table_name": "records",
                    "column_name": attr_id,
                },
                "referenced_columns": [],
            }
            for attr_id in attr_ids
        },
    )
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "fetch_semantic_fk_join_groups",
        lambda _database_name, *, attribute_ids: [
            {
                "target_attribute_id": attribute_ids[0],
                "source_table_name": "record_details",
                "source_column_name": "record_id",
                "target_table_name": "records",
                "target_column_name": "id",
                "use_in_join_paths": True,
            }
        ],
    )
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "fetch_term_table_pairs",
        lambda **_kwargs: [
            {
                "term_id": "subject-term",
                "table_id": "subject-table",
                "path": "REPRESENTS",
            }
        ],
    )


def test_batches_groups_and_uses_reasoning_llm_with_evidence_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        _hit("direct-location", original_text="location"),
        _hit("related-location", original_text="location", score=0.1),
        _hit(
            "direct-category",
            original_text="category",
            meaning="record category",
        ),
        _hit(
            "related-category",
            original_text="category",
            meaning="record category",
            score=0.1,
        ),
        _hit(
            "value-suspect",
            entity_type="exact_value",
            original_text="requested text",
            meaning="record",
        ),
    ]
    state = _state(hits, evidence="The location field uses the direct attribute.")
    reasoning_llm = state["llm"]
    captured: dict[str, Any] = {"calls": 0}

    def fake_invoke(llm: object, messages: list[Any], schema: type) -> Any:
        captured["calls"] += 1
        captured["llm"] = llm
        captured["prompt"] = messages[0].content
        assert schema is ColumnAmbiguityResolutionModel
        return schema(
            selections=[
                {
                    "group_id": "group_1",
                    "selected_attribute_id": "direct-location",
                    "reasoning": "The evidence identifies the direct field.",
                },
                {
                    "group_id": "group_2",
                    "selected_attribute_id": "direct-category",
                    "reasoning": "The question targets the record itself.",
                },
            ]
        )

    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        fake_invoke,
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert captured["calls"] == 1
    assert captured["llm"] is reasoning_llm
    assert "EVIDENCE IS PRIMARY" in captured["prompt"]
    assert "The location field uses the direct attribute." in captured["prompt"]
    assert "subject_semantic_links: REPRESENTS" in captured["prompt"]
    assert "default_for_joins=True" in captured["prompt"]
    assert [
        hit["id"] for hit in result["path_state"]["retrieved_column_attributes"]
    ] == ["direct-location", "direct-category", "value-suspect"]


def test_question_and_semantic_layer_are_used_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(
        [
            _hit("direct-location"),
            _hit("related-location", score=0.1),
        ]
    )
    captured: dict[str, str] = {}

    def fake_invoke(_llm: object, messages: list[Any], schema: type) -> Any:
        captured["prompt"] = messages[0].content
        return schema(
            selections=[
                {
                    "group_id": "group_1",
                    "selected_attribute_id": "direct-location",
                    "reasoning": "The question and subject ownership agree.",
                }
            ]
        )

    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        fake_invoke,
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert "Evidence:\n(none)" in captured["prompt"]
    assert "Original question:" in captured["prompt"]
    assert "physical_column: main.records.direct-location" in captured["prompt"]
    assert [
        hit["id"] for hit in result["path_state"]["retrieved_column_attributes"]
    ] == ["direct-location"]


@pytest.mark.parametrize(
    "selection",
    [
        {
            "group_id": "group_1",
            "selected_attribute_id": None,
            "reasoning": "Unresolved.",
        },
        {
            "group_id": "group_1",
            "selected_attribute_id": "unknown-id",
            "reasoning": "Invalid.",
        },
    ],
)
def test_unresolved_or_invalid_selection_retains_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, Any],
) -> None:
    hits = [_hit("direct-location"), _hit("related-location")]
    state = _state(hits)
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        lambda _llm, _messages, schema: schema(selections=[selection]),
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert result["path_state"]["retrieved_column_attributes"] == hits


def test_llm_failure_retains_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [_hit("direct-location"), _hit("related-location")]
    state = _state(hits)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        fail,
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert result["path_state"]["retrieved_column_attributes"] == hits


def test_field_selection_preserves_shared_exact_value_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _hit("direct-location")
    shared = _hit("related-location")
    shared["query_entity_specs"] = [
        shared["query_entity_spec"],
        {
            "type": "exact_value",
            "original_text": "requested text",
            "meaning": "record",
        },
    ]
    state = _state([selected, shared])
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        lambda _llm, _messages, schema: schema(
            selections=[
                {
                    "group_id": "group_1",
                    "selected_attribute_id": "direct-location",
                    "reasoning": "The field meaning selects the direct attribute.",
                }
            ]
        ),
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert result["path_state"]["retrieved_column_attributes"] == [selected, shared]


def test_exact_value_selection_prioritizes_candidate_for_database_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailing_city = _hit(
        "mailing-city",
        entity_type="exact_value",
        original_text="requested place",
        meaning="city",
        score=0.1,
    )
    physical_city = _hit(
        "physical-city",
        entity_type="exact_value",
        original_text="requested place",
        meaning="city",
        score=0.2,
    )
    state = _state([mailing_city, physical_city])
    captured: dict[str, str] = {}

    def fake_invoke(_llm: object, messages: list[Any], schema: type) -> Any:
        captured["prompt"] = messages[0].content
        return schema(
            selections=[
                {
                    "group_id": "group_1",
                    "selected_attribute_id": "physical-city",
                    "reasoning": "The question asks for the physical location.",
                }
            ]
        )

    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        fake_invoke,
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert "entity_type: exact_value" in captured["prompt"]
    assert [
        hit["id"] for hit in result["path_state"]["retrieved_column_attributes"]
    ] == ["physical-city", "mailing-city"]


def test_single_field_candidate_skips_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        _hit("direct-location"),
        _hit(
            "value-suspect",
            entity_type="exact_value",
            original_text="requested text",
            meaning="record",
        ),
    ]
    state = _state(hits)
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: pytest.fail("LLM should not be called"),
    )

    result = ColumnAmbiguityResolverAgent().execute(state)

    assert result["path_state"]["retrieved_column_attributes"] == hits


def test_empty_retrieval_is_a_valid_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state([])
    monkeypatch.setattr(
        column_ambiguity_resolver,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: pytest.fail("LLM should not be called"),
    )

    assert ColumnAmbiguityResolverAgent().validate_input(state) is True
    assert ColumnAmbiguityResolverAgent().execute(state) == {
        "path_state": state["path_state"]
    }
