# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from auto_ontology.retrieval.text_to_sql.agents import semantic_context_extraction
from auto_ontology.retrieval.text_to_sql.agents.semantic_context_extraction import (
    ExtractedSemanticConstraint,
    SemanticContextExtractionAgent,
    SemanticContextExtractionResult,
    build_semantic_sources,
    format_semantic_constraints,
    validate_constraints,
)


def _state() -> dict:
    return {
        "llm": object(),
        "reasoning_llm": object(),
        "initial_question": "List active account labels.",
        "evidence": "Active accounts use status = 'A'.",
        "value_anchors": [
            {
                "id": "anchor-1",
                "phrase": "active",
                "kind": "value",
                "tbl": "accounts",
                "col": "status",
                "stored_value": "A",
                "n_rows": 12,
            }
        ],
        "path_state": {
            "normalized_question": "List active account labels.",
            "relevant_tables": [
                {
                    "id": "table-1",
                    "database_name": "db",
                    "schema_name": "public",
                    "name": "accounts",
                    "description": "One row per account.",
                    "columns": [
                        {
                            "id": "column-1",
                            "name": "status",
                            "data_type": "text",
                            "description": "Lifecycle status code.",
                            "usage_evidence": "Use this field for account state.",
                            "constraints": "Do not compare display labels here.",
                            "format": "single uppercase character",
                            "sample_values": ["A", "I"],
                            "is_unique": False,
                            "is_nullable": False,
                        }
                    ],
                }
            ],
            "retrieved_column_attributes": [
                {
                    "id": "attribute-1",
                    "name": "Account state",
                    "term_name": "Account",
                    "table_name": "accounts",
                    "source_column": "status",
                    "text": (
                        "ColumnAttribute: Account state of Term Account. "
                        "Use this field for account state. Do not compare display "
                        "labels here. Values include A and I."
                    ),
                },
                {
                    "id": "attribute-without-text",
                    "name": "Ignored structured description",
                    "description": "This must not become a source.",
                },
            ],
            "term_synonyms": {"account": ["buyer", "customer"]},
            "custom_analyses": [],
            "sql_attributes": [],
            "attribute_join_paths": [],
        },
    }


def test_build_sources_uses_only_column_attribute_text() -> None:
    sources = build_semantic_sources(_state())
    rendered = "\n".join(source["content"] for source in sources.values())

    assert len(sources) == 1
    assert next(iter(sources)).startswith("column_attribute:")
    assert "Use this field for account state." in rendered
    assert "Do not compare display labels here." in rendered
    assert "Values include A and I." in rendered
    assert "One row per account." not in rendered
    assert "single uppercase character" not in rendered
    assert "buyer" not in rendered
    assert "stored value" not in rendered
    assert "This must not become a source." not in rendered


def test_validation_drops_unknown_sources_and_duplicate_constraints() -> None:
    sources = {"column:1": {"label": "public.accounts.status", "content": "status"}}
    result = SemanticContextExtractionResult(
        constraints=[
            ExtractedSemanticConstraint(
                kind="field_mapping",
                instruction="Use status for lifecycle state.",
                source_ids=["column:1"],
            ),
            ExtractedSemanticConstraint(
                kind="field_mapping",
                instruction="Use status for lifecycle state!",
                source_ids=["column:1"],
            ),
            ExtractedSemanticConstraint(
                kind="required_usage",
                instruction="Use a missing field.",
                source_ids=["column:missing"],
            ),
        ]
    )

    constraints = validate_constraints(result, sources)

    assert len(constraints) == 1
    context = format_semantic_constraints(constraints, sources)
    assert "public.accounts.status" in context
    assert "Authoritative Evidence overrides this context" in context


def test_agent_uses_reasoning_model_and_keeps_only_grounded_context(
    monkeypatch,
) -> None:
    state = _state()
    reasoning_llm = state["reasoning_llm"]
    captured: dict = {}

    def fake_invoke(llm, messages, schema):
        captured["llm"] = llm
        captured["messages"] = messages
        assert schema is SemanticContextExtractionResult
        source_id = next(
            source_id
            for source_id in build_semantic_sources(state)
            if source_id.startswith("column_attribute:")
        )
        return SemanticContextExtractionResult(
            reasoning="The status metadata resolves the requested state field.",
            constraints=[
                ExtractedSemanticConstraint(
                    kind="field_mapping",
                    instruction="Use the account status field for lifecycle state.",
                    source_ids=[source_id],
                ),
                ExtractedSemanticConstraint(
                    kind="required_usage",
                    instruction="Invent an unsupported requirement.",
                    source_ids=["column:not-present"],
                ),
            ],
        )

    monkeypatch.setattr(
        semantic_context_extraction,
        "safe_invoke_with_structured_output",
        fake_invoke,
    )

    result = SemanticContextExtractionAgent().execute(state)
    path_state = result["path_state"]

    assert captured["llm"] is reasoning_llm
    assert "Values mentioned as examples" in (captured["messages"][0].content)
    assert state["evidence"] in captured["messages"][1].content
    assert len(path_state["semantic_constraints"]) == 1
    assert "account status field" in path_state["semantic_constraints_context"]
    assert path_state["thoughts_log"][-1]["node"] == "extract_semantic_context"


def test_agent_fails_open_when_extraction_fails(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(
        semantic_context_extraction,
        "safe_invoke_with_structured_output",
        fail,
    )

    result = SemanticContextExtractionAgent().execute(_state())

    assert result["path_state"]["semantic_constraints"] == []
    assert result["path_state"]["semantic_constraints_context"] == ""
