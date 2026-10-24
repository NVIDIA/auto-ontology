# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared table/column prompt formatting in sql_from_semantic.py."""

from __future__ import annotations

from auto_ontology.retrieval.text_to_sql.agents import sql_from_semantic
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
    format_calculation_sql_template,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_important_columns_for_prompt,
    format_tables_for_prompt,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    DIRECT_ANSWER_FIELD_RULES,
    create_sql_from_candidates_prompt,
    create_sql_user_prompt,
    format_authoritative_evidence,
    format_projection_rules,
)
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel


def test_prompt_renders_date_format_when_present() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "matches",
                "database_name": "cricket",
                "schema_name": "public",
                "columns": [
                    {
                        "name": "Match_Date",
                        "data_type": "text",
                        "description": "Date the match was played.",
                        "format": "YYMMDD",
                    }
                ],
            }
        ]
    )
    assert "description | format" in rendered
    assert "YYMMDD" in rendered


def test_prompt_omits_format_when_absent() -> None:
    rendered = format_tables_for_prompt(
        [
            {
                "name": "matches",
                "schema_name": "public",
                "columns": [
                    {"name": "team_name", "data_type": "text", "description": ""}
                ],
            }
        ]
    )
    assert "format:" not in rendered


def test_prompt_does_not_prefer_fewer_joins() -> None:
    system_prompt = create_sql_from_candidates_prompt()

    assert "fewest joins" not in system_prompt
    assert "shorter join path" not in system_prompt
    assert "Use only the hops" not in system_prompt
    assert "If removing a join would not change the answer, omit it" not in (
        create_sql_user_prompt
    )
    assert "Never join a table solely because its path is listed" not in (
        create_sql_user_prompt
    )
    assert "include the joins that connect those tables" in create_sql_user_prompt
    assert "include every table and every hop exactly as listed" in (
        create_sql_user_prompt
    )
    assert "every intermediate/bridge table and every listed hop is mandatory" in (
        system_prompt
    )


def test_prompt_filters_nulls_only_when_the_request_requires_values() -> None:
    assert "Do not add `IS NOT NULL` solely because" in create_sql_user_prompt
    assert "question or evidence requires the requested value" in (
        create_sql_user_prompt
    )
    assert "preserve rows containing NULLs" in create_sql_user_prompt


def test_prompt_does_not_infer_distinct_from_foreign_keys() -> None:
    projection_rules = format_projection_rules(False)
    assert "Use DISTINCT only when the question or Authoritative Evidence" in (
        projection_rules
    )
    assert "COUNT(fk) counts matching child rows" in projection_rules
    assert "Never infer DISTINCT solely from an FK" in projection_rules
    assert "necessary one-to-many join" not in create_sql_user_prompt


def test_projection_strictness_is_configured_per_request() -> None:
    strict_rule = "- Return exactly the requested output fields and NO others."
    combined_aggregate_rule = (
        "return ONE aggregate and combine the category predicates with OR"
    )

    assert strict_rule not in format_projection_rules(shorten_answer=False)
    assert strict_rule in format_projection_rules(shorten_answer=True)
    assert combined_aggregate_rule not in format_projection_rules(False)
    assert combined_aggregate_rule in format_projection_rules(True)


def test_sql_generation_plans_query_structure_before_sql() -> None:
    field_order = list(SQLGenerationModel.model_fields)
    assert field_order[:3] == ["plan", "thought", "sql_code"]

    prompt = create_sql_from_candidates_prompt()
    plan_position = prompt.index("- plan:")
    thought_position = prompt.index("- thought:")
    sql_position = prompt.index("- sql_code:")
    assert plan_position < thought_position < sql_position
    assert "exact join-key pairs" in prompt
    assert "result-row grain" in prompt
    assert "Explicitly justify every DISTINCT, CAST, projected field" in prompt
    assert "Do not include SQL in this field" in " ".join(prompt.split())
    assert "\nExample:\n" not in prompt
    assert "Every projected field and JOIN must be required" in create_sql_user_prompt
    assert "prevent integer truncation in division is valid" in create_sql_user_prompt
    assert "evidence-defined direct comparison operand" in create_sql_user_prompt


def test_authoritative_evidence_column_mapping_overrides_candidates() -> None:
    rendered = format_authoritative_evidence(
        "The requested identifier maps to records.canonical_identifier.",
        evidence_first_sql=True,
    )
    system_prompt = create_sql_from_candidates_prompt(has_sql_examples=True)

    assert "Explicit mappings in this evidence are binding" in rendered
    assert "Do not substitute a semantically similar column" in rendered
    assert "retrieved candidates" in rendered
    assert "canonical_identifier" in rendered
    assert "## Authoritative Evidence Checklist" in rendered
    assert "enumerate every explicit evidence" in system_prompt
    assert system_prompt.index("1. Explicit Authoritative Evidence") < (
        system_prompt.index("2. Verified schema facts")
    )
    assert system_prompt.index("2. Verified schema facts") < (
        system_prompt.index("3. Semantic hints")
    )
    assert system_prompt.index("3. Semantic hints") < (
        system_prompt.index("4. Reference SQL examples")
    )
    assert "authoritative only for stored literal values" in system_prompt


def test_authoritative_evidence_preserves_all_mapped_outputs() -> None:
    rendered = format_authoritative_evidence(
        "The requested classification maps to records.primary_class "
        "and records.secondary_class."
    )

    assert "If several output columns are explicitly mapped, preserve all of them" in (
        rendered
    )
    assert "primary_class" in rendered
    assert "secondary_class" in rendered


def test_authoritative_evidence_does_not_broaden_relationship_role() -> None:
    rendered = format_authoritative_evidence(
        "Match the requested entity through relations.source_entity_id."
    )

    assert "Do not broaden an explicit predicate" in rendered
    assert "endpoint, relationship role, or OR branch" in rendered
    assert "source_entity_id" in rendered


def test_authoritative_evidence_min_max_preserves_ties() -> None:
    rendered = format_authoritative_evidence(
        "Use MIN(records.score) to identify the lowest score."
    )

    assert "MIN and MAX in evidence are aggregate semantics" in rendered
    assert "equivalent tie-preserving form" in rendered
    assert "Do not replace this with ORDER BY and LIMIT 1" in rendered


def test_authoritative_evidence_preserves_schema_confirmed_spelling() -> None:
    rendered = format_authoritative_evidence(
        "Use records.CreationTimestmp as the event timestamp."
    )

    assert "Preserve the exact spelling and case" in rendered
    assert "when that identifier exists in the Available Schema" in rendered
    assert "never invent it" in rendered
    assert "CreationTimestmp" in rendered


def test_value_only_evidence_does_not_resolve_other_ambiguity() -> None:
    rendered = format_authoritative_evidence(
        "The stored label for the requested state is 'enabled'."
    )

    assert "If it only explains a literal value" in rendered
    assert "does not resolve an otherwise ambiguous table, column" in rendered
    assert "Do not infer additional requirements" in rendered
    assert "do not copy arbitrary prose into SQL" in rendered


def test_calculation_sql_template_is_structural_guidance() -> None:
    rendered = format_calculation_sql_template(
        {
            "calculation_subtype": "ranking",
            "sql_template": (
                "SELECT Title FROM posts ORDER BY ViewCount DESC LIMIT 5;"
            ),
        }
    )

    assert "classified as `ranking`" in rendered
    assert "SELECT Title FROM posts" in rendered
    assert "Never copy its table names" in rendered
    assert format_calculation_sql_template({}) == ""


def test_sql_generation_uses_reasoning_client(monkeypatch) -> None:
    non_reasoning_llm = object()
    reasoning_llm = object()
    invoked_with: list[object] = []
    invoked_messages: list = []

    def fake_invoke(llm, messages, _schema):
        invoked_with.append(llm)
        invoked_messages.extend(messages)
        return SQLGenerationModel(
            plan="Use the accounts table and return one aggregate row.",
            thought="No assumptions.",
            sql_code="SELECT COUNT(*) FROM accounts",
            response="Counts all accounts.",
        )

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )

    result = SQLFromCandidatesAgent().execute(
        {
            "llm": non_reasoning_llm,
            "reasoning_llm": reasoning_llm,
            "initial_question": "How many accounts?",
            "evidence": "Active means accounts.status = 'active'.",
            "value_anchors": [
                {
                    "phrase": "active",
                    "tbl": "accounts",
                    "col": "status",
                    "stored_value": "active",
                    "n_rows": 3,
                }
            ],
            "sql_examples": [
                {
                    "question": "Count matching records",
                    "sql": "SELECT COUNT(*) FROM records WHERE state = 'ready'",
                }
            ],
            "messages": [],
            "connectors": [],
            "path_state": {
                "primary_attribute": None,
                "relevant_tables": [],
                "relevant_queries": [],
            },
        }
    )

    assert result["decision"] == "constructable"
    assert invoked_with == [reasoning_llm]
    evidence = "Active means accounts.status = 'active'."
    assert sum(evidence in message.content for message in invoked_messages) == 2
    assert evidence in invoked_messages[-1].content
    assert "## Authoritative Evidence" in invoked_messages[-1].content
    assert invoked_messages[-2].content.startswith("## Authoritative Evidence")
    anchor_index = next(
        index
        for index, message in enumerate(invoked_messages)
        if message.content.startswith("## Verified Database Values")
    )
    examples_index = next(
        index
        for index, message in enumerate(invoked_messages)
        if message.content.startswith("## Reference Query Patterns")
    )
    assert anchor_index < examples_index < len(invoked_messages) - 2
    direct_answer_messages = [
        message
        for message in invoked_messages
        if message.content == DIRECT_ANSWER_FIELD_RULES
    ]
    assert len(direct_answer_messages) == 1
    assert invoked_messages.index(direct_answer_messages[0]) < len(invoked_messages) - 1
    assert "project that column unchanged" in DIRECT_ANSWER_FIELD_RULES
    assert "explicitly requests a normalized representation" in (
        DIRECT_ANSWER_FIELD_RULES
    )
    normalized_rules = " ".join(DIRECT_ANSWER_FIELD_RULES.split())
    assert "only when no direct answer field exists" in normalized_rules
    assert "including a field named with a synonym" in normalized_rules
    assert "Apply an outer aggregation only when the question explicitly requests" in (
        normalized_rules
    )
    assert "only when no matching direct field exists" in normalized_rules


def test_sql_generation_includes_selected_junction_guidance(monkeypatch) -> None:
    invoked_messages: list = []

    def fake_invoke(_llm, messages, _schema):
        invoked_messages.extend(messages)
        return SQLGenerationModel(
            plan="Use the selected association and preserve its join path.",
            thought="Use the selected association.",
            sql_code="SELECT 1",
            response="Generated SQL.",
        )

    monkeypatch.setattr(
        sql_from_semantic,
        "safe_invoke_with_structured_output",
        fake_invoke,
    )
    SQLFromCandidatesAgent().execute(
        {
            "llm": object(),
            "reasoning_llm": object(),
            "initial_question": "Use the relationship",
            "evidence": "",
            "messages": [],
            "connectors": [],
            "path_state": {
                "primary_attribute": None,
                "attribute_join_paths": [
                    {
                        "path": [
                            {
                                "source_schema": "main",
                                "source_table": "entity_link",
                                "source_column": "entity_id",
                                "target_schema": "main",
                                "target_table": "entity",
                                "target_column": "id",
                            }
                        ]
                    }
                ],
                "relevant_tables": [
                    {
                        "name": "entity_link",
                        "schema_name": "main",
                        "database_name": "db",
                        "is_junction_table": True,
                        "junction_table_kind": "bridge",
                        "junction_table_rationale": "Associates endpoint roles.",
                        "relationship_semantics": [
                            {
                                "relationship_type": "reciprocal_undirected_edge",
                                "endpoint_columns": ["entity_id", "related_entity_id"],
                                "edge_key_columns": ["relationship_id"],
                                "target_table": "entity",
                                "target_column": "id",
                                "reciprocal_rows_verified": True,
                            }
                        ],
                        "columns": [{"name": "entity_id", "data_type": "integer"}],
                    }
                ],
                "relevant_queries": [],
            },
        }
    )

    prompt = invoked_messages[-1].content
    assert "## Junction Tables" in prompt
    assert "Associates endpoint roles." in prompt
    assert "JOIN PATHS (VERIFIED OPTIONS" in prompt
    assert "## Verified Reciprocal-Row Relationships" in prompt
    assert "entity_id, related_entity_id" in prompt


def test_important_columns_include_full_details_and_ignore_structural_paths() -> None:
    rendered = format_important_columns_for_prompt(
        {
            "attr_name": "Charter Number",
            "col_name": "CharterNum",
            "table_name": "schools",
            "schema_name": "main",
            "database_name": "california_schools",
            "datatype": "text",
        },
        [
            {
                "attr_name": "District Name",
                "col_name": "District",
                "table_name": "schools",
                "schema_name": "main",
                "database_name": "california_schools",
                "datatype": "text",
                "path": [],
            },
            {
                "attr_name": "Removed Table Value",
                "col_name": "value",
                "table_name": "filtered_out",
                "schema_name": "main",
                "database_name": "california_schools",
                "datatype": "text",
                "path": [],
            },
            {"path": [{"source_table": "schools", "target_table": "districts"}]},
        ],
        [
            {
                "name": "schools",
                "schema_name": "main",
                "database_name": "california_schools",
                "columns": [
                    {
                        "name": "CharterNum",
                        "data_type": "text",
                        "description": "Four-character charter number.",
                        "sample_values": ["0040", "0728"],
                        "format": "NNNN",
                    },
                    {
                        "name": "District",
                        "data_type": "text",
                        "description": "District overseeing the school.",
                    },
                ],
            }
        ],
    )

    assert "Semantic match: Charter Number" in rendered
    assert "california_schools.main.schools" in rendered
    assert "Four-character charter number." in rendered
    assert "sample values: 0040, 0728" in rendered
    assert "format: NNNN" in rendered
    assert "Semantic match: District Name" in rendered
    assert "Removed Table Value" not in rendered
    assert "structural bridge" not in rendered
