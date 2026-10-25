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
    create_intent_validation_prompt,
    create_sql_from_candidates_prompt,
    create_sql_user_prompt,
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


def test_generation_prompt_does_not_decide_null_result_filtering() -> None:
    assert "IS NOT NULL" not in create_sql_user_prompt
    assert "NULL filtering" not in create_sql_user_prompt


def test_prompt_uses_distinct_when_source_grain_repeats_requested_entities() -> None:
    assert "identifier marked `is_unique: false`" in create_sql_user_prompt
    assert "both within one table and after joins" in create_sql_user_prompt
    assert "Count DISTINCT entity IDs" in create_sql_user_prompt
    assert "deduplicate/group to one row per entity before LIMIT" in (
        create_sql_user_prompt
    )
    assert "question asks for child/detail records themselves" in create_sql_user_prompt
    assert "Authoritative Evidence formula's numerator" in create_sql_user_prompt


def test_intent_validation_checks_repeated_entity_grain_without_a_join() -> None:
    prompt = create_intent_validation_prompt(
        "How many customers have active events?",
        "How many customers have active events?",
        "How many customers have active events?",
        "SELECT COUNT(customer_id) FROM events WHERE status = 'active'",
        used_schema_context=(
            "events.customer_id | foreign_key: customers.id | is_unique: false"
        ),
    )

    assert "whether those repeated IDs occur in one" in prompt
    assert "base table or after a join" in prompt
    assert "COUNT(DISTINCT entity_id)" in prompt
    assert "finer-grained records themselves" in prompt


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
    assert "Do not include SQL in this field" in prompt
    assert "\nExample:\n" not in prompt


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
    semantic_context = (
        "## Important Semantic Context\n"
        "- [field_mapping] Use account status for lifecycle state."
    )

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
            "messages": [],
            "connectors": [],
            "path_state": {
                "primary_attribute": None,
                "relevant_tables": [],
                "relevant_queries": [],
                "semantic_constraints": [
                    {
                        "kind": "field_mapping",
                        "instruction": "Use account status for lifecycle state.",
                        "source_ids": ["column:status"],
                    }
                ],
                "semantic_constraints_context": semantic_context,
            },
        }
    )

    assert result["decision"] == "constructable"
    assert invoked_with == [reasoning_llm]
    evidence = "Active means accounts.status = 'active'."
    assert sum(evidence in message.content for message in invoked_messages) == 2
    assert evidence in invoked_messages[-1].content
    assert "## Authoritative Evidence" in invoked_messages[-1].content
    semantic_messages = [
        message for message in invoked_messages if message.content == semantic_context
    ]
    assert len(semantic_messages) == 1
    evidence_message = next(
        message
        for message in invoked_messages
        if message.content.startswith("## Authoritative Evidence")
    )
    assert invoked_messages.index(evidence_message) < invoked_messages.index(
        semantic_messages[0]
    )
    assert "## Available Schema" in invoked_messages[-1].content
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
