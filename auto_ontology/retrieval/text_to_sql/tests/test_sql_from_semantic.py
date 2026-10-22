# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared table/column prompt formatting in sql_from_semantic.py."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

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


def test_prompt_filters_nulls_only_when_the_request_requires_values() -> None:
    assert "Do not add `IS NOT NULL` solely because" in create_sql_user_prompt
    assert "question or evidence requires the requested value" in (
        create_sql_user_prompt
    )
    assert "preserve rows containing NULLs" in create_sql_user_prompt


def test_prompt_does_not_infer_distinct_from_foreign_keys() -> None:
    assert "Use DISTINCT only when the question or authoritative evidence" in (
        create_sql_user_prompt
    )
    assert "COUNT(fk) counts matching child rows" in create_sql_user_prompt
    assert "Never infer DISTINCT solely from an FK" in create_sql_user_prompt
    assert "necessary one-to-many join" not in create_sql_user_prompt


def test_projection_strictness_is_configured_per_request() -> None:
    strict_rule = "- Return exactly the requested output fields and NO others."
    combined_aggregate_rule = (
        "return ONE aggregate and combine the category predicates with OR"
    )
    field_order_rule = "Preserve the question's field order in SELECT"

    assert strict_rule not in format_projection_rules(shorten_answer=False)
    assert strict_rule in format_projection_rules(shorten_answer=True)
    assert combined_aggregate_rule not in format_projection_rules(False)
    assert combined_aggregate_rule in format_projection_rules(True)
    assert field_order_rule in format_projection_rules(False)
    assert field_order_rule in format_projection_rules(True)


def test_bridge_role_projection_rule_is_absent_without_relevant_tables() -> None:
    assert "multiple foreign-key columns" not in format_projection_rules(
        relevant_tables=None
    )


def test_bridge_role_projection_rule_is_absent_for_different_parents() -> None:
    tables = [
        {
            "name": "relationship",
            "columns": [
                {"name": "customer_id", "semantic_fk": "main.customers.id"},
                {"name": "order_id", "semantic_fk": "main.orders.id"},
            ],
        }
    ]

    assert "multiple foreign-key columns" not in format_projection_rules(
        relevant_tables=tables
    )


def test_projection_rules_combine_alternative_bridge_roles() -> None:
    tables = [
        {
            "name": "relationship",
            "columns": [
                {"name": "source_id", "semantic_fk": '"main"."people"."id"'},
                {"name": "target_id", "semantic_fk": "people.person_id"},
            ],
        }
    ]
    rules = format_projection_rules(
        shorten_answer=False,
        relevant_tables=tables,
    )

    assert (
        "bridge table has multiple foreign-key columns referencing the same "
        "parent entity type"
    ) in rules
    assert "return ONE output column" in rules
    assert "combining role-specific SELECTs with UNION" in rules
    assert "unless the question explicitly requests separate role-specific outputs" in (
        rules
    )


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
            "messages": [
                SystemMessage(content="Follow the deployment policy."),
                HumanMessage(content="How many accounts?"),
            ],
            "connectors": [],
            "path_state": {
                "primary_attribute": None,
                "relevant_tables": [
                    {
                        "name": "account_relationship",
                        "columns": [
                            {
                                "name": "source_account_id",
                                "semantic_fk": "main.accounts.id",
                                "semantic_fk_role": "source side of the relationship",
                                "semantic_fk_how_to_use": (
                                    "Join to the account that initiated the relationship."
                                ),
                                "semantic_fk_when_to_use": [
                                    "the initiating account is requested"
                                ],
                                "semantic_fk_avoid_when": [
                                    "the receiving account is requested"
                                ],
                            },
                            {
                                "name": "target_account_id",
                                "semantic_fk": "accounts.id",
                            },
                        ],
                    }
                ],
                "relevant_queries": [],
                "custom_analyses": [
                    {
                        "name": "Active account count",
                        "description": "Counts active accounts.",
                        "sql": "SELECT COUNT(*) FROM accounts WHERE status = 'active'",
                    }
                ],
                "custom_analyses_str": ["DUPLICATE CUSTOM ANALYSIS REPRESENTATION"],
                "table_relevance_reasoning": "INTERNAL TABLE SELECTION REASONING",
            },
        }
    )

    assert result["decision"] == "constructable"
    assert invoked_with == [reasoning_llm]
    evidence = "Active means accounts.status = 'active'."
    assert len(invoked_messages) == 2
    assert isinstance(invoked_messages[0], SystemMessage)
    assert isinstance(invoked_messages[1], HumanMessage)
    assert "Follow the deployment policy." in invoked_messages[0].content
    assert sum(evidence in message.content for message in invoked_messages) == 1
    assert evidence in invoked_messages[-1].content
    assert "## Authoritative Evidence" in invoked_messages[-1].content
    assert invoked_messages[-1].content.count("How many accounts?") == 1
    assert invoked_messages[-1].content.index("## Authoritative Evidence") < (
        invoked_messages[-1].content.index("## Available Schema")
    )
    assert invoked_messages[-1].content.index("## Rules") < (
        invoked_messages[-1].content.index("## Final Checklist")
    )
    assert "Active account count" in invoked_messages[-1].content
    assert "DUPLICATE CUSTOM ANALYSIS REPRESENTATION" not in (
        invoked_messages[-1].content
    )
    assert "INTERNAL TABLE SELECTION REASONING" not in invoked_messages[-1].content
    assert "multiple foreign-key columns referencing the same" in (
        invoked_messages[-1].content
    )
    prompt = invoked_messages[-1].content
    assert "## Semantic Relationship Guidance" in prompt
    assert "Use this relationship when:" in prompt
    assert "Avoid this relationship when:" in prompt
    assert prompt.index("## Semantic Relationship Guidance") > prompt.index(
        "## Available Schema"
    )
    assert prompt.index("## Semantic Relationship Guidance") < prompt.index(
        "## Semantically Important Columns"
    )


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
