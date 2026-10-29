# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared table/column prompt formatting in sql_from_semantic.py."""

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_from_semantic
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
    format_calculation_sql_template,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_tables_for_prompt,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    DIRECT_ANSWER_FIELD_RULES,
    create_sql_from_candidates_prompt,
    create_sql_user_prompt,
    format_dialect_rules,
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


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_prompt_forbids_joins_outside_paths_and_foreign_keys(dialect: str) -> None:
    system_prompt = create_sql_from_candidates_prompt(dialect=dialect)

    assert "NEVER invent a join" in system_prompt
    assert "foreign key reference in AVAILABLE TABLES" in system_prompt


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


def test_mysql_rules_lift_the_division_cast_and_forbid_rounding_casts() -> None:
    rules = format_dialect_rules("MySQL")

    assert rules == format_dialect_rules("mysql")
    # The generic rule the MySQL block overrides is still in the shared prompt.
    assert "Cast to REAL before dividing" in create_sql_user_prompt
    assert "`/` never truncates in MySQL" in rules
    assert "4 more decimal places" in rules
    assert "rule does not apply" in rules
    assert "DECIMAL(p, s), FLOAT, DOUBLE or REAL" in rules
    # Only a ratio too small for 4 decimal places may be widened.
    assert "1 / 1000000 = 0.0000" in rules
    assert "POPULATION statistics" in rules
    # Other dialects keep their own rules.
    for dialect in ("sqlite", "postgres", "snowflake", None):
        assert "MySQL" not in format_dialect_rules(dialect)


def test_projection_strictness_is_configured_per_request() -> None:
    strict_rule = "- Return exactly the requested output fields and NO others."
    combined_aggregate_rule = (
        "return ONE aggregate and combine the category predicates with OR"
    )

    assert strict_rule not in format_projection_rules(shorten_answer=False)
    assert strict_rule in format_projection_rules(shorten_answer=True)
    assert combined_aggregate_rule not in format_projection_rules(False)
    assert combined_aggregate_rule in format_projection_rules(True)
    for shorten_only_rule in (
        "Never use SELECT * or table.*",
        "return only the table's primary key column(s)",
    ):
        assert shorten_only_rule not in format_projection_rules(False)
        assert shorten_only_rule in format_projection_rules(True)


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


def test_sql_description_is_a_user_question_separate_from_response() -> None:
    description = SQLGenerationModel.model_fields["description"].description or ""
    response = SQLGenerationModel.model_fields["response"].description or ""

    assert "After writing sql_code" in description
    assert "one concise natural-language question" in description
    for forbidden_detail in ("tables", "columns", "joins", "clauses", "aliases"):
        assert forbidden_detail in description
    assert "2-4 sentences" in response


def test_sql_generation_uses_reasoning_client(monkeypatch) -> None:
    non_reasoning_llm = object()
    reasoning_llm = object()
    invoked_with: list[object] = []
    invocations: list[list] = []

    def fake_invoke(llm, messages, _schema):
        invoked_with.append(llm)
        invocations.append(messages)
        return SQLGenerationModel(
            thought="No assumptions.",
            sql_code="SELECT COUNT(*) FROM accounts",
            description="How many accounts are there?",
            response="Counts every row in the accounts table.",
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
                "matched_attributes": [],
                "relevant_tables": [],
                "relevant_queries": [],
            },
        }
    )

    assert result["decision"] == "constructable"
    assert invoked_with == [reasoning_llm]
    evidence = "Active means accounts.status = 'active'."
    (messages,) = invocations
    assert sum(evidence in message.content for message in messages) == 2
    assert evidence in messages[-1].content
    assert "## Authoritative Evidence" in messages[-1].content
    direct_answer_messages = [
        message for message in messages if message.content == DIRECT_ANSWER_FIELD_RULES
    ]
    assert len(direct_answer_messages) == 1
    assert messages.index(direct_answer_messages[0]) < len(messages) - 1
    assert "project that column unchanged" in DIRECT_ANSWER_FIELD_RULES
    assert "explicitly requests a normalized representation" in (
        DIRECT_ANSWER_FIELD_RULES
    )
    normalized_rules = " ".join(DIRECT_ANSWER_FIELD_RULES.split())
    assert "only when no direct answer field exists" in normalized_rules
    assert (
        result["path_state"]["sql_generation_result"].sql_code
        == "SELECT COUNT(*) FROM accounts"
    )


def test_sql_generation_is_unconstructable_when_every_attempt_fails(
    monkeypatch,
) -> None:
    calls = 0

    def fail_invoke(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fail_invoke
    )

    result = SQLFromCandidatesAgent().execute(
        {
            "llm": object(),
            "initial_question": "Count accounts",
            "evidence": "",
            "messages": [],
            "connectors": [],
            "path_state": {
                "matched_attributes": [],
                "relevant_tables": [],
                "relevant_queries": [],
            },
        }
    )

    assert calls == 3
    assert result["decision"] == "unconstructable"
    assert result["path_state"]["unconstructable_explanation"] == (
        "LLM failed to produce a response."
    )
