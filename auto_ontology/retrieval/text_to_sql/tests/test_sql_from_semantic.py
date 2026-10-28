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
        is_schema_only = messages[-1].content.startswith("### Question")
        return SQLGenerationModel(
            thought="No assumptions.",
            sql_code=(
                "SELECT COUNT(*) FROM accounts"
                if not is_schema_only
                else "SELECT COUNT(id) FROM accounts"
            ),
            description=(
                "How many accounts are there?"
                if not is_schema_only
                else "How many accounts have an identifier?"
            ),
            response=(
                "Counts every row in the accounts table."
                if not is_schema_only
                else "Counts every non-null account identifier."
            ),
        )

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(
        sql_from_semantic,
        "rerank_hits",
        lambda query, hits, **kwargs: [
            {
                "_rerank_score": 1.0,
                **next(hit for hit in hits if hit["id"] == "semantic"),
            }
        ],
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
            },
        }
    )

    assert result["decision"] == "constructable"
    assert invoked_with == [reasoning_llm, reasoning_llm]
    assert len(invocations) == 2
    evidence = "Active means accounts.status = 'active'."
    semantic_messages, schema_messages = invocations
    assert len(semantic_messages) == 2
    assert isinstance(semantic_messages[0], SystemMessage)
    assert isinstance(semantic_messages[1], HumanMessage)
    assert sum(evidence in message.content for message in semantic_messages) == 2
    assert evidence in semantic_messages[-1].content
    assert "## Authoritative Evidence" in semantic_messages[-1].content
    assert len(schema_messages) == 2
    assert isinstance(schema_messages[0], SystemMessage)
    assert isinstance(schema_messages[1], HumanMessage)
    assert schema_messages[-1].content.startswith("### Question")
    assert "### Evidence" in schema_messages[-1].content
    assert evidence in schema_messages[-1].content
    assert "## Semantic Hints" not in schema_messages[-1].content
    assert "## Example SQL Queries" not in schema_messages[-1].content
    assert (
        result["path_state"]["sql_generation_result"].sql_code
        == "SELECT COUNT(*) FROM accounts"
    )


def test_sql_generation_reranks_descriptions_without_sending_sql(monkeypatch) -> None:
    seen: dict = {}

    def fake_invoke(_llm, messages, _schema):
        schema_only = messages[-1].content.startswith("### Question")
        if schema_only:
            seen["schema_prompt"] = messages[-1].content
        return SQLGenerationModel(
            thought="No assumptions.",
            sql_code=(
                "SELECT account_id FROM accounts"
                if not schema_only
                else "SELECT name FROM accounts"
            ),
            description=(
                "What are the account identifiers?"
                if not schema_only
                else "What are the account names?"
            ),
            response=(
                "Semantic candidate user response."
                if not schema_only
                else "Schema-only candidate user response."
            ),
        )

    def fake_rerank(query, hits, **kwargs):
        seen.update(query=query, hits=hits, kwargs=kwargs)
        return [
            {
                "_rerank_score": 0.9,
                **next(hit for hit in hits if hit["id"] == "schema-only"),
            }
        ]

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(sql_from_semantic, "rerank_hits", fake_rerank)
    monkeypatch.setattr(
        sql_from_semantic,
        "get_rerank_kwargs",
        lambda: {
            "rerank_invoke_url": "https://rerank.example/rerank",
            "model_name": "reranker",
            "api_key": "",
        },
    )

    result = SQLFromCandidatesAgent().execute(
        {
            "llm": object(),
            "initial_question": "List account names",
            "evidence": "Use the account name.",
            "messages": [],
            "connectors": [],
            "path_state": {
                "primary_attribute": None,
                "relevant_tables": [
                    {
                        "name": "accounts",
                        "columns": [
                            {"name": "account_id", "data_type": "integer"},
                            {"name": "name", "data_type": "text"},
                        ],
                    }
                ],
                "relevant_queries": [],
            },
        }
    )

    assert seen["query"] == (
        "Question:\nList account names\n\nEvidence:\nUse the account name."
    )
    assert seen["hits"] == [
        {"id": "semantic", "text": "What are the account identifiers?"},
        {
            "id": "schema-only",
            "text": "What are the account names?",
        },
    ]
    assert seen["kwargs"]["top_n"] == 1
    assert all("SELECT" not in hit["text"] for hit in seen["hits"])
    assert "table\naccounts" in seen["schema_prompt"]
    assert "column | type | is_nullable" in seen["schema_prompt"]
    assert "list of important semantic entities" not in seen["schema_prompt"]
    winner = result["path_state"]["sql_generation_result"]
    assert winner.sql_code == "SELECT name FROM accounts"
    assert winner.description == "What are the account names?"
    assert winner.response == "Schema-only candidate user response."


def test_sql_generation_rerank_failure_falls_back_to_semantic(monkeypatch) -> None:
    responses = iter(
        [
            SQLGenerationModel(
                thought="Semantic.",
                sql_code="SELECT semantic_value FROM facts",
                description="What is the semantic value?",
                response="Returns the semantic value.",
            ),
            SQLGenerationModel(
                thought="Schema only.",
                sql_code="SELECT schema_value FROM facts",
                description="What is the schema-only value?",
                response="Returns the schema-only value.",
            ),
        ]
    )
    monkeypatch.setattr(
        sql_from_semantic,
        "safe_invoke_with_structured_output",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(
        sql_from_semantic,
        "rerank_hits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = SQLFromCandidatesAgent().execute(
        {
            "llm": object(),
            "initial_question": "Return a value",
            "evidence": "",
            "messages": [],
            "connectors": [],
            "path_state": {
                "primary_attribute": None,
                "relevant_tables": [],
                "relevant_queries": [],
            },
        }
    )

    assert (
        result["path_state"]["sql_generation_result"].sql_code
        == "SELECT semantic_value FROM facts"
    )


def test_sql_generation_uses_only_successful_candidate(monkeypatch) -> None:
    def fake_invoke(_llm, messages, _schema):
        if messages[-1].content.startswith("### Question"):
            return None
        return SQLGenerationModel(
            thought="No assumptions.",
            sql_code="SELECT COUNT(*) FROM accounts",
            description="How many accounts are there?",
            response="Counts all account rows.",
        )

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(
        sql_from_semantic,
        "rerank_hits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rerank must not run for one candidate")
        ),
    )

    result = SQLFromCandidatesAgent().execute(
        {
            "llm": object(),
            "initial_question": "Count accounts",
            "evidence": "",
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
    assert (
        result["path_state"]["sql_generation_result"].sql_code
        == "SELECT COUNT(*) FROM accounts"
    )


def test_sql_generation_is_unconstructable_when_both_candidates_fail(
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
                "primary_attribute": None,
                "relevant_tables": [],
                "relevant_queries": [],
            },
        }
    )

    assert calls == 6
    assert result["decision"] == "unconstructable"
    assert result["path_state"]["unconstructable_explanation"] == (
        "LLM failed to produce a response."
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
