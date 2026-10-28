# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared table/column prompt formatting in sql_from_semantic.py."""

from __future__ import annotations

from auto_ontology.retrieval.text_to_sql.agents import (
    sql_candidate_selection,
    sql_from_semantic,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
    format_calculation_sql_template,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_important_columns_for_prompt,
    format_tables_for_prompt,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    SCHEMA_ONLY_SQL_SYSTEM_PROMPT,
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


def _generation(sql: str, description: str, response: str) -> SQLGenerationModel:
    return SQLGenerationModel(
        thought="No assumptions.",
        sql_code=sql,
        description=description,
        response=response,
    )


def _is_schema_only(messages: list) -> bool:
    return messages[-1].content.startswith("### Question")


def test_sql_description_is_a_user_question_separate_from_response() -> None:
    fields = list(SQLGenerationModel.model_fields)
    description = SQLGenerationModel.model_fields["description"].description or ""
    response = SQLGenerationModel.model_fields["response"].description or ""

    assert fields.index("sql_code") < fields.index("description")
    assert fields.index("description") < fields.index("response")
    assert "phrased the way a user would ask" in description
    assert "not the question you were given" in description
    for forbidden_detail in ("tables", "columns", "joins", "sources"):
        assert forbidden_detail in description
    assert "2-4 sentences" in response
    assert SCHEMA_ONLY_SQL_SYSTEM_PROMPT.startswith("Please reason step by step")
    assert "Return exactly the requested output fields and NO others." in (
        SCHEMA_ONLY_SQL_SYSTEM_PROMPT
    )


def test_sql_generation_uses_reasoning_client(monkeypatch) -> None:
    non_reasoning_llm = object()
    reasoning_llm = object()
    invoked_with: list[object] = []
    invocations: dict[str, list] = {}

    def fake_invoke(llm, messages, schema):
        assert schema is SQLGenerationModel
        invoked_with.append(llm)
        if _is_schema_only(messages):
            invocations["schema-only"] = messages
            return _generation(
                "SELECT COUNT(id) FROM accounts",
                "How many accounts have an identifier?",
                "Counts every non-null account identifier.",
            )
        invocations["semantic"] = messages
        return _generation(
            "SELECT COUNT(*) FROM accounts",
            "How many accounts are there?",
            "Counts every row in the accounts table.",
        )

    judged_with: list[object] = []

    def fake_judge(ctx, first, _second):
        judged_with.append(ctx.llm)
        return "semantic"

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(sql_candidate_selection, "_judge_once", fake_judge)

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
    assert judged_with == [reasoning_llm, reasoning_llm]
    assert result["path_state"]["sql_candidate_selection"]["stage"] == "judge"
    evidence = "Active means accounts.status = 'active'."
    semantic_messages = invocations["semantic"]
    schema_messages = invocations["schema-only"]
    assert sum(evidence in message.content for message in semantic_messages) == 2
    assert evidence in semantic_messages[-1].content
    assert "## Authoritative Evidence" in semantic_messages[-1].content
    assert len(schema_messages) == 2
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
        if _is_schema_only(messages):
            seen["schema_messages"] = messages
            return _generation(
                "SELECT name FROM accounts",
                "What are the account names?",
                "Schema-only candidate user response.",
            )
        return _generation(
            "SELECT account_id FROM accounts",
            "What are the account identifiers?",
            "Semantic candidate user response.",
        )

    def fake_rerank(query, hits, **kwargs):
        seen.update(query=query, hits=hits, kwargs=kwargs)
        return [
            {"_rerank_score": 0.9, **hits[1]},
            {"_rerank_score": 0.2, **hits[0]},
        ]

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(sql_candidate_selection, "_judge_once", lambda *_args: None)
    monkeypatch.setattr(sql_candidate_selection, "rerank_hits", fake_rerank)
    monkeypatch.setattr(
        sql_candidate_selection,
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
    assert "top_n" not in seen["kwargs"]
    assert all("SELECT" not in hit["text"] for hit in seen["hits"])
    system_prompt, schema_prompt = (m.content for m in seen["schema_messages"])
    assert system_prompt == SCHEMA_ONLY_SQL_SYSTEM_PROMPT
    assert "table\naccounts" in schema_prompt
    assert "column | type | is_nullable" in schema_prompt
    assert "table view that describes the database" in schema_prompt
    assert "list of important semantic entities" not in schema_prompt
    winner = result["path_state"]["sql_generation_result"]
    assert winner.sql_code == "SELECT name FROM accounts"
    assert winner.description == "What are the account names?"
    assert winner.response == "Schema-only candidate user response."
    assert result["path_state"]["sql_candidate_selected"] == "schema-only"
    assert result["path_state"]["sql_candidate_selection"]["stage"] == "rerank"
    common = {"static_issue": None, "execution": None, "judge_votes": []}
    assert result["path_state"]["sql_candidates"] == [
        {
            "source": "semantic",
            "sql_code": "SELECT account_id FROM accounts",
            "description": "What are the account identifiers?",
            "rerank_score": 0.2,
            **common,
        },
        {
            "source": "schema-only",
            "sql_code": "SELECT name FROM accounts",
            "description": "What are the account names?",
            "rerank_score": 0.9,
            **common,
        },
    ]


def test_sql_generation_rerank_failure_falls_back_to_semantic(monkeypatch) -> None:
    def fake_invoke(_llm, messages, _schema):
        if _is_schema_only(messages):
            return _generation(
                "SELECT schema_value FROM facts",
                "What is the schema-only value?",
                "Returns the schema-only value.",
            )
        return _generation(
            "SELECT semantic_value FROM facts",
            "What is the semantic value?",
            "Returns the semantic value.",
        )

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(sql_candidate_selection, "_judge_once", lambda *_args: None)
    monkeypatch.setattr(
        sql_candidate_selection,
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
    schema_only_calls = 0

    def fake_invoke(_llm, messages, _schema):
        nonlocal schema_only_calls
        if _is_schema_only(messages):
            schema_only_calls += 1
            return None
        return _generation(
            "SELECT COUNT(*) FROM accounts",
            "How many accounts are there?",
            "Counts all account rows.",
        )

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    monkeypatch.setattr(
        sql_candidate_selection,
        "rerank_hits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rerank must not run for one candidate")
        ),
    )
    monkeypatch.setattr(
        sql_candidate_selection,
        "_judge",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("judge must not run for one candidate")
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
    assert schema_only_calls == 3
    assert (
        result["path_state"]["sql_generation_result"].sql_code
        == "SELECT COUNT(*) FROM accounts"
    )
    assert result["path_state"]["sql_candidate_selected"] == "semantic"
    assert len(result["path_state"]["sql_candidates"]) == 1


def test_sql_generation_is_unconstructable_when_both_candidates_fail(
    monkeypatch,
) -> None:
    calls: list[None] = []

    def fail_invoke(*_args, **_kwargs):
        calls.append(None)
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

    assert len(calls) == 6
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
