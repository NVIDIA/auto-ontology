# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_value_validation
from auto_ontology.retrieval.text_to_sql.agents import combined_precheck
from auto_ontology.retrieval.text_to_sql.agents.combined_precheck import (
    CombinedPrecheckAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_value_validation import (
    SQLValueValidationAgent,
    ValueQueryArguments,
    ValueReview,
    ValueSubstitution,
    ValueValidationAction,
    ValueValidationActionType,
    extract_value_checks,
)
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    create_graph,
    route_generated_sql,
    route_reconstructed_sql,
)


def _generation(sql: str) -> SQLGenerationModel:
    return SQLGenerationModel(
        thought="No assumptions.",
        sql_code=sql,
        response="Account rows are returned.",
    )


def _tables() -> list[dict]:
    return [
        {
            "schema_name": "main",
            "name": "accounts",
            "columns": [
                {"name": "status", "data_type": "text"},
                {"name": "country", "data_type": "text"},
                {"name": "age", "data_type": "integer"},
            ],
        }
    ]


def _state(sql: str, *, enabled: bool = True, evidence: str = "") -> dict:
    return {
        "llm": object(),
        "initial_question": "Show active accounts",
        "evidence": evidence,
        "validate_sql_values": enabled,
        "sql_value_validation_cache": {},
        "connectors": [],
        "path_state": {
            "sql_generation_result": _generation(sql),
            "relevant_tables": _tables(),
        },
    }


class _Connector:
    dialect = "sqlite"
    database_name = "test"


class _Executor:
    calls = 0

    def __init__(self, _connector, max_calls: int) -> None:
        self.max_calls = max_calls

    @property
    def budget_left(self) -> int:
        return self.max_calls - self.calls

    def run(self, sql: str, purpose: str = "") -> dict:
        del purpose
        type(self).calls += 1
        rows = [] if "'actve'" in sql else [{"status": "Active"}]
        return {
            "sql": sql,
            "ok": True,
            "rows": rows,
            "row_count": len(rows),
            "error": None,
        }


def _query_action(proof_query: str) -> ValueValidationAction:
    return ValueValidationAction(
        action=ValueValidationActionType.QUERY,
        arguments=ValueQueryArguments(
            sql=proof_query,
            purpose="Find the stored status representation.",
        ),
    )


def _finish_action(
    original_sql: str,
    absence_query: str,
    proof_query: str,
    *,
    corrected_sql: str | None = None,
    corrected_evidence: str | None = None,
) -> ValueValidationAction:
    return ValueValidationAction(
        action=ValueValidationActionType.FINISH,
        corrected_sql=corrected_sql or original_sql,
        corrected_evidence=corrected_evidence,
        reviewed_check_ids=["value_1"],
        reviews=[
            ValueReview(
                check_id="value_1",
                original_literal="'actve'",
                final_literal="'Active'",
                status="corrected",
                proof_query=proof_query,
            )
        ],
        substitutions=[
            ValueSubstitution(
                check_id="value_1",
                original_literal="'actve'",
                replacement_literal="'Active'",
                absence_query=absence_query,
                proof_query=proof_query,
            )
        ],
        reasoning="The database stores the status as Active.",
    )


def test_extracts_filter_and_case_values_but_not_case_output_labels() -> None:
    checks = extract_value_checks(
        "SELECT CASE WHEN status = 'active' THEN 'YES' ELSE 'NO' END "
        "FROM accounts WHERE country IN ('US', 'CA') AND age BETWEEN 10 AND 20",
        "sqlite",
    )

    assert [check["literal"] for check in checks] == [
        "'active'",
        "'US'",
        "'CA'",
        "10",
        "20",
    ]
    assert "'YES'" not in {check["literal"] for check in checks}
    assert "'NO'" not in {check["literal"] for check in checks}


def test_prompt_requires_trim_and_like_before_semantic_synonyms() -> None:
    prompt = sql_value_validation._SYSTEM_PROMPT

    assert prompt.index("trim leading and trailing whitespace") < prompt.index(
        "semantic\n   synonyms"
    )
    assert prompt.index("then search the same column with LIKE") < prompt.index(
        "semantic\n   synonyms"
    )
    assert (
        "Never jump to a synonym before testing the trimmed literal and LIKE" in prompt
    )


def test_feature_flag_off_skips_llm_and_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sql_value_validation,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: pytest.fail("LLM should not run"),
    )

    assert (
        SQLValueValidationAgent().execute(
            _state("SELECT * FROM accounts WHERE status = 'active'", enabled=False)
        )
        == {}
    )


def test_agent_repairs_sql_and_matching_evidence_and_caches_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sql = "SELECT * FROM accounts WHERE status = 'actve'"
    corrected_sql = "SELECT * FROM accounts WHERE status = 'Active'"
    absence_query = "SELECT status FROM accounts WHERE status = 'actve'"
    proof_query = "SELECT status FROM accounts WHERE status = 'Active'"
    actions = iter(
        [
            _query_action(absence_query),
            _query_action(proof_query),
            _finish_action(
                original_sql,
                absence_query,
                proof_query,
                corrected_sql=corrected_sql,
                corrected_evidence="active status means accounts.status = 'Active'",
            ),
        ]
    )
    monkeypatch.setattr(
        sql_value_validation,
        "resolve_connector_from_tables",
        lambda *_args: _Connector(),
    )
    monkeypatch.setattr(sql_value_validation, "ProbeExecutor", _Executor)
    monkeypatch.setattr(
        sql_value_validation,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )
    state = _state(
        original_sql,
        evidence="active status means accounts.status = 'actve'",
    )

    result = SQLValueValidationAgent().execute(state)

    assert result["path_state"]["sql_code"] == corrected_sql
    assert result["path_state"]["sql_generation_result"].sql_code == corrected_sql
    assert result["evidence"] == "active status means accounts.status = 'Active'"
    assert len(result["path_state"]["sql_value_substitutions"]) == 1
    assert result["sql_value_validation_cache"]["queries"]
    assert result["sql_value_validation_cache"]["findings"]


def test_cached_probe_is_reused_without_database_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Executor.calls = 0
    original_sql = "SELECT * FROM accounts WHERE status = 'actve'"
    corrected_sql = "SELECT * FROM accounts WHERE status = 'Active'"
    absence_query = "SELECT status FROM accounts WHERE status = 'actve'"
    proof_query = "SELECT status FROM accounts WHERE status = 'Active'"
    first_actions = iter(
        [
            _query_action(absence_query),
            _query_action(proof_query),
            _finish_action(
                original_sql,
                absence_query,
                proof_query,
                corrected_sql=corrected_sql,
            ),
        ]
    )
    monkeypatch.setattr(
        sql_value_validation,
        "resolve_connector_from_tables",
        lambda *_args: _Connector(),
    )
    monkeypatch.setattr(sql_value_validation, "ProbeExecutor", _Executor)
    monkeypatch.setattr(
        sql_value_validation,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(first_actions),
    )
    first = SQLValueValidationAgent().execute(_state(original_sql))
    assert _Executor.calls == 2

    second_state = _state(original_sql)
    second_state["sql_value_validation_cache"] = first["sql_value_validation_cache"]
    second_actions = iter(
        [
            _query_action(absence_query),
            _query_action(proof_query),
            _finish_action(
                original_sql,
                absence_query,
                proof_query,
                corrected_sql=corrected_sql,
            ),
        ]
    )
    monkeypatch.setattr(
        sql_value_validation,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(second_actions),
    )

    second = SQLValueValidationAgent().execute(second_state)

    assert _Executor.calls == 2
    assert second["path_state"]["sql_code"] == corrected_sql


def test_unproven_or_structural_sql_change_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sql = "SELECT * FROM accounts WHERE status = 'actve'"
    absence_query = "SELECT status FROM accounts WHERE status = 'actve'"
    proof_query = "SELECT status FROM accounts WHERE status = 'Active'"
    action = _finish_action(
        original_sql,
        absence_query,
        proof_query,
        corrected_sql=("SELECT country FROM accounts WHERE status = 'Active'"),
    )
    monkeypatch.setattr(
        sql_value_validation,
        "resolve_connector_from_tables",
        lambda *_args: _Connector(),
    )
    monkeypatch.setattr(
        sql_value_validation,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: action,
    )

    result = SQLValueValidationAgent().execute(_state(original_sql))

    assert "path_state" not in result
    assert result["sql_value_validation_cache"]["findings"] == {}


def test_post_generation_routes_are_feature_flagged() -> None:
    assert (
        route_generated_sql({"decision": "constructable", "validate_sql_values": True})
        == "validate_sql_values"
    )
    assert (
        route_generated_sql({"decision": "constructable", "validate_sql_values": False})
        == "validate_sql_query"
    )
    assert (
        route_reconstructed_sql(
            {"decision": "invalid_sql", "validate_sql_values": True}
        )
        == "validate_sql_values"
    )
    assert (
        route_reconstructed_sql(
            {"decision": "unconstructable", "validate_sql_values": True}
        )
        == "validate_sql_query"
    )


def test_graph_places_value_validation_after_both_sql_writers() -> None:
    edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in create_graph().compile().get_graph().edges
    }

    assert (
        "construct_sql_from_candidates",
        "validate_sql_values",
        True,
    ) in edges
    assert ("reconstruct_sql", "validate_sql_values", True) in edges
    assert ("validate_sql_values", "validate_sql_query", False) in edges


def test_agentic_flag_skips_duplicate_proactive_value_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(combined_precheck, "is_db_probe_proactive", lambda: True)
    monkeypatch.setattr(combined_precheck, "is_db_probe_join_path_check", lambda: False)
    monkeypatch.setattr(
        combined_precheck,
        "is_db_probe_jsonb_path_check",
        lambda: False,
    )
    agent = CombinedPrecheckAgent()
    monkeypatch.setattr(
        agent._value_check,
        "execute",
        lambda _state: pytest.fail("duplicate value probe should not run"),
    )
    state = _state("SELECT * FROM accounts WHERE status = 'active'")

    result = agent.execute(state)

    assert result["decision"] == "valid_sql"
