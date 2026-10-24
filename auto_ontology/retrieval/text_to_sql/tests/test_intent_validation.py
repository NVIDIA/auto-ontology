# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the intent phase of unified SQL validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    INTENT_VALIDATION_SKIPPED_AFTER,
    IntentIssue,
    IntentValidationModel,
    SQLValidationAgent,
)


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(is_valid=True),
)
def test_intent_validation_keeps_authoritative_attribute_join(
    mock_invoke: MagicMock,
) -> None:
    path_state = {
        "sql_generation_result": SimpleNamespace(
            sql_code=(
                "SELECT customers.name FROM orders "
                "JOIN customers ON orders.customer_id = customers.id"
            )
        ),
        "primary_attribute": {
            "attr_name": "Order Customer",
            "col_name": "customer_id",
            "table_name": "orders",
            "schema_name": "public",
        },
        "attribute_join_paths": [
            {
                "attr_name": "Customer Name",
                "col_name": "name",
                "table_name": "customers",
                "schema_name": "public",
                "path": [
                    {
                        "source_schema": "public",
                        "source_table": "orders",
                        "source_column": "customer_id",
                        "target_schema": "public",
                        "target_table": "customers",
                        "target_column": "id",
                    }
                ],
            }
        ],
        "relevant_tables": [
            {
                "name": "orders",
                "schema_name": "public",
                "description": "Customer purchase records.",
                "columns": [
                    {
                        "name": "customer_id",
                        "data_type": "integer",
                        "description": "Customer placing the order.",
                        "sample_values": [10, 20],
                    },
                    {
                        "name": "internal_note",
                        "data_type": "text",
                        "description": "Not referenced by this query.",
                    },
                ],
            },
            {
                "name": "customers",
                "schema_name": "public",
                "description": "Customer directory.",
                "columns": [
                    {
                        "name": "name",
                        "data_type": "text",
                        "description": "Display name.",
                        "sample_values": ["Ada", "Grace"],
                    }
                ],
            },
        ],
    }
    state = {
        "llm": MagicMock(),
        "initial_question": "List customer names for orders",
        "connectors": [],
        "path_state": path_state,
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {**path_state, "sql_code": path_state["sql_generation_result"].sql_code},
        ["public.orders", "public.customers"],
        ["public.orders.customer_id", "public.customers.name"],
    )

    assert result["decision"] == "valid_sql"
    messages = mock_invoke.call_args.args[1]
    prompt = messages[1].content
    assert "AUTHORITATIVE JOIN PATHS" in prompt
    assert "JOIN PATHS (VERIFIED OPTIONS" in prompt
    assert "public.orders.customer_id = public.customers.id" in prompt
    assert "do not flag a generated join that follows" in prompt
    assert "SEMANTIC HINT" not in prompt
    assert "likely starting table" not in prompt
    assert "Column: customer_id  (Order Customer)" not in prompt
    assert "RELEVANT CANDIDATE FIELDS (NON-AUTHORITATIVE)" in prompt
    assert "Semantic match: Order Customer" in prompt
    assert "not proof that a plausible projected column should be replaced" in prompt
    assert "USED SQL OBJECTS" in prompt
    assert "- public.orders | description: Customer purchase records." in prompt
    assert "public.orders.customer_id (integer) - Customer placing the order." in prompt
    assert "sample values: 10, 20" in prompt
    assert "- public.customers | description: Customer directory." in prompt
    assert "public.customers.name (text) - Display name." in prompt
    assert "sample values: Ada, Grace" in prompt
    assert "internal_note" not in prompt


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(is_valid=True),
)
def test_intent_validation_receives_verbatim_authoritative_evidence(
    mock_invoke: MagicMock,
) -> None:
    evidence = (
        "Revenue must use orders.net_amount * orders.exchange_rate, "
        "even if orders.converted_amount exists."
    )
    state = {
        "llm": MagicMock(),
        "initial_question": "Calculate revenue",
        "evidence": evidence,
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT SUM(net_amount * exchange_rate) FROM orders"
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.orders"],
        ["public.orders.net_amount", "public.orders.exchange_rate"],
    )

    assert result["decision"] == "valid_sql"
    messages = mock_invoke.call_args.args[1]
    assert len(messages) == 2
    validation_prompt = messages[1].content
    assert validation_prompt.count(evidence) == 1
    assert "AUTHORITATIVE EVIDENCE TO VALIDATE AGAINST" in validation_prompt
    assert "preserve the semantic requirements" in validation_prompt
    assert "not a requirement to reproduce its literal text as SQL" in (
        validation_prompt
    )
    assert "surrounding whitespace differs" in validation_prompt
    assert "Type coercion needed for correct arithmetic is valid" in validation_prompt


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(is_valid=True),
)
def test_intent_validation_omits_generation_only_direct_answer_rules(
    mock_invoke: MagicMock,
) -> None:
    path_state = {
        "sql_generation_result": SimpleNamespace(
            sql_code="SELECT status_flag FROM accounts WHERE account_id = 7"
        ),
        "relevant_tables": [
            {
                "name": "accounts",
                "schema_name": "public",
                "columns": [
                    {
                        "name": "status_flag",
                        "data_type": "boolean",
                        "description": "Stored account status.",
                    }
                ],
            }
        ],
    }
    state = {
        "llm": MagicMock(),
        "initial_question": "Return the stored account status",
        "connectors": [],
        "path_state": path_state,
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {**path_state, "sql_code": path_state["sql_generation_result"].sql_code},
        ["public.accounts"],
        ["public.accounts.status_flag"],
    )

    assert result["decision"] == "valid_sql"
    messages = mock_invoke.call_args.args[1]
    assert "## Direct Answer Fields" not in messages[0].content
    assert "DIRECT ANSWER FIELD CHECK" not in messages[-1].content
    assert "normalization was explicitly requested" not in messages[-1].content


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=True,
        evidence_issues=[
            IntentIssue(
                message=(
                    "Uses converted_amount instead of the required "
                    "net_amount * exchange_rate."
                ),
                basis="authoritative_evidence",
                confidence="certain",
            )
        ],
    ),
)
def test_intent_validation_rejects_and_reports_evidence_violation(
    _mock_invoke: MagicMock,
) -> None:
    state = {
        "llm": MagicMock(),
        "initial_question": "Calculate revenue",
        "evidence": "Revenue must use net_amount * exchange_rate.",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT SUM(converted_amount) FROM orders"
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.orders"],
        ["public.orders.converted_amount"],
    )

    assert result["decision"] == "invalid_sql"
    assert "Critical evidence issues" in result["path_state"]["error"]
    assert "net_amount * exchange_rate" in result["path_state"]["error"]
    assert result["path_state"]["last_intent_rejected_sql"] == (
        "SELECT SUM(converted_amount) FROM orders"
    )


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=True,
        projection_issues=[
            IntentIssue(
                message="The SQL returns account_id, which was not requested.",
                basis="explicit_question",
                confidence="certain",
            )
        ],
    ),
)
def test_short_answer_validation_requires_exact_projection(
    mock_invoke: MagicMock,
) -> None:
    state = {
        "llm": MagicMock(),
        "initial_question": "How many accounts are active?",
        "shorten_answer": True,
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code=(
                    "SELECT COUNT(*) AS active_count, account_id "
                    "FROM accounts WHERE status = 'active' GROUP BY account_id"
                )
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.accounts"],
        ["public.accounts.account_id", "public.accounts.status"],
    )

    assert result["decision"] == "invalid_sql"
    assert "Critical projection issues" in result["path_state"]["error"]
    assert "account_id, which was not requested" in result["path_state"]["error"]
    messages = mock_invoke.call_args.args[1]
    assert "STRICT SHORT-ANSWER OVERRIDE" in messages[0].content
    assert "STRICT PROJECTION CHECK" in messages[1].content
    assert "every field requested" in messages[1].content
    assert "NO additional fields" in messages[1].content
    assert "extra, missing, or duplicated fields" in messages[1].content
    assert "combining them into one output" in messages[1].content
    assert "reordered" not in messages[1].content
    assert "reshaped" not in messages[1].content
    assert "row-versus-column" in messages[1].content


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(is_valid=True),
)
def test_non_short_answer_validation_omits_strict_projection_check(
    mock_invoke: MagicMock,
) -> None:
    state = {
        "llm": MagicMock(),
        "initial_question": "How many accounts are active?",
        "shorten_answer": False,
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT COUNT(*) FROM accounts WHERE status = 'active'"
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.accounts"],
        ["public.accounts.status"],
    )

    assert result["decision"] == "valid_sql"
    messages = mock_invoke.call_args.args[1]
    assert "STRICT SHORT-ANSWER OVERRIDE" not in messages[0].content
    assert "STRICT PROJECTION CHECK" not in messages[1].content
    assert "Every outermost projected field must be requested" in messages[1].content
    assert "Flag DISTINCT only when it demonstrably changes" in messages[1].content
    assert "Do not require an explicit uniqueness phrase" in messages[1].content
    assert "Flag CAST only when it demonstrably changes" in messages[1].content
    assert "never flag numeric coercion needed to avoid integer" in messages[1].content
    assert "SCHEMA METADATA IS CONTEXTUAL" in messages[1].content
    assert "Do not reject a projected column merely because" in messages[1].content
    assert (
        "Candidate fields may reveal that a requested concept is entirely missing"
        in (messages[1].content)
    )
    assert "Never flag harmless aliases or equivalent SQL forms" in messages[1].content
    assert "missing restriction" in messages[1].content
    assert "highest/top result" in messages[1].content
    assert "grouping at a grain different" in messages[1].content
    assert "explicitly requests separately" in messages[1].content
    assert "parallel category counts explicitly requested separately" in (
        messages[1].content
    )
    assert "explicit 'in total' request" in messages[1].content
    assert "direct stored field unambiguously matches" in messages[1].content
    assert "basis: exactly one of explicit_question" in messages[1].content
    assert "confidence: certain only when the error is directly demonstrated" in (
        messages[1].content
    )
    assert '"probably", "might", "prefer", and' in messages[1].content


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=True,
        join_issues=[
            IntentIssue(
                message="The join demonstrably duplicates the requested entity rows.",
                basis="sql_semantics",
                confidence="certain",
            )
        ],
        aggregation_issues=[
            IntentIssue(
                message="Verified key metadata proves grouping occurs at child grain.",
                basis="verified_schema",
                confidence="certain",
            )
        ],
        shape_issues=[
            IntentIssue(
                message="DISTINCT demonstrably changes the count from rows to entities.",
                basis="sql_semantics",
                confidence="certain",
            )
        ],
        projection_issues=[
            IntentIssue(
                message="The projected identifier was not requested.",
                basis="explicit_question",
                confidence="certain",
            )
        ],
    ),
)
def test_intent_validation_rejects_unsupported_query_shape(
    _mock_invoke: MagicMock,
) -> None:
    state = {
        "llm": MagicMock(),
        "initial_question": "Count active accounts",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code=(
                    "SELECT COUNT(DISTINCT account_id) "
                    "FROM accounts WHERE status = 'active'"
                )
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.accounts"],
        ["public.accounts.account_id", "public.accounts.status"],
    )

    assert result["decision"] == "invalid_sql"
    assert "Critical join issues" in result["path_state"]["error"]
    assert "Critical aggregation issues" in result["path_state"]["error"]
    assert "Critical query-shape issues" in result["path_state"]["error"]
    assert "Critical projection issues" in result["path_state"]["error"]
    assert "DISTINCT demonstrably changes" in result["path_state"]["error"]


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=False,
        projection_issues=[
            IntentIssue(
                message="A semantic candidate suggests a different identifier.",
                basis="semantic_hint",
                confidence="certain",
            ),
            IntentIssue(
                message="Another output representation may be preferable.",
                basis="preference",
                confidence="certain",
            ),
            IntentIssue(
                message="The requested representation is ambiguous.",
                basis="ambiguity",
                confidence="certain",
            ),
        ],
        shape_issues=[
            IntentIssue(
                message="The cast might affect comparison behavior.",
                basis="sql_semantics",
                confidence="uncertain",
            )
        ],
    ),
)
def test_intent_validation_ignores_preferences_hints_and_uncertainty(
    _mock_invoke: MagicMock,
) -> None:
    state = {
        "llm": MagicMock(),
        "initial_question": "Return the account identifier",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT external_id AS account_identifier FROM accounts"
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.accounts"],
        ["public.accounts.external_id"],
    )

    assert result["decision"] == "valid_sql"
    assert "error" not in result["path_state"]


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output"
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.get_schemas_by_ids",
    return_value={},
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.fetch_all_schema_ids",
    return_value=[],
)
def test_parse_failure_does_not_call_intent_llm(
    _mock_schema_ids: MagicMock,
    _mock_schemas: MagicMock,
    mock_invoke: MagicMock,
) -> None:
    agent = SQLValidationAgent()
    agent._sql_parse_validation = MagicMock(
        return_value={"error": "unknown column", "another_try": 1}
    )
    state = {
        "llm": MagicMock(),
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(sql_code="SELECT bad FROM t"),
            "relevant_tables": [],
        },
    }

    result = agent.execute(state)

    assert result["decision"] == "invalid_sql"
    assert result["path_state"]["error"] == "unknown column"
    mock_invoke.assert_not_called()


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.parse_query_single"
)
def test_parse_result_populates_catalog_ids_and_qualified_names(
    mock_parse: MagicMock,
) -> None:
    mock_parse.return_value = SimpleNamespace(
        get_tables_ids=lambda: ["table-id"],
        get_column_ids=lambda: ["column-id"],
    )
    schema = SimpleNamespace(
        id_to_node={
            "table-id": "public.orders",
            "column-id": "public.orders.total",
        }
    )

    result = SQLValidationAgent._sql_parse_validation(
        {"public": schema}, "SELECT total FROM orders", ["postgres"]
    )

    assert result["sql_tables"] == ["table-id"]
    assert result["sql_columns"] == ["column-id"]
    assert result["used_tables"] == ["public.orders"]
    assert result["used_columns"] == ["public.orders.total"]


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.parse_query_single",
    return_value=None,
)
def test_parse_result_without_catalog_table_is_invalid(_mock_parse: MagicMock) -> None:
    result = SQLValidationAgent._sql_parse_validation(
        {}, "SELECT * FROM missing", ["postgres"]
    )

    assert "error" in result
    assert "known to the catalog" in result["error"]


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output"
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.get_schemas_by_ids",
    return_value={},
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.fetch_all_schema_ids",
    return_value=[],
)
def test_retry_threshold_still_runs_intent_llm(
    _mock_schema_ids: MagicMock,
    _mock_schemas: MagicMock,
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = None
    agent = SQLValidationAgent()
    agent._sql_parse_validation = MagicMock(
        return_value={
            "success": True,
            "sql_tables": ["table-id"],
            "sql_columns": ["column-id"],
            "used_tables": ["public.orders"],
            "used_columns": ["public.orders.total"],
        }
    )
    state = {
        "llm": MagicMock(),
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT total FROM orders"
            ),
            "relevant_tables": [],
            "failed_attempts": [{} for _ in range(INTENT_VALIDATION_SKIPPED_AFTER + 1)],
        },
    }

    result = agent.execute(state)

    assert result["decision"] == "valid_sql"
    assert result["path_state"]["sql_tables"] == ["table-id"]
    assert result["path_state"]["sql_columns"] == ["column-id"]
    agent._sql_parse_validation.assert_called_once()
    mock_invoke.assert_called_once()


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output"
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.get_schemas_by_ids",
    return_value={},
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.fetch_all_schema_ids",
    return_value=[],
)
def test_final_full_pipeline_attempt_still_runs_intent_llm(
    _mock_schema_ids: MagicMock,
    _mock_schemas: MagicMock,
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = None
    agent = SQLValidationAgent()
    agent._sql_parse_validation = MagicMock(
        return_value={
            "success": True,
            "sql_tables": ["table-id"],
            "sql_columns": ["column-id"],
            "used_tables": ["public.orders"],
            "used_columns": ["public.orders.total"],
        }
    )
    state = {
        "llm": MagicMock(),
        "connectors": [],
        "full_pipeline_attempt": 2,
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT total FROM orders"
            ),
            "relevant_tables": [],
        },
    }

    result = agent.execute(state)

    assert result["decision"] == "valid_sql"
    assert result["path_state"]["sql_tables"] == ["table-id"]
    assert result["path_state"]["sql_columns"] == ["column-id"]
    agent._sql_parse_validation.assert_called_once()
    mock_invoke.assert_called_once()


def test_unconstructable_decision_passes_through_unified_validation() -> None:
    state = {"decision": "unconstructable", "path_state": {"error": "gave up"}}

    agent = SQLValidationAgent()

    assert agent.validate_input(state)
    assert agent.execute(state) == {
        "decision": "unconstructable",
        "path_state": {"error": "gave up"},
    }
