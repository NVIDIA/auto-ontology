# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the intent phase of unified SQL validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    INTENT_VALIDATION_SKIPPED_AFTER,
    IntentRejectionReviewModel,
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
    assert "public.orders.customer_id = public.customers.id" in prompt
    assert "do not flag a generated join that follows" in prompt
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
    assert evidence in messages[1].content
    assert "MUST follow every instruction" in messages[1].content
    assert "Validation authority hierarchy" in messages[1].content
    assert "direct contradiction" in messages[1].content
    assert "Evidence compliance is strict, not lenient" in messages[2].content
    assert "original request wins" in messages[0].content
    assert "COUNT(DISTINCT ...)" in messages[0].content
    assert "derived yes/no" in messages[0].content


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=True,
        evidence_issues=[
            "Uses converted_amount instead of the required net_amount * exchange_rate."
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
    assert result["path_state"]["intent_repair_contract"] is True


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output"
)
def test_reasoning_review_blocks_unconfirmed_intent_rejection(
    mock_invoke: MagicMock,
) -> None:
    fast_llm = MagicMock(name="fast_llm")
    reasoning_llm = MagicMock(name="reasoning_llm")
    mock_invoke.side_effect = [
        IntentValidationModel(
            is_valid=False,
            aggregation_issues=["COUNT(*) should be COUNT(DISTINCT patient_id)."],
        ),
        IntentRejectionReviewModel(
            reject_sql=False,
            reasoning=(
                "The question does not explicitly request distinct patients, so "
                "the existing joined-row grain is a reasonable interpretation."
            ),
        ),
    ]
    state = {
        "llm": fast_llm,
        "reasoning_llm": reasoning_llm,
        "initial_question": "How many qualifying patients are there?",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code=(
                    "SELECT COUNT(*) FROM patient JOIN laboratory USING (patient_id)"
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
        ["public.patient", "public.laboratory"],
        ["public.patient.patient_id", "public.laboratory.patient_id"],
    )

    assert result["decision"] == "valid_sql"
    assert mock_invoke.call_args_list[0].args[0] is fast_llm
    assert mock_invoke.call_args_list[1].args[0] is reasoning_llm
    review_prompt = mock_invoke.call_args_list[1].args[1][-1].content
    assert "COUNT(*) should be COUNT(DISTINCT patient_id)" in review_prompt
    assert "proposed issues above" in review_prompt


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output"
)
def test_reasoning_review_confirms_only_supported_intent_issues(
    mock_invoke: MagicMock,
) -> None:
    fast_llm = MagicMock(name="fast_llm")
    reasoning_llm = MagicMock(name="reasoning_llm")
    mock_invoke.side_effect = [
        IntentValidationModel(
            is_valid=False,
            projection_issues=["The requested account name is missing."],
            evidence_issues=["An alias could be improved."],
            semantic_issues=["The account id might belong to another role."],
        ),
        IntentRejectionReviewModel(
            reject_sql=True,
            reasoning="The account name is explicitly requested.",
            projection_issues=["The requested account name is missing."],
        ),
    ]
    state = {
        "llm": fast_llm,
        "reasoning_llm": reasoning_llm,
        "initial_question": "List each account id and account name.",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT account_id FROM accounts"
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
        ["public.accounts.account_id"],
    )

    assert result["decision"] == "invalid_sql"
    assert "The requested account name is missing" in result["path_state"]["error"]
    assert "alias could be improved" not in result["path_state"]["error"]
    assert "another role" not in result["path_state"]["error"]
    assert result["path_state"]["intent_repair_contract"] is True


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output"
)
def test_reasoning_review_confirms_calculation_contract_failure(
    mock_invoke: MagicMock,
) -> None:
    fast_llm = MagicMock(name="fast_llm")
    reasoning_llm = MagicMock(name="reasoning_llm")
    issue = (
        "The denominator counts all transactions, but 'among active accounts' "
        "requires the population of active accounts."
    )
    mock_invoke.side_effect = [
        IntentValidationModel(
            is_valid=False,
            requested_outputs=["percentage of active accounts"],
            answer_grain="single result",
            calculation_contract=(
                "active accounts / all accounts in the requested region * 100"
            ),
            calculation_issues=[issue],
        ),
        IntentRejectionReviewModel(
            reject_sql=True,
            reasoning="The question explicitly names accounts as the population.",
            calculation_issues=[issue],
        ),
    ]
    state = {
        "llm": fast_llm,
        "reasoning_llm": reasoning_llm,
        "initial_question": "What percentage of active accounts are in region A?",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code=(
                    "SELECT 100.0 * SUM(amount) / "
                    "(SELECT SUM(amount) FROM transactions) FROM accounts "
                    "JOIN transactions USING (account_id) WHERE status = 'active'"
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
        ["public.accounts", "public.transactions"],
        ["public.accounts.status", "public.transactions.amount"],
    )

    assert result["decision"] == "invalid_sql"
    assert "Critical calculation issues" in result["path_state"]["error"]
    assert "active accounts" in result["path_state"]["error"]
    assert result["path_state"]["intent_repair_contract"] is True
    review_prompt = mock_invoke.call_args_list[1].args[1][-1].content
    assert issue in review_prompt


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=False,
        grain_issues=["The top-N may be at the wrong entity grain."],
    ),
)
def test_new_recall_issue_cannot_rewrite_without_independent_review(
    _mock_invoke: MagicMock,
) -> None:
    llm = MagicMock(name="shared_llm")
    state = {
        "llm": llm,
        "reasoning_llm": llm,
        "initial_question": "List the top five drivers by their fastest lap.",
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT driver_id FROM laps ORDER BY milliseconds LIMIT 5"
            )
        },
    }

    result = SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.laps"],
        ["public.laps.driver_id", "public.laps.milliseconds"],
    )

    assert result["decision"] == "valid_sql"
    assert "intent_repair_contract" not in result["path_state"]


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(is_valid=True),
)
def test_intent_prompt_includes_recall_checks_and_benchmark_traps(
    mock_invoke: MagicMock,
) -> None:
    state = {
        "llm": MagicMock(),
        "initial_question": (
            "What percentage of active accounts are in region A? Return their rank."
        ),
        "evidence": "Use transaction rows as the denominator.",
        "shorten_answer": True,
        "connectors": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code=(
                    "SELECT account_id, COUNT(*) FROM accounts "
                    "JOIN transactions USING (account_id) GROUP BY account_id"
                )
            )
        },
    }

    SQLValidationAgent()._validate_intent(
        state,
        {
            **state["path_state"],
            "sql_code": state["path_state"]["sql_generation_result"].sql_code,
        },
        ["public.accounts", "public.transactions"],
        ["public.accounts.account_id"],
    )

    messages = mock_invoke.call_args.args[1]
    combined = "\n".join(str(message.content) for message in messages)
    assert "answer entity and grain" in combined
    assert "numerator population" in combined
    assert "sorting alone is not that output" in combined
    assert "original request wins" in combined
    assert "Do not infer DISTINCT or a tie policy" in combined
    assert "raw stored code/label with a derived yes/no" in combined


@patch(
    "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=True,
        projection_issues=["The SQL returns account_id, which was not requested."],
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
    assert "every requested field, exactly once" in messages[1].content
    assert "NO additional fields" in messages[1].content


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
