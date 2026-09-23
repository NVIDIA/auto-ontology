"""Tests for the intent phase of unified SQL validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    INTENT_VALIDATION_SKIPPED_AFTER,
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
    assert messages[1].content.endswith(evidence)
    assert "MUST follow every instruction" in messages[1].content
    assert "Evidence compliance is strict, not lenient" in messages[2].content


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
def test_retry_threshold_skips_only_intent_llm(
    _mock_schema_ids: MagicMock,
    _mock_schemas: MagicMock,
    mock_invoke: MagicMock,
) -> None:
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
    mock_invoke.assert_not_called()


def test_unconstructable_decision_passes_through_unified_validation() -> None:
    state = {"decision": "unconstructable", "path_state": {"error": "gave up"}}

    agent = SQLValidationAgent()

    assert agent.validate_input(state)
    assert agent.execute(state) == {
        "decision": "unconstructable",
        "path_state": {"error": "gave up"},
    }
