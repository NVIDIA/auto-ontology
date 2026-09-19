"""Tests for semantic context in intent validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gsf.retrieval.text_to_sql.agents.intent_validation import (
    IntentValidationAgent,
    IntentValidationModel,
)


@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output",
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
    }
    state = {
        "llm": MagicMock(),
        "initial_question": "List customer names for orders",
        "path_state": path_state,
    }

    result = IntentValidationAgent().execute(state)

    assert result["decision"] == "intent_valid"
    messages = mock_invoke.call_args.args[1]
    prompt = messages[1].content
    assert "AUTHORITATIVE JOIN PATHS" in prompt
    assert "public.orders.customer_id = public.customers.id" in prompt
    assert "do not flag a generated join that follows" in prompt


@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output",
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
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT SUM(net_amount * exchange_rate) FROM orders"
            )
        },
    }

    result = IntentValidationAgent().execute(state)

    assert result["decision"] == "intent_valid"
    messages = mock_invoke.call_args.args[1]
    assert messages[1].content.endswith(evidence)
    assert "MUST follow every instruction" in messages[1].content
    assert "Evidence compliance is strict, not lenient" in messages[2].content


@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation._JOINS_VALIDATED_ELSEWHERE",
    False,
)
@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output",
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
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT SUM(converted_amount) FROM orders"
            )
        },
    }

    result = IntentValidationAgent().execute(state)

    assert result["decision"] == "intent_invalid"
    assert "Critical evidence issues" in result["path_state"]["error"]
    assert "net_amount * exchange_rate" in result["path_state"]["error"]
