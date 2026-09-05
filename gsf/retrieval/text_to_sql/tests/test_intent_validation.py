"""Tests for semantic context in intent validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gsf.retrieval.text_to_sql.agents.intent_validation import (
    ConstraintDisposition,
    IntentValidationAgent,
    IntentValidationModel,
)
from gsf.retrieval.text_to_sql.text_to_sql_graph import route_sql_validation


@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output",
    return_value=IntentValidationModel(
        is_valid=True, constraint_validation_complete=True
    ),
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


def _validation_state(question: str, sql: str) -> dict:
    return {
        "llm": MagicMock(),
        "initial_question": question,
        "path_state": {
            "processing_question": question,
            "normalized_question": question,
            "sql_generation_result": SimpleNamespace(sql_code=sql),
        },
    }


def _violation(kind: str, source: str, explanation: str) -> ConstraintDisposition:
    return ConstraintDisposition(
        constraint_type=kind,
        source="submitted_question",
        source_text=source,
        status="violated",
        explanation=explanation,
    )


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_unsupported_status_filter_is_a_typed_intent_failure(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=True,
        constraint_validation_complete=True,
        constraint_dispositions=[
            _violation(
                "literal",
                "No status was requested",
                "SQL invents status = 'active'",
            )
        ],
    )

    result = IntentValidationAgent().execute(
        _validation_state("List suppliers", "SELECT * FROM suppliers WHERE status='active'")
    )

    assert result["decision"] == "intent_invalid"
    assert result["path_state"]["intent_constraint_dispositions"][0]["status"] == "violated"
    assert "invents status" in result["path_state"]["error"]


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_label_rewrite_is_a_typed_intent_failure(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=False,
        constraint_validation_complete=True,
        constraint_dispositions=[
            _violation("literal", "label = Enterprise", "SQL filters label = Commercial")
        ],
    )

    result = IntentValidationAgent().execute(
        _validation_state(
            "Show Enterprise accounts",
            "SELECT * FROM accounts WHERE label='Commercial'",
        )
    )

    assert result["decision"] == "intent_invalid"
    assert "label = Enterprise" in result["path_state"]["error"]


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_spurious_distinct_is_a_typed_intent_failure(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=True,
        constraint_validation_complete=True,
        constraint_dispositions=[
            _violation(
                "distinctness",
                "Return every transaction",
                "DISTINCT collapses repeated transaction amounts",
            )
        ],
    )

    result = IntentValidationAgent().execute(
        _validation_state(
            "Return every transaction amount", "SELECT DISTINCT amount FROM transactions"
        )
    )

    assert result["decision"] == "intent_invalid"


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_formula_drift_is_a_typed_intent_failure(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=True,
        constraint_validation_complete=True,
        constraint_dispositions=[
            _violation(
                "formula",
                "margin = revenue - cost",
                "SQL divides revenue by cost",
            )
        ],
    )

    result = IntentValidationAgent().execute(
        _validation_state(
            "Compute margin as revenue minus cost",
            "SELECT revenue / cost AS margin FROM sales",
        )
    )

    assert result["decision"] == "intent_invalid"


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_retained_constraints_are_observable_and_pass(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=True,
        constraint_validation_complete=True,
        constraint_dispositions=[
            ConstraintDisposition(
                constraint_type="ordering",
                source="submitted_question",
                source_text="top 5 by descending revenue",
                status="retained",
                explanation="ORDER BY revenue DESC LIMIT 5",
            )
        ],
    )

    result = IntentValidationAgent().execute(
        _validation_state(
            "Top 5 suppliers by revenue",
            "SELECT supplier, SUM(revenue) revenue FROM sales GROUP BY supplier "
            "ORDER BY revenue DESC LIMIT 5",
        )
    )

    assert result["decision"] == "intent_valid"
    assert result["path_state"]["intent_constraint_dispositions"][0]["status"] == "retained"


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_incomplete_constraint_validation_fails_closed(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=True,
        constraint_validation_complete=False,
    )

    result = IntentValidationAgent().execute(
        _validation_state("List suppliers", "SELECT * FROM suppliers")
    )

    assert result["decision"] == "intent_invalid"
    assert result["path_state"]["intent_constraint_validation_complete"] is False
    assert "incomplete" in result["path_state"]["error"]


@patch("gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output")
def test_processing_question_constraint_keeps_its_source(
    mock_invoke: MagicMock,
) -> None:
    mock_invoke.return_value = IntentValidationModel(
        is_valid=True,
        constraint_validation_complete=True,
        constraint_dispositions=[
            ConstraintDisposition(
                constraint_type="literal",
                source="processing_question",
                source_text="supplier status is active",
                status="retained",
                explanation="SQL uses status = 'active'",
            )
        ],
    )
    state = _validation_state(
        "What about active ones?",
        "SELECT * FROM suppliers WHERE status = 'active'",
    )
    state["path_state"]["processing_question"] = "List active suppliers"
    state["path_state"]["normalized_question"] = "suppliers with active status"

    result = IntentValidationAgent().execute(state)

    assert result["decision"] == "intent_valid"
    disposition = result["path_state"]["intent_constraint_dispositions"][0]
    assert disposition["source"] == "processing_question"
    prompt = mock_invoke.call_args.args[1][1].content
    assert "Submitted question:\nWhat about active ones?" in prompt
    assert "Standalone processing question:\nList active suppliers" in prompt
    assert "Normalized retrieval question:\nsuppliers with active status" in prompt


@patch(
    "gsf.retrieval.text_to_sql.agents.intent_validation.invoke_with_structured_output",
    return_value=None,
)
def test_unavailable_constraint_validation_fails_closed(
    mock_invoke: MagicMock,
) -> None:
    result = IntentValidationAgent().execute(
        _validation_state("List suppliers", "SELECT * FROM suppliers")
    )

    assert result["decision"] == "intent_invalid"
    assert result["path_state"]["intent_constraint_dispositions"] == []
    assert result["path_state"]["intent_constraint_validation_complete"] is False
    assert "unavailable" in result["path_state"]["error"]


def test_repeated_unavailable_validation_never_skips_to_execution() -> None:
    state = {
        "decision": "valid_sql",
        "path_state": {
            "reconstruction_count": 6,
            "intent_constraint_validation_complete": False,
        },
    }

    assert route_sql_validation(state) == "unconstructable"
