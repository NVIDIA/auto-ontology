# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_from_semantic
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
)
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel


def _table(
    table_id: str,
    name: str,
    identifier: str | None,
) -> dict:
    return {
        "id": table_id,
        "database_name": "db",
        "schema_name": "public",
        "name": name,
        "default_identifier": identifier,
        "columns": [],
    }


def test_builds_instruction_only_for_the_subject_tables_identifier() -> None:
    path_state = {
        "subject": "customer",
        "subject_table_ids": ["customers"],
        "target_db": "db",
        "relevant_tables": [
            _table("customers", "customers", "customer_number"),
            _table("accounts", "accounts", "account_id"),
        ],
    }

    result = sql_from_semantic._subject_identifier_instruction(path_state, None)

    assert '"db"."public"."customers"."customer_number"' in result
    assert "account_id" not in result


def test_ambiguous_subject_tables_do_not_add_identifier_instruction() -> None:
    path_state = {
        "subject_table_ids": ["customers-v1", "customers-v2"],
        "relevant_tables": [
            _table("customers-v1", "customers", "customer_number"),
            _table("customers-v2", "archived_customers", "customer_number"),
        ],
    }

    assert sql_from_semantic._subject_identifier_instruction(path_state, None) == ""


def test_primary_attribute_disambiguates_subject_tables() -> None:
    path_state = {
        "subject": "customer",
        "subject_table_ids": ["customers-v1", "customers-v2"],
        "primary_attribute": {"table_id": "customers-v2"},
        "relevant_tables": [
            _table("customers-v1", "customers", "customer_number"),
            _table("customers-v2", "archived_customers", "archive_id"),
        ],
    }

    result = sql_from_semantic._subject_identifier_instruction(path_state, None)

    assert '"archived_customers"."archive_id"' in result
    assert '"customers"."customer_number"' not in result


def test_subject_table_without_identifier_does_not_borrow_another_tables() -> None:
    path_state = {
        "subject_table_ids": ["customers"],
        "relevant_tables": [
            _table("customers", "customers", None),
            _table("accounts", "accounts", "account_id"),
        ],
    }

    assert sql_from_semantic._subject_identifier_instruction(path_state, None) == ""


@pytest.mark.parametrize("shorten_answer", [False, True])
def test_identifier_instruction_is_added_inside_generation_only_for_short_answers(
    monkeypatch: pytest.MonkeyPatch,
    shorten_answer: bool,
) -> None:
    captured: list = []

    def fake_invoke(_llm, messages, _schema):
        captured.extend(messages)
        return SQLGenerationModel(
            thought="No assumptions.",
            sql_code="SELECT customer_number FROM customers",
            response="Returns customers.",
        )

    monkeypatch.setattr(
        sql_from_semantic,
        "_subject_identifier_instruction",
        lambda *_args: "SUBJECT IDENTIFIER INSTRUCTION",
    )
    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", fake_invoke
    )
    state = {
        "llm": object(),
        "reasoning_llm": object(),
        "initial_question": "Which customers are active?",
        "evidence": "Active customers have status = 'active'.",
        "shorten_answer": shorten_answer,
        "messages": [],
        "connectors": [],
        "path_state": {
            "retrieved_column_attributes": [],
            "relevant_tables": [],
            "relevant_queries": [],
        },
    }

    SQLFromCandidatesAgent().execute(state)

    rendered = "\n".join(message.content for message in captured)
    assert ("SUBJECT IDENTIFIER INSTRUCTION" in rendered) is shorten_answer
    if shorten_answer:
        assert any(
            message.content == "SUBJECT IDENTIFIER INSTRUCTION" for message in captured
        )
        assert state["evidence"] == "Active customers have status = 'active'."
