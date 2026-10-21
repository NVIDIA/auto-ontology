# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from auto_ontology.retrieval.text_to_sql.agents.identifier_evidence import (
    IdentifierEvidenceAgent,
)


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


def test_adds_only_the_subject_tables_identifier() -> None:
    state = {
        "shorten_answer": True,
        "evidence": "Active means accounts.status = 'active'.",
        "connectors": [],
        "path_state": {
            "subject": "customer",
            "subject_table_ids": ["customers"],
            "target_db": "db",
            "relevant_tables": [
                _table("customers", "customers", "customer_number"),
                _table("accounts", "accounts", "account_id"),
            ],
        },
    }

    result = IdentifierEvidenceAgent().execute(state)

    assert result["evidence"].startswith(state["evidence"])
    assert '"db"."public"."customers"."customer_number"' in result["evidence"]
    assert "account_id" not in result["evidence"]


def test_does_not_add_identifier_evidence_for_regular_answers() -> None:
    state = {
        "shorten_answer": False,
        "connectors": [],
        "path_state": {
            "subject_table_ids": ["customers"],
            "relevant_tables": [_table("customers", "customers", "customer_number")],
        },
    }

    assert IdentifierEvidenceAgent().execute(state) == {}


def test_ambiguous_subject_tables_do_not_add_identifier_evidence() -> None:
    state = {
        "shorten_answer": True,
        "connectors": [],
        "path_state": {
            "subject_table_ids": ["customers-v1", "customers-v2"],
            "relevant_tables": [
                _table("customers-v1", "customers", "customer_number"),
                _table("customers-v2", "archived_customers", "customer_number"),
            ],
        },
    }

    assert IdentifierEvidenceAgent().execute(state) == {}


def test_primary_attribute_disambiguates_subject_tables() -> None:
    state = {
        "shorten_answer": True,
        "connectors": [],
        "path_state": {
            "subject": "customer",
            "subject_table_ids": ["customers-v1", "customers-v2"],
            "primary_attribute": {"table_id": "customers-v2"},
            "relevant_tables": [
                _table("customers-v1", "customers", "customer_number"),
                _table("customers-v2", "archived_customers", "archive_id"),
            ],
        },
    }

    result = IdentifierEvidenceAgent().execute(state)

    assert '"archived_customers"."archive_id"' in result["evidence"]
    assert '"customers"."customer_number"' not in result["evidence"]


def test_subject_table_without_identifier_does_not_borrow_another_tables() -> None:
    state = {
        "shorten_answer": True,
        "connectors": [],
        "path_state": {
            "subject_table_ids": ["customers"],
            "relevant_tables": [
                _table("customers", "customers", None),
                _table("accounts", "accounts", "account_id"),
            ],
        },
    }

    assert IdentifierEvidenceAgent().execute(state) == {}
