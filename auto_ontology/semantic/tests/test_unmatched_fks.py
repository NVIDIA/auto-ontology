# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for attaching unmatched suspected FK columns as ColumnAttributes."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic import unmatched_fks
from auto_ontology.semantic.models import (
    TableTermsResult,
    TermAttributeAssignment,
    TermProposal,
)

_MODULE = "auto_ontology.semantic.unmatched_fks"

_TABLES = {
    "employees": {
        "id": "employees",
        "name": "employees",
        "schema_name": "hr",
        "description": "Staff",
        "pk": ["id"],
    },
    "employee_project": {
        "id": "employee_project",
        "name": "employee_project",
        "schema_name": "hr",
        "description": "",
        "pk": None,
    },
}
_CONTEXTS = {
    "employees": {
        "columns": [
            {"name": "id", "data_type": "integer", "description": "Employee id"},
            {
                "name": "manager_id",
                "data_type": "integer",
                "description": "Reports to",
                "sample_values": "[7, 9]",
            },
            {"name": "legacy_ref", "data_type": "text", "description": "Old ref"},
        ],
        "fks": [],
    },
    "employee_project": {
        "columns": [
            {"name": "employee_id", "data_type": "integer", "description": "Who"},
            {"name": "project_id", "data_type": "integer", "description": "What"},
        ],
        "fks": [],
    },
}


def _unlinked(table: str, name: str) -> dict:
    return {
        "id": f"{table}.{name}",
        "name": name,
        "table_id": table,
        "table_name": table,
    }


@pytest.fixture
def store() -> Iterator[dict[str, MagicMock]]:
    terms: dict[str, list[dict]] = {"employees": [{"id": "t1", "name": "Employee"}]}
    with (
        patch(f"{_MODULE}.fetch_table_by_id", side_effect=_TABLES.get),
        patch(f"{_MODULE}.fetch_table_context", side_effect=_CONTEXTS.__getitem__),
        patch(
            f"{_MODULE}.fetch_terms_and_attributes_for_table",
            side_effect=lambda table_id: (terms.get(table_id, []), []),
        ),
        patch(f"{_MODULE}.merge_column_attribute", return_value="attr") as merge,
        patch(
            "auto_ontology.semantic.visit_enter.merge_column_attribute",
            return_value="attr",
        ) as commit_merge,
        patch(
            "auto_ontology.semantic.visit_enter.upsert_table_term",
            return_value=("t2", "Employee Assignment"),
        ) as upsert,
        patch(f"{_MODULE}.extract_term") as extract,
        patch(f"{_MODULE}.find_unlinked_fk_columns") as unlinked,
    ):
        yield {
            "merge": merge,
            "commit_merge": commit_merge,
            "upsert": upsert,
            "extract": extract,
            "unlinked": unlinked,
        }


def test_self_reference_joins_the_tables_term(store) -> None:
    store["unlinked"].return_value = [
        _unlinked("employees", "manager_id"),
    ]
    embedder = MagicMock()
    embedder.embed_term_rows.return_value = [{"metadata": {"id": "t1"}}]

    with patch(f"{_MODULE}.write_embedded_rows") as write:
        assert unmatched_fks.attach_unmatched_fk_columns("hr", embedder) == 1

    store["merge"].assert_called_once_with(
        term_name="Employee",
        table_id="employees",
        source_column="manager_id",
        attr_name="managerId",
        datatype="integer",
    )
    store["extract"].assert_not_called()
    embedder.embed_term_rows.assert_called_once_with(
        {"id": "t1", "name": "Employee"}, []
    )
    embedder.vdb.delete_by_id.assert_called_once_with("t1")
    write.assert_called_once_with(embedder.vdb, [{"metadata": {"id": "t1"}}])


def _embed_with(rows_or_error: list | Exception) -> tuple[MagicMock, MagicMock]:
    terms = [{"id": "t1", "name": "Employee"}]
    attrs = [
        {"id": "a1", "term_name": "Employee"},
        {"id": "a2", "term_name": "Employee"},
    ]
    embedder = MagicMock()
    if isinstance(rows_or_error, Exception):
        embedder.embed_term_rows.side_effect = rows_or_error
    else:
        embedder.embed_term_rows.return_value = rows_or_error
    with (
        patch(
            f"{_MODULE}.fetch_terms_and_attributes_for_table",
            return_value=(terms, attrs),
        ),
        patch(f"{_MODULE}.write_embedded_rows") as write,
    ):
        unmatched_fks._embed_table_terms(_TABLES["employees"], embedder)
    return embedder, write


@pytest.mark.parametrize("failure", [[], RuntimeError("embed endpoint down")])
def test_a_failed_reembed_keeps_the_old_vectors(failure: list | Exception) -> None:
    embedder, write = _embed_with(failure)

    embedder.vdb.delete_by_id.assert_not_called()
    write.assert_not_called()


def test_only_nodes_with_a_replacement_lose_their_old_vector() -> None:
    rows = [{"metadata": {"id": "t1"}}, {"metadata": {"id": "a1"}}]
    embedder, write = _embed_with(rows)

    assert [c.args[0] for c in embedder.vdb.delete_by_id.call_args_list] == [
        "t1",
        "a1",
    ]
    write.assert_called_once_with(embedder.vdb, rows)


def test_table_without_a_term_gets_one(store) -> None:
    store["unlinked"].return_value = [
        _unlinked("employee_project", "employee_id"),
        _unlinked("employee_project", "project_id"),
    ]
    store["extract"].return_value = TableTermsResult(
        terms=[
            TermProposal(
                name="Employee Assignment",
                description="Who works on what",
                attributes=[
                    TermAttributeAssignment(
                        source_column="employee_id", display_name="Employee"
                    ),
                    TermAttributeAssignment(
                        source_column="project_id", display_name="Project"
                    ),
                ],
            )
        ]
    )

    assert unmatched_fks.attach_unmatched_fk_columns("hr") == 2

    specs = store["extract"].call_args.args[2]
    assert [spec.source_column for spec in specs] == ["employee_id", "project_id"]
    store["upsert"].assert_called_once_with(
        "Employee Assignment",
        "Who works on what",
        "employee_project",
        synonyms=[],
    )
    assert [c.kwargs["attr_name"] for c in store["commit_merge"].call_args_list] == [
        "Employee",
        "Project",
    ]
    store["merge"].assert_not_called()


def test_nothing_unlinked_writes_nothing(store) -> None:
    store["unlinked"].return_value = []

    assert unmatched_fks.attach_unmatched_fk_columns("hr") == 0

    store["merge"].assert_not_called()
    store["extract"].assert_not_called()


def test_one_failing_table_does_not_stop_the_rest(store) -> None:
    store["unlinked"].return_value = [
        _unlinked("employee_project", "employee_id"),
        _unlinked("employees", "manager_id"),
    ]
    store["extract"].side_effect = RuntimeError("LLM down")

    assert unmatched_fks.attach_unmatched_fk_columns("hr") == 1
    store["merge"].assert_called_once()


def test_cancellation_stops_between_tables(store) -> None:
    store["unlinked"].return_value = [_unlinked("employees", "manager_id")]
    with patch(f"{_MODULE}.is_cancelled", return_value=True):
        assert unmatched_fks.attach_unmatched_fk_columns("hr") == 0

    store["merge"].assert_not_called()
