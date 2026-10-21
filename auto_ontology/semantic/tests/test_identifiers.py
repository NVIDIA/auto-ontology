# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from auto_ontology.semantic import identifiers
from auto_ontology.semantic.models import IdentifierSelectionResult


def _table(
    *,
    pk: list[str] | None = None,
    current: str | None = None,
    columns: list[dict] | None = None,
) -> dict:
    return {
        "id": "table-1",
        "name": "orders",
        "description": "One row per order.",
        "schema_name": "public",
        "pk": pk or [],
        "default_identifier_column_id": current,
        "columns": columns
        or [
            {
                "id": "column-1",
                "name": "order_number",
                "description": "Stable order identifier.",
                "data_type": "text",
                "is_unique": True,
            }
        ],
    }


def test_valid_existing_identifier_is_preserved_without_llm(monkeypatch) -> None:
    table = _table(current="column-1")
    monkeypatch.setattr(
        identifiers, "fetch_tables_for_identifier_selection", lambda _database: [table]
    )
    monkeypatch.setattr(
        identifiers,
        "get_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )
    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        identifiers,
        "set_default_table_identifier",
        lambda table_id, column_id: writes.append((table_id, column_id)),
    )

    assert identifiers.select_default_identifiers("database") == 0
    assert writes == []


def test_single_declared_pk_is_selected_without_llm(monkeypatch) -> None:
    table = _table(pk=["order_number"])
    monkeypatch.setattr(
        identifiers, "fetch_tables_for_identifier_selection", lambda _database: [table]
    )
    monkeypatch.setattr(
        identifiers,
        "get_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )
    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        identifiers,
        "set_default_table_identifier",
        lambda table_id, column_id: writes.append((table_id, column_id)) or True,
    )

    assert identifiers.select_default_identifiers("database") == 1
    assert writes == [("table-1", "column-1")]


def test_exact_id_is_selected_without_llm(monkeypatch) -> None:
    table = _table(
        columns=[
            {
                "id": "column-id",
                "name": "ID",
                "description": "",
                "data_type": "integer",
                "is_unique": None,
            },
            {
                "id": "column-name",
                "name": "name",
                "description": "",
                "data_type": "text",
                "is_unique": None,
            },
        ]
    )
    monkeypatch.setattr(
        identifiers, "fetch_tables_for_identifier_selection", lambda _database: [table]
    )
    monkeypatch.setattr(
        identifiers,
        "get_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )
    selected: list[str] = []
    monkeypatch.setattr(
        identifiers,
        "set_default_table_identifier",
        lambda _table_id, column_id: selected.append(column_id) or True,
    )

    assert identifiers.select_default_identifiers("database") == 1
    assert selected == ["column-id"]


def test_composite_pk_component_is_not_selected_as_identifier(monkeypatch) -> None:
    table = _table(
        pk=["id", "version"],
        columns=[
            {
                "id": "column-id",
                "name": "id",
                "description": "",
                "data_type": "integer",
                "is_unique": None,
            },
            {
                "id": "column-version",
                "name": "version",
                "description": "",
                "data_type": "integer",
                "is_unique": None,
            },
        ],
    )
    monkeypatch.setattr(
        identifiers, "fetch_tables_for_identifier_selection", lambda _database: [table]
    )
    monkeypatch.setattr(identifiers, "get_llm_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        identifiers,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: IdentifierSelectionResult(
            selected_column_id="column-id"
        ),
    )
    writes: list[str] = []
    monkeypatch.setattr(
        identifiers,
        "set_default_table_identifier",
        lambda _table_id, column_id: writes.append(column_id) or True,
    )

    assert identifiers.select_default_identifiers("database") == 0
    assert writes == []


def test_ambiguous_identifier_uses_validated_llm_choice(monkeypatch) -> None:
    table = _table()
    monkeypatch.setattr(
        identifiers, "fetch_tables_for_identifier_selection", lambda _database: [table]
    )
    monkeypatch.setattr(identifiers, "get_llm_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        identifiers,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: IdentifierSelectionResult(
            selected_column_id="column-1"
        ),
    )
    selected: list[str] = []
    monkeypatch.setattr(
        identifiers,
        "set_default_table_identifier",
        lambda _table_id, column_id: selected.append(column_id) or True,
    )

    assert identifiers.select_default_identifiers("database") == 1
    assert selected == ["column-1"]


def test_invalid_llm_column_is_ignored(monkeypatch) -> None:
    table = _table()
    monkeypatch.setattr(
        identifiers, "fetch_tables_for_identifier_selection", lambda _database: [table]
    )
    monkeypatch.setattr(identifiers, "get_llm_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        identifiers,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: IdentifierSelectionResult(
            selected_column_id="other-table-column"
        ),
    )
    writes: list[str] = []
    monkeypatch.setattr(
        identifiers,
        "set_default_table_identifier",
        lambda _table_id, column_id: writes.append(column_id) or True,
    )

    assert identifiers.select_default_identifiers("database") == 0
    assert writes == []
