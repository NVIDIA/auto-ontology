# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from auto_ontology.semantic import junction_classifier
from auto_ontology.semantic.junction_classifier import classify_junction_tables


def _column(name: str) -> dict[str, Any]:
    return {"name": name, "data_type": "text", "description": ""}


def test_classifies_structural_and_payload_junctions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = [
        {
            "id": "pure",
            "name": "pure_link",
            "schema_name": "main",
            "pk": [],
        },
        {
            "id": "payload",
            "name": "payload_link",
            "schema_name": "main",
            "pk": ["id"],
        },
        {
            "id": "detail",
            "name": "detail",
            "schema_name": "main",
            "pk": ["id"],
        },
    ]
    contexts = {
        "pure": {
            "columns": [_column("left_id"), _column("right_id")],
            "fks": [
                {
                    "source_column": "left_id",
                    "target_table": "entity",
                    "target_column": "id",
                },
                {
                    "source_column": "right_id",
                    "target_table": "entity",
                    "target_column": "id",
                },
            ],
        },
        "payload": {
            "columns": [
                _column("id"),
                _column("owner_id"),
                _column("member_id"),
                _column("status"),
            ],
            "fks": [
                {
                    "source_column": "owner_id",
                    "target_table": "owner",
                    "target_column": "id",
                },
                {
                    "source_column": "member_id",
                    "target_table": "member",
                    "target_column": "id",
                },
            ],
        },
        "detail": {
            "columns": [_column("id"), _column("parent_id"), _column("description")],
            "fks": [
                {
                    "source_column": "parent_id",
                    "target_table": "parent",
                    "target_column": "id",
                }
            ],
        },
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        junction_classifier,
        "fetch_table_headers_for_database",
        lambda _database_name: tables,
    )
    monkeypatch.setattr(
        junction_classifier,
        "fetch_table_context",
        lambda table_id: contexts[table_id],
    )
    monkeypatch.setattr(
        junction_classifier,
        "fetch_semantic_fk_join_groups",
        lambda _database_name: [],
    )

    def fake_invoke(_llm: object, messages: list[Any], schema: type) -> Any:
        captured["prompt"] = messages[1].content
        return schema(
            decisions=[
                {
                    "table_id": "payload",
                    "is_junction_table": True,
                    "rationale": "The row grain is the relationship.",
                }
            ]
        )

    monkeypatch.setattr(
        junction_classifier, "get_llm_client", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        junction_classifier,
        "invoke_with_structured_output",
        fake_invoke,
    )
    monkeypatch.setattr(
        junction_classifier,
        "replace_junction_table_classifications",
        lambda database_name, rows: captured.update(
            database_name=database_name,
            rows=rows,
        ),
    )

    assert classify_junction_tables("db") == 2
    assert captured["database_name"] == "db"
    assert {row["table_id"]: row["kind"] for row in captured["rows"]} == {
        "pure": "bridge",
        "payload": "association",
    }
    assert "table_id: payload" in captured["prompt"]
    assert "table_id: detail" not in captured["prompt"]


def test_shared_parent_children_are_not_junctions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = [
        {"id": "left", "name": "left_detail", "schema_name": "main", "pk": []},
        {"id": "right", "name": "right_detail", "schema_name": "main", "pk": []},
    ]
    contexts = {
        table["id"]: {
            "columns": [_column("parent_id"), _column("value")],
            "fks": [
                {
                    "source_column": "parent_id",
                    "target_table": "parent",
                    "target_column": "id",
                }
            ],
        }
        for table in tables
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        junction_classifier,
        "fetch_table_headers_for_database",
        lambda _database_name: tables,
    )
    monkeypatch.setattr(
        junction_classifier,
        "fetch_table_context",
        lambda table_id: contexts[table_id],
    )
    monkeypatch.setattr(
        junction_classifier,
        "fetch_semantic_fk_join_groups",
        lambda _database_name: [],
    )
    monkeypatch.setattr(
        junction_classifier,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: pytest.fail("LLM should not be called"),
    )
    monkeypatch.setattr(
        junction_classifier,
        "replace_junction_table_classifications",
        lambda database_name, rows: captured.update(
            database_name=database_name,
            rows=rows,
        ),
    )

    assert classify_junction_tables("db") == 0
    assert captured["rows"] == []
