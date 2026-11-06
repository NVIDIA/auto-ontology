# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table roles, grain and sibling notes in the SQL prompt."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from auto_ontology.retrieval.text_to_sql.agents import candidates_preparation as cp
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_tables_for_prompt,
    format_used_schema_for_prompt,
)


def _table(table_id: str, name: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": table_id,
        "name": name,
        "schema_name": "public",
        "database_name": "shop",
        "columns": [{"name": "ORDER_KEY", "data_type": "int"}],
        **extra,
    }


def test_role_and_grain_render_with_the_table() -> None:
    rendered = format_tables_for_prompt(
        [
            _table(
                "o",
                "orders",
                table_role="fact",
                grain="one row per order line",
            )
        ]
    )

    assert "role | grain" in rendered
    assert "fact | one row per order line" in rendered
    assert rendered.startswith("A table's role says how to join it.")
    assert "Do not join two fact tables" in rendered


def test_no_role_note_or_columns_without_roles() -> None:
    rendered = format_tables_for_prompt([_table("o", "orders")])

    assert "role" not in rendered
    assert "grain" not in rendered
    assert "similar to" not in rendered


def test_sibling_notes_name_the_other_table() -> None:
    sibling = {
        "id": "s",
        "name": "orders_summary",
        "schema_name": "public",
        "database_name": "shop",
        "note": "Use for order lines; the summary holds one row per order.",
    }
    rendered = format_tables_for_prompt(
        [
            _table("o", "orders", sibling_notes=[sibling]),
            _table("s", "orders_summary"),
        ]
    )

    assert (
        "similar to shop.public.orders_summary; when to use this table instead: "
        "Use for order lines; the summary "
        "holds one row per order." in rendered
    )
    assert "Some tables have near-identical columns" in rendered


def test_used_schema_carries_role_and_grain() -> None:
    rendered = format_used_schema_for_prompt(
        [_table("o", "orders", table_role="fact", grain="one row per order")],
        ["public.orders"],
        [],
    )

    assert "| role: fact | grain: one row per order" in rendered


def test_attach_keeps_only_siblings_in_scope() -> None:
    tables = [_table("o", "orders"), _table("s", "orders_summary")]
    semantics = {
        "o": {
            "role": "fact",
            "grain": "one row per order line",
            "siblings": [
                {"id": "s", "name": "orders_summary", "note": "lines"},
                {"id": "x", "name": "orders_archive", "note": "history"},
            ],
        },
        "s": {"role": "fact", "grain": None, "siblings": []},
    }
    with patch.object(cp, "fetch_table_semantics", return_value=semantics):
        cp.attach_table_semantics(tables)

    orders, summary = tables
    assert orders["table_role"] == "fact"
    assert orders["grain"] == "one row per order line"
    assert [s["id"] for s in orders["sibling_notes"]] == ["s"]
    assert summary["table_role"] == "fact"
    assert "grain" not in summary
    assert "sibling_notes" not in summary


def test_attach_leaves_tables_without_semantics_alone() -> None:
    tables = [_table("o", "orders")]
    with patch.object(cp, "fetch_table_semantics", return_value={}):
        cp.attach_table_semantics(tables)

    assert tables == [_table("o", "orders")]
