# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concept merge candidates and the concept's embedding row."""

from __future__ import annotations

from auto_ontology.semantic import concepts
from auto_ontology.semantic.constants import LABEL_CONCEPT
from auto_ontology.semantic.embed import _concept_row


def _concept(concept_id: str, table: str, *columns: tuple[str, str]) -> dict:
    return {
        "id": concept_id,
        "name": concept_id.title(),
        "description": None,
        "terms": [{"id": f"{concept_id}-term", "name": concept_id.title()}],
        "keys": {
            f"{table}-id": {
                "table_name": table,
                "schema_name": "public",
                "columns": [
                    {"id": f"{table}.{name}", "name": name, "data_type": data_type}
                    for name, data_type in columns
                ],
            }
        },
    }


def _pairs(*items: dict) -> set[tuple[str, str]]:
    return {
        (a["table_name"], b["table_name"])
        for a, b in concepts._candidate_pairs(concepts._key_entries(list(items)))
    }


def test_tables_keyed_on_the_same_name_are_candidates() -> None:
    assert _pairs(
        _concept("order", "orders", ("order_id", "integer")),
        _concept("detail", "order_details", ("ORDER_ID", "bigint")),
    ) == {("orders", "order_details")}


def test_composite_keys_match_regardless_of_column_order() -> None:
    assert _pairs(
        _concept("line", "lines", ("order_id", "int"), ("line_no", "int")),
        _concept("note", "line_notes", ("line_no", "int"), ("order_id", "int")),
    ) == {("lines", "line_notes")}


def test_generic_key_names_are_not_candidates() -> None:
    assert not _pairs(
        _concept("order", "orders", ("id", "integer")),
        _concept("customer", "customers", ("id", "integer")),
    )


def test_incompatible_types_are_not_candidates() -> None:
    assert not _pairs(
        _concept("order", "orders", ("order_ref", "integer")),
        _concept("legacy", "legacy_orders", ("order_ref", "varchar(20)")),
    )


def test_a_concept_is_not_a_candidate_with_itself() -> None:
    order = _concept("order", "orders", ("order_id", "int"))
    order["keys"]["details-id"] = {
        "table_name": "order_details",
        "schema_name": "public",
        "columns": [{"id": "d.order_id", "name": "order_id", "data_type": "int"}],
    }
    assert not _pairs(order)


def test_concept_row_names_its_terms_and_keys() -> None:
    row = _concept_row(
        "shop", _concept("line", "lines", ("order_id", "int"), ("line_no", "int"))
    )

    assert row["text"] == "Concept: Line. Terms: Line. Key: lines.(order_id, line_no)"
    assert row["metadata"]["label"] == LABEL_CONCEPT
    assert row["metadata"]["table_ids"] == ["lines-id"]
    assert row["metadata"]["database_name"] == "shop"
