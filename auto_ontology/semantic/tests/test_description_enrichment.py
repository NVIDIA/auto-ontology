# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-compilation description enrichment."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic import compile as compile_module
from auto_ontology.semantic import description_enrichment as de
from auto_ontology.semantic.models import ColumnDescription, TableDescriptionsResult


def _column(
    name: str,
    samples: list[str] | None = None,
    *,
    data_type: str = "TEXT",
    attribute_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"col-{name}",
        "name": name,
        "data_type": data_type,
        "sample_values": json.dumps(samples) if samples is not None else None,
        "attribute_id": attribute_id,
        "source_description": extra.pop("source_description", ""),
        **extra,
    }


def _orders() -> tuple[dict, list[dict], dict]:
    table = {
        "id": "t-orders",
        "name": "orders",
        "schema_name": "main",
        "source_description": "",
        "columns": [
            _column("StoreKey", ["10020030004567", "10020030004589"]),
            _column("Store Code", ["0004567", "0004589"], attribute_id="a-sc"),
            _column("Store Name", ["Main Street Market"], attribute_id="a-sn"),
        ],
    }
    links = [
        {
            "local_column": "StoreKey",
            "other_table_id": "t-stores",
            "other_table_name": "stores",
            "other_column": "StoreKey",
            "direction": "out",
        }
    ]
    neighbours = {
        "t-stores": {
            "id": "t-stores",
            "name": "stores",
            "description": "Retail stores.",
            "columns": [
                _column("StoreKey", ["10020030004567", "10020030004589"]),
                _column("Store", ["Main Street Market"]),
            ],
        }
    }
    return table, links, neighbours


def test_value_shape() -> None:
    assert de.value_shape(["0004567", "0004589"], "TEXT") == "7 digits stored as text"
    assert de.value_shape(["12", "345"], "INTEGER") == "2-3 digits"
    assert de.value_shape(["1.5", "20"], "TEXT") == "numbers stored as text"
    assert de.value_shape(["Y", "N"], "TEXT") == "1-letter upper-case codes"
    assert de.value_shape([], "TEXT") is None
    assert de.value_shape(["Oakland", "San Diego"], "TEXT") is None


def test_name_tokens_split_camel_and_spaces() -> None:
    assert de.name_tokens("StoreKey") == {"store", "key"}
    assert de.name_tokens("Store Code") == {"store", "code"}
    assert de.name_tokens("district_id") == {"district", "id"}


def test_hint_flags_a_code_that_is_not_the_join_key() -> None:
    table, links, neighbours = _orders()
    hints = de.collision_hints(table["columns"][1:], table, links, neighbours)

    assert len(hints) == 1
    hint = hints[0]
    assert hint.startswith("Store Code names table stores")
    assert "orders.StoreKey = stores.StoreKey" in hint
    assert "this column is 7 digits stored as text" in hint
    assert "stores.StoreKey is 14 digits stored as text" in hint


def test_no_hint_for_the_join_column_itself() -> None:
    table, links, neighbours = _orders()
    assert de.collision_hints(table["columns"][:1], table, links, neighbours) == []


def test_context_lists_joins_and_neighbour_columns() -> None:
    table, links, neighbours = _orders()
    context = de.build_table_context(table, links, neighbours)

    assert "Table: main.orders" in context
    assert "StoreKey -> stores.StoreKey (14 digits stored as text)" in context
    assert "Joined table: stores" in context
    assert "Description: Retail stores." in context
    assert "  - Store (TEXT) | samples: Main Street Market" in context


def test_resolve_column_names_accepts_echoed_annotation() -> None:
    result = TableDescriptionsResult(
        table_description="",
        columns=[
            ColumnDescription(column_name="Store Code (TEXT)", description="A."),
            ColumnDescription(column_name="unknown", description="B."),
            ColumnDescription(column_name="Store Name", description=""),
        ],
    )
    assert de.resolve_column_names(result, ["Store Code", "Store Name"]) == {
        "Store Code": "A."
    }


def _patch_reads(table: dict, links: list[dict], neighbours: dict):
    return (
        patch.object(de, "fetch_table_for_enrichment", return_value=table),
        patch.object(de, "fetch_table_links", return_value=links),
        patch.object(de, "fetch_neighbour_tables", return_value=neighbours),
    )


def test_enrich_writes_only_columns_with_an_attribute() -> None:
    table, links, neighbours = _orders()
    table["columns"][1]["format"] = "NNNNNNN"
    result = TableDescriptionsResult(
        table_description="One row per store and month.",
        columns=[
            ColumnDescription(column_name="Store Code", description="Local code."),
            ColumnDescription(column_name="Store Name", description="Name."),
            ColumnDescription(column_name="StoreKey", description="Should be ignored."),
        ],
    )
    written = {"table": True, "columns": ["col-Store Code"], "attributes": []}
    reads = _patch_reads(table, links, neighbours)
    with (
        reads[0],
        reads[1],
        reads[2],
        patch.object(de, "invoke_with_structured_output", return_value=result),
        patch.object(de, "get_llm_client"),
        patch.object(de, "write_enriched_descriptions", return_value=written) as write,
        patch.object(de, "_reembed") as reembed,
    ):
        assert de.enrich_table_descriptions("t-orders") is True

    write.assert_called_once_with(
        "t-orders",
        "One row per store and month.",
        {"col-Store Code": "Local code.", "col-Store Name": "Name."},
        {"a-sc": "Local code. — format: NNNNNNN", "a-sn": "Name."},
    )
    reembed.assert_called_once_with("t-orders", written, None)


def test_prompt_carries_existing_description_and_hint() -> None:
    table, links, neighbours = _orders()
    table["columns"][1]["source_description"] = "Store Code, 1 = x"
    reads = _patch_reads(table, links, neighbours)
    with (
        reads[0],
        reads[1],
        reads[2],
        patch.object(de, "invoke_with_structured_output", return_value=None) as llm,
        patch.object(de, "get_llm_client"),
        patch.object(de, "write_enriched_descriptions") as write,
    ):
        assert de.enrich_table_descriptions("t-orders") is False

    prompt = llm.call_args.args[1][1].content
    assert "existing description: Store Code, 1 = x" in prompt
    assert "Hints:\n- Store Code names table stores" in prompt
    write.assert_not_called()


def test_table_without_attributes_is_skipped() -> None:
    table, links, neighbours = _orders()
    for column in table["columns"]:
        column["attribute_id"] = None
    with (
        patch.object(de, "fetch_table_for_enrichment", return_value=table),
        patch.object(de, "invoke_with_structured_output") as llm,
    ):
        assert de.enrich_table_descriptions("t-orders") is False
    llm.assert_not_called()


def test_only_tables_that_got_a_term_are_enriched() -> None:
    with (
        patch.object(de, "fetch_tables_with_term", return_value={"t1", "t3"}),
        patch.object(de, "enrich_table_descriptions", return_value=True) as one,
    ):
        assert de.enrich_descriptions("db", ["t1", "t2", "t3"]) == 2
    assert sorted(c.args[0] for c in one.call_args_list) == ["t1", "t3"]


def test_cancelled_run_enriches_nothing() -> None:
    with (
        patch.object(de, "fetch_tables_with_term", return_value={"t1"}),
        patch.object(de, "is_cancelled", return_value=True),
        patch.object(de, "enrich_table_descriptions") as one,
    ):
        assert de.enrich_descriptions("db", ["t1"]) == 0
    one.assert_not_called()


@pytest.mark.parametrize("cancelled", [False, True])
def test_compile_enriches_the_tables_it_started_with(cancelled: bool) -> None:
    tables = [{"id": "t1", "name": "a"}, {"id": "t2", "name": "b"}]
    with (
        patch.object(compile_module, "load_domain_summary"),
        patch.object(compile_module, "build_semantic_embedder", return_value=None),
        patch.object(
            compile_module, "fetch_all_tables_without_term", return_value=tables
        ),
        patch.object(compile_module, "compile_semantic_layer", return_value=2) as c,
        patch.object(compile_module, "resolve_semantic_fks", return_value=0),
        patch(
            "auto_ontology.semantic.sql_attribute_suggester.suggest_sql_attributes",
            return_value=0,
        ),
        patch.object(
            compile_module, "build_bridge_tables_sql_attributes", return_value=0
        ),
        patch.object(
            compile_module,
            "is_cancelled",
            side_effect=[False, cancelled],
        ),
        patch.object(compile_module, "enrich_descriptions", return_value=2) as enrich,
    ):
        compile_module.run_semantic_compilation("db", domain_summary=MagicMock())

    assert c.call_args.kwargs["tables"] is tables
    if cancelled:
        enrich.assert_not_called()
    else:
        enrich.assert_called_once_with("db", ["t1", "t2"], embedder=None)
