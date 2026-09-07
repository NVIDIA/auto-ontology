# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catalog embedding-row builder.

These rows are what actually gets embedded, so the text templates are pinned
character-for-character below. A change that looks cosmetic here silently
invalidates every stored embedding — the assertions are deliberately exact
rather than "contains the table name".
"""

from __future__ import annotations

import pandas as pd
import pytest

from gsf.catalog.constants import Labels
from gsf.utils.embedding_rows import (
    CatalogEmbeddingRowsOp,
    build_column_text,
    build_embed_row,
    build_table_text,
)

_COLS = ["text", "_embed_modality", "path", "page_number", "metadata"]


def _op(database_name: str = "db") -> CatalogEmbeddingRowsOp:
    return CatalogEmbeddingRowsOp(database_name=database_name)


def _tables(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── Text templates ───────────────────────────────────────────────────────────


def test_table_text_is_exact() -> None:
    text = build_table_text(
        table_name="orders",
        table_description="All orders.",
        columns=[{"column_name": "id", "data_type": "int", "description": "PK"}],
        schema_name="public",
        database_name="shop",
    )
    assert text == (
        "database_name: shop, schema_name: public, table_name: orders"
        ", table_description: All orders."
        ", columns: {name: id, data_type: int, description: PK}"
    )


def test_table_text_omits_blank_description_but_keeps_columns_key() -> None:
    """An empty ``columns:`` tail is still emitted — the key is unconditional."""
    text = build_table_text(
        table_name="t",
        table_description="",
        columns=[],
        schema_name="s",
        database_name="d",
    )
    assert text == "database_name: d, schema_name: s, table_name: t, columns: "


def test_column_text_is_exact() -> None:
    text = build_column_text(
        column_name="total",
        column_description="Order total.",
        data_type="numeric",
        sample_values=[1, 2],
        table_name="orders",
        schema_name="public",
        database_name="shop",
    )
    assert text == (
        "database_name: shop, schema_name: public, table_name: orders"
        ", column_name: total, data_type: numeric"
        ", column_description: Order total."
        ", sample_values: 1, 2"
    )


def test_column_text_drops_empty_optional_parts() -> None:
    text = build_column_text(
        column_name="c",
        column_description="",
        data_type="text",
        sample_values=[],
        table_name="t",
        schema_name="s",
        database_name="d",
    )
    assert text == (
        "database_name: d, schema_name: s, table_name: t"
        ", column_name: c, data_type: text"
    )


# ── Row shape ────────────────────────────────────────────────────────────────


def test_row_uses_the_gsf_path_prefix() -> None:
    """The prefix is provenance only, but it must match the rest of GSF.

    Every other emit site in the codebase writes ``gsf:``; a row builder that
    disagreed would make stored provenance depend on which code path ingested
    it.
    """
    row = build_embed_row(
        text="  t  ",
        node_id="abc",
        label=Labels.TABLE,
        name="n",
        schema_name="s",
        database_name="d",
    )
    assert row["path"] == "gsf:abc"
    assert row["metadata"]["source_path"] == "gsf:abc"
    assert row["text"] == "t"
    assert row["page_number"] == -1
    assert row["_embed_modality"] == "text"


def test_row_falls_back_when_node_id_is_missing() -> None:
    for missing in (None, ""):
        row = build_embed_row(
            text="t",
            node_id=missing,
            label=Labels.COLUMN,
            name="n",
            schema_name="s",
            database_name="d",
        )
        assert row["path"] == "gsf:unknown"


def test_identifiers_are_duplicated_into_content_metadata() -> None:
    """The nesting is load-bearing: only content_metadata survives the VDB write."""
    row = build_embed_row(
        text="t",
        node_id="i",
        label=Labels.TABLE,
        name="n",
        schema_name="s",
        database_name="d",
    )
    meta = row["metadata"]
    nested = meta["content_metadata"]
    for key in ("id", "label", "name", "source_path", "schema_name", "database_name"):
        assert meta[key] == nested[key]
    assert nested is not meta, "content_metadata must be a copy, not an alias"


# ── Operator ─────────────────────────────────────────────────────────────────


def test_process_emits_one_table_row_and_one_row_per_column() -> None:
    tables = _tables(
        {
            "id": "t1",
            "table_name": "orders",
            "table_schema": "public",
            "description": "d",
        }
    )
    columns = pd.DataFrame(
        [
            {
                "id": "c1",
                "table_name": "orders",
                "table_schema": "public",
                "column_name": "id",
                "data_type": "int",
                "description": None,
                "sample_values": None,
            },
            {
                "id": "c2",
                "table_name": "orders",
                "table_schema": "public",
                "column_name": "total",
                "data_type": "numeric",
                "description": None,
                "sample_values": None,
            },
        ]
    )
    out = _op().process((tables, columns))
    assert list(out.columns) == _COLS
    labels = [r["metadata"]["label"] for _, r in out.iterrows()]
    assert labels == [Labels.TABLE, Labels.COLUMN, Labels.COLUMN]


def test_same_table_name_in_two_schemas_does_not_merge_columns() -> None:
    """The bucketing key includes the schema.

    Keyed on table name alone, ``public.users`` would absorb ``other.users``'s
    columns into its embedding text and every column row would be emitted
    twice.
    """
    tables = _tables(
        {
            "id": "t1",
            "table_name": "users",
            "table_schema": "public",
            "description": "",
        },
        {"id": "t2", "table_name": "users", "table_schema": "other", "description": ""},
    )
    columns = pd.DataFrame(
        [
            {
                "id": "c1",
                "table_name": "users",
                "table_schema": "public",
                "column_name": "email",
                "data_type": "text",
                "description": None,
                "sample_values": None,
            },
            {
                "id": "c2",
                "table_name": "users",
                "table_schema": "other",
                "column_name": "ssn",
                "data_type": "text",
                "description": None,
                "sample_values": None,
            },
        ]
    )
    out = _op().process((tables, columns))
    assert len(out) == 4, "two table rows + exactly one column row each"

    by_id = {r["metadata"]["id"]: r for _, r in out.iterrows()}
    assert "email" in by_id["t1"]["text"] and "ssn" not in by_id["t1"]["text"]
    assert "ssn" in by_id["t2"]["text"] and "email" not in by_id["t2"]["text"]
    assert by_id["c1"]["metadata"]["schema_name"] == "public"
    assert by_id["c2"]["metadata"]["schema_name"] == "other"


def test_bucketing_is_case_insensitive() -> None:
    tables = _tables(
        {
            "id": "t1",
            "table_name": "Orders",
            "table_schema": "Public",
            "description": "",
        }
    )
    columns = pd.DataFrame(
        [
            {
                "id": "c1",
                "table_name": "orders",
                "table_schema": "public",
                "column_name": "id",
                "data_type": "int",
                "description": None,
                "sample_values": None,
            }
        ]
    )
    out = _op().process((tables, columns))
    assert len(out) == 2, "the column must attach despite the case difference"


def test_sample_values_are_capped_at_five() -> None:
    tables = _tables(
        {"id": "t1", "table_name": "t", "table_schema": "s", "description": ""}
    )
    columns = pd.DataFrame(
        [
            {
                "id": "c1",
                "table_name": "t",
                "table_schema": "s",
                "column_name": "c",
                "data_type": "int",
                "description": None,
                "sample_values": [[1, 2, 3, 4, 5, 6, 7]][0],
            }
        ]
    )
    out = _op().process((tables, columns))
    column_text = out.iloc[1]["text"]
    assert column_text.endswith("sample_values: 1, 2, 3, 4, 5")
    assert "6" not in column_text.split("sample_values: ")[1]


def test_nan_descriptions_do_not_leak_the_string_nan() -> None:
    """``pd.isna`` guards exist because a missing description reads as NaN."""
    tables = _tables(
        {
            "id": "t1",
            "table_name": "t",
            "table_schema": "s",
            "description": float("nan"),
        }
    )
    columns = pd.DataFrame(
        [
            {
                "id": "c1",
                "table_name": "t",
                "table_schema": "s",
                "column_name": "c",
                "data_type": float("nan"),
                "description": float("nan"),
                "sample_values": None,
            }
        ]
    )
    out = _op().process((tables, columns))
    for text in out["text"]:
        assert "nan" not in text.lower()


def test_empty_input_returns_the_declared_columns() -> None:
    """Downstream concatenates these frames; a bare DataFrame() would break it."""
    out = _op().process((pd.DataFrame(), pd.DataFrame()))
    assert out.empty
    assert list(out.columns) == _COLS


def test_non_tuple_input_names_the_database() -> None:
    with pytest.raises(TypeError, match="shop"):
        _op("shop").process(pd.DataFrame())


def test_preprocess_and_postprocess_are_pass_through() -> None:
    op = _op()
    sentinel = object()
    assert op.preprocess(sentinel) is sentinel
    assert op.postprocess(sentinel) is sentinel
