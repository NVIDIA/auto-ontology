# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for refreshing catalog Table/Column vectors after an edit."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from auto_ontology.utils import catalog_reembed

_MODULE = "auto_ontology.utils.catalog_reembed"

_TABLES = pd.DataFrame(
    [{"id": "t1", "table_name": "orders", "table_schema": "main", "description": ""}]
)
_COLUMNS = pd.DataFrame(
    [
        {
            "id": cid,
            "table_name": "orders",
            "table_schema": "main",
            "column_name": name,
            "data_type": "INTEGER",
            "description": "",
            "sample_values": [],
        }
        for cid, name in (("c1", "order_id"), ("c2", "total"))
    ]
)


def test_a_column_alone_is_refreshed_through_its_table() -> None:
    vdb = MagicMock()
    with (
        patch(f"{_MODULE}.fetch_parent_table_id_for_column", return_value="t1"),
        patch(
            f"{_MODULE}.fetch_tables_and_columns_by_node_ids",
            return_value=(_TABLES, _COLUMNS, "db"),
        ) as fetch,
        patch(f"{_MODULE}.get_embed_params"),
        patch(f"{_MODULE}.embed_rows", side_effect=lambda rows, *_, **__: rows),
        patch(f"{_MODULE}.get_data_vdb", return_value=vdb),
        patch(f"{_MODULE}.replace_embedded_rows", return_value=1) as replace,
    ):
        catalog_reembed.refresh_catalog_embeddings(["c2"])

    fetch.assert_called_once_with(["c2", "t1"])
    store, ids, written = replace.call_args.args
    assert store is vdb
    assert ids == {"c2"}
    assert [row["metadata"]["id"] for row in written] == ["c2"]


def test_a_partly_written_refresh_raises() -> None:
    with (
        patch(f"{_MODULE}.fetch_parent_table_id_for_column", return_value=None),
        patch(
            f"{_MODULE}.fetch_tables_and_columns_by_node_ids",
            return_value=(_TABLES, _COLUMNS, "db"),
        ),
        patch(f"{_MODULE}.get_embed_params"),
        patch(f"{_MODULE}.embed_rows", side_effect=lambda rows, *_, **__: rows),
        patch(f"{_MODULE}.get_data_vdb"),
        patch(f"{_MODULE}.replace_embedded_rows", return_value=1),
        pytest.raises(RuntimeError, match="1/2 catalog rows"),
    ):
        catalog_reembed.refresh_catalog_embeddings(["c1", "c2"])
