# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-table semantic foreign-key resolution."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from auto_ontology.semantic import semantic_fk
from auto_ontology.semantic.embed import _build_rows
from auto_ontology.semantic.models import SemanticFkJoinDefaultSelection


def _hit(
    attr_id: str,
    table_id: str,
    *,
    score: float = 0.5,
    table_name: str = "target_table",
    schema_name: str = "public",
    source_column: str = "id",
) -> dict:
    return {
        "metadata": {
            "id": attr_id,
            "table_id": table_id,
            "table_name": table_name,
            "schema_name": schema_name,
            "source_column": source_column,
            "is_unique": True,
        },
        "_distance": score,
        "text": f"ColumnAttribute {attr_id}",
    }


def test_format_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert semantic_fk._format_sample_values('["1", "2"]') == "sample_values: 1, 2"
    # Current writers store a native list.
    assert semantic_fk._format_sample_values(["1", "2"]) == "sample_values: 1, 2"
    assert semantic_fk._format_sample_values(None) == ""
    assert semantic_fk._format_sample_values(["1", None, "2"]) == "sample_values: 1, 2"
    assert (
        semantic_fk._format_sample_values('["1", null, "2"]') == "sample_values: 1, 2"
    )


def _key_row(
    attr_id: str,
    table_id: str,
    values: list | None,
    *,
    table_name: str = "target_table",
    column_name: str = "id",
) -> dict:
    return {
        "attribute_id": attr_id,
        "column_id": f"{attr_id}-column",
        "column_name": column_name,
        "data_type": "integer",
        "distinct_values": json.dumps(values) if values is not None else None,
        "table_id": table_id,
        "table_name": table_name,
        "schema_name": "public",
    }


def _keys(
    rows: list[dict], connector: MagicMock | None = None
) -> semantic_fk._KeyIndex:
    with patch(
        "auto_ontology.semantic.semantic_fk.fetch_value_key_candidates",
        return_value=rows,
    ):
        return semantic_fk._KeyIndex("database", connector or MagicMock())


def _fk_column(values: list | None, *, name: str = "customer_ref") -> dict:
    return {
        "id": "source-column",
        "name": name,
        "table_id": "table-1",
        "table_name": "orders",
        "schema_name": "public",
        "distinct_values": json.dumps(values) if values is not None else None,
    }


def _retriever(*hits: dict) -> MagicMock:
    retriever = MagicMock()
    retriever.query.side_effect = [list(hits), []]
    return retriever


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="other-attr")
def test_vdb_resolution_excludes_same_table_candidates(
    mock_llm_pick: MagicMock,
) -> None:
    retriever = MagicMock()
    retriever.query.side_effect = [
        [_hit("same-attr", "table-1"), _hit("other-attr", "table-2")],
        [_hit("same-attr", "table-1")],
    ]
    column = {
        "id": "source-column",
        "name": "id",
        "table_id": "table-1",
        "table_name": "tags",
    }

    selected = semantic_fk._resolve_via_vdb(column, retriever, "database")

    assert selected == "other-attr"
    assert mock_llm_pick.call_args.args[1] == [_hit("other-attr", "table-2")]
    assert all(
        call.kwargs["vdb_kwargs"]["where"]["is_unique"] is True
        for call in retriever.query.call_args_list
    )


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit")
def test_vdb_resolution_skips_llm_when_only_same_table_candidate_exists(
    mock_llm_pick: MagicMock,
) -> None:
    retriever = MagicMock()
    retriever.query.side_effect = [[_hit("same-attr", "table-1")], []]
    column = {
        "id": "source-column",
        "name": "id",
        "table_id": "table-1",
        "table_name": "tags",
    }

    assert semantic_fk._resolve_via_vdb(column, retriever, "database") is None
    mock_llm_pick.assert_not_called()


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="picked")
def test_pick_stands_unconfirmed_when_fk_column_has_no_values(
    _mock_llm_pick: MagicMock,
) -> None:
    keys = _keys([_key_row("picked", "table-2", [999])])

    selected = semantic_fk._resolve_via_vdb(
        _fk_column(None), _retriever(_hit("picked", "table-2")), "database", keys
    )

    assert selected == "picked"


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="picked")
def test_llm_pick_is_confirmed_by_values(_mock_llm_pick: MagicMock) -> None:
    keys = _keys(
        [
            _key_row("picked", "table-2", [100, 101, 102]),
            _key_row("closer", "table-3", [100, 101, 102]),
        ]
    )

    selected = semantic_fk._resolve_via_vdb(
        _fk_column([100, 102]),
        _retriever(_hit("closer", "table-3", score=0.1), _hit("picked", "table-2")),
        "database",
        keys,
    )

    assert selected == "picked"


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="picked")
def test_unconfirmed_pick_falls_to_next_hit_by_distance(
    _mock_llm_pick: MagicMock,
) -> None:
    keys = _keys(
        [
            _key_row("picked", "table-2", [900, 901]),
            _key_row("far", "table-3", [100, 101, 102]),
            _key_row("near", "table-4", [100, 101, 102]),
        ]
    )

    selected = semantic_fk._resolve_via_vdb(
        _fk_column([100, 101]),
        _retriever(
            _hit("picked", "table-2", score=0.1),
            _hit("far", "table-3", score=0.9),
            _hit("near", "table-4", score=0.3),
        ),
        "database",
        keys,
    )

    assert selected == "near"


def test_value_only_search_shortlists_containing_keys_for_the_llm() -> None:
    keys = _keys(
        [
            _key_row("vector-hit", "table-2", [900, 901]),
            _key_row("holder", "table-3", [1000, 1001, 1002], table_name="clients"),
            _key_row("stranger", "table-4", [5000, 5001]),
            _key_row("own-table", "table-1", [1000, 1001, 1002]),
        ]
    )
    with patch(
        "auto_ontology.semantic.semantic_fk._llm_pick_hit",
        side_effect=[None, "holder"],
    ) as pick:
        selected = semantic_fk._resolve_via_vdb(
            _fk_column([1000, 1002]),
            _retriever(_hit("vector-hit", "table-2")),
            "database",
            keys,
        )

    assert selected == "holder"
    fallback = pick.call_args_list[1]
    assert [hit["metadata"]["id"] for hit in fallback.args[1]] == ["holder"]
    assert fallback.kwargs["system_prompt"] == semantic_fk._VALUE_SYSTEM_PROMPT
    assert "public.clients.id" in fallback.args[1][0]["text"]


def test_value_only_search_runs_without_vector_hits() -> None:
    keys = _keys([_key_row("holder", "table-3", [1000, 1001])])
    with patch(
        "auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="holder"
    ) as pick:
        selected = semantic_fk._resolve_via_vdb(
            _fk_column([1000, 1001]), _retriever(), "database", keys
        )

    assert selected == "holder"
    pick.assert_called_once()


@pytest.mark.parametrize(
    ("name", "values"),
    [("customer_ref", [1, 2, 3]), ("customer_name", ["Alice", "Bob"])],
    ids=["short-values", "label-column"],
)
def test_value_only_search_skips_ambiguous_columns(name: str, values: list) -> None:
    keys = _keys([_key_row("holder", "table-3", values)])
    with patch("auto_ontology.semantic.semantic_fk._llm_pick_hit") as pick:
        selected = semantic_fk._resolve_via_vdb(
            _fk_column(values, name=name), _retriever(), "database", keys
        )

    assert selected is None
    pick.assert_not_called()


def _families(*members: tuple[str, str, str]) -> semantic_fk._Families:
    return semantic_fk._Families(
        [
            {"table_id": table, "family_id": family, "base_name": base}
            for table, family, base in members
        ]
    )


# table-1 OLD_ROOMS, table-2 NEW_HOTELS, table-3 OLD_HOTELS, table-4 NEW_SPA
_FAMILIES = _families(
    ("table-1", "old", "ROOMS"),
    ("table-2", "new", "HOTELS"),
    ("table-3", "old", "HOTELS"),
    ("table-4", "new", "SPA"),
)


@pytest.mark.parametrize(
    ("source", "target", "crosses"),
    [
        ("table-1", "table-2", True),
        ("table-1", "table-3", False),
        ("table-1", "table-4", False),
        ("table-1", "shared-dimension", False),
        ("standalone", "table-2", False),
    ],
    ids=[
        "own-family-has-a-copy",
        "same-family",
        "own-family-has-no-copy",
        "target-in-no-family",
        "source-in-no-family",
    ],
)
def test_family_crossing(source: str, target: str, crosses: bool) -> None:
    assert _FAMILIES.crosses(source, target) is crosses


def test_a_target_sharing_any_family_is_not_crossing() -> None:
    families = _families(
        ("table-1", "old", "ROOMS"),
        ("table-2", "new", "HOTELS"),
        ("table-2", "old", "HOTELS"),
    )

    assert not families.crosses("table-1", "table-2")


def test_join_probe_uses_tsql_top_for_sql_server() -> None:
    connector = MagicMock(dialect="tsql")
    connector.qualify.side_effect = lambda schema, table: f"[{schema}].[{table}]"
    sql = semantic_fk._join_probe_sql(
        connector,
        {"schema_name": "dbo", "table_name": "orders", "name": "customer_id"},
        {
            "schema_name": "dbo",
            "table_name": "customers",
            "column_name": "id",
        },
        as_text=False,
    )

    assert sql.startswith("SELECT TOP 1 1 AS hit")
    assert "LIMIT" not in sql


@patch("auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="same-family")
def test_vector_hits_from_another_family_are_never_offered(
    mock_llm_pick: MagicMock,
) -> None:
    retriever = _retriever(
        _hit("other-family", "table-2", score=0.1),
        _hit("same-family", "table-3", score=0.5),
    )

    selected = semantic_fk._resolve_via_vdb(
        _fk_column(None), retriever, "database", families=_FAMILIES
    )

    assert selected == "same-family"
    assert [h["metadata"]["id"] for h in mock_llm_pick.call_args.args[1]] == [
        "same-family"
    ]


def test_value_only_search_skips_keys_from_another_family() -> None:
    keys = _keys(
        [
            _key_row("other-family", "table-2", [1000, 1001]),
            _key_row("same-family", "table-3", [1000, 1001]),
        ]
    )
    with patch(
        "auto_ontology.semantic.semantic_fk._llm_pick_hit", return_value="same-family"
    ) as pick:
        semantic_fk._resolve_via_vdb(
            _fk_column([1000, 1001]), _retriever(), "database", keys, _FAMILIES
        )

    assert [h["metadata"]["id"] for h in pick.call_args.args[1]] == ["same-family"]


def test_key_index_probes_missing_values_once_and_stores_them() -> None:
    keys = _keys([_key_row("attr", "table-2", None, column_name="code")])
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.probe_ascending_distinct_values",
            return_value=["A01", "A02"],
        ) as probe,
        patch(
            "auto_ontology.semantic.semantic_fk.store_column_distinct_values"
        ) as store,
    ):
        assert keys.values("attr") == ["A01", "A02"]
        assert keys.values("attr") == ["A01", "A02"]

    probe.assert_called_once_with(keys.connector, "public", "target_table", "code")
    store.assert_called_once_with("table-2", {"code": ["A01", "A02"]})
    assert keys.stored()[0][1] == ["A01", "A02"]


def test_key_index_remembers_a_failed_probe() -> None:
    keys = _keys([_key_row("attr", "table-2", None)])
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.probe_ascending_distinct_values",
            return_value=None,
        ) as probe,
        patch(
            "auto_ontology.semantic.semantic_fk.store_column_distinct_values"
        ) as store,
    ):
        assert keys.values("attr") is None
        assert keys.values("attr") is None

    probe.assert_called_once()
    store.assert_not_called()


def _join_keys(*results: object) -> semantic_fk._KeyIndex:
    connector = MagicMock(dialect="postgres")
    connector.qualify.side_effect = lambda schema, table: f'"{schema}"."{table}"'
    connector.supports_statement_timeout = False
    connector.execute.side_effect = list(results)
    return _keys([_key_row("attr", "table-2", [1])], connector)


def test_join_validation_accepts_a_returned_row() -> None:
    keys = _join_keys(pd.DataFrame({"hit": [1]}))

    assert semantic_fk._join_returns_rows(keys, _fk_column([1]), "attr")
    sql = keys.connector.execute.call_args.args[0]
    assert sql == (
        'SELECT 1 AS hit FROM "public"."orders" s JOIN "public"."target_table" t '
        'ON s."customer_ref" = t."id" LIMIT 1'
    )


def test_join_validation_rejects_an_empty_join_without_retrying() -> None:
    keys = _join_keys(pd.DataFrame({"hit": []}))

    assert not semantic_fk._join_returns_rows(keys, _fk_column([1]), "attr")
    assert keys.connector.execute.call_count == 1


def test_join_validation_retries_as_text_after_a_type_error() -> None:
    keys = _join_keys(
        RuntimeError("operator does not exist"), pd.DataFrame({"hit": [1]})
    )

    assert semantic_fk._join_returns_rows(keys, _fk_column([1]), "attr")
    retry = keys.connector.execute.call_args.args[0]
    assert 'CAST(s."customer_ref" AS TEXT) = CAST(t."id" AS TEXT)' in retry


def test_join_validation_rejects_when_both_attempts_fail() -> None:
    keys = _join_keys(RuntimeError("bad"), RuntimeError("still bad"))

    assert not semantic_fk._join_returns_rows(keys, _fk_column([1]), "attr")


def test_join_validation_rejects_an_unknown_target() -> None:
    keys = _join_keys()

    assert not semantic_fk._join_returns_rows(keys, _fk_column([1]), "missing")
    keys.connector.execute.assert_not_called()


def test_column_attribute_embedding_includes_physical_table() -> None:
    rows = _build_rows(
        "database",
        {},
        [
            {
                "id": "attr-1",
                "name": "Tag id",
                "term_name": "Tag",
                "source_column": "id",
                "table_id": "table-1",
                "table_name": "tags",
                "is_unique": True,
            }
        ],
    )

    assert rows[0]["metadata"]["table_id"] == "table-1"
    assert rows[0]["metadata"]["table_name"] == "tags"
    assert rows[0]["metadata"]["is_unique"] is True
    assert rows[0]["metadata"]["content_metadata"]["table_id"] == "table-1"
    assert rows[0]["metadata"]["content_metadata"]["table_name"] == "tags"
    assert rows[0]["metadata"]["content_metadata"]["is_unique"] is True


def test_resolution_logs_candidate_progress(caplog: pytest.LogCaptureFixture) -> None:
    candidates = [
        {"id": "col-1", "name": "customer_id", "table_name": "orders"},
        {"id": "col-2", "name": "product_id", "table_name": "order_items"},
    ]
    caplog.set_level(logging.INFO, logger=semantic_fk.__name__)

    with (
        patch(
            "auto_ontology.semantic.semantic_fk.find_unlinked_fk_columns",
            return_value=candidates,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._build_retriever",
            return_value=MagicMock(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_table_families",
            return_value={},
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._resolve_connector", return_value=None
        ),
        patch("auto_ontology.semantic.semantic_fk._resolve_via_vdb", return_value=None),
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=[],
        ),
    ):
        assert semantic_fk.resolve_semantic_fks("database") == 0

    assert "Resolving semantic FK edges… (1/2): orders.customer_id" in caplog.messages
    assert (
        "Resolving semantic FK edges… (2/2): order_items.product_id" in caplog.messages
    )


def _join_group_row(column_id: str, column_name: str) -> dict:
    return {
        "source_table_id": "orders-table",
        "source_table_name": "orders",
        "source_table_description": "Customer orders.",
        "source_column_id": column_id,
        "source_column_name": column_name,
        "source_column_description": f"The {column_name} role.",
        "target_attribute_id": "customer-attribute",
        "target_attribute_name": "Customer ID",
        "target_table_id": "customers-table",
        "target_table_name": "customers",
        "target_table_description": "Customers.",
        "target_column_id": "customer-id",
        "target_column_name": "id",
        "target_column_description": "Customer identifier.",
        "use_in_join_paths": True,
    }


def test_single_semantic_fk_is_default_without_llm() -> None:
    group = [_join_group_row("customer-id-column", "customer_id")]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
        patch("auto_ontology.semantic.semantic_fk.get_llm_client") as get_llm_client,
    ):
        assert semantic_fk.select_semantic_fk_join_defaults("database") == 1

    get_llm_client.assert_not_called()
    set_default.assert_called_once_with(
        "orders-table",
        "customer-attribute",
        "customer-id-column",
    )


def test_llm_selects_default_from_parallel_semantic_fks() -> None:
    group = [
        _join_group_row("billing-column", "billing_customer_id"),
        _join_group_row("customer-column", "customer_id"),
    ]
    result = SemanticFkJoinDefaultSelection(
        selected_column_id="customer-column",
        rationale="The unqualified customer relationship.",
    )
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.get_llm_client",
            return_value=object(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.invoke_with_structured_output",
            return_value=result,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    set_default.assert_called_once_with(
        "orders-table",
        "customer-attribute",
        "customer-column",
    )


def test_invalid_llm_selection_uses_deterministic_default() -> None:
    group = [
        _join_group_row("z-column", "z_customer_id"),
        _join_group_row("a-column", "a_customer_id"),
    ]
    result = SemanticFkJoinDefaultSelection(
        selected_column_id="invented-column",
        rationale="Invalid.",
    )
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.get_llm_client",
            return_value=object(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.invoke_with_structured_output",
            return_value=result,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    set_default.assert_called_once_with(
        "orders-table",
        "customer-attribute",
        "a-column",
    )


def test_missing_llm_uses_deterministic_default() -> None:
    group = [
        _join_group_row("z-column", "z_customer_id"),
        _join_group_row("a-column", "a_customer_id"),
    ]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=group,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.get_llm_client",
            side_effect=EnvironmentError("not configured"),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.set_semantic_fk_join_default"
        ) as set_default,
    ):
        semantic_fk.select_semantic_fk_join_defaults("database")

    set_default.assert_called_once_with(
        "orders-table",
        "customer-attribute",
        "a-column",
    )


def test_resolution_always_normalizes_join_defaults() -> None:
    with (
        patch(
            "auto_ontology.semantic.semantic_fk._resolve_semantic_fks",
            return_value=3,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.select_semantic_fk_join_defaults"
        ) as select_defaults,
    ):
        assert semantic_fk.resolve_semantic_fks("database") == 3

    select_defaults.assert_called_once_with("database")


def _run_inferred_resolution(
    *, join_ok: bool, prevented: bool = False
) -> tuple[int, MagicMock, MagicMock, MagicMock]:
    candidates = [{"id": "col-1", "name": "customer_ref", "table_name": "orders"}]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.find_unlinked_fk_columns",
            return_value=candidates,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._build_retriever",
            return_value=MagicMock(),
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_table_families",
            return_value={},
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.is_semantic_fk_db_query_prevented",
            return_value=prevented,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._resolve_connector",
            return_value=MagicMock(),
        ) as resolve_connector,
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_value_key_candidates",
            return_value=[],
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._resolve_via_vdb", return_value="attr"
        ),
        patch(
            "auto_ontology.semantic.semantic_fk._join_returns_rows",
            return_value=join_ok,
        ) as join,
        patch(
            "auto_ontology.semantic.semantic_fk.merge_semantic_fk", return_value=True
        ) as merge,
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=[],
        ),
    ):
        created = semantic_fk.resolve_semantic_fks("database")
    return created, resolve_connector, join, merge


def test_inferred_fk_is_stored_when_the_join_returns_rows() -> None:
    created, _connector, join, merge = _run_inferred_resolution(join_ok=True)

    assert created == 1
    join.assert_called_once()
    merge.assert_called_once_with("col-1", "attr")


def test_inferred_fk_is_dropped_when_the_join_returns_nothing() -> None:
    created, _connector, _join, merge = _run_inferred_resolution(join_ok=False)

    assert created == 0
    merge.assert_not_called()


def test_prevented_db_queries_store_the_pick_without_validation() -> None:
    created, resolve_connector, join, merge = _run_inferred_resolution(
        join_ok=False, prevented=True
    )

    assert created == 1
    resolve_connector.assert_not_called()
    join.assert_not_called()
    merge.assert_called_once_with("col-1", "attr")


def test_declared_fk_is_trusted_without_a_join() -> None:
    candidates = [
        {
            "id": "col-1",
            "name": "customer_id",
            "table_name": "orders",
            "table_id": "orders-table",
            "fk_target_col_id": "customers-id",
            "fk_target_table_id": "customers-table",
        }
    ]
    with (
        patch(
            "auto_ontology.semantic.semantic_fk.find_unlinked_fk_columns",
            return_value=candidates,
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.find_column_attribute_by_column_id",
            return_value="attr",
        ),
        patch(
            "auto_ontology.semantic.semantic_fk.merge_semantic_fk", return_value=True
        ) as merge,
        patch("auto_ontology.semantic.semantic_fk._join_returns_rows") as join,
        patch(
            "auto_ontology.semantic.semantic_fk.fetch_semantic_fk_join_groups",
            return_value=[],
        ),
    ):
        assert semantic_fk.resolve_semantic_fks("database") == 1

    merge.assert_called_once_with("col-1", "attr")
    join.assert_not_called()
