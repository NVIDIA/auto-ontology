# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole-catalog schema map is built once per catalog state, never stale.

SQL validation parses every attempt against it. Rebuilding it each time made a
validation as slow as the catalog is large; keeping it is only safe if a
changed catalog is always noticed, and if what a parse returns from the kept
map is what it returned from a fresh one.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from auto_ontology.catalog.parsers.sqlglot_extractor import (
    SchemaCatalog,
    extract_tables_and_columns,
)
from auto_ontology.catalog.sql_parse import parse_query_single
from auto_ontology.retrieval.data_access import graph_schemas


def _rows(tables: dict[str, list[str]], schema: str = "shop") -> list[dict[str, str]]:
    """Catalog rows in the shape ``fetch_schemas_by_ids`` returns."""
    rows = []
    for table_name, columns in tables.items():
        table_id = str(uuid.uuid4())
        for column_name in columns:
            rows.append(
                {
                    "column_name": column_name,
                    "column_id": str(uuid.uuid4()),
                    "table_name": table_name,
                    "table_id": table_id,
                    "database_name": "db",
                    "table_schema": schema,
                    "data_type": "text",
                }
            )
    return rows


class _Catalog:
    """A stand-in for the catalog tables, counting how often it is read.

    ``fingerprint`` is what the cheap check reports. ``rows_fingerprint`` is
    what comes back with the rows, and is the same thing unless a test is
    simulating a commit that landed between the two reads.
    """

    def __init__(self, tables: dict[str, list[str]]) -> None:
        self.rows = _rows(tables)
        self.fingerprint = "1"
        self.rows_fingerprint: str | None = None
        self.reads = 0

    def fetch_rows(self, schema_ids: list | None = None) -> list[dict[str, str]]:
        self.reads += 1
        return self.rows

    def fetch_snapshot(self) -> tuple[str, list[dict[str, str]]]:
        self.reads += 1
        return self.rows_fingerprint or self.fingerprint, self.rows

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(graph_schemas, "fetch_schemas_by_ids", self.fetch_rows)
        monkeypatch.setattr(graph_schemas, "fetch_all_schema_ids", lambda: [])
        monkeypatch.setattr(
            graph_schemas, "fetch_catalog_schema_snapshot", self.fetch_snapshot
        )
        monkeypatch.setattr(
            graph_schemas,
            "fetch_catalog_schema_fingerprint",
            lambda: self.fingerprint,
        )


@pytest.fixture(autouse=True)
def _fresh_snapshot() -> Any:
    graph_schemas.reset_catalog_schemas()
    yield
    graph_schemas.reset_catalog_schemas()


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> _Catalog:
    catalog = _Catalog(
        {
            "orders": ["order_id", "customer_id", "total"],
            "customers": ["customer_id", "name"],
        }
    )
    catalog.install(monkeypatch)
    return catalog


def test_unchanged_catalog_is_read_once(catalog: _Catalog) -> None:
    first = graph_schemas.get_catalog_schemas()
    second = graph_schemas.get_catalog_schemas()

    assert second is first
    assert catalog.reads == 1


def test_changed_catalog_is_rebuilt_on_the_next_call(catalog: _Catalog) -> None:
    before = graph_schemas.get_catalog_schemas()
    assert not before["shop"].table_exists("refunds")

    catalog.rows = catalog.rows + _rows({"refunds": ["refund_id"]})
    catalog.fingerprint = "2"
    after = graph_schemas.get_catalog_schemas()

    assert after is not before
    assert after["shop"].table_exists("refunds")
    assert catalog.reads == 2


def test_map_is_filed_under_the_fingerprint_of_its_own_rows(catalog: _Catalog) -> None:
    """A commit between the check and the read must not mislabel the map.

    The check sees the catalog without ``refunds``; by the time the rows are
    read, ``refunds`` has been committed. Filed under the fingerprint from the
    check, that map would be served again the moment ``refunds`` was dropped
    and the catalog's fingerprint returned to the earlier value -- validating
    SQL against a table that no longer exists.
    """
    without_refunds = catalog.rows
    catalog.fingerprint = "without-refunds"
    catalog.rows = without_refunds + _rows({"refunds": ["refund_id"]})
    catalog.rows_fingerprint = "with-refunds"

    assert graph_schemas.get_catalog_schemas()["shop"].table_exists("refunds")

    # refunds is dropped again: the catalog is back to the state the first
    # check saw, and reports the same fingerprint it did then.
    catalog.rows = without_refunds
    catalog.rows_fingerprint = None

    assert not graph_schemas.get_catalog_schemas()["shop"].table_exists("refunds")


def test_unreadable_fingerprint_builds_fresh_every_time(
    catalog: _Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> str:
        raise RuntimeError("function hashtextextended does not exist")

    monkeypatch.setattr(graph_schemas, "fetch_catalog_schema_fingerprint", unavailable)

    first = graph_schemas.get_catalog_schemas()
    second = graph_schemas.get_catalog_schemas()

    assert first["shop"].table_exists("orders")
    assert second is not first
    assert catalog.reads == 2


def test_a_failed_build_is_not_kept(
    catalog: _Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty catalog raises, as it always did, and leaves nothing behind."""
    populated = catalog.rows
    catalog.rows = []
    with pytest.raises(KeyError):
        graph_schemas.get_catalog_schemas()

    # Same fingerprint on purpose: had the failure been cached, this would
    # return whatever was stored instead of building.
    catalog.rows = populated
    assert graph_schemas.get_catalog_schemas()["shop"].table_exists("orders")


_QUERIES = [
    "SELECT name FROM customers",
    "SELECT * FROM shop.orders",
    "SELECT o.total, c.name FROM orders o JOIN customers c "
    "ON o.customer_id = c.customer_id WHERE total > 10",
    "SELECT customer_id FROM orders JOIN customers USING (customer_id)",
    "WITH big AS (SELECT customer_id, total FROM orders WHERE total > 100) "
    "SELECT name, total FROM big JOIN customers "
    "ON big.customer_id = customers.customer_id",
    "SELECT order_id FROM orders UNION ALL SELECT customer_id FROM customers",
    "SELECT missing_column FROM orders, customers WHERE name = 'x'",
    "SELECT 1 FROM not_in_the_catalog",
]


def _extraction(sql: str, schemas: dict, dialect: str) -> Any:
    result = extract_tables_and_columns(
        sql=sql, dialects=[dialect], all_schemas=schemas
    )
    return (
        {name: (sorted(m.columns), m.schema_name) for name, m in result.tables.items()},
        result.ast_node_count,
        result.joins,
        result.unions,
    )


@pytest.mark.parametrize("dialect", ["postgres", "snowflake", "databricks"])
@pytest.mark.parametrize("sql", _QUERIES)
def test_shared_map_extracts_what_a_fresh_map_does(
    catalog: _Catalog, sql: str, dialect: str
) -> None:
    """Twice against the shared map, so the second run reads its caches."""
    fresh = graph_schemas.get_schemas_by_ids([])
    shared = graph_schemas.get_catalog_schemas()
    assert isinstance(shared, SchemaCatalog)

    expected = _extraction(sql, fresh, dialect)

    assert _extraction(sql, shared, dialect) == expected
    assert _extraction(sql, shared, dialect) == expected


def test_shared_map_resolves_the_same_catalog_ids(catalog: _Catalog) -> None:
    sql = "SELECT o.total, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
    fresh = parse_query_single(
        sql=sql, schemas=graph_schemas.get_schemas_by_ids([]), dialects=["postgres"]
    )
    shared_schemas = graph_schemas.get_catalog_schemas()

    for _ in range(2):
        with graph_schemas.catalog_schemas_lock:
            shared = parse_query_single(
                sql=sql, schemas=shared_schemas, dialects=["postgres"]
            )
        assert sorted(shared.get_tables_ids()) == sorted(fresh.get_tables_ids())
        assert sorted(shared.get_column_ids()) == sorted(fresh.get_column_ids())


def test_a_parse_does_not_see_nodes_an_earlier_parse_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same SQL must report the same columns on the first parse and the tenth.

    A column stored with its quotes fails the lookup that builds its node and
    is left out, but the half-built node stays behind; asked again, the schema
    answers from it and the column appears. With one map shared by every
    validation, what a question was told would depend on the questions before.
    """
    catalog = _Catalog({"orders": ["order_id", '"Total"']})
    catalog.install(monkeypatch)
    sql = "SELECT order_id, total FROM orders"

    def parsed_columns(schemas: dict) -> list[str]:
        with graph_schemas.catalog_schemas_lock:
            graph_schemas.forget_parsed_nodes(schemas)
            query = parse_query_single(sql=sql, schemas=schemas, dialects=["postgres"])
        return sorted(query.get_column_ids())

    fresh = parsed_columns(graph_schemas.get_schemas_by_ids([]))
    assert len(fresh) == 1, "the quoted column is skipped by a fresh schema"

    shared = graph_schemas.get_catalog_schemas()
    assert [parsed_columns(shared) for _ in range(3)] == [fresh] * 3


def test_forgetting_nodes_leaves_a_one_off_map_alone(catalog: _Catalog) -> None:
    one_off = graph_schemas.get_schemas_by_ids([])
    node = one_off["shop"].get_table_node("orders")

    graph_schemas.forget_parsed_nodes(one_off)

    assert one_off["shop"].get_table_node("orders") is node


def test_column_index_follows_a_replaced_frame(catalog: _Catalog) -> None:
    """The per-table index is tied to the frame it was read from."""
    schema = graph_schemas.get_schemas_by_ids([])["shop"]
    assert schema.column_names_by_table()["customers"] == ("customer_id", "name")

    schema.columns_df = schema.columns_df[
        schema.columns_df["column_name"] != "name"
    ].copy()

    assert schema.column_names_by_table()["customers"] == ("customer_id",)


def test_reused_sqlglot_schema_does_not_grow_with_parses(catalog: _Catalog) -> None:
    """A worker parses for days; per-query lookups must not accumulate.

    Measured over every dict sqlglot keeps on the schema except the catalog
    mapping itself, so a later sqlglot that renames or adds a memo table fails
    here instead of leaking quietly.
    """
    shared = graph_schemas.get_catalog_schemas()

    def parse(index: int) -> None:
        sql = (
            f"SELECT o{index}.total, c{index}.name FROM orders o{index} "
            f"JOIN customers c{index} ON o{index}.customer_id = c{index}.customer_id"
        )
        with graph_schemas.catalog_schemas_lock:
            extract_tables_and_columns(
                sql=sql, dialects=["postgres"], all_schemas=shared
            )

    def retained() -> int:
        schema = shared._qualify_schemas["postgres"]
        return sum(
            len(value)
            for name, value in vars(schema).items()
            if isinstance(value, dict) and name not in ("mapping", "udf_mapping")
        )

    for index in range(5):
        parse(index)
    after_a_few = retained()
    for index in range(5, 300):
        parse(index)

    assert retained() == after_a_few
