# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate enrichment.

``expand_info`` turns ``(label, id)`` pairs from the vector store into the rows
a prompt is built from. The output is what matters, and two parts of it are easy
to get subtly wrong:

* **``relevant_tables``** — a column hit must bring back its *whole* parent
  table, not just the column that matched, because the generator needs every
  column to write a query;
* **``sample_values``** — emitted only when non-empty. An empty list rendered
  into a prompt reads as "this column has no values", which is a different claim
  from "we never profiled it".

Malformed input is dropped rather than raising: these pairs come from vector
metadata, where one bad row must not cost the whole retrieval.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from gsf.catalog.constants import Labels  # noqa: E402
from gsf.dal import candidates as c  # noqa: E402
from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.sql_attribute LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="public")
        self.table = _add(s.catalog_table, schema_id=self.schema, name="orders")
        self.columns = {
            name: _add(
                s.catalog_column,
                table_id=self.table,
                name=name,
                data_type=data_type,
                ordinal_position=position,
            )
            for position, (name, data_type) in enumerate(
                [("id", "integer"), ("total", None)], start=1
            )
        }
        self.queries: list[str] = []
        self.terms: list[str] = []
        self.owners: list[tuple] = []

    def statement(self, label: str) -> str:
        qid = _add(s.sql_query, sql_full_query=f"SELECT 1 -- {self.prefix}-{label}")
        self.queries.append(qid)
        _link(s.sql_query_table, sql_query_id=qid, table_id=self.table)
        return qid


@pytest.fixture
def world():
    w = World(f"e-{uuid.uuid4().hex[:8]}")
    yield w
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.database)
    )
    for table in (s.sql_attribute, s.custom_analysis, s.term):
        store().query_write(table.delete().where(table.c.name.like(f"{w.prefix}%")))
    for query_id in w.queries:
        store().query_write(s.sql_query.delete().where(s.sql_query.c.id == query_id))


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def test_malformed_pairs_are_dropped_not_raised() -> None:
    """One bad vector-metadata row must not cost the whole retrieval."""
    assert (
        c.expand_info(
            [
                None,
                "not a dict",
                {"label": "Table"},
                {"id": "x"},
                {"id": "x", "label": "   "},
            ]
        )
        == {}
    )


def test_none_and_empty_are_empty() -> None:
    assert c.expand_info(None) == {}
    assert c.expand_info([]) == {}


def test_an_unknown_label_is_skipped_with_a_warning(world, caplog) -> None:
    result = c.expand_info([{"id": world.table, "label": "Nonsense"}])
    assert result == {}
    assert "Nonsense" in caplog.text


# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------


def test_a_column_brings_back_its_whole_parent_table(world) -> None:
    """A column hit is really a table hit.

    The generator needs every column to write a query, so the parent arrives
    with all of them — not just the one that matched.
    """
    result = c.expand_info([{"id": world.columns["id"], "label": Labels.COLUMN}])
    entry = result[world.columns["id"]]

    assert entry["table_name"] == "orders"
    assert entry["parent_id"] == world.table
    assert [col["name"] for col in entry["relevant_tables"][0]["columns"]] == [
        "id",
        "total",
    ]


def test_a_null_data_type_becomes_an_empty_string(world) -> None:
    """`toString(coalesce(...))`: callers concatenate this, and None renders."""
    result = c.expand_info([{"id": world.columns["id"], "label": Labels.COLUMN}])
    columns = {
        col["name"]: col
        for col in result[world.columns["id"]]["relevant_tables"][0]["columns"]
    }
    assert columns["total"]["data_type"] == ""
    assert columns["id"]["data_type"] == "integer"


def test_sample_values_are_omitted_when_absent(world) -> None:
    """None, not an empty list — the two make different claims in a prompt."""
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == world.columns["id"])
        .values(sample_values='["1", "2"]')
    )
    result = c.expand_info([{"id": world.columns["id"], "label": Labels.COLUMN}])
    columns = {
        col["name"]: col
        for col in result[world.columns["id"]]["relevant_tables"][0]["columns"]
    }
    assert columns["id"]["sample_values"] == '["1", "2"]'
    assert columns["total"]["sample_values"] is None


def test_two_columns_of_one_table_are_both_expanded(world) -> None:
    result = c.expand_info(
        [
            {"id": world.columns["id"], "label": Labels.COLUMN},
            {"id": world.columns["total"], "label": Labels.COLUMN},
        ]
    )
    assert set(result) == set(world.columns.values())


# --------------------------------------------------------------------------
# SqlAttribute
# --------------------------------------------------------------------------


def test_a_sql_attribute_carries_its_sql_term_and_tables(world) -> None:
    attribute = _add(s.sql_attribute, name=f"{world.prefix}-revenue")
    term = _add(s.term, name=f"{world.prefix}-Revenue")
    _link(s.sql_attribute_term, attribute_id=attribute, term_id=term)
    query = world.statement("revenue")
    _link(s.sql_attribute_sql, attribute_id=attribute, sql_query_id=query)

    entry = c.expand_info([{"id": attribute, "label": "SqlAttribute"}])[attribute]

    assert entry["sql"] == f"SELECT 1 -- {world.prefix}-revenue"
    assert entry["term_name"] == f"{world.prefix}-Revenue"
    assert entry["term_id"] == term
    assert [t["name"] for t in entry["relevant_tables"]] == ["orders"]


def test_a_sql_attribute_without_a_statement_does_not_appear(world) -> None:
    """This match was not optional, unlike the analysis branch. Preserved.

    Such an attribute has nothing to contribute to a prompt, and the caller
    reads a missing id as "no context available".
    """
    attribute = _add(s.sql_attribute, name=f"{world.prefix}-orphan")
    term = _add(s.term, name=f"{world.prefix}-Revenue")
    _link(s.sql_attribute_term, attribute_id=attribute, term_id=term)

    assert c.expand_info([{"id": attribute, "label": "SqlAttribute"}]) == {}


# --------------------------------------------------------------------------
# CustomAnalysis
# --------------------------------------------------------------------------


def test_an_analysis_carries_its_sql_and_tables(world) -> None:
    analysis = _add(s.custom_analysis, name=f"{world.prefix}-revenue")
    query = world.statement("analysis")
    _link(s.custom_analysis_sql, analysis_id=analysis, sql_query_id=query)

    entry = c.expand_info([{"id": analysis, "label": Labels.CUSTOM_ANALYSIS}])[analysis]

    assert entry["sql"] == f"SELECT 1 -- {world.prefix}-analysis"
    assert [t["name"] for t in entry["relevant_tables"]] == ["orders"]


def test_an_analysis_without_a_statement_still_appears(world) -> None:
    """Optional here, unlike the SqlAttribute branch."""
    analysis = _add(s.custom_analysis, name=f"{world.prefix}-orphan")

    entry = c.expand_info([{"id": analysis, "label": Labels.CUSTOM_ANALYSIS}])[analysis]

    assert entry["sql"] == ""
    assert entry["relevant_tables"] == []


# --------------------------------------------------------------------------
# The default branch
# --------------------------------------------------------------------------


def test_a_table_returns_its_plain_properties(world) -> None:
    """No special branch: properties and nothing else."""
    entry = c.expand_info([{"id": world.table, "label": Labels.TABLE}])[world.table]
    assert entry["name"] == "orders"
    assert "relevant_tables" not in entry


def test_a_database_returns_its_plain_properties(world) -> None:
    entry = c.expand_info([{"id": world.database, "label": Labels.DB}])[world.database]
    assert entry["name"] == world.prefix


def test_labels_are_expanded_independently(world) -> None:
    """One batch, several labels — each dispatched to its own branch."""
    result = c.expand_info(
        [
            {"id": world.table, "label": Labels.TABLE},
            {"id": world.columns["id"], "label": Labels.COLUMN},
        ]
    )
    assert set(result) == {world.table, world.columns["id"]}
    assert "relevant_tables" in result[world.columns["id"]]
    assert "relevant_tables" not in result[world.table]


def test_a_missing_id_yields_no_entry(world) -> None:
    assert c.expand_info([{"id": "no-such-id", "label": Labels.TABLE}]) == {}
