# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog introspection tests for the Postgres connector.

Runs against the Pagila fixture, which is the only fixture that has a
partitioned table and a materialized view — the two relation kinds this
connector used to drop silently. Skipped when no fixture database is reachable.

Set up with::

    docker compose up -d postgres
    uv run --no-sync python -m dev_tools.fixtures.seed_fixtures
"""

from __future__ import annotations

import os

import pytest

from gsf.catalog.constants import TableTypes
from gsf.connectors.registry import create_connector


@pytest.fixture(scope="module")
def pagila():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set; run dev_tools.fixtures.seed_fixtures first")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    try:
        connector = create_connector(
            f"postgresql://{user}:{password}@{host}:{port}/pagila"
        )
        connector.get_tables()
    except Exception as exc:  # noqa: BLE001 — any failure means "no fixture DB"
        pytest.skip(f"pagila not reachable: {exc}")
    yield connector
    connector.close()


@pytest.fixture(scope="module")
def public_tables(pagila):
    tables = pagila.get_tables()
    return tables[tables.table_schema == "public"]


def test_partitioned_parent_is_catalogued(public_tables) -> None:
    """``payment`` is partitioned. It used to vanish entirely.

    ``relispartition = false`` correctly hid the children, but ``relkind`` did
    not allow ``'p'``, so the parent was dropped too and an entire queryable
    table was missing from the catalog.
    """
    payment = public_tables[public_tables.table_name == "payment"]
    assert len(payment) == 1, "partitioned parent missing from the catalog"
    assert payment.table_type.iloc[0] == TableTypes.BASE_TABLE


def test_partition_children_are_not_catalogued(public_tables) -> None:
    """Children are an implementation detail; listing both double-counts rows."""
    children = public_tables[public_tables.table_name.str.startswith("payment_p")]
    assert children.empty, f"partition children leaked: {list(children.table_name)}"


def test_materialized_view_is_catalogued(public_tables) -> None:
    """``rental_by_category`` is a matview.

    Matviews are absent from ``information_schema.tables``, so driving the query
    from there meant ``TableTypes.MATERIALIZED_VIEW`` could never be produced —
    the ``relkind = 'm'`` branch was unreachable.
    """
    matview = public_tables[public_tables.table_name == "rental_by_category"]
    assert len(matview) == 1, "materialized view missing from the catalog"
    assert matview.table_type.iloc[0] == TableTypes.MATERIALIZED_VIEW


def test_all_three_table_types_are_produced(public_tables) -> None:
    """Every ``TableTypes`` value is reachable, not just ``base table``."""
    assert set(public_tables.table_type) == {
        TableTypes.BASE_TABLE,
        TableTypes.VIEW,
        TableTypes.MATERIALIZED_VIEW,
    }


def test_columns_reach_every_catalogued_relation(pagila, public_tables) -> None:
    """A relation with no columns is useless downstream.

    Matview columns are missing from ``information_schema.columns`` too, so this
    would have been empty for ``rental_by_category`` even once the table itself
    was listed.
    """
    columns = pagila.get_columns()
    public_columns = columns[columns.table_schema == "public"]
    with_columns = set(public_columns.table_name)
    missing = set(public_tables.table_name) - with_columns
    assert not missing, f"catalogued relations with no columns: {sorted(missing)}"


def test_declared_types_are_reported_not_placeholders(pagila) -> None:
    """Enums, arrays and domains report their real type.

    ``information_schema.columns.data_type`` collapses these to
    ``USER-DEFINED`` and ``ARRAY``, which tells a SQL-generating model nothing.
    """
    columns = pagila.get_columns()
    film = columns[
        (columns.table_schema == "public") & (columns.table_name == "film")
    ].set_index("column_name")

    assert film.loc["rating", "data_type"] == "mpaa_rating"
    assert film.loc["special_features", "data_type"] == "text[]"
    assert "USER-DEFINED" not in set(columns.data_type)
    assert "ARRAY" not in set(columns.data_type)


def test_system_schemas_are_excluded(pagila) -> None:
    schemas = set(pagila.get_tables().table_schema)
    assert schemas == {"public", "analytics"}, schemas


def test_nullability_is_a_real_boolean(pagila) -> None:
    """Not the ``'YES'``/``'NO'`` text ``information_schema`` reports.

    Both spellings are truthy, so leaking them makes every column read as
    nullable — the bug this connector's ``NOT a.attnotnull`` avoids by never
    producing a string in the first place.
    """
    columns = pagila.get_columns()
    assert columns.is_nullable.dtype == bool
    assert set(columns.is_nullable) <= {True, False}

    film = columns[
        (columns.table_schema == "public") & (columns.table_name == "film")
    ].set_index("column_name")
    assert bool(film.loc["film_id", "is_nullable"]) is False, "film_id is NOT NULL"
    assert bool(film.loc["description", "is_nullable"]) is True

    # The real assertion: a mix, not a column of all-True. The bug's signature
    # was every column agreeing.
    assert 0 < int(columns.is_nullable.sum()) < len(columns)
