# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``fetch_databases``: the connector facts a data page shows on each card.

Besides its name, owner and schema count, a database is returned with its
``dialect`` (the connector type), ``created`` and ``pulled`` timestamps. Both
``dialect`` and ``pulled`` are nullable in the table -- a database added through
a path that records no type, or one never pulled -- and the zone-scoped read
must return the same fields as the unscoped one.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import Table  # noqa: E402
from sqlalchemy.exc import OperationalError, ProgrammingError  # noqa: E402

from auto_ontology.dal import datasources as ds  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402

PULLED = datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module", autouse=True)
def require_schema() -> None:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_database LIMIT 1")
    except (OperationalError, ProgrammingError) as exc:
        # Only "no database" or "no table yet" is a reason to skip; anything else
        # is a real failure and must surface.
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table: Table, **values: Any) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


class Databases:
    """A database that has a type and a pull time, and one that has neither."""

    def __init__(self) -> None:
        self.prefix = f"d-{uuid.uuid4().hex[:8]}"
        self.full_name = f"{self.prefix}-full"
        self.bare_name = f"{self.prefix}-bare"
        self.full = _add(
            s.catalog_database,
            name=self.full_name,
            dialect="snowflake",
            pulled=PULLED,
        )
        self.bare = _add(s.catalog_database, name=self.bare_name)
        # A zone sees a database through a table of one of its schemas.
        schema = _add(s.catalog_schema, database_id=self.full, name="public")
        table = _add(s.catalog_table, schema_id=schema, name="t")
        _add(s.catalog_schema, database_id=self.bare, name="public")
        self.zone = _add(s.zone, name=f"{self.prefix}-zone")
        store().query_write(
            s.zone_target.insert().values(zone_id=self.zone, table_id=table)
        )

    def mine(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {r["name"]: r for r in rows if r["name"].startswith(self.prefix)}


@pytest.fixture
def databases() -> Iterator[Databases]:
    d = Databases()
    yield d
    store().query_write(s.zone.delete().where(s.zone.c.id == d.zone))
    for database_id in (d.full, d.bare):
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == database_id)
        )


def test_returns_dialect_created_and_pulled(databases: Databases) -> None:
    rows = databases.mine(ds.fetch_databases())

    full = rows[databases.full_name]
    assert full["dialect"] == "snowflake"
    assert full["pulled"] == PULLED
    assert isinstance(full["created"], datetime)


def test_null_dialect_and_pulled_stay_null(databases: Databases) -> None:
    rows = databases.mine(ds.fetch_databases())

    bare = rows[databases.bare_name]
    assert bare["dialect"] is None
    assert bare["pulled"] is None
    # `created` has a server default, so it is always set.
    assert isinstance(bare["created"], datetime)


def test_zone_scoped_read_returns_the_same_fields(databases: Databases) -> None:
    rows = databases.mine(ds.fetch_databases(zone_ids=[databases.zone]))

    assert set(rows) == {databases.full_name}
    assert rows[databases.full_name]["dialect"] == "snowflake"
    assert rows[databases.full_name]["pulled"] == PULLED
    assert isinstance(rows[databases.full_name]["created"], datetime)
