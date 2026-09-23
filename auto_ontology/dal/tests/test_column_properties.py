# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``store_column_sample_values``/``store_column_uniqueness``/``store_column_date_formats``.

All three are thin wrappers over ``_set_column_property``, which had no direct
test coverage of its own — only mocked out in ``auto_ontology/semantic/tests/test_visit_enter.py``,
which checks that profiling *calls* these with the right arguments, never that
the arguments actually land in the database the way each function's docstring
promises (JSON-encoding, bool coercion, skip-if-falsy).
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from auto_ontology.dal import datasources as d  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM catalog_column LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


class World:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.db = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.db, name="public")
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}

    def table(self, name: str) -> str:
        tid = _add(s.catalog_table, schema_id=self.schema, name=name)
        self.tables[name] = tid
        return tid

    def column(self, table: str, name: str, position: int = 1) -> str:
        cid = _add(
            s.catalog_column,
            table_id=self.tables[table],
            name=name,
            ordinal_position=position,
        )
        self.columns[f"{table}.{name}"] = cid
        return cid


@pytest.fixture
def world():
    w = World(f"cp-{uuid.uuid4().hex[:8]}")
    yield w
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == w.db)
    )


def _row(column_id: str) -> dict:
    return store().query_read(
        select(
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_column.c.format,
        ).where(s.catalog_column.c.id == column_id)
    )[0]


# --------------------------------------------------------------------------
# store_column_sample_values
# --------------------------------------------------------------------------


def test_store_column_sample_values_writes_json_encoded_lists(world) -> None:
    world.table("orders")
    world.column("orders", "status")

    d.store_column_sample_values(world.tables["orders"], {"status": ["open", "closed"]})

    row = _row(world.columns["orders.status"])
    assert json.loads(row["sample_values"]) == ["open", "closed"]


def test_store_column_sample_values_is_a_noop_for_empty_input(world) -> None:
    world.table("orders")
    world.column("orders", "status")

    d.store_column_sample_values(world.tables["orders"], {})

    assert _row(world.columns["orders.status"])["sample_values"] is None


# --------------------------------------------------------------------------
# store_column_uniqueness
# --------------------------------------------------------------------------


def test_store_column_uniqueness_writes_bool_flags(world) -> None:
    world.table("orders")
    world.column("orders", "id")
    world.column("orders", "status", position=2)

    d.store_column_uniqueness(world.tables["orders"], {"id": True, "status": False})

    assert _row(world.columns["orders.id"])["is_unique"] is True
    assert _row(world.columns["orders.status"])["is_unique"] is False


def test_store_column_uniqueness_coerces_truthy_values(world) -> None:
    """The docstring promises ``bool(flag)``, not "whatever truthy value was passed"."""
    world.table("orders")
    world.column("orders", "id")

    d.store_column_uniqueness(world.tables["orders"], {"id": 1})

    assert _row(world.columns["orders.id"])["is_unique"] is True


# --------------------------------------------------------------------------
# store_column_date_formats
# --------------------------------------------------------------------------


def test_store_column_date_formats_writes_the_inferred_notation(world) -> None:
    world.table("events")
    world.column("events", "occurred_on")

    d.store_column_date_formats(world.tables["events"], {"occurred_on": "YYYY-MM-DD"})

    assert _row(world.columns["events.occurred_on"])["format"] == "YYYY-MM-DD"


def test_store_column_date_formats_skips_falsy_formats(world) -> None:
    """A column the inference declined to guess on (falsy format) is left alone.

    Not written as NULL -- that would discard a notation a previous run
    established, per the function's own docstring.
    """
    world.table("events")
    world.column("events", "occurred_on")
    d.store_column_date_formats(world.tables["events"], {"occurred_on": "YYYY-MM-DD"})

    d.store_column_date_formats(world.tables["events"], {"occurred_on": ""})

    assert _row(world.columns["events.occurred_on"])["format"] == "YYYY-MM-DD"


def test_store_column_date_formats_is_a_noop_for_empty_input(world) -> None:
    world.table("events")
    world.column("events", "occurred_on")

    d.store_column_date_formats(world.tables["events"], {})

    assert _row(world.columns["events.occurred_on"])["format"] is None


# --------------------------------------------------------------------------
# _set_column_property -- shared scoping behavior
# --------------------------------------------------------------------------


def test_set_column_property_only_touches_named_columns(world) -> None:
    """A column not named in *values* keeps whatever it already had."""
    world.table("orders")
    world.column("orders", "id")
    world.column("orders", "status", position=2)
    d.store_column_uniqueness(world.tables["orders"], {"id": True, "status": True})

    d.store_column_uniqueness(world.tables["orders"], {"id": False})

    assert _row(world.columns["orders.id"])["is_unique"] is False
    assert _row(world.columns["orders.status"])["is_unique"] is True


def test_set_column_property_scopes_to_the_given_table(world) -> None:
    """Same column name in a different table must not be touched."""
    world.table("orders")
    world.column("orders", "id")
    world.table("line_items")
    world.column("line_items", "id")

    d.store_column_uniqueness(world.tables["orders"], {"id": True})

    assert _row(world.columns["orders.id"])["is_unique"] is True
    assert _row(world.columns["line_items.id"])["is_unique"] is None
