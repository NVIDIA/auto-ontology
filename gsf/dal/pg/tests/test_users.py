# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone scoping — the access-control boundary.

A mistake here does not surface as a wrong answer, it surfaces as one user
seeing another's data, so each expansion rule is tested on its own rather than
in aggregate. The rules are asymmetric and easy to get subtly wrong:

* granting a **database** admits its schemas and their tables (downward);
* granting a **schema** admits its parent database and its own tables (both);
* granting a **table** admits its schema and database (upward) — but **not**
  that schema's other tables, which is the case a transitive implementation
  would silently get wrong.

Needs a migrated database and skips without one.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from gsf.dal.pg import schema as s  # noqa: E402
from gsf.dal.pg.session import store  # noqa: E402
from gsf.dal.pg.users import (  # noqa: E402
    get_accessible_catalog_ids_for_zones,
    resolve_accessible_catalog_ids,
    resolve_table_filter,
)


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.zone_target LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


@pytest.fixture
def catalog():
    """Two databases, each with two schemas of two tables.

    Enough shape that "everything" and "the right subset" are different
    answers — a filter that silently matched everything would pass against a
    single-table fixture.
    """
    prefix = f"z-{uuid.uuid4().hex[:8]}"
    ids: dict[str, str] = {"prefix": prefix}

    def insert(table, **values):
        row = store().query_write(table.insert().values(**values).returning(table.c.id))
        return row[0]["id"]

    for db_n in (1, 2):
        db_name = f"{prefix}-db{db_n}"
        db_id = insert(s.catalog_database, name=db_name)
        ids[f"db{db_n}"] = db_id
        for sch_n in (1, 2):
            schema_id = insert(s.catalog_schema, database_id=db_id, name=f"sch{sch_n}")
            ids[f"db{db_n}.sch{sch_n}"] = schema_id
            for tbl_n in (1, 2):
                table_id = insert(
                    s.catalog_table, schema_id=schema_id, name=f"t{tbl_n}"
                )
                ids[f"db{db_n}.sch{sch_n}.t{tbl_n}"] = table_id

    yield ids

    # Zones share the fixture's prefix so they can be cleaned up with it. They
    # did not, originally, and leaked ~8 rows per run into a shared store --
    # which `test_golden`'s `zones.list_zones` comparison eventually caught,
    # since it reads whatever is actually there.
    store().query_write(s.zone.delete().where(s.zone.c.name.like(f"{prefix}%")))
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name.like(f"{prefix}%"))
    )


def _zone_over(prefix: str, **target) -> str:
    zone_id = store().query_write(
        s.zone.insert()
        .values(name=f"{prefix}-zone-{uuid.uuid4().hex[:8]}")
        .returning(s.zone.c.id)
    )[0]["id"]
    store().query_write(s.zone_target.insert().values(zone_id=zone_id, **target))
    return zone_id


# --------------------------------------------------------------------------
# Expansion rules, one at a time
# --------------------------------------------------------------------------


def test_database_grant_admits_its_schemas_and_tables(catalog) -> None:
    zone = _zone_over(catalog["prefix"], database_id=catalog["db1"])
    got = get_accessible_catalog_ids_for_zones([zone])

    assert got["db_ids"] == {catalog["db1"]}
    assert got["schema_ids"] == {catalog["db1.sch1"], catalog["db1.sch2"]}
    assert got["table_ids"] == {
        catalog["db1.sch1.t1"],
        catalog["db1.sch1.t2"],
        catalog["db1.sch2.t1"],
        catalog["db1.sch2.t2"],
    }


def test_schema_grant_admits_its_parent_and_its_tables(catalog) -> None:
    zone = _zone_over(catalog["prefix"], schema_id=catalog["db1.sch1"])
    got = get_accessible_catalog_ids_for_zones([zone])

    assert got["db_ids"] == {catalog["db1"]}, "the parent database is needed to render"
    assert got["schema_ids"] == {catalog["db1.sch1"]}
    assert got["table_ids"] == {catalog["db1.sch1.t1"], catalog["db1.sch1.t2"]}


def test_table_grant_admits_its_ancestors_only(catalog) -> None:
    """The case a transitive expansion gets wrong.

    Granting one table admits its schema so the tree can be drawn — but must
    **not** then admit that schema's other tables, which would hand over data
    the zone never granted.
    """
    zone = _zone_over(catalog["prefix"], table_id=catalog["db1.sch1.t1"])
    got = get_accessible_catalog_ids_for_zones([zone])

    assert got["db_ids"] == {catalog["db1"]}
    assert got["schema_ids"] == {catalog["db1.sch1"]}
    assert got["table_ids"] == {catalog["db1.sch1.t1"]}
    assert catalog["db1.sch1.t2"] not in got["table_ids"], (
        "a sibling table leaked in — expansion is transitive when it must not be"
    )


def test_grants_from_several_zones_are_unioned(catalog) -> None:
    zone_a = _zone_over(catalog["prefix"], table_id=catalog["db1.sch1.t1"])
    zone_b = _zone_over(catalog["prefix"], table_id=catalog["db2.sch2.t2"])

    got = get_accessible_catalog_ids_for_zones([zone_a, zone_b])

    assert got["table_ids"] == {catalog["db1.sch1.t1"], catalog["db2.sch2.t2"]}
    assert got["db_ids"] == {catalog["db1"], catalog["db2"]}


def test_other_databases_are_never_admitted(catalog) -> None:
    zone = _zone_over(catalog["prefix"], database_id=catalog["db1"])
    got = get_accessible_catalog_ids_for_zones([zone])

    assert catalog["db2"] not in got["db_ids"]
    assert catalog["db2.sch1.t1"] not in got["table_ids"]


def test_no_zones_grants_nothing(catalog) -> None:
    got = get_accessible_catalog_ids_for_zones([])
    assert got == {"db_ids": set(), "schema_ids": set(), "table_ids": set()}


def test_unknown_zone_grants_nothing(catalog) -> None:
    got = get_accessible_catalog_ids_for_zones([str(uuid.uuid4())])
    assert got["table_ids"] == set()


# --------------------------------------------------------------------------
# The filter contract
# --------------------------------------------------------------------------


def test_none_means_unscoped_not_empty(catalog) -> None:
    """``None`` is the whole catalog; ``[]`` is nothing. Conflating them is a bug."""
    assert resolve_accessible_catalog_ids(None) is None

    predicate, _ = resolve_table_filter(None, s.catalog_table.c.id)
    assert predicate is None, "None must mean no restriction at all"


def test_empty_zone_list_denies_everything(catalog) -> None:
    predicate, _ = resolve_table_filter([], s.catalog_table.c.id)
    assert predicate is not None, "an empty zone list must deny, not permit"

    rows = store().query_read(s.catalog_table.select().where(predicate))
    assert rows == []


def test_filter_admits_exactly_the_granted_tables(catalog) -> None:
    zone = _zone_over(catalog["prefix"], schema_id=catalog["db1.sch1"])
    predicate, _ = resolve_table_filter([zone], s.catalog_table.c.id)

    rows = store().query_read(s.catalog_table.select().where(predicate))
    assert {r["id"] for r in rows} == {
        catalog["db1.sch1.t1"],
        catalog["db1.sch1.t2"],
    }


def test_filter_works_on_a_plain_id_column_too(catalog) -> None:
    """``column_attribute.table_id`` is a bare text column, not a foreign key.

    The filter has to apply to it as readily as to a real key, since that is
    how the semantic tier is scoped.
    """
    zone = _zone_over(catalog["prefix"], table_id=catalog["db1.sch1.t1"])
    predicate, _ = resolve_table_filter([zone], s.column_attribute.c.table_id)
    assert predicate is not None


def test_extra_params_pass_through(catalog) -> None:
    """Kept for signature compatibility; Core binds its own parameters."""
    _, params = resolve_table_filter(None, s.catalog_table.c.id, extra_params={"a": 1})
    assert params == {"a": 1}


def test_prefetched_ids_are_reused_not_requeried(catalog) -> None:
    """Threading a resolved result through must not re-resolve it."""
    prefetched = {"db_ids": set(), "schema_ids": set(), "table_ids": {"sentinel"}}
    predicate, _ = resolve_table_filter(
        ["ignored"], s.catalog_table.c.id, data_ids_by_zone=prefetched
    )
    compiled = str(predicate.compile(compile_kwargs={"literal_binds": True}))
    assert "sentinel" in compiled
