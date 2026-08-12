# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone CRUD on Postgres.

Concentrated on the two things that do not survive the move unchanged:

* **polymorphic targets** — callers pass a flat list of ids without saying
  whether each is a database, schema or table, which a graph never needed to
  know and three nullable foreign keys very much do;
* **the name rule** — case-insensitive on the trimmed name, and scoped to zones
  sharing a database. It is enforced in application code because a ``UNIQUE``
  column would be wrong in both directions at once, so it needs tests that would
  catch either direction.

Needs a migrated database and skips without one.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402
from gsf.dal.zones import (  # noqa: E402
    create_zone,
    delete_zone,
    get_zone_by_id,
    list_zones,
    set_zone_enabled,
    update_zone,
)


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.zone LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


@pytest.fixture
def catalog():
    """Two databases, each with a schema and a table."""
    prefix = f"z-{uuid.uuid4().hex[:8]}"
    ids: dict[str, str] = {"prefix": prefix}

    def insert(table, **values):
        return store().query_write(
            table.insert().values(**values).returning(table.c.id)
        )[0]["id"]

    for n in (1, 2):
        db_id = insert(s.catalog_database, name=f"{prefix}-db{n}")
        schema_id = insert(s.catalog_schema, database_id=db_id, name="shop")
        table_id = insert(s.catalog_table, schema_id=schema_id, name="customer")
        ids[f"db{n}"] = db_id
        ids[f"db{n}.schema"] = schema_id
        ids[f"db{n}.table"] = table_id

    yield ids

    store().query_write(s.zone.delete().where(s.zone.c.name.like(f"{prefix}%")))
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name.like(f"{prefix}%"))
    )


def _name(catalog, suffix: str = "") -> str:
    return f"{catalog['prefix']}-zone{suffix}"


# --------------------------------------------------------------------------
# Polymorphic targets
# --------------------------------------------------------------------------


def test_a_zone_can_target_all_three_tiers_at_once(catalog) -> None:
    """The flat id list carries no tier; each has to be resolved to a column."""
    zone = create_zone(
        name=_name(catalog),
        description="mixed",
        color="#fff",
        item_ids=[catalog["db1"], catalog["db2.schema"], catalog["db2.table"]],
    )

    fetched = get_zone_by_id(zone["id"])
    by_label = {item["label"] for item in fetched["items"]}
    assert by_label == {"database", "schema", "table"}
    assert {item["id"] for item in fetched["items"]} == {
        catalog["db1"],
        catalog["db2.schema"],
        catalog["db2.table"],
    }


def test_each_target_lands_in_its_own_column(catalog) -> None:
    """``CHECK (num_nonnulls(...) = 1)`` would reject a row filled in twice."""
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"], catalog["db1.table"]],
    )
    rows = store().query_read(
        s.zone_target.select().where(s.zone_target.c.zone_id == zone["id"])
    )
    assert len(rows) == 2
    for row in rows:
        filled = [
            row["database_id"] is not None,
            row["schema_id"] is not None,
            row["table_id"] is not None,
        ]
        assert sum(filled) == 1


def test_unknown_item_ids_are_dropped_not_rejected(catalog) -> None:
    """Nothing to link, and that is not an error."""
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"], str(uuid.uuid4())],
    )
    assert zone["items"] == [catalog["db1"]]


def test_item_order_is_preserved(catalog) -> None:
    """The API returns the order and the UI renders it; a set would lose it."""
    ordered = [catalog["db2.table"], catalog["db1"], catalog["db2.schema"]]
    zone = create_zone(
        name=_name(catalog), description=None, color="#fff", item_ids=ordered
    )
    assert zone["items"] == ordered


# --------------------------------------------------------------------------
# The name rule — wrong in both directions if left to a UNIQUE column
# --------------------------------------------------------------------------


def test_same_name_in_the_same_database_is_rejected(catalog) -> None:
    create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    with pytest.raises(ValueError):
        create_zone(
            name=_name(catalog),
            description=None,
            color="#fff",
            item_ids=[catalog["db1.table"]],
        )


def test_the_check_is_case_and_whitespace_insensitive(catalog) -> None:
    """A plain UNIQUE column would admit both of these."""
    create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    with pytest.raises(ValueError):
        create_zone(
            name=f"  {_name(catalog).upper()}  ",
            description=None,
            color="#fff",
            item_ids=[catalog["db1"]],
        )


def test_same_name_in_a_different_database_is_allowed(catalog) -> None:
    """A plain UNIQUE column would reject this one."""
    create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    other = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db2"]],
    )
    assert other["id"]


def test_renaming_a_zone_to_itself_is_allowed(catalog) -> None:
    """``exclude_id`` — otherwise a zone conflicts with its own name."""
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    updated = update_zone(
        zone_id=zone["id"],
        updates={"name": _name(catalog), "description": "edited"},
        item_ids=[catalog["db1"]],
    )
    assert updated["description"] == "edited"


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------


def test_update_with_none_items_leaves_membership_alone(catalog) -> None:
    """``None`` means "not editing membership"; ``[]`` means "clear it".

    Collapsing the two would strip a zone's items on any metadata-only edit.
    """
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    updated = update_zone(
        zone_id=zone["id"], updates={"description": "just the text"}, item_ids=None
    )
    assert [i["id"] for i in updated["items"]] == [catalog["db1"]]


def test_update_with_empty_items_clears_membership(catalog) -> None:
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    updated = update_zone(zone_id=zone["id"], updates={}, item_ids=[])
    assert updated["items"] == []


def test_update_replaces_rather_than_appends(catalog) -> None:
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    updated = update_zone(zone_id=zone["id"], updates={}, item_ids=[catalog["db2"]])
    assert [i["id"] for i in updated["items"]] == [catalog["db2"]]


def test_disabling_keeps_the_zone_visible(catalog) -> None:
    """The label swap becomes a boolean; a disabled zone still lists."""
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    disabled = set_zone_enabled(zone["id"], False)
    assert disabled["enabled"] is False

    listed = {z["id"]: z for z in list_zones()}
    assert zone["id"] in listed, "a disabled zone vanished from the admin listing"
    assert listed[zone["id"]]["enabled"] is False

    assert set_zone_enabled(zone["id"], True)["enabled"] is True


def test_delete_removes_the_zone_and_its_targets(catalog) -> None:
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1"]],
    )
    assert delete_zone(zone["id"]) is True
    assert get_zone_by_id(zone["id"]) is None
    assert (
        store().query_read(
            s.zone_target.select().where(s.zone_target.c.zone_id == zone["id"])
        )
        == []
    )


def test_deleting_a_table_removes_its_zone_membership(catalog) -> None:
    """A dangling grant is an access-control bug, which is why these cascade."""
    zone = create_zone(
        name=_name(catalog),
        description=None,
        color="#fff",
        item_ids=[catalog["db1.table"]],
    )
    store().query_write(
        s.catalog_table.delete().where(s.catalog_table.c.id == catalog["db1.table"])
    )
    assert get_zone_by_id(zone["id"])["items"] == []


def test_missing_zone_reads_as_none(catalog) -> None:
    assert get_zone_by_id(str(uuid.uuid4())) is None
    assert delete_zone(str(uuid.uuid4())) is False
    assert set_zone_enabled(str(uuid.uuid4()), False) is None
    assert update_zone(zone_id=str(uuid.uuid4()), updates={}, item_ids=None) is None
