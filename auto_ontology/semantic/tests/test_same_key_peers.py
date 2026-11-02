# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for linking tables that store the same primary key."""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from auto_ontology.semantic import same_key_peers as peers


def _profile(
    table: str,
    column: str,
    groups: dict[str, set[tuple[str | None, ...]]],
    *,
    names: list[str] | None = None,
    ids: list[str] | None = None,
    rows: int | None = None,
) -> peers._KeyProfile:
    return peers._KeyProfile(
        table_id=table,
        table_name=table,
        column_id=f"{table}.{column}",
        column_name=column,
        attribute_id=f"{table}.{column}-attr",
        entity=peers._stem(peers._KEY_NAME, column) or table,
        descriptive={"name": names or [], "id": ids or []},
        rows=rows if rows is not None else len(groups),
        groups=groups,
    )


def _named(keys: range, label: str = "Org") -> dict[str, set[tuple[str | None, ...]]]:
    return {str(key): {(f"{label.lower()} {key}",)} for key in keys}


def test_identifiers_are_the_tables_own_key_not_its_parent() -> None:
    assert peers._identifiers(
        "FCLT_ORGANIZATION",
        None,
        [
            "FCLT_ORGANIZATION_KEY",
            "FCLT_ORG_PARENT_KEY",
            "FCLT_ORG_DLC_KEY",
            "ORGANIZATION_NAME",
        ],
    ) == {"FCLT_ORGANIZATION_KEY"}
    assert peers._identifiers(
        "FAC_ORGANIZATION", None, ["ORGANIZATION_KEY", "ORGANIZATION_NAME"]
    ) == {"ORGANIZATION_KEY"}
    assert peers._identifiers(
        "FCLT_ROOMS", None, ["FCLT_ROOM_KEY", "FCLT_ORGANIZATION_KEY", "SPACE_UNIT"]
    ) == {"FCLT_ROOM_KEY"}


def test_declared_primary_key_wins_and_composite_key_has_no_identifier() -> None:
    assert peers._identifiers("orders", ["order_no"], ["order_no", "order_id"]) == {
        "order_no"
    }
    assert (
        peers._identifiers("order_lines", ["order_id", "line"], ["order_id"]) == set()
    )


def test_fact_key_repeated_per_row_is_not_the_grain() -> None:
    rooms_per_org = {
        str(key): {(f"room {key}-{room}",) for room in range(5)}
        for key in range(100, 130)
    }
    fact = _profile("FCLT_ROOMS", "FCLT_ORGANIZATION_KEY", rooms_per_org, rows=150)

    assert not fact.is_grain


def test_repeated_rows_of_one_entity_are_still_the_grain() -> None:
    repeated = _profile(
        "FCLT_ORGANIZATION",
        "FCLT_ORGANIZATION_KEY",
        _named(range(100, 130)),
        names=["ORGANIZATION_NAME"],
        rows=60,
    )

    assert repeated.is_grain


def test_same_key_requires_names_to_agree() -> None:
    fac = _profile(
        "FAC_ORGANIZATION",
        "ORGANIZATION_KEY",
        _named(range(100, 131)),
        names=["ORGANIZATION_NAME"],
    )
    fclt = _profile(
        "FCLT_ORGANIZATION",
        "FCLT_ORGANIZATION_KEY",
        _named(range(100, 130)),
        names=["ORGANIZATION_NAME"],
    )
    rooms = _profile(
        "FCLT_ROOMS",
        "FCLT_ROOM_KEY",
        _named(range(100, 200), label="Room"),
        names=["ROOM_NAME"],
    )

    assert peers._same_key(fac, fclt)
    assert not peers._same_key(fac, rooms)


def test_without_descriptions_column_names_must_name_one_entity() -> None:
    customers = _profile("customers", "customer_key", {str(k): {()} for k in range(40)})
    products = _profile("products", "product_key", {str(k): {()} for k in range(30)})
    copy = _profile("crm_customers", "customer_key", {str(k): {()} for k in range(30)})

    assert not peers._same_key(customers, products)
    assert peers._same_key(customers, copy)


def test_tiny_or_mostly_disjoint_key_sets_do_not_match() -> None:
    small = _profile("a_org", "org_key", {str(k): {()} for k in range(10)})
    large = _profile("b_org", "org_key", {str(k): {()} for k in range(100)})
    shifted = _profile("c_org", "org_key", {str(k): {()} for k in range(90, 130)})

    assert not peers._same_key(small, large)
    assert not peers._same_key(large, shifted)


# ---------------------------------------------------------------------------
# End to end: SQLite source, Postgres catalog
# ---------------------------------------------------------------------------


@pytest.fixture
def store_ready() -> None:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    pytest.importorskip("sqlalchemy")
    from auto_ontology.dal.session import store

    try:
        store().query_read("SELECT 1 FROM catalog_table LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _write_source(path: Path) -> None:
    """The ``dw`` organization case, shrunk: FAC has one extra key, FCLT repeats some."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE FAC_ORGANIZATION (
            ORGANIZATION_KEY INTEGER, ORGANIZATION_ID TEXT, ORGANIZATION_NAME TEXT
        );
        CREATE TABLE FCLT_ORGANIZATION (
            FCLT_ORGANIZATION_KEY INTEGER, FCLT_ORG_PARENT_KEY INTEGER,
            ORGANIZATION_ID TEXT, ORGANIZATION_NAME TEXT
        );
        CREATE TABLE FCLT_ROOMS (
            FCLT_ROOM_KEY INTEGER, FCLT_ORGANIZATION_KEY INTEGER, ROOM_NAME TEXT
        );
        """
    )
    keys = range(101, 126)
    connection.executemany(
        "INSERT INTO FAC_ORGANIZATION VALUES (?, ?, ?)",
        [(key, f"ORG-{key}", f"Org {key}") for key in [*keys, 239]],
    )
    # Six keys appear twice, identical on name and id: 25 keys over 31 rows.
    connection.executemany(
        "INSERT INTO FCLT_ORGANIZATION VALUES (?, ?, ?, ?)",
        [
            (key, key - 1 if key > 101 else None, f"ORG-{key}", f"Org {key}")
            for key in [*keys, *range(101, 107)]
        ],
    )
    # Room keys cover every organization key, but name rooms, not organizations.
    connection.executemany(
        "INSERT INTO FCLT_ROOMS VALUES (?, ?, ?)",
        [(room, 101 + room % 25, f"Room {room}") for room in range(101, 161)],
    )
    connection.commit()
    connection.close()


class _Catalog:
    def __init__(self, prefix: str) -> None:
        from auto_ontology.dal import schema as s

        self.s = s
        self.prefix = prefix
        self.database_name = f"{prefix}-dw"
        self.database_id = self._add(s.catalog_database, name=self.database_name)
        self.schema_id = self._add(
            s.catalog_schema, database_id=self.database_id, name="main"
        )
        self.tables: dict[str, str] = {}
        self.columns: dict[str, str] = {}
        self.attributes: dict[str, str] = {}

    def _add(self, table, **values) -> str:
        from auto_ontology.dal.session import store

        return store().query_write(
            table.insert().values(**values).returning(table.c.id)
        )[0]["id"]

    def table(self, name: str, *columns: str) -> None:
        self.tables[name] = self._add(
            self.s.catalog_table, schema_id=self.schema_id, name=name
        )
        for position, column in enumerate(columns, 1):
            self.columns[f"{name}.{column}"] = self._add(
                self.s.catalog_column,
                table_id=self.tables[name],
                name=column,
                ordinal_position=position,
            )

    def attribute(self, column: str) -> str:
        from auto_ontology.dal.session import store
        from auto_ontology.semantic.constants import SEMANTIC_SOURCE

        table, name = column.split(".")
        attribute_id = self._add(
            self.s.column_attribute,
            name=f"{self.prefix}-{column}",
            source=SEMANTIC_SOURCE,
            source_column=name,
            term_name=f"{self.prefix}-{table}",
            table_id=self.tables[table],
        )
        store().query_write(
            self.s.column__has_attribute.insert().values(
                column_id=self.columns[column], attribute_id=attribute_id
            )
        )
        self.attributes[column] = attribute_id
        return attribute_id

    def cleanup(self) -> None:
        from auto_ontology.dal.session import store

        store().query_write(
            self.s.catalog_database.delete().where(
                self.s.catalog_database.c.id == self.database_id
            )
        )
        store().query_write(
            self.s.column_attribute.delete().where(
                self.s.column_attribute.c.term_name.like(f"{self.prefix}%")
            )
        )


@pytest.fixture
def dw(store_ready: None, tmp_path: Path):
    from auto_ontology.connectors.sqlite import SQLiteDatabase

    path = tmp_path / "dw.sqlite"
    _write_source(path)
    catalog = _Catalog(f"skp-{uuid.uuid4().hex[:8]}")
    catalog.table(
        "FAC_ORGANIZATION", "ORGANIZATION_KEY", "ORGANIZATION_ID", "ORGANIZATION_NAME"
    )
    catalog.table(
        "FCLT_ORGANIZATION",
        "FCLT_ORGANIZATION_KEY",
        "FCLT_ORG_PARENT_KEY",
        "ORGANIZATION_ID",
        "ORGANIZATION_NAME",
    )
    catalog.table("FCLT_ROOMS", "FCLT_ROOM_KEY", "FCLT_ORGANIZATION_KEY", "ROOM_NAME")
    for column in (
        "FAC_ORGANIZATION.ORGANIZATION_KEY",
        "FCLT_ORGANIZATION.FCLT_ORGANIZATION_KEY",
        "FCLT_ORGANIZATION.FCLT_ORG_PARENT_KEY",
        "FCLT_ROOMS.FCLT_ROOM_KEY",
        "FCLT_ROOMS.FCLT_ORGANIZATION_KEY",
    ):
        catalog.attribute(column)
    from auto_ontology.dal.attributes import merge_semantic_fk

    assert merge_semantic_fk(
        catalog.columns["FCLT_ROOMS.FCLT_ORGANIZATION_KEY"],
        catalog.attributes["FAC_ORGANIZATION.ORGANIZATION_KEY"],
    )
    connector = SQLiteDatabase(str(path))
    with patch.object(peers, "_resolve_connector", return_value=connector):
        yield catalog
    connector.close()
    catalog.cleanup()


def _semantic_fks(catalog: _Catalog) -> set[tuple[str, str, bool]]:
    from sqlalchemy import select

    from auto_ontology.dal.session import store

    s = catalog.s
    column_names = {v: k for k, v in catalog.columns.items()}
    attribute_names = {v: k for k, v in catalog.attributes.items()}
    rows = store().query_read(
        select(
            s.column__semantic_fk.c.column_id,
            s.column__semantic_fk.c.attribute_id,
            s.column__semantic_fk.c.use_in_join_paths,
        ).where(s.column__semantic_fk.c.column_id.in_(list(catalog.columns.values())))
    )
    return {
        (
            column_names[row["column_id"]],
            attribute_names[row["attribute_id"]],
            row["use_in_join_paths"],
        )
        for row in rows
    }


def test_links_same_primary_key_both_ways_and_nothing_else(dw: _Catalog) -> None:
    from auto_ontology.dal.attributes import _find_table_join_hops

    assert peers.link_same_key_peers(dw.database_name) == 1

    assert _semantic_fks(dw) == {
        (
            "FCLT_ORGANIZATION.FCLT_ORGANIZATION_KEY",
            "FAC_ORGANIZATION.ORGANIZATION_KEY",
            True,
        ),
        (
            "FAC_ORGANIZATION.ORGANIZATION_KEY",
            "FCLT_ORGANIZATION.FCLT_ORGANIZATION_KEY",
            True,
        ),
        # The fact table's existing edge stays, and it gains no peer.
        (
            "FCLT_ROOMS.FCLT_ORGANIZATION_KEY",
            "FAC_ORGANIZATION.ORGANIZATION_KEY",
            True,
        ),
    }
    fac_key = dw.columns["FAC_ORGANIZATION.ORGANIZATION_KEY"]
    fclt_key = dw.columns["FCLT_ORGANIZATION.FCLT_ORGANIZATION_KEY"]
    assert _find_table_join_hops(
        dw.tables["FCLT_ORGANIZATION"], dw.tables["FAC_ORGANIZATION"], dw.database_id
    ) == [(fclt_key, fac_key)]
    assert _find_table_join_hops(
        dw.tables["FAC_ORGANIZATION"], dw.tables["FCLT_ORGANIZATION"], dw.database_id
    ) == [(fac_key, fclt_key)]


def test_rerun_is_idempotent(dw: _Catalog) -> None:
    peers.link_same_key_peers(dw.database_name)
    first = _semantic_fks(dw)

    peers.link_same_key_peers(dw.database_name)

    assert _semantic_fks(dw) == first
