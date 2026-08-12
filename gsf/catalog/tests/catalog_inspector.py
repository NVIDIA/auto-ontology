# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the catalog back, in the vocabulary of the catalog.

Ingestion tests assert on what the catalog *contains*, and this answers that in
terms of databases, schemas, tables and columns rather than rows and joins — so
a test reads as a statement about the catalog rather than about the schema that
happens to hold it.

It existed to ask the same question of two backends while the port was in
flight. Phase 11 left only one, and the indirection is still worth keeping for
the reason above.
"""

from __future__ import annotations

from typing import Any


def wipe(database_name: str) -> None:
    """Remove a database and everything under it."""
    from gsf.dal import schema as s
    from gsf.dal.session import store

    # One statement; the whole tree follows by ON DELETE CASCADE.
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name == database_name)
    )


def tables(database_name: str) -> set[str]:
    rows = _pg(
        "SELECT t.name AS name FROM gsf.catalog_table t "
        "JOIN gsf.catalog_schema s ON s.id = t.schema_id "
        "JOIN gsf.catalog_database d ON d.id = s.database_id "
        "WHERE d.name = :name",
        name=database_name,
    )
    return {r["name"] for r in rows}


def columns(database_name: str, table: str) -> set[str]:
    rows = _pg(
        "SELECT c.name AS name FROM gsf.catalog_column c "
        "JOIN gsf.catalog_table t ON t.id = c.table_id "
        "JOIN gsf.catalog_schema s ON s.id = t.schema_id "
        "JOIN gsf.catalog_database d ON d.id = s.database_id "
        "WHERE d.name = :name AND t.name = :table",
        name=database_name,
        table=table,
    )
    return {r["name"] for r in rows}


def column_property(database_name: str, table: str, column: str, prop: str) -> Any:
    """One column property. *prop* is a node property or a column name."""
    rows = _pg(
        f"SELECT c.{prop} AS value FROM gsf.catalog_column c "
        "JOIN gsf.catalog_table t ON t.id = c.table_id "
        "JOIN gsf.catalog_schema s ON s.id = t.schema_id "
        "JOIN gsf.catalog_database d ON d.id = s.database_id "
        "WHERE d.name = :name AND t.name = :table AND c.name = :column",
        name=database_name,
        table=table,
        column=column,
    )
    return rows[0]["value"] if rows else None


def entity_counts(database_name: str) -> dict[str, int]:
    """How many of each catalog entity a database holds.

    Keyed by entity rather than by label or table name, so the two backends
    produce comparable answers. A duplicate-creating bug preserves every *name*,
    so counts are what a no-op re-ingest has to be checked against.
    """
    rows = _pg(
        """
        SELECT
          (SELECT count(*) FROM gsf.catalog_schema s
             JOIN gsf.catalog_database d ON d.id = s.database_id
            WHERE d.name = :name) AS schemas,
          (SELECT count(*) FROM gsf.catalog_table t
             JOIN gsf.catalog_schema s ON s.id = t.schema_id
             JOIN gsf.catalog_database d ON d.id = s.database_id
            WHERE d.name = :name) AS tables,
          (SELECT count(*) FROM gsf.catalog_column c
             JOIN gsf.catalog_table t ON t.id = c.table_id
             JOIN gsf.catalog_schema s ON s.id = t.schema_id
             JOIN gsf.catalog_database d ON d.id = s.database_id
            WHERE d.name = :name) AS columns
        """,
        name=database_name,
    )
    row = rows[0]
    return {
        "Schema": row["schemas"],
        "Table": row["tables"],
        "Column": row["columns"],
    }


def foreign_keys(database_name: str) -> set[tuple[str, str]]:
    """``(source column name, target column name)`` for each foreign key."""
    rows = _pg(
        "SELECT sc.name AS src, tc.name AS dst "
        "FROM gsf.column_foreign_key fk "
        "JOIN gsf.catalog_column sc ON sc.id = fk.source_column_id "
        "JOIN gsf.catalog_column tc ON tc.id = fk.target_column_id "
        "JOIN gsf.catalog_table t ON t.id = sc.table_id "
        "JOIN gsf.catalog_schema s ON s.id = t.schema_id "
        "JOIN gsf.catalog_database d ON d.id = s.database_id "
        "WHERE d.name = :name",
        name=database_name,
    )
    return {(r["src"], r["dst"]) for r in rows}


def _pg(sql: str, **params):
    from sqlalchemy import text

    from gsf.dal.session import store

    return store().query_read(text(sql), params)
