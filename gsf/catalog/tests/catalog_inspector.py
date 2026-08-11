# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the catalog back, whichever store it was written to.

Ingestion tests assert on what the catalog *contains*, and that has to be the
same question in both backends or the two are not really being compared. This
answers it in the vocabulary of the catalog — databases, schemas, tables,
columns — rather than in nodes-and-labels or rows-and-joins.

Dispatch is on :data:`gsf.infra.store.USE_PG`, read once at import like
everything else, so a test module using this exercises whichever backend the
process was started with. Running both means running the tests twice, in two
processes; that is the cost of the switch being a constant, and the constant is
what keeps a single process from reading one store and writing another.
"""

from __future__ import annotations

from typing import Any

from gsf.infra.store import USE_PG


def wipe(database_name: str) -> None:
    """Remove a database and everything under it."""
    if USE_PG:
        from gsf.dal.pg import schema as s
        from gsf.dal.pg.session import store

        store().query_write(
            s.catalog_database.delete().where(
                s.catalog_database.c.name == database_name
            )
        )
        return

    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    get_neo4j_conn().query_write(
        "MATCH (d:Database {name:$name}) "
        "CALL apoc.path.subgraphNodes(d, {}) YIELD node DETACH DELETE node",
        {"name": database_name},
    )


def tables(database_name: str) -> set[str]:
    if USE_PG:
        rows = _pg(
            "SELECT t.name AS name FROM gsf.catalog_table t "
            "JOIN gsf.catalog_schema s ON s.id = t.schema_id "
            "JOIN gsf.catalog_database d ON d.id = s.database_id "
            "WHERE d.name = :name",
            name=database_name,
        )
    else:
        rows = _neo4j(
            "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)"
            "-[:CONTAINS]->(t:Table) RETURN t.name AS name",
            name=database_name,
        )
    return {r["name"] for r in rows}


def columns(database_name: str, table: str) -> set[str]:
    if USE_PG:
        rows = _pg(
            "SELECT c.name AS name FROM gsf.catalog_column c "
            "JOIN gsf.catalog_table t ON t.id = c.table_id "
            "JOIN gsf.catalog_schema s ON s.id = t.schema_id "
            "JOIN gsf.catalog_database d ON d.id = s.database_id "
            "WHERE d.name = :name AND t.name = :table",
            name=database_name,
            table=table,
        )
    else:
        rows = _neo4j(
            "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)-[:CONTAINS]->"
            "(t:Table {name:$table})-[:CONTAINS]->(c:Column) RETURN c.name AS name",
            name=database_name,
            table=table,
        )
    return {r["name"] for r in rows}


def column_property(database_name: str, table: str, column: str, prop: str) -> Any:
    """One column property. *prop* is a node property or a column name."""
    if USE_PG:
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
    else:
        rows = _neo4j(
            "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)-[:CONTAINS]->"
            "(:Table {name:$table})-[:CONTAINS]->(c:Column {name:$column}) "
            f"RETURN c.`{prop}` AS value",
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
    if USE_PG:
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

    rows = _neo4j(
        "MATCH (d:Database {name:$name}) "
        "CALL apoc.path.subgraphNodes(d, {}) YIELD node "
        "RETURN labels(node)[0] AS label, count(*) AS n",
        name=database_name,
    )
    counts = {r["label"]: r["n"] for r in rows}
    return {key: counts.get(key, 0) for key in ("Schema", "Table", "Column")}


def foreign_keys(database_name: str) -> set[tuple[str, str]]:
    """``(source column name, target column name)`` for each foreign key."""
    if USE_PG:
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
    else:
        rows = _neo4j(
            "MATCH (:Database {name:$name})-[:CONTAINS]->(:Schema)-[:CONTAINS]->"
            "(:Table)-[:CONTAINS]->(c:Column)-[:FOREIGN_KEY]->(t:Column) "
            "RETURN c.name AS src, t.name AS dst",
            name=database_name,
        )
    return {(r["src"], r["dst"]) for r in rows}


def _pg(sql: str, **params):
    from sqlalchemy import text

    from gsf.dal.pg.session import store

    return store().query_read(text(sql), params)


def _neo4j(cypher: str, **params):
    from gsf.catalog.store.neo4j.connection import get_neo4j_conn

    return get_neo4j_conn().query_read(cypher, params)
