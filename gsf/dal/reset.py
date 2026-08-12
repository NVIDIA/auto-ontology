# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-database reset: drop a database's catalog rows and pgvector embeddings.

The single source of truth for wiping one database's ingested data.

**This module is where the schema pays for itself.** The Cypher walked
``apoc.path.subgraphNodes`` from the ``Database`` node and deleted whatever it
reached, in ``apoc.periodic.iterate`` batches so a large graph did not exhaust
transaction memory. Here, ``CONTAINS`` is a parent FK column, so deleting one
row removes the whole catalog tier by ``ON DELETE CASCADE`` — no traversal, no
batching, and no way for a newly added child table to be missed.

Two behaviour changes come with that, both deliberate and both recorded in
DECISIONS.md:

* **B1 — the deletes are narrower.** ``subgraphNodes`` followed *any*
  relationship, so from a ``Database`` it reached that database's Terms, and
  from a shared Term it reached a *different* database's tables. Resetting one
  database could therefore delete another's data. A cascade follows foreign
  keys, which only ever point downward within one database.
* **B2 — the scoped semantic reset still does not reach ``PqlAnalysis``.** It
  is never attached to a database, so the traversal never found it. Preserved
  rather than fixed: a scoped reset silently deleting every predictive analysis
  in the deployment would be a worse surprise than the current gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select

from gsf.dal import schema as s
from gsf.dal.session import store
from gsf.vdb import get_data_vdb, get_semantic_vdb

logger = logging.getLogger(__name__)


@dataclass
class ResetResult:
    """Summary of what a :func:`delete_all_data` call removed."""

    database_name: str
    data_rows: int
    semantic_rows: int


def _table_ids(database_name: str):
    """The tables of one database, as a subquery."""
    return (
        select(s.catalog_table.c.id)
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
    )


def _delete_scoped_semantic(database_name: str) -> int:
    """Delete the semantic rows belonging to one database.

    "Belonging" is resolved per entity, because each reaches a database by a
    different path:

    * a **Term** through the tables that represent it;
    * a **ColumnAttribute** through its ``table_id``;
    * a **SqlAttribute** and a **CustomAnalysis** through the tables their SQL
      references.

    Each is deleted only when **every** table it touches belongs to this
    database — the same all-or-nothing shape the zone reads use, and for the
    same reason inverted: a Term shared with another database is that other
    database's data too, and a reset of this one must not take it.

    ``PqlAnalysis``, ``TextAttribute`` and ``Analysis`` are absent here on
    purpose (B2): nothing connects them to a database, so the Cypher traversal
    never reached them either.
    """
    tables = _table_ids(database_name)
    deleted = 0

    def only_ours(link_table, owner_column, table_column):
        """Rows whose links all land inside this database, and at least one does."""
        inside = (
            select(owner_column)
            .where(table_column.in_(tables))
            .distinct()
            .scalar_subquery()
        )
        outside = (
            select(owner_column)
            .where(table_column.notin_(tables))
            .distinct()
            .scalar_subquery()
        )
        return owner_column.in_(inside) & owner_column.notin_(outside)

    # Term: reached through table_term.
    term_ids = select(s.table_term.c.term_id).where(
        only_ours(s.table_term, s.table_term.c.term_id, s.table_term.c.table_id)
    )
    deleted += len(
        store().query_write(
            delete(s.term).where(s.term.c.id.in_(term_ids)).returning(s.term.c.id)
        )
    )

    # ColumnAttribute: owned outright by exactly one table.
    deleted += len(
        store().query_write(
            delete(s.column_attribute)
            .where(s.column_attribute.c.table_id.in_(tables))
            .returning(s.column_attribute.c.id)
        )
    )

    # SqlAttribute and CustomAnalysis: reached through the tables their SQL hits.
    for owner_table, link_table, owner_column in (
        (s.sql_attribute, s.sql_attribute_sql, s.sql_attribute_sql.c.attribute_id),
        (
            s.custom_analysis,
            s.custom_analysis_sql,
            s.custom_analysis_sql.c.analysis_id,
        ),
    ):
        touched = (
            select(owner_column, s.sql_query_table.c.table_id)
            .select_from(
                link_table.join(
                    s.sql_query_table,
                    s.sql_query_table.c.sql_query_id == link_table.c.sql_query_id,
                )
            )
            .subquery("touched")
        )
        inside = (
            select(touched.c[owner_column.name])
            .where(touched.c.table_id.in_(tables))
            .distinct()
        )
        outside = (
            select(touched.c[owner_column.name])
            .where(touched.c.table_id.notin_(tables))
            .distinct()
        )
        deleted += len(
            store().query_write(
                delete(owner_table)
                .where(
                    owner_table.c.id.in_(inside),
                    owner_table.c.id.notin_(outside),
                )
                .returning(owner_table.c.id)
            )
        )
    return deleted


def _delete_all_semantic() -> int:
    """Delete every semantic row, in every database.

    Unscoped, so ``PqlAnalysis``, ``TextAttribute`` and ``Analysis`` *are*
    included — the Cypher matched on label alone here, which reached nodes
    orphaned from every ``Database``.
    """
    deleted = 0
    for table in (
        s.term,
        s.column_attribute,
        s.sql_attribute,
        s.text_attribute,
        s.analysis,
        s.pql_analysis,
        s.custom_analysis,
    ):
        deleted += len(store().query_write(delete(table).returning(table.c.id)))
    return deleted


def delete_semantic_layer(database_name: str | None = None) -> int:
    """Delete semantic rows and their pgvector embeddings. Catalog rows survive.

    Custom analyses, SQL and predictive alike, are part of what goes: they are
    user-authored, so nothing recompiles them afterwards.

    Returns the number of **pgvector rows** deleted, not database rows — the
    caller reports it as "embeddings removed". The row count is logged.
    """
    if database_name is None:
        deleted = _delete_all_semantic()
    else:
        deleted = _delete_scoped_semantic(database_name)
    logger.info(
        "delete_semantic_layer: removed %d semantic rows for database %s",
        deleted,
        database_name or "<all>",
    )

    semantic_vdb = get_semantic_vdb()
    if database_name is None:
        semantic_deleted = semantic_vdb.delete_all()
    else:
        semantic_deleted = len(semantic_vdb.delete_by_database(database_name))

    logger.info(
        "delete_semantic_layer: removed %d semantic pgvector rows for database %s",
        semantic_deleted,
        database_name or "<all>",
    )
    return semantic_deleted


def delete_data_layer(database_name: str | None = None) -> int:
    """Delete a database's catalog rows and its pgvector embeddings.

    One ``DELETE`` on ``catalog_database``. Schemas, tables, columns, foreign
    keys, joins, statement links and zone targets all follow by cascade — which
    is the point of modelling ``CONTAINS`` as a parent FK rather than an edge
    table, and means a child table added later cannot be forgotten here.

    Returns the number of pgvector rows deleted.
    """
    statement = delete(s.catalog_database)
    if database_name is not None:
        statement = statement.where(s.catalog_database.c.name == database_name)
    deleted = len(store().query_write(statement.returning(s.catalog_database.c.id)))
    logger.info(
        "delete_data_layer: removed %d database rows for database %s",
        deleted,
        database_name or "<all>",
    )

    data_vdb = get_data_vdb()
    if database_name is None:
        data_deleted = data_vdb.delete_all()
    else:
        data_deleted = len(data_vdb.delete_by_database(database_name))

    logger.info(
        "delete_data_layer: removed %d data pgvector rows for database %s",
        data_deleted,
        database_name or "<all>",
    )
    return data_deleted


def delete_all_data(database_name: str | None = None) -> ResetResult:
    """Delete every trace of a database, both layers included.

    **Order matters, and for the same reason it did in Cypher**: the semantic
    rows are identified *through* the catalog — a Term by the tables that
    represent it, a SqlAttribute by the tables its SQL hits. Delete the catalog
    first and those links are already gone, so the semantic pass would find
    nothing to scope and leave every Term behind.
    """
    semantic_rows = delete_semantic_layer(database_name)
    data_rows = delete_data_layer(database_name)

    result = ResetResult(
        database_name=database_name,
        data_rows=data_rows,
        semantic_rows=semantic_rows,
    )
    logger.info(
        "delete_all_data: removed %d pgvector rows for database %s "
        "(%d data, %d semantic)",
        result.data_rows + result.semantic_rows,
        database_name,
        result.data_rows,
        result.semantic_rows,
    )
    return result
