# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-database reset: drop a database's Neo4j subgraph and pgvector embeddings.

This is the single source of truth for wiping one database's ingested data. It
removes the Neo4j subgraph reachable from the ``Database`` node (catalog and
semantic nodes alike) and the corresponding rows in both pgvector collections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gsf.catalog.constants import Labels
from gsf.catalog.store.neo4j.connection import get_neo4j_conn

from gsf.semantic.constants import (
    LABEL_ANALYSIS,
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_PQL_ANALYSIS,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    LABEL_TEXT_ATTRIBUTE,
)
from gsf.vdb import get_data_vdb, get_semantic_vdb

logger = logging.getLogger(__name__)


@dataclass
class ResetResult:
    """Summary of what a :func:`delete_all_data` call removed."""

    database_name: str
    data_rows: int
    semantic_rows: int


_SEMANTIC_LABELS = (
    LABEL_TERM,
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TEXT_ATTRIBUTE,
    LABEL_ANALYSIS,
    LABEL_PQL_ANALYSIS,
    Labels.CUSTOM_ANALYSIS,
)


def _delete_nodes_in_batches(node_match: str, database_name: str | None) -> int:
    """``DETACH DELETE`` every node returned as ``n`` by *node_match*.

    *node_match* is the driving statement of ``apoc.periodic.iterate``, which
    batches the deletion so very large graphs do not exhaust transaction
    memory. It may reference ``$database_name``, which is forwarded to both
    statements and simply left unused by the unscoped variants.

    Returns the number of nodes deleted.
    """
    rows = get_neo4j_conn().query_write(
        f"""
        CALL apoc.periodic.iterate(
            "{node_match}",
            "DETACH DELETE n",
            {{batchSize: 1000, params: {{database_name: $database_name}}}}
        )
        YIELD total
        RETURN total
        """,
        {"database_name": database_name},
    )
    if not rows:
        return 0
    return int(rows[0].get("total") or 0)


def _delete_database_nodes(database_name: str | None = None) -> int:
    """Delete a database's every node, catalog and semantic alike.

    Traverses out from the ``Database`` node following any relationship type,
    so everything reachable from it goes. When ``database_name`` is ``None``,
    every ``Database`` node and its graph is removed. Returns the number of
    nodes deleted.
    """
    # Pin the traversal to one Database node, or start from every one of them.
    database_filter = "" if database_name is None else " {name: $database_name}"
    deleted = _delete_nodes_in_batches(
        f"""
        MATCH (db:{Labels.DB}{database_filter})
        CALL apoc.path.subgraphNodes(db, {{}}) YIELD node AS n
        RETURN DISTINCT n
        """,
        database_name,
    )
    logger.info(
        "_delete_database_nodes: removed %d nodes for database %s",
        deleted,
        database_name or "<all>",
    )
    return deleted


def _delete_semantic_nodes(database_name: str | None = None) -> int:
    """Delete semantic Neo4j nodes, leaving data nodes intact.

    Deletes only nodes carrying one of :data:`_SEMANTIC_LABELS`; data nodes
    (``DB``/``Schema``/``Table``/``Column``) are untouched. Scoped to one
    database, semantic nodes are found by traversing out from its ``Database``
    node; for a full wipe they are matched on label alone, so nodes orphaned
    from every ``Database`` node are collected too — including ``PqlAnalysis``,
    which is never attached to a ``Database``. Returns the number of nodes
    deleted.
    """
    labels = "|".join(_SEMANTIC_LABELS)
    if database_name is None:
        node_match = f"MATCH (n:{labels}) RETURN n"
    else:
        label_predicate = " OR ".join(f"n:{label}" for label in _SEMANTIC_LABELS)
        node_match = f"""
        MATCH (db:{Labels.DB} {{name: $database_name}})
        CALL apoc.path.subgraphNodes(db, {{}}) YIELD node AS n
        WHERE {label_predicate}
        RETURN DISTINCT n
        """
    deleted = _delete_nodes_in_batches(node_match, database_name)
    logger.info(
        "_delete_semantic_nodes: removed %d semantic nodes for database %s",
        deleted,
        database_name or "<all>",
    )
    return deleted


def delete_semantic_layer(database_name: str | None = None) -> int:
    """Delete semantic Neo4j nodes and pgvector embeddings.

    When ``database_name`` is given, removes only that database's semantic nodes
    (see :data:`_SEMANTIC_LABELS`) reachable from the ``Database`` node and its
    ``semantic_layer`` rows. When ``None``, removes semantic nodes and
    embeddings across every database. Data nodes are left intact.

    Custom analyses, SQL and predictive alike, are part of what goes: they are
    user-authored, so nothing recompiles them afterwards. Returns the number of
    pgvector rows deleted.
    """
    _delete_semantic_nodes(database_name)

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
    """Delete a database's Neo4j nodes and pgvector embeddings.

    When ``database_name`` is given, removes that ``Database`` node and
    everything reachable from it in Neo4j plus its ``data_objects_layer`` rows.
    When ``None``, removes every ``Database`` node's graph and all data
    embeddings. Returns the number of pgvector rows deleted.
    """
    _delete_database_nodes(database_name)

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

    Removes the ``Database`` node and everything reachable from it in Neo4j,
    then deletes every ``data_objects_layer`` and ``semantic_layer`` row tagged
    with ``database_name``. Other databases are untouched.

    Semantic nodes are removed first (while still reachable from the
    ``Database`` node) before :func:`delete_data_layer` removes that node.
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
