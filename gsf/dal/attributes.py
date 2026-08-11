# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write for ColumnAttribute and SemanticFK entities.

Also contains find_join_path, which traverses SEMANTIC_FK / HAS_ATTRIBUTE /
CONTAINS edges to resolve multi-hop join routes at retrieval time.
"""

from __future__ import annotations

import logging
from typing import Any

from gsf.catalog.constants import Edges, Labels
from gsf.catalog.store.neo4j.connection import get_neo4j_conn

from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_SEMANTIC_FK,
    SEMANTIC_SOURCE,
)
from gsf.dal.datasources import fetch_col_table_contexts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ColumnAttribute CRUD
# ---------------------------------------------------------------------------


def merge_column_attribute(
    *,
    term_name: str,
    table_id: str,
    source_column: str,
    attr_name: str,
    datatype: str,
    description: str | None,
) -> str | None:
    """Merge the ColumnAttribute node and return its persistent ``id`` (UUID)."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN} {{name: $source_column}})
        MATCH (term:{LABEL_TERM} {{name: $term_name, source: $source}})
        MERGE (attr:{LABEL_COLUMN_ATTRIBUTE} {{
            name: $attr_name,
            source_column: $source_column,
            term_name: $term_name,
            table_id: $table_id,
            source: $source
        }})
        ON CREATE SET attr.id = randomUUID()
        SET attr.datatype = $datatype,
            attr.description = coalesce($description, attr.description)
        MERGE (col)-[:{REL_HAS_ATTRIBUTE}]->(attr)
        MERGE (attr)-[:{REL_PROPERTY_OF}]->(term)
        RETURN attr.id AS id
        """,
        {
            "table_id": table_id,
            "source_column": source_column,
            "term_name": term_name,
            "attr_name": attr_name,
            "datatype": datatype,
            "description": description,
            "source": SEMANTIC_SOURCE,
        },
    )
    return rows[0]["id"] if rows else None


def update_column_attribute(
    attr_id: str,
    term_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    certified: bool | None = None,
) -> dict[str, Any] | None:
    """Update ColumnAttribute metadata and return its embedding context."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: $attr_id}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM} {{id: $term_id}})
        SET attr.name = coalesce($name, attr.name),
            attr.description = coalesce($description, attr.description),
            attr.certified = coalesce($certified, attr.certified)
        WITH attr, term
        OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->(attr)
        OPTIONAL MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->
              (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
              (:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN attr.id AS id,
               attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.id AS column_id,
               col.sample_values AS sample_values,
               term.id AS term_id,
               term.synonyms AS term_synonyms,
               head(collect(DISTINCT db.name)) AS database_name,
               coalesce(attr.certified, false) AS certified
        """,
        {
            "attr_id": attr_id,
            "term_id": term_id,
            "name": name,
            "description": description,
            "certified": certified,
        },
    )
    return dict(rows[0]) if rows else None


def find_column_attribute_by_column_id(column_id: str) -> str | None:
    """Return the id of the ColumnAttribute connected to a given Column, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (col:{Labels.COLUMN} {{id: $col_id}})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        RETURN attr.id AS id LIMIT 1
        """,
        {"col_id": column_id, "source": SEMANTIC_SOURCE},
    )
    return rows[0]["id"] if rows else None


def fetch_attr_column_contexts(
    attr_ids: list[str],
    *,
    database_name: str | None,
) -> dict[str, dict]:
    """Fetch Column + Table + Schema context for ColumnAttribute IDs.

    Returns a mapping of attr_id -> {attr_name, attr_description, col_id,
    col_name, table_id, table_name, schema_name, database_name, term_name}.
    """
    if not attr_ids:
        return {}
    query = """
    UNWIND $attr_ids AS attr_id
    MATCH (attr:ColumnAttribute {id: attr_id})
    OPTIONAL MATCH (col:Column)-[:SEMANTIC_FK|HAS_ATTRIBUTE]->(attr)
    OPTIONAL MATCH (col)<-[:CONTAINS]-(tbl:Table)<-[:CONTAINS]-(sch:Schema)
    OPTIONAL MATCH (sch)<-[:CONTAINS]-(db:Database)
    WHERE $database_name IS NULL OR db.name = $database_name
    OPTIONAL MATCH (attr)-[:PROPERTY_OF]->(term:Term)
    RETURN attr.id AS attr_id, attr.name AS attr_name,
           attr.description AS attr_description,
           col.id AS col_id, col.name AS col_name,
           tbl.id AS table_id, tbl.name AS table_name, sch.name AS schema_name,
           db.name AS database_name, term.name AS term_name
    """
    try:
        rows = get_neo4j_conn().query_read(
            query,
            {"attr_ids": attr_ids, "database_name": database_name},
        )
    except Exception:
        logger.warning("fetch_attr_column_contexts: Neo4j query failed", exc_info=True)
        return {}
    result: dict[str, dict] = {}
    for row in rows:
        aid = row.get("attr_id")
        if not aid:
            continue
        result[aid] = {
            "attr_name": row.get("attr_name") or "",
            "attr_description": row.get("attr_description") or "",
            "col_id": row.get("col_id"),
            "col_name": row.get("col_name") or "",
            "table_id": row.get("table_id"),
            "table_name": row.get("table_name") or "",
            "schema_name": row.get("schema_name") or "",
            "database_name": row.get("database_name") or "",
            "term_name": row.get("term_name") or "",
        }
    return result


# ---------------------------------------------------------------------------
# SemanticFK
# ---------------------------------------------------------------------------


def find_unlinked_fk_columns(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return Column nodes with no SEMANTIC_FK and no HAS_ATTRIBUTE edge.

    These are FK columns that have not yet been linked to a ColumnAttribute.

    When *database_name* is provided, only columns belonging to that database
    are returned. Multiple databases can be co-resident in the same Neo4j
    graph (e.g. the BIRD benchmark), so scoping keeps each compile pass'
    FK-resolution isolated to a single database. When omitted, every unlinked
    FK column in the graph is returned.
    """
    if database_name is not None:
        result = get_neo4j_conn().query_read(
            f"""
            MATCH (d:{Labels.DB} {{name: $database_name}})-[:{Edges.CONTAINS}]->
                  (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->
                  (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
            WHERE NOT (col)-[:{REL_SEMANTIC_FK}]->()
              AND NOT (col)-[:{REL_HAS_ATTRIBUTE}]->()
            OPTIONAL MATCH (col)-[:{Edges.FOREIGN_KEY}]->(tgt:{Labels.COLUMN})
            RETURN col.id          AS id,
                   col.name        AS name,
                   col.description AS description,
                   col.sample_values AS sample_values,
                   t.name          AS table_name,
                   tgt.id          AS fk_target_col_id
            """,
            {"database_name": database_name},
        )
    else:
        result = get_neo4j_conn().query_read(
            f"""
            MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
            WHERE NOT (col)-[:{REL_SEMANTIC_FK}]->()
              AND NOT (col)-[:{REL_HAS_ATTRIBUTE}]->()
            OPTIONAL MATCH (col)-[:{Edges.FOREIGN_KEY}]->(tgt:{Labels.COLUMN})
            RETURN col.id          AS id,
                   col.name        AS name,
                   col.description AS description,
                   col.sample_values AS sample_values,
                   t.name          AS table_name,
                   tgt.id          AS fk_target_col_id
            """
        )
    return result


_COLUMN_PATH_RETURN = (
    "col.id AS id, col.name AS column_name, "
    "t.id AS table_id, t.name AS table_name, "
    "sch.id AS schema_id, db.id AS db_id"
)


def _column_path_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Project a Neo4j column-path row into the API column-ref shape."""
    return {
        "id": row["id"],
        "column_name": row["column_name"],
        "table_id": row["table_id"],
        "table_name": row["table_name"],
        "schema_id": row["schema_id"],
        "db_id": row["db_id"],
    }


def fetch_column_attribute_columns_map(
    attr_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Return primary/referenced columns keyed by ColumnAttribute id.

    Each value is ``{primary_column, referenced_columns}``. Missing
    attributes are omitted; callers should default to
    ``primary_column=None`` / ``referenced_columns=[]``.

    Each column dict includes catalog path ids (``db_id``, ``schema_id``,
    ``table_id``, ``id``) plus display names so the UI can navigate to
    ``/data?focus=db|schema|table|column``.
    """
    if not attr_ids:
        return {}

    conn = get_neo4j_conn()
    primary_rows = conn.query_read(
        f"""
        UNWIND $attr_ids AS attr_id
        MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: attr_id}})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN attr.id AS attr_id, {_COLUMN_PATH_RETURN}
        """,
        {"attr_ids": attr_ids},
    )
    referenced_rows = conn.query_read(
        f"""
        UNWIND $attr_ids AS attr_id
        MATCH (col:{Labels.COLUMN})-[:{REL_SEMANTIC_FK}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: attr_id}})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(col)
        RETURN attr.id AS attr_id, {_COLUMN_PATH_RETURN}
        ORDER BY t.name, col.name
        """,
        {"attr_ids": attr_ids},
    )

    result: dict[str, dict[str, Any]] = {
        attr_id: {"primary_column": None, "referenced_columns": []}
        for attr_id in attr_ids
    }
    for row in primary_rows:
        attr_id = row["attr_id"]
        if attr_id in result and result[attr_id]["primary_column"] is None:
            result[attr_id]["primary_column"] = _column_path_dict(row)
    for row in referenced_rows:
        attr_id = row["attr_id"]
        if attr_id in result:
            result[attr_id]["referenced_columns"].append(_column_path_dict(row))
    return result


def merge_semantic_fk(src_column_id: str, tgt_attr_id: str) -> None:
    """Create a SEMANTIC_FK edge from a source Column to a target ColumnAttribute."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (src:{Labels.COLUMN} {{id: $src_id}})
        MATCH (tgt:{LABEL_COLUMN_ATTRIBUTE} {{id: $tgt_id}})
        MERGE (src)-[:{REL_SEMANTIC_FK}]->(tgt)
        """,
        {"src_id": src_column_id, "tgt_id": tgt_attr_id},
    )


# ---------------------------------------------------------------------------
# Join path traversal
# ---------------------------------------------------------------------------


def find_join_path(anchor_col_id: str, dest_col_id: str) -> list[dict]:
    """Find the shortest semantic join path between two Column nodes.

    SEMANTIC_FK is directional (Column -> ColumnAttribute) and is followed
    only in that outgoing direction: an FK column points at the attribute it
    references. Traversing it undirected would hop from one FK column up to a
    shared target attribute and back down a *different* FK column, fabricating
    a join between two unrelated columns that merely reference the same target
    (e.g. two person-id columns). HAS_ATTRIBUTE and CONTAINS stay undirected.

    Returns a list of hop dicts:
        [{source_schema, source_table, source_column,
          target_schema, target_table, target_column}, ...]
    Returns [] when anchor == dest or no path exists.
    """
    if anchor_col_id == dest_col_id:
        return []

    # apoc.path.expandConfig is used instead of shortestPath because Cypher's
    # variable-length patterns apply a single direction to every relationship
    # type, whereas we need SEMANTIC_FK outgoing-only (">") while keeping
    # HAS_ATTRIBUTE and CONTAINS bidirectional. bfs + limit:1 yields the
    # shortest path; labelFilter "-Schema" keeps Schema nodes out of the path.
    path_query = """
    MATCH (col_anchor:Column {id: $anchor_col_id})
    MATCH (col_dest:Column {id: $dest_col_id})
    CALL apoc.path.expandConfig(col_anchor, {
        relationshipFilter: 'SEMANTIC_FK>|HAS_ATTRIBUTE|CONTAINS',
        labelFilter: '-Schema',
        terminatorNodes: [col_dest],
        bfs: true,
        uniqueness: 'NODE_GLOBAL',
        minLevel: 1,
        maxLevel: 30,
        limit: 1
    }) YIELD path
    RETURN [n IN nodes(path) | {
        id: n.id,
        name: n.name,
        label: labels(n)[0]
    }] AS path_nodes
    """
    try:
        rows = get_neo4j_conn().query_read(
            path_query,
            {"anchor_col_id": anchor_col_id, "dest_col_id": dest_col_id},
        )
    except Exception:
        logger.warning(
            "find_join_path: Neo4j query failed for %s -> %s",
            anchor_col_id,
            dest_col_id,
            exc_info=True,
        )
        return []

    if not rows:
        return []

    path_nodes: list[dict] = rows[0].get("path_nodes") or []
    col_nodes = [n for n in path_nodes if n.get("label") == "Column"]
    if len(col_nodes) < 2:
        return []

    col_ids = [n["id"] for n in col_nodes if n.get("id")]
    col_ctx = fetch_col_table_contexts(col_ids)
    database_names = {
        context.get("database_name")
        for context in col_ctx.values()
        if context.get("database_name")
    }
    if len(database_names) > 1:
        logger.warning(
            "find_join_path: rejected cross-database path %s -> %s (%s)",
            anchor_col_id,
            dest_col_id,
            ", ".join(sorted(database_names)),
        )
        return []

    hops: list[dict] = []
    for i in range(0, len(col_nodes) - 1, 2):
        src = col_nodes[i]
        tgt = col_nodes[i + 1]
        src_ctx = col_ctx.get(src.get("id") or "", {})
        tgt_ctx = col_ctx.get(tgt.get("id") or "", {})
        hops.append(
            {
                "source_database": src_ctx.get("database_name", ""),
                "source_schema": src_ctx.get("schema_name", ""),
                "source_table": src_ctx.get("table_name", ""),
                "source_column": src.get("name", ""),
                "target_database": tgt_ctx.get("database_name", ""),
                "target_schema": tgt_ctx.get("schema_name", ""),
                "target_table": tgt_ctx.get("table_name", ""),
                "target_column": tgt.get("name", ""),
            }
        )
    return hops
