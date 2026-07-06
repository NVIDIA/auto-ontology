# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for SqlAttribute / Sql subgraph.

Contains only functions that call ``get_neo4j_conn()`` directly.

Orchestration (SQL validation, connector resolution, VDB lifecycle)
lives in ``gsf/server/sql_attributes/service.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import (
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_PROPERTY_OF,
)
from gsf.server.sql_utils import SqlParseError

logger = logging.getLogger(__name__)

# Source values stored on SqlAttribute nodes.
SQL_ATTR_SOURCE_MANUAL = "manual"
SQL_ATTR_SOURCE_SQL = "sql"


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


class SqlAttributeNameConflict(Exception):
    """Raised when a write would collide with another SqlAttribute name."""


SqlAttributeSqlError = SqlParseError


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_sql_attributes() -> list[dict[str, Any]]:
    """Return every SqlAttribute with its connected Term and SQL text."""
    return get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})
        MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        MATCH (attr)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        RETURN attr.id          AS id,
               attr.name        AS name,
               attr.description AS description,
               attr.expression  AS expression,
               attr.source      AS source,
               term.id          AS term_id,
               term.name        AS term_name,
               sql.sql_full_query AS sql
        ORDER BY attr.name
        """
    )


def get_sql_attribute(attr_id: str) -> dict[str, Any] | None:
    """Return a single SqlAttribute by id, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        MATCH (attr)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        RETURN attr.id          AS id,
               attr.name        AS name,
               attr.description AS description,
               attr.expression  AS expression,
               attr.source      AS source,
               term.id          AS term_id,
               term.name        AS term_name,
               sql.sql_full_query AS sql
        """,
        {"id": attr_id},
    )
    return rows[0] if rows else None


def find_attr_by_name(name: str, exclude_id: str | None) -> dict[str, str] | None:
    """Return ``{id, name}`` of a SqlAttribute using *name*, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (a:{LABEL_SQL_ATTRIBUTE} {{name: $name}})
        WHERE $exclude_id IS NULL OR a.id <> $exclude_id
        RETURN a.id AS id, a.name AS name
        LIMIT 1
        """,
        {"name": name, "exclude_id": exclude_id},
    )
    return {"id": rows[0]["id"], "name": rows[0]["name"]} if rows else None


def get_sql_attribute_by_id(attr_id: str) -> str | None:
    """Return the id of the SqlAttribute, or None if it doesn't exist."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (a:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        RETURN a.id AS id
        LIMIT 1
        """,
        {"id": attr_id},
    )
    return rows[0]["id"] if rows else None


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def detach_existing_sql_edges(attr_id: str) -> None:
    """Drop every HAS_SQL edge leaving the SqlAttribute."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (a:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
              -[r:{Edges.HAS_SQL}]->(:{Labels.SQL})
        DELETE r
        """,
        {"id": attr_id},
    )


def link_to_term(attr_id: str, term_id: str) -> None:
    """Set the PROPERTY_OF edge from SqlAttribute to Term, replacing any prior link."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
        OPTIONAL MATCH (attr)-[old:{REL_PROPERTY_OF}]->(existing)
        WHERE existing.id <> $term_id
        DELETE old
        WITH attr
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MERGE (attr)-[:{REL_PROPERTY_OF}]->(term)
        """,
        {"attr_id": attr_id, "term_id": term_id},
    )


def update_sql_attribute_props(
    attr_id: str,
    *,
    name: str,
    description: str,
    expression: str,
    source: str,
) -> None:
    """SET properties on an existing SqlAttribute node."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        SET attr.name        = $name,
            attr.description = $description,
            attr.expression  = $expression,
            attr.source      = $source
        """,
        {
            "id": attr_id,
            "name": name,
            "description": description,
            "expression": expression,
            "source": source,
        },
    )


def delete_sql_attribute_node(attr_id: str) -> None:
    """DETACH DELETE the SqlAttribute node."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        DETACH DELETE attr
        """,
        {"id": attr_id},
    )


# ---------------------------------------------------------------------------
# Retrieval-time helpers
# ---------------------------------------------------------------------------


def fetch_sql_attributes_with_sql(attr_ids: list[str]) -> list[dict[str, str]]:
    """Fetch id, name, description, expression, and SQL for each SqlAttribute.

    Returns a list of dicts with keys: id, name, description, expression, sql.
    """
    if not attr_ids:
        return []

    query = f"""
    UNWIND $ids AS attr_id
    MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: attr_id}})
    MATCH (attr)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
    MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
    RETURN attr.id AS attr_id, attr.name AS name,
           attr.description AS description,
           attr.expression AS expression,
           sql.sql_full_query AS sql_text,
           term.name AS term_name
    """
    try:
        rows = get_neo4j_conn().query_read(query, {"ids": attr_ids})
    except Exception:
        logger.warning(
            "fetch_sql_attributes_with_sql: Neo4j query failed",
            exc_info=True,
        )
        return []

    seen_ids: set[str] = set()
    result: list[dict[str, str]] = []
    for row in rows:
        aid = row.get("attr_id") or ""
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        result.append(
            {
                "id": aid,
                "name": row.get("name") or "",
                "description": row.get("description") or "",
                "expression": row.get("expression") or "",
                "sql": row.get("sql_text") or "",
                "term_name": row.get("term_name") or "",
            }
        )
    return result


def fetch_tables_from_sql_attributes(
    attr_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch Tables referenced by SqlAttribute nodes via HAS_SQL -> Sql -> SQL -> Table."""
    if not attr_ids:
        return []
    query = f"""
    UNWIND $ids AS attr_id
    MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: attr_id}})
          -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
          -[:{Edges.SQL}]->(tbl:{Labels.TABLE})
    OPTIONAL MATCH (tbl)<-[:CONTAINS]-(sch:Schema)
    OPTIONAL MATCH (tbl)-[:CONTAINS]->(col:Column)
    WITH tbl, sch, collect({{name: col.name, data_type: col.data_type,
                             description: col.description}}) AS cols
    RETURN tbl.id AS id, tbl.name AS name, tbl.description AS description,
           sch.name AS schema_name, cols
    """
    try:
        rows = get_neo4j_conn().query_read(query, {"ids": attr_ids})
    except Exception:
        logger.warning(
            "fetch_tables_from_sql_attributes: Neo4j query failed",
            exc_info=True,
        )
        return []
    tables: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        tid = row.get("id") or ""
        if not tid or tid in seen:
            continue
        seen.add(tid)
        cols = [c for c in (row.get("cols") or []) if c.get("name")]
        tables.append(
            {
                "id": tid,
                "name": row.get("name") or "",
                "description": row.get("description") or "",
                "schema_name": row.get("schema_name") or "",
                "label": Labels.TABLE,
                "columns": cols,
            }
        )
    return tables


# ---------------------------------------------------------------------------
# Embedding data fetch
# ---------------------------------------------------------------------------


def fetch_sql_attribute_docs(attr_id: str) -> list[dict[str, Any]]:
    """Fetch one SqlAttribute from Neo4j as embedding-ready docs.

    Returns a list of dicts with keys: text, name, label, id.
    Empty list when the attribute or its Sql node is missing.
    """
    result = get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
              -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        RETURN collect({{
            text: 'sql_attribute: ' + attr.name +
                  CASE WHEN attr.description IS NOT NULL
                       AND trim(toString(attr.description)) <> ''
                       THEN ', description: ' + attr.description
                       ELSE '' END +
                  CASE WHEN term.name IS NOT NULL
                       THEN ', term: ' + term.name
                       ELSE '' END +
                  ', sql: ' + sql.sql_full_query,
            name: attr.name,
            label: labels(attr)[0],
            id: attr.id
        }}) AS docs
        """,
        {"attr_id": attr_id},
    )
    docs = result[0].get("docs") if result else None
    return docs or []
