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

from gsf.dal.cypher_fragments import column_description_expr
from gsf.dal.users import resolve_accessible_catalog_ids
from gsf.semantic.constants import (
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_PROPERTY_OF,
)
from gsf.server.sql_utils import SqlParseError
from gsf.server.zones.constants import (
    LABEL_ZONE_DISABLED,
    REL_ZONE_OF,
    ZONE_LABEL_PATTERN,
)

logger = logging.getLogger(__name__)

# Source values stored on SqlAttribute nodes.
SQL_ATTR_SOURCE_MANUAL = "manual"
SQL_ATTR_SOURCE_SQL = "sql"


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


class SqlAttributeNameConflict(Exception):
    """Raised when a write would collide with another SqlAttribute name."""


class SqlAttributeExpressionConflict(Exception):
    """Raised when a Term already has another SqlAttribute with this SQL."""


SqlAttributeSqlError = SqlParseError


# Shared RETURN projection — keep list/get/fetch-by-term queries in sync.
_SQL_ATTRIBUTE_FIELDS = """attr.id            AS id,
               attr.name          AS name,
               attr.description   AS description,
               attr.description_suggestion AS description_suggestion,
               attr.expression    AS expression,
               attr.source        AS source,
               sql.sql_full_query AS sql"""


def _sql_attr_zone_filter(
    zone_ids: list[str] | None,
    extra_params: dict[str, Any] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a Cypher ``WHERE`` clause and params for SqlAttribute zone scoping.

    The tables that scope a SqlAttribute are the ones its SQL actually
    references (``SqlAttribute -[HAS_SQL]-> Sql -[SQL]-> Table``), not the
    tables that represent its parent Term. The attribute is visible in the
    given zones only when every table its SQL touches is accessible. Pass a
    pre-resolved *data_ids_by_zone* (see ``resolve_accessible_catalog_ids``)
    to avoid a repeat Neo4j round trip when the caller already has it.
    """
    params = dict(extra_params or {})
    if zone_ids is None:
        return "", params
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    table_ids = list(resolved["table_ids"])
    attr_filter = (
        f"WHERE NOT EXISTS {{"
        f" (attr)-[:{Edges.HAS_SQL}]->(:{Labels.SQL})"
        f"-[:{Edges.SQL}]->(tbl:{Labels.TABLE})"
        f" WHERE NOT tbl.id IN $table_ids"
        f" }}"
    )
    params["table_ids"] = table_ids
    return attr_filter, params


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _query_sql_attributes(
    *,
    attr_id: str | None = None,
    term_id: str | None = None,
    zone_ids: list[str] | None = None,
    order_by: str | None = None,
) -> list[dict[str, Any]]:
    """Run the shared SqlAttribute ↔ Term ↔ Sql traversal behind every read below.

    Every SqlAttribute read joins the same three things: the SqlAttribute
    itself, its Term (via PROPERTY_OF), and its SQL text (via HAS_SQL) —
    this factors that join, the zone scoping (``_sql_attr_zone_filter``), and
    the shared RETURN projection into one place. Anchor on *attr_id* or
    *term_id* (mutually exclusive) to scope to one attribute/term, or leave
    both ``None`` for every SqlAttribute. Pass *order_by* as a raw ``ORDER
    BY`` expression (e.g. ``"attr.name"``); omitted when ``None``.
    """
    extra_params: dict[str, Any] = {}
    if attr_id is not None:
        extra_params["id"] = attr_id
    if term_id is not None:
        extra_params["term_id"] = term_id
    attr_filter, params = _sql_attr_zone_filter(zone_ids, extra_params=extra_params)

    if term_id is not None:
        anchor = f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(term)
        {attr_filter}
        """
    elif attr_id is not None:
        anchor = f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {attr_filter}
        """
    else:
        anchor = f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {attr_filter}
        """

    return get_neo4j_conn().query_read(
        f"""
        {anchor}
        MATCH (attr)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        RETURN {_SQL_ATTRIBUTE_FIELDS},
               term.id   AS term_id,
               term.name AS term_name
        {f"ORDER BY {order_by}" if order_by else ""}
        """,
        params,
    )


def list_sql_attributes() -> list[dict[str, Any]]:
    """Return every SqlAttribute with its connected Term and SQL text."""
    return _query_sql_attributes(order_by="attr.name")


def fetch_sql_attribute_counts(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-term SqlAttribute counts, zone-scoped when zone_ids are provided.

    Uses the same ``_sql_attr_zone_filter`` as every other SqlAttribute read:
    an attribute only counts when every table its SQL actually references is
    reachable through *zone_ids*. Pass a pre-resolved *data_ids_by_zone*
    (see ``resolve_accessible_catalog_ids``) when the caller already
    resolved *zone_ids* for this request, to skip a repeat Neo4j round trip.

    Each entry is ``{term_id: str, count: int}``. Terms with zero
    SqlAttributes are omitted.
    """
    attr_filter, params = _sql_attr_zone_filter(
        zone_ids, data_ids_by_zone=data_ids_by_zone
    )
    return get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {attr_filter}
        RETURN term.id AS term_id, count(DISTINCT attr) AS count
        """,
        params,
    )


def get_full_sql_attribute_by_id(
    attr_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return a single SqlAttribute with its term, SQL text and resolved zones.

    Zones are resolved through the tables the attribute's own SQL actually
    references (``SqlAttribute -[HAS_SQL]-> Sql -[SQL]-> Table``), matching
    ``_sql_attr_zone_filter`` — not through the parent Term's tables, which
    can differ from what the SQL text queries.  When *zone_ids* is supplied
    the attribute must pass the all-or-nothing zone check, otherwise
    ``None`` is returned so viewers cannot read out-of-zone attributes.
    When *zone_ids* is supplied, disabled zones are excluded outright from
    the result — a viewer never sees a disabled zone chip even if its id
    ended up in *zone_ids*. Pass ``None`` to skip the check and include
    disabled zones (with ``enabled: False``) so admins can see and manage
    them.
    """
    conn = get_neo4j_conn()
    rows = _query_sql_attributes(attr_id=attr_id, zone_ids=zone_ids)
    if not rows:
        return None
    result = dict(rows[0])

    zone_filter = (
        ""
        if zone_ids is None
        else f"AND z.id IN $zone_ids AND NOT z:{LABEL_ZONE_DISABLED}"
    )
    zone_params: dict[str, Any] = {"attr_id": attr_id}
    if zone_ids is not None:
        zone_params["zone_ids"] = zone_ids

    zone_rows = conn.query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
              -[:{Edges.HAS_SQL}]->(:{Labels.SQL})
              -[:{Edges.SQL}]->(t:{Labels.TABLE})
        MATCH (z:{ZONE_LABEL_PATTERN})-[:{REL_ZONE_OF}]->(item)
        WHERE (item = t
           OR (item)-[:{Edges.CONTAINS}*1..2]->(t))
              {zone_filter}
        RETURN DISTINCT z.id    AS id,
                        z.name  AS name,
                        z.color AS color,
                        NOT z:{LABEL_ZONE_DISABLED} AS enabled
        ORDER BY z.name
        """,
        zone_params,
    )
    result["zones"] = [dict(r) for r in zone_rows]
    return result


def fetch_sql_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return SqlAttribute nodes linked to a single Term via PROPERTY_OF.

    When *zone_ids* is supplied, each attribute is scoped independently via
    ``_sql_attr_zone_filter`` — an attribute of an otherwise-visible term is
    still excluded when its own SQL touches a table outside *zone_ids*.
    """
    return _query_sql_attributes(
        term_id=term_id, zone_ids=zone_ids, order_by="attr.name"
    )


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


def find_attr_by_expression(
    *,
    term_id: str,
    expression: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    """Return a same-term SqlAttribute with equivalent SQL, or None.

    Mirrors the legacy snippet validation behavior by ignoring case and
    collapsing whitespace before comparison.
    """
    normalized_expression = " ".join(expression.split()).lower()
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(:{LABEL_TERM} {{id: $term_id}})
        WHERE $exclude_id IS NULL OR attr.id <> $exclude_id
        RETURN attr.id AS id,
               attr.name AS name,
               attr.expression AS expression
        """,
        {"term_id": term_id, "exclude_id": exclude_id},
    )
    for row in rows:
        row_expression = row.get("expression")
        if not isinstance(row_expression, str):
            continue
        if " ".join(row_expression.split()).lower() == normalized_expression:
            return {"id": row["id"], "name": row["name"]}
    return None


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


def update_sql_attribute(
    attr_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    expression: str | None = None,
    source: str | None = None,
) -> None:
    """SET properties on an existing SqlAttribute node.

    Omitted fields (``None``) are left unchanged.
    """
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        SET attr.name        = coalesce($name, attr.name),
            attr.description = coalesce($description, attr.description),
            attr.expression  = coalesce($expression, attr.expression),
            attr.source      = coalesce($source, attr.source)
        """,
        {
            "id": attr_id,
            "name": name,
            "description": description,
            "expression": expression,
            "source": source,
        },
    )


def set_sql_attribute_description_suggestion(
    attr_id: str,
    description_suggestion: str,
) -> None:
    """SET the cached LLM description suggestion on a SqlAttribute node."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        SET attr.description_suggestion = $description_suggestion
        """,
        {"id": attr_id, "description_suggestion": description_suggestion},
    )


def clear_sql_attribute_description_suggestion(attr_id: str) -> None:
    """REMOVE the cached LLM description suggestion from a SqlAttribute node."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        REMOVE attr.description_suggestion
        """,
        {"id": attr_id},
    )


def clear_sql_attribute_description_suggestions_for_term(term_id: str) -> None:
    """REMOVE cached LLM suggestions from every SqlAttribute of a Term."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->
              (term:{LABEL_TERM} {{id: $term_id}})
        REMOVE attr.description_suggestion
        """,
        {"term_id": term_id},
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
    MATCH (tbl)<-[:CONTAINS]-(sch:Schema)
    MATCH (tbl)-[:CONTAINS]->(col:Column)
    WITH tbl, sch, collect({{name: col.name, data_type: col.data_type,
                             description: {column_description_expr("col")}}}) AS cols
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
