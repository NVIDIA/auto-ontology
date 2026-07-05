# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write for Term nodes.

ColumnAttribute and SemanticFK operations live in gsf/dal/attributes.py.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Edges,
    Labels,
)
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.dal.users import resolve_table_filter
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    SEMANTIC_SOURCE,
)
from gsf.server.zones.constants import LABEL_ZONE, REL_ZONE_OF

logger = logging.getLogger(__name__)

# Shared RETURN projection for ColumnAttribute rows — keep fetch_column_attributes
# and fetch_column_attributes_by_term_id in sync when adding/removing fields.
_COLUMN_ATTRIBUTE_FIELDS = """attr.id            AS id,
               attr.name          AS name,
               attr.description   AS description,
               attr.term_name     AS term_name,
               attr.source_column AS source_column,
               attr.datatype      AS datatype,
               attr.table_id      AS table_id"""


def table_has_term(table_id: str) -> bool:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.name AS name LIMIT 1
        """,
        {"table_id": table_id, "source": SEMANTIC_SOURCE},
    )
    return bool(rows)


def get_term_for_table(table_id: str) -> str | None:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.name AS name LIMIT 1
        """,
        {"table_id": table_id, "source": SEMANTIC_SOURCE},
    )
    return rows[0]["name"] if rows else None


def get_slim_term_by_id(term_id: str) -> dict[str, str] | None:
    """Return ``{id, name}`` of a Term, or None."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (t:{LABEL_TERM} {{id: $term_id}})
        RETURN t.id AS id, t.name AS name LIMIT 1
        """,
        {"term_id": term_id},
    )
    return rows[0] if rows else None


def merge_term(
    name: str,
    description: str,
    table_id: str,
    synonyms: list[str] | None = None,
) -> str | None:
    """Merge the Term node and return its persistent ``id`` (UUID)."""
    rows = get_neo4j_conn().query_write(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})
        MERGE (term:{LABEL_TERM} {{name: $name, source: $source}})
        ON CREATE SET term.id = randomUUID()
        SET term.description = $description,
            term.synonyms = $synonyms
        MERGE (t)-[:{REL_REPRESENTS}]->(term)
        RETURN term.id AS id
        """,
        {
            "table_id": table_id,
            "name": name,
            "description": description,
            "synonyms": synonyms or [],
            "source": SEMANTIC_SOURCE,
        },
    )
    return rows[0]["id"] if rows else None


def fetch_term_synonyms(attr_ids: list[str]) -> dict[str, list[str]]:
    """Fetch synonyms for Terms connected to the given ColumnAttribute IDs.

    Returns a mapping of term_name -> list[synonym].
    """
    if not attr_ids:
        return {}
    query = """
    UNWIND $attr_ids AS attr_id
    MATCH (attr:ColumnAttribute {id: attr_id})-[:PROPERTY_OF]->(term:Term)
    WHERE term.synonyms IS NOT NULL AND size(term.synonyms) > 0
    RETURN DISTINCT term.name AS term_name, term.synonyms AS synonyms
    """
    try:
        rows = get_neo4j_conn().query_read(query, {"attr_ids": attr_ids})
    except Exception:
        logger.warning("fetch_term_synonyms: Neo4j query failed", exc_info=True)
        return {}
    result: dict[str, list[str]] = {}
    for row in rows:
        name = row.get("term_name")
        syns = row.get("synonyms") or []
        if name and syns:
            result[name] = [s for s in syns if s]
    return result


def fetch_all_terms_and_attributes(
    zone_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan semantic Term and ColumnAttribute nodes in Neo4j.

    When *zone_ids* is supplied the result is restricted to terms and attributes
    that belong to tables reachable through those zones.  Pass ``None`` (or omit)
    to return all data (admin / internal callers).
    """
    conn = get_neo4j_conn()
    table_filter, params = resolve_table_filter(
        zone_ids, "t.id", extra_params={"source": SEMANTIC_SOURCE}
    )

    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        {table_filter}
        RETURN DISTINCT term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id
        """,
        params,
    )
    attrs = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        {table_filter}
        RETURN attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.name AS column_name,
               col.sample_values AS sample_values,
               attr.id AS id
        """,
        params,
    )
    return terms, attrs


def get_full_term_by_id(term_id: str) -> dict[str, Any] | None:
    """Return a single Term node by its id with table count and zones, or None.

    ``zones`` is resolved via the attribute → column → table → zone path:
    a term participates in a zone when at least one of its ColumnAttributes
    is linked to a column whose parent table belongs to that zone.
    """
    conn = get_neo4j_conn()
    rows = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        OPTIONAL MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
        RETURN term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id,
               count(DISTINCT t) AS table_count
        LIMIT 1
        """,
        {"term_id": term_id},
    )
    if not rows:
        return None
    result = dict(rows[0])
    zone_rows = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (term)<-[:{REL_PROPERTY_OF}]-(:{LABEL_COLUMN_ATTRIBUTE})
              <-[:{REL_HAS_ATTRIBUTE}]-(:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(t:{Labels.TABLE})
        MATCH (z:{LABEL_ZONE})-[:{REL_ZONE_OF}]->(item)
        WHERE item = t
           OR (item)-[:{Edges.CONTAINS}*1..2]->(t)
        RETURN DISTINCT z.id    AS id,
                        z.name  AS name,
                        z.color AS color
        ORDER BY z.name
        """,
        {"term_id": term_id},
    )
    result["zones"] = [dict(r) for r in zone_rows]
    return result


def fetch_table_schema_map(database_name: str) -> dict[str, str]:
    """Return ``{table_name_lower: schema_name}`` for every table in *database_name*.

    Used by the SqlAttribute suggester to qualify bare table names in
    generated SELECT statements with their canonical schema prefix.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB} {{name: $db_name}})-[:{Edges.CONTAINS}]->
              (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
        RETURN t.name AS table_name, sch.name AS schema_name
        """,
        {"db_name": database_name},
    )
    return {
        row["table_name"].lower(): row["schema_name"]
        for row in rows
        if row.get("table_name") and row.get("schema_name")
    }


def fetch_terms_with_sqls(source: str) -> list[dict[str, Any]]:
    """Return every Term with the SQL queries from its connected tables.

    Each row contains:
      term_id, term_name, term_description,
      sqls — list of {sql_text, props} where *props* holds all Sql node
              properties (including count_monthly_YYYY_MM counters).

    Only terms that have at least one associated Sql query are returned.
    """
    return get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{source: $source}})
        MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
        MATCH (sql:{Labels.SQL})-[:{Edges.SQL}]->(t)
        WITH term,
             collect({{sql_text: sql.sql_full_query,
                       sql_id:   sql.id,
                       props:    properties(sql)}}) AS sqls
        RETURN term.id          AS term_id,
               term.name        AS term_name,
               term.description AS term_description,
               sqls
        """,
        {"source": source},
    )


def fetch_terms_and_attributes_for_table(
    table_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (terms, attrs) written for a single table — used for inline embedding."""
    conn = get_neo4j_conn()
    params = {"table_id": table_id, "source": SEMANTIC_SOURCE}
    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        RETURN term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id
        """,
        params,
    )
    attrs = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE} {{id: $table_id}})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        RETURN attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.sample_values AS sample_values,
               attr.id AS id
        """,
        params,
    )
    return terms, attrs


def fetch_column_attributes(
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return ColumnAttribute nodes, optionally restricted to *zone_ids*."""
    table_filter, params = resolve_table_filter(
        zone_ids, "attr.table_id", extra_params={"source": SEMANTIC_SOURCE}
    )

    return get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        {table_filter}
        RETURN {_COLUMN_ATTRIBUTE_FIELDS}
        ORDER BY attr.term_name, attr.name
        """,
        params,
    )


def fetch_column_attributes_by_term_id(term_id: str) -> list[dict[str, Any]]:
    """Return ColumnAttribute nodes for a single Term."""
    return get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{term_name: term.name, source: $source}})
        RETURN {_COLUMN_ATTRIBUTE_FIELDS}
        ORDER BY attr.name
        """,
        {"term_id": term_id, "source": SEMANTIC_SOURCE},
    )


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


def fetch_related_terms(term_id: str) -> list[dict[str, Any]]:
    """Return Term nodes related to *term_id* by co-location in the same table.

    Two terms are considered related when they are both connected to the same
    Table node — either directly via a REPRESENTS edge, or indirectly through
    the Column → ColumnAttribute → PROPERTY_OF path.

    Step 1 — collect every table connected to *term_id* (both paths).
    Step 2 — collect every other term connected to those same tables (both paths).
    """
    conn = get_neo4j_conn()
    params = {"term_id": term_id}

    # Step 1: tables for this term
    table_rows = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{id: $term_id}})
        RETURN ta.id AS table_id
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM} {{id: $term_id}})
        RETURN ta.id AS table_id
        """,
        params,
    )
    if not table_rows:
        return []

    table_ids = [r["table_id"] for r in table_rows if r.get("table_id")]
    if not table_ids:
        return []

    # Step 2: other terms in those tables
    term_rows = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term_b:{LABEL_TERM})
        WHERE ta.id IN $table_ids AND term_b.id <> $term_id
        RETURN DISTINCT term_b.id AS id, term_b.name AS name,
                        term_b.description AS description
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE})
              -[:{REL_PROPERTY_OF}]->(term_b:{LABEL_TERM})
        WHERE ta.id IN $table_ids AND term_b.id <> $term_id
        RETURN DISTINCT term_b.id AS id, term_b.name AS name,
                        term_b.description AS description
        """,
        {"table_ids": table_ids, "term_id": term_id},
    )
    return [dict(r) for r in term_rows]


def fetch_related_terms_counts(
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-term related-term counts for all terms.

    Builds a ``term_id → set(table_id)`` map and a ``table_id → set(term_id)``
    reverse map from Neo4j, then computes for each term the number of distinct
    other terms that share at least one table with it.

    When *zone_ids* is supplied the result is restricted to terms reachable
    through those zones.  Pass ``None`` to return counts for all terms.

    Each entry is ``{term_id: str, count: int}``.
    """
    conn = get_neo4j_conn()
    table_filter, params = resolve_table_filter(
        zone_ids, "ta.id", extra_params={"source": SEMANTIC_SOURCE}
    )

    pairs = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{source: $source}})
        {table_filter}
        RETURN term.id AS term_id, ta.id AS table_id
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {table_filter}
        RETURN term.id AS term_id, ta.id AS table_id
        """,
        params,
    )

    term_tables: dict[str, set[str]] = {}
    table_terms: dict[str, set[str]] = {}
    for row in pairs:
        tid = row.get("term_id")
        tab = row.get("table_id")
        if tid and tab:
            term_tables.setdefault(tid, set()).add(tab)
            table_terms.setdefault(tab, set()).add(tid)

    result: list[dict[str, Any]] = []
    for term_id, tables in term_tables.items():
        related: set[str] = set()
        for tab_id in tables:
            related.update(table_terms.get(tab_id, set()))
        related.discard(term_id)
        result.append({"term_id": term_id, "count": len(related)})
    return result
