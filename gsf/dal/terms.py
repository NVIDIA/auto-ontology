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

from gsf.dal.users import get_accessible_catalog_ids_for_zones, resolve_table_filter
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

    *zone_ids* is a hard authorization boundary (the zones the requesting
    user has been granted access to), not a relevance filter.  A Term can be
    represented by more than one Table (see ``merge_term``), so when
    *zone_ids* is supplied a term is only returned when **every** table that
    represents it is reachable through those zones — a term that also
    represents an out-of-zone table is excluded entirely, mirroring the
    all-or-nothing rule used for CustomAnalysis
    (see ``gsf.dal.custom_analyses.list_custom_analyses``).  ColumnAttribute
    rows are naturally owned by exactly one table (via CONTAINS), so a plain
    accessible-table filter is correct for them without this check.  Pass
    ``None`` (or omit) to return all data (admin / internal callers).
    """
    conn = get_neo4j_conn()
    attr_filter, attr_params = resolve_table_filter(
        zone_ids, "t.id", extra_params={"source": SEMANTIC_SOURCE}
    )

    if zone_ids is None:
        term_filter = ""
        term_params: dict[str, Any] = {"source": SEMANTIC_SOURCE}
    else:
        table_ids = list(get_accessible_catalog_ids_for_zones(zone_ids)["table_ids"])
        term_filter = (
            f"WHERE NOT EXISTS {{"
            f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)"
            f" WHERE NOT other.id IN $table_ids"
            f" }}"
        )
        term_params = {"source": SEMANTIC_SOURCE, "table_ids": table_ids}

    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
        {term_filter}
        RETURN DISTINCT term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id
        """,
        term_params,
    )
    attrs = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->
              (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        {attr_filter}
        RETURN attr.name AS name,
               attr.description AS description,
               attr.term_name AS term_name,
               attr.source_column AS source_column,
               col.name AS column_name,
               col.sample_values AS sample_values,
               attr.id AS id
        """,
        attr_params,
    )
    return terms, attrs


def get_full_term_by_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return a single Term node by its id with table count and zones, or None.

    *zone_ids* is a hard authorization boundary, not a relevance filter. A
    Term can be represented by more than one Table (see ``merge_term``), so
    when *zone_ids* is supplied the term is only returned when **every**
    table that represents it is reachable through those zones — matching the
    all-or-nothing rule used by ``fetch_all_terms_and_attributes`` for the
    ``/terms`` list, so a viewer can't bypass list-level zone scoping by
    requesting a term directly by id.  Returns ``None`` (→ 404) when the
    check fails.  Pass ``None`` to skip the check (admin / internal callers).

    ``zones`` is resolved via the attribute → column → table → zone path:
    a term participates in a zone when at least one of its ColumnAttributes
    is linked to a column whose parent table belongs to that zone.  When
    *zone_ids* is supplied, the returned ``zones`` are additionally
    restricted to that set, so a viewer never sees zone names/colors they
    don't have access to.
    """
    conn = get_neo4j_conn()
    if zone_ids is None:
        term_filter = ""
        term_params: dict[str, Any] = {"term_id": term_id}
    else:
        table_ids = list(get_accessible_catalog_ids_for_zones(zone_ids)["table_ids"])
        term_filter = (
            f"WHERE NOT EXISTS {{"
            f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)"
            f" WHERE NOT other.id IN $table_ids"
            f" }}"
        )
        term_params = {"term_id": term_id, "table_ids": table_ids}

    rows = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        {term_filter}
        OPTIONAL MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
        RETURN term.name AS name, term.description AS description,
               term.synonyms AS synonyms, term.id AS id,
               count(DISTINCT t) AS table_count
        LIMIT 1
        """,
        term_params,
    )
    if not rows:
        return None
    result = dict(rows[0])

    zone_filter = "" if zone_ids is None else "AND z.id IN $zone_ids"
    zone_params: dict[str, Any] = {"term_id": term_id}
    if zone_ids is not None:
        zone_params["zone_ids"] = zone_ids

    zone_rows = conn.query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (term)<-[:{REL_PROPERTY_OF}]-(:{LABEL_COLUMN_ATTRIBUTE})
              <-[:{REL_HAS_ATTRIBUTE}]-(:{Labels.COLUMN})
              <-[:{Edges.CONTAINS}]-(t:{Labels.TABLE})
        MATCH (z:{LABEL_ZONE})-[:{REL_ZONE_OF}]->(item)
        WHERE (item = t
           OR (item)-[:{Edges.CONTAINS}*1..2]->(t))
              {zone_filter}
        RETURN DISTINCT z.id    AS id,
                        z.name  AS name,
                        z.color AS color
        ORDER BY z.name
        """,
        zone_params,
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


def fetch_column_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return ColumnAttribute nodes for a single Term.

    *zone_ids* is a hard authorization boundary; when supplied, only
    attributes owned by tables reachable through those zones are returned —
    a ColumnAttribute is owned by exactly one table, so a plain membership
    filter is sufficient here (no all-or-nothing check needed).
    """
    table_filter, params = resolve_table_filter(
        zone_ids,
        "attr.table_id",
        extra_params={"term_id": term_id, "source": SEMANTIC_SOURCE},
    )
    return get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{term_name: term.name, source: $source}})
        {table_filter}
        RETURN {_COLUMN_ATTRIBUTE_FIELDS}
        ORDER BY attr.name
        """,
        params,
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


def fetch_related_terms(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return Term nodes related to *term_id* by co-location in the same table.

    Two terms are considered related when they are both connected to the same
    Table node — either directly via a REPRESENTS edge, or indirectly through
    the Column → ColumnAttribute → PROPERTY_OF path.

    Step 1 — collect every table connected to *term_id* (both paths).
    Step 2 — collect every other term connected to those same tables (both paths).

    *zone_ids* is a hard authorization boundary, not a relevance filter.
    When supplied: Step 1 only considers tables reachable through those
    zones, so sharing a table the caller can't see never surfaces a related
    term.  Step 2 additionally requires that a candidate related term's own
    REPRESENTS-tables are *all* within *zone_ids* — matching the
    all-or-nothing rule used by ``fetch_all_terms_and_attributes`` for the
    ``/terms`` list — so a related term the caller couldn't otherwise open
    (it would 404 via ``get_full_term_by_id``) is never shown as a chip.
    Pass ``None`` to skip zone scoping (admin / internal callers).
    """
    conn = get_neo4j_conn()
    params: dict[str, Any] = {"term_id": term_id}

    if zone_ids is None:
        step1_filter = ""
    else:
        params["zone_table_ids"] = list(
            get_accessible_catalog_ids_for_zones(zone_ids)["table_ids"]
        )
        step1_filter = "AND ta.id IN $zone_table_ids"

    # Step 1: tables for this term
    table_rows = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{id: $term_id}})
        WHERE true {step1_filter}
        RETURN ta.id AS table_id
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM} {{id: $term_id}})
        WHERE true {step1_filter}
        RETURN ta.id AS table_id
        """,
        params,
    )
    if not table_rows:
        return []

    shared_table_ids = [r["table_id"] for r in table_rows if r.get("table_id")]
    if not shared_table_ids:
        return []

    term_b_filter = (
        ""
        if zone_ids is None
        else (
            f"AND NOT EXISTS {{"
            f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term_b)"
            f" WHERE NOT other.id IN $zone_table_ids"
            f" }}"
        )
    )
    step2_params: dict[str, Any] = {
        "shared_table_ids": shared_table_ids,
        "term_id": term_id,
    }
    if zone_ids is not None:
        step2_params["zone_table_ids"] = params["zone_table_ids"]

    # Step 2: other terms in those tables
    term_rows = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term_b:{LABEL_TERM})
        WHERE ta.id IN $shared_table_ids AND term_b.id <> $term_id
              {term_b_filter}
        RETURN DISTINCT term_b.id AS id, term_b.name AS name,
                        term_b.description AS description
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE})
              -[:{REL_PROPERTY_OF}]->(term_b:{LABEL_TERM})
        WHERE ta.id IN $shared_table_ids AND term_b.id <> $term_id
              {term_b_filter}
        RETURN DISTINCT term_b.id AS id, term_b.name AS name,
                        term_b.description AS description
        """,
        step2_params,
    )
    return [dict(r) for r in term_rows]


def fetch_related_terms_counts(
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-term related-term counts for all terms.

    Builds a ``term_id → set(table_id)`` map and a ``table_id → set(term_id)``
    reverse map from Neo4j, then computes for each term the number of distinct
    other terms that share at least one table with it.

    *zone_ids* is a hard authorization boundary, not a relevance filter. A
    Term can be represented by more than one Table (see ``merge_term``), so
    a term only contributes pairs — via **either** the REPRESENTS or the
    ColumnAttribute path — when **every** table that represents it (via
    REPRESENTS) is reachable through *zone_ids*.  This is the same
    all-or-nothing rule applied consistently to both paths — a term whose
    ColumnAttribute happens to live on an in-zone table doesn't get a free
    pass if it also REPRESENTS an out-of-zone table — so this function stays
    in agreement with ``fetch_all_terms_and_attributes`` (the ``/terms``
    list) and ``fetch_related_terms`` (the single-term related list): a
    term's count here always matches how many terms actually show up on its
    detail page.  Pass ``None`` to return counts for all terms (admin /
    internal callers).

    Each entry is ``{term_id: str, count: int}``.
    """
    conn = get_neo4j_conn()
    if zone_ids is None:
        filter_clause = ""
        params: dict[str, Any] = {"source": SEMANTIC_SOURCE}
    else:
        table_ids = list(get_accessible_catalog_ids_for_zones(zone_ids)["table_ids"])
        filter_clause = (
            f"WHERE ta.id IN $table_ids"
            f" AND NOT EXISTS {{"
            f" (other:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)"
            f" WHERE NOT other.id IN $table_ids"
            f" }}"
        )
        params = {"source": SEMANTIC_SOURCE, "table_ids": table_ids}

    pairs = conn.query_read(
        f"""
        MATCH (ta:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{source: $source}})
        {filter_clause}
        RETURN term.id AS term_id, ta.id AS table_id
        UNION
        MATCH (ta:{Labels.TABLE})-[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}]->(:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
              -[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        {filter_clause}
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
