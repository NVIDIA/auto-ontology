# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write for Term nodes.

ColumnAttribute and SemanticFK operations live in gsf/neo4j/attributes.py.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Edges,
    Labels,
)
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    LABEL_ZONE,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
    REL_ZONE_OF,
    SEMANTIC_SOURCE,
)

logger = logging.getLogger(__name__)


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


def fetch_all_terms_and_attributes() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    """Scan all semantic Term and ColumnAttribute nodes in Neo4j for embedding."""
    conn = get_neo4j_conn()
    params = {"source": SEMANTIC_SOURCE}
    terms = conn.query_read(
        f"""
        MATCH (t:{Labels.TABLE})-[:{REL_REPRESENTS}]->
              (term:{LABEL_TERM} {{source: $source}})
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


def fetch_term_by_id(term_id: str) -> dict[str, Any] | None:
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


def fetch_column_attributes_with_fk_count() -> list[dict[str, Any]]:
    """Return all ColumnAttribute nodes with:

    * ``fk_count``   — number of Column nodes that point to this attribute
                       via a SEMANTIC_FK edge (i.e. it is used as a FK target).
    * ``is_primary_key`` — True when at least one FK references this attribute,
                           meaning it acts as the primary-key anchor for its Term.
    """
    return get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{source: $source}})
        OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_SEMANTIC_FK}]->(attr)
        WITH attr, count(col) AS fk_count
        RETURN attr.id           AS id,
               attr.name         AS name,
               attr.description  AS description,
               attr.term_name    AS term_name,
               attr.source_column AS source_column,
               attr.datatype     AS datatype,
               attr.table_id     AS table_id,
               fk_count,
               fk_count > 0      AS is_primary_key
        ORDER BY attr.term_name, attr.name
        """,
        {"source": SEMANTIC_SOURCE},
    )


def fetch_column_attributes_by_term_id(term_id: str) -> list[dict[str, Any]]:
    """Return ColumnAttribute nodes for a single Term, enriched with FK count."""
    return get_neo4j_conn().query_read(
        f"""
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{term_name: term.name, source: $source}})
        OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_SEMANTIC_FK}]->(attr)
        WITH attr, count(col) AS fk_count
        RETURN attr.id            AS id,
               attr.name          AS name,
               attr.description   AS description,
               attr.term_name     AS term_name,
               attr.source_column AS source_column,
               attr.datatype      AS datatype,
               attr.table_id      AS table_id,
               fk_count,
               fk_count > 0       AS is_primary_key
        ORDER BY attr.name
        """,
        {"term_id": term_id, "source": SEMANTIC_SOURCE},
    )


def find_unlinked_fk_columns() -> list[dict[str, Any]]:
    """Return Column nodes that have no SEMANTIC_FK edge and no HAS_ATTRIBUTE edge.

    These are FK columns that have not yet been linked to a ColumnAttribute.
    Each row includes ``fk_target_col_id`` (the id of the declared FK target
    Column, or ``None`` when no FOREIGN_KEY edge exists).
    """
    return get_neo4j_conn().query_read(
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
