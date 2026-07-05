# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write for Term nodes.

ColumnAttribute and SemanticFK operations live in gsf/dal/attributes.py.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_REPRESENTS,
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


def get_term_by_id(term_id: str) -> dict[str, str] | None:
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
