"""Neo4j read/write for semantic entities — single source of truth, no in-memory graph."""

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
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
    SEMANTIC_SOURCE,
)

logger = logging.getLogger(__name__)


def fetch_all_tables() -> list[dict[str, Any]]:
    """Return tables that have not yet been assigned a Term."""
    return get_neo4j_conn().query_read(
        f"""
        MATCH (t:{Labels.TABLE})
        WHERE NOT (t)-[:{REL_REPRESENTS}]->()
        RETURN t.id AS id, t.name AS name, t.description AS description
        ORDER BY t.name
        """
    )


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


def fetch_all_terms_and_attributes() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    """Scan all semantic nodes in Neo4j for embedding.

    Each attribute row includes ``sample_values`` (the JSON string stored on
    the physical Column node by the ingestion pipeline), which is incorporated
    into the embedding text.
    """
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
    """Return a single Term node by its id with table count, or None if not found."""
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
    return rows[0] if rows else None


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
