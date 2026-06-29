# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — Neo4j queries for ``SqlAttribute`` nodes.

A SqlAttribute represents a derived business metric or formula
(e.g. ``SUM(amount) / COUNT(DISTINCT customer_id)``) linked to a
semantic ``Term`` via a ``PROPERTY_OF`` edge and to a ``Sql`` node
via ``HAS_SQL`` (same pattern as ``CustomAnalysis``).

Graph shape::

    (term:Term) <-[:PROPERTY_OF]- (attr:SqlAttribute)
                                           |
                                     [:HAS_SQL]
                                           v
                                      (sql:Sql) -[:SQL]-> (Table/Column)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Edges,
    Labels,
    Props,
)
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.operators.vdb import IngestVdbOperator
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed

from gsf.semantic.constants import (
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_PROPERTY_OF,
)
from gsf.connectors import get_connectors
from gsf.server.sql_utils import SqlParseError, get_dialects, get_schemas, validate_sql
from gsf.utils import get_embed_params
from gsf.vdb import get_semantic_vdb

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)


class SqlAttributeNameConflict(Exception):
    pass


SqlAttributeSqlError = SqlParseError


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def list_sql_attributes() -> list[dict[str, Any]]:
    """Return every SqlAttribute with its connected Term and SQL text."""
    return get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE})
        OPTIONAL MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        OPTIONAL MATCH (attr)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
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
        OPTIONAL MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
        OPTIONAL MATCH (attr)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
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


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _find_attr_by_name(name: str, exclude_id: str | None) -> dict[str, str] | None:
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


def _persist_attr_with_sql(
    attr_node: Neo4jNode,
    sql: str,
    query_obj: Any,
) -> dict[str, Any]:
    """Create the Sql node via add_query and link it to the SqlAttribute."""
    query_obj.sql_node.match_props = {"sql_full_query": sql}

    edge_props = {Props.ANALYSIS_ID: attr_node.get_id()}
    query_obj.edges.append((attr_node, query_obj.sql_node, edge_props))

    add_query(query_obj.get_edges())

    props = attr_node.get_properties()
    return {
        "id": attr_node.get_id(),
        "name": props["name"],
        "description": props.get("description", ""),
        "expression": props.get("expression", ""),
        "source": props.get("source", ""),
        "sql": sql,
    }


def _detach_existing_sql_edges(attr_id: str) -> None:
    """Drop every HAS_SQL edge leaving the SqlAttribute."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (a:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
              -[r:{Edges.HAS_SQL}]->(:{Labels.SQL})
        DELETE r
        """,
        {"id": attr_id},
    )


def _link_to_term(attr_id: str, term_id: str) -> None:
    """Create PROPERTY_OF edge from SqlAttribute to Term (replacing any old one)."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
        OPTIONAL MATCH (attr)-[old:{REL_PROPERTY_OF}]->()
        DELETE old
        WITH attr
        MATCH (term:{LABEL_TERM} {{id: $term_id}})
        MERGE (attr)-[:{REL_PROPERTY_OF}]->(term)
        """,
        {"attr_id": attr_id, "term_id": term_id},
    )


def _resolve_connector(connector: str) -> str:
    """Look up a loaded connector by name and return its ``database_name``.

    Raises ``ValueError`` if no connector matches.
    """
    for c in get_connectors():
        if getattr(c, "database_name", None) == connector:
            return c.database_name
    raise ValueError(f"Connector {connector!r} not found among loaded connectors")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_sql_attribute(
    embed_params: "EmbedParams",
    vdb: "VDB",
    attr_id: str,
    database_name: str | None = None,
) -> None:
    """Fetch one SqlAttribute from Neo4j, embed it, and append to *vdb*.

    The embedded text includes the term name so retrieval can match on
    business-level semantics, not just the raw SQL expression.
    """
    import pandas as pd

    result = get_neo4j_conn().query_read(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $attr_id}})
              -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        OPTIONAL MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
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
    if not docs:
        logger.info(
            "No SqlAttribute rows found for attr_id=%r; skipping VDB upsert.",
            attr_id,
        )
        return

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
        tabular_fields = {
            "id": node_id,
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "source_path": path,
            "database_name": database_name,
        }
        rows.append(
            {
                "text": (item.get("text") or "").strip(),
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {
                    **tabular_fields,
                    "content_metadata": dict(tabular_fields),
                },
            }
        )
    df = pd.DataFrame(rows)

    before = time.time()
    embedded = embed_text_main_text_embed(
        df,
        model_name=embed_params.model_name,
        embed_invoke_url=embed_params.embed_invoke_url,
        api_key=embed_params.api_key,
        embed_modality=embed_params.embed_modality,
    )

    with_embeddings = [
        row
        for row in embedded.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        raise RuntimeError(
            f"Embedding step produced 0/{len(embedded)} SqlAttribute rows "
            f"with embeddings; check upstream embed errors (often a transient "
            f"{embed_params.embed_invoke_url} 5xx)."
        )

    IngestVdbOperator(vdb=vdb)(with_embeddings)
    logger.info(
        "Embedded and appended %d/%d SqlAttribute row(s) via %s in %.2fs.",
        len(with_embeddings),
        len(embedded),
        type(vdb).__name__,
        time.time() - before,
    )


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def create_sql_attribute(
    *,
    name: str,
    description: str,
    expression: str,
    term_id: str,
    connector: str,
    source: str = "manual",
) -> dict[str, Any]:
    """Create a SqlAttribute, its Sql node, link to a Term, and embed."""
    database_name = _resolve_connector(connector)

    conflict = _find_attr_by_name(name, exclude_id=None)
    if conflict is not None:
        raise SqlAttributeNameConflict(
            f"SqlAttribute with name {name!r} already exists (id={conflict['id']!r})"
        )

    conn = get_neo4j_conn()
    term_rows = conn.query_read(
        f"""
        MATCH (t:{LABEL_TERM} {{id: $term_id}})
        RETURN t.id AS id, t.name AS name LIMIT 1
        """,
        {"term_id": term_id},
    )
    if not term_rows:
        raise ValueError(f"Term with id {term_id!r} not found")

    query_obj = validate_sql(
        expression,
        get_dialects(database_name),
        get_schemas(database_name),
    )

    attr_node = Neo4jNode(
        name=name,
        label=LABEL_SQL_ATTRIBUTE,
        props={
            "name": name,
            "description": description,
            "expression": expression,
            "source": source,
        },
        match_props={"name": name},
    )

    row = _persist_attr_with_sql(attr_node, expression, query_obj)
    _link_to_term(row["id"], term_id)
    row["term_name"] = term_rows[0]["name"]
    row["term_id"] = term_id
    row["database_name"] = database_name

    vdb = get_semantic_vdb()
    _embed_sql_attribute(
        embed_params=get_embed_params(),
        vdb=vdb,
        attr_id=row["id"],
        database_name=database_name,
    )

    return row


def update_sql_attribute(
    *,
    attr_id: str,
    name: str,
    description: str,
    expression: str,
    term_id: str,
    connector: str,
    source: str = "manual",
) -> dict[str, Any] | None:
    """Replace a SqlAttribute, re-parse SQL, re-link Term, and re-embed."""
    database_name = _resolve_connector(connector)
    conn = get_neo4j_conn()

    existing = conn.query_read(
        f"""
        MATCH (a:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        RETURN a.id AS id
        """,
        {"id": attr_id},
    )
    if not existing:
        return None

    conflict = _find_attr_by_name(name, exclude_id=attr_id)
    if conflict is not None:
        raise SqlAttributeNameConflict(
            f"SqlAttribute with name {name!r} already exists (id={conflict['id']!r})"
        )

    term_rows = conn.query_read(
        f"""
        MATCH (t:{LABEL_TERM} {{id: $term_id}})
        RETURN t.id AS id, t.name AS name LIMIT 1
        """,
        {"term_id": term_id},
    )
    if not term_rows:
        raise ValueError(f"Term with id {term_id!r} not found")

    query_obj = validate_sql(
        expression,
        get_dialects(database_name),
        get_schemas(database_name),
    )

    _detach_existing_sql_edges(attr_id)

    conn.query_write(
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

    attr_node = Neo4jNode(
        name=name,
        label=LABEL_SQL_ATTRIBUTE,
        props={
            "name": name,
            "description": description,
            "expression": expression,
            "source": source,
        },
        match_props={"id": attr_id},
        existing_id=attr_id,
    )

    _persist_attr_with_sql(attr_node, expression, query_obj)
    _link_to_term(attr_id, term_id)

    vdb = get_semantic_vdb()
    vdb.delete_by_id(attr_id)
    _embed_sql_attribute(
        embed_params=get_embed_params(),
        vdb=vdb,
        attr_id=attr_id,
        database_name=database_name,
    )

    return {
        "id": attr_id,
        "name": name,
        "description": description,
        "expression": expression,
        "source": source,
        "database_name": database_name,
        "term_name": term_rows[0]["name"],
        "term_id": term_id,
        "sql": expression,
    }


def delete_sql_attribute(attr_id: str) -> dict[str, Any] | None:
    """Delete a SqlAttribute, its edges, and its VDB embedding."""
    existing = get_neo4j_conn().query_read(
        f"""
        MATCH (a:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        RETURN a.id AS id
        LIMIT 1
        """,
        {"id": attr_id},
    )
    if not existing:
        return None

    get_neo4j_conn().query_write(
        f"""
        MATCH (attr:{LABEL_SQL_ATTRIBUTE} {{id: $id}})
        WITH attr, attr.id AS id
        DETACH DELETE attr
        RETURN id
        """,
        {"id": attr_id},
    )

    get_semantic_vdb().delete_by_id(attr_id)

    return {"id": attr_id}
