# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for CustomAnalysis / Sql subgraph.

Contains only functions that call ``graph()`` directly.

Non-Neo4j helpers remain in their original locations:
  - get_custom_analyses_ids, build_custom_analyses_section,
    get_relevant_queries  →  retrieval/data_access/custom_analyses.py
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels

from gsf.dal.cypher_fragments import column_description_expr
from gsf.dal.neo4j_tx import graph
from gsf.dal.users import resolve_accessible_catalog_ids
from gsf.server.sql_utils import SqlParseError

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain errors (raised by write helpers; surfaced as HTTP 409/422 by router)
# ---------------------------------------------------------------------------


class CustomAnalysisNameConflict(Exception):
    """Raised when a write would collide with another CustomAnalysis name."""


class CustomAnalysisSqlConflict(Exception):
    """Raised when the submitted SQL is already linked to a different analysis."""


class CustomAnalysisSqlError(SqlParseError):
    """Raised when the SQL can't be parsed against the current catalog."""


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_custom_analyses(
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return ``CustomAnalysis`` nodes joined with their ``Sql`` node.

    *zone_ids* is a hard authorization boundary (the zones the requesting
    user has been granted access to), not a relevance filter.  When supplied,
    an analysis is only returned when **every** table its SQL references is
    reachable through those zones — an analysis that touches even one
    out-of-zone table is excluded entirely, since its SQL text would
    otherwise leak the names/columns of tables the caller isn't authorized
    to see.  Pass ``None`` (or omit) to return all analyses (admin /
    internal callers).
    """
    params: dict[str, Any] = {}
    if zone_ids is not None:
        data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
        params["table_ids"] = list(data_ids_by_zone["table_ids"])
        zone_filter = (
            f"WHERE NOT EXISTS {{"
            f" (sql)-[:{Edges.SQL}]->(tbl:{Labels.TABLE})"
            f" WHERE NOT tbl.id IN $table_ids"
            f" }}"
        )
    else:
        zone_filter = ""

    rows = graph().query_read(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        {zone_filter}
        WITH ca, sql
        ORDER BY ca.name

        RETURN collect({{
            id: ca.id,
            name: ca.name,
            description: ca.description,
            sql: sql.sql_full_query
        }}) AS analyses
        """,
        params,
    )
    return rows[0]["analyses"]


def find_analysis_by_name(
    name: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    rows = graph().query_read(
        f"""
        MATCH (other:{Labels.CUSTOM_ANALYSIS} {{name: $name}})
        WHERE $exclude_id IS NULL OR other.id <> $exclude_id
        RETURN other.id AS id, other.name AS name
        LIMIT 1
        """,
        {"name": name, "exclude_id": exclude_id},
    )
    if not rows:
        return None
    return {"id": rows[0]["id"], "name": rows[0]["name"]}


def find_analysis_by_sql(
    sql: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    rows = graph().query_read(
        f"""
        MATCH (other:{Labels.CUSTOM_ANALYSIS})
              -[:{Edges.HAS_SQL}]->(:{Labels.SQL} {{sql_full_query: $sql}})
        WHERE $exclude_id IS NULL OR other.id <> $exclude_id
        RETURN other.id AS id, other.name AS name
        LIMIT 1
        """,
        {"sql": sql, "exclude_id": exclude_id},
    )
    if not rows:
        return None
    return {"id": rows[0]["id"], "name": rows[0]["name"]}


def detach_existing_sql_edges(analysis_id: str) -> None:
    """Drop every ``HAS_SQL`` edge leaving the given CustomAnalysis."""
    graph().query_write(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
              -[r:{Edges.HAS_SQL}]->(:{Labels.SQL})
        DELETE r
        """,
        {"analysis_id": analysis_id},
    )


def fetch_custom_analyses() -> list[dict[str, str]]:
    """Fetch all CustomAnalysis nodes from Neo4j and return as domain rules.

    Each analysis becomes ``{"name": <name>, "description": <sql>}``.
    """
    query = (
        f"MATCH (n:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL}) "
        "RETURN n.name AS name, n.description AS description, sql.sql_full_query AS sql_code"
    )
    try:
        results = graph().query_read(query=query, parameters={})
    except Exception as e:
        logger.warning("Failed to fetch custom analyses from Neo4j: %s", e)
        return []

    rules: list[dict[str, str]] = []
    for row in results or []:
        name = row.get("name", "")
        description = row.get("description", "")
        sql_code = row.get("sql_code", "")
        if not name:
            continue
        parts = []
        if description:
            parts.append(description)
        if sql_code:
            parts.append(f"SQL: {sql_code}")
        if parts:
            rules.append({"name": name, "description": "\n".join(parts)})
    logger.info("Fetched %d custom analyses from Neo4j as domain rules", len(rules))
    return rules


# ---------------------------------------------------------------------------
# Embedding helper (called by server/custom_analyses/service.py write path)
# ---------------------------------------------------------------------------


def embed_custom_analyses(
    embed_params: "EmbedParams",
    vdb: "VDB",
    analysis_id: str | None = None,
    database_name: str | None = None,
) -> None:
    """Fetch ``CustomAnalysis`` docs from Neo4j, embed them, and append to *vdb*."""
    import pandas as pd
    from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
    from nemo_retriever.operators.vdb import IngestVdbOperator

    query = f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        WHERE $analysis_id IS NULL OR ca.id = $analysis_id
        WITH DISTINCT ca, sql,
             CASE
                 WHEN ca.description IS NOT NULL AND trim(toString(ca.description)) <> ''
                 THEN ca.description
                 ELSE ''
             END AS desc,
             CASE
                 WHEN sql.sql_full_query IS NOT NULL
                 THEN ', sql: ' + sql.sql_full_query
                 ELSE ''
             END AS sql_text
        RETURN collect({{
            text: 'custom_analysis: ' + ca.name +
                  CASE WHEN desc <> '' THEN ', description: ' + desc ELSE '' END +
                  sql_text,
            name: ca.name,
            label: labels(ca)[0],
            id: ca.id
        }}) AS docs
    """
    result = graph().query_read(
        query,
        parameters={"analysis_id": analysis_id},
    )
    docs = result[0].get("docs") if result else None
    if not docs:
        logger.info(
            "No CustomAnalysis rows found for analysis_id=%r; skipping VDB upsert.",
            analysis_id,
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
            f"Embedding step produced 0/{len(embedded)} CustomAnalysis rows "
            f"with embeddings; check upstream embed errors (often a transient "
            f"{embed_params.embed_invoke_url} 5xx)."
        )

    IngestVdbOperator(vdb=vdb)(with_embeddings)
    logger.info(
        "Embedded and appended %d/%d CustomAnalysis row(s) via %s in %.2fs.",
        len(with_embeddings),
        len(embedded),
        type(vdb).__name__,
        time.time() - before,
    )


# ---------------------------------------------------------------------------
# Retrieval-time helpers (extracted from candidates_preparation.py)
# ---------------------------------------------------------------------------


def fetch_custom_analyses_with_sql(analysis_ids: list[str]) -> list[dict[str, str]]:
    """Fetch id, name, description and sql for each CustomAnalysis via HAS_SQL -> Sql.

    Returns a list of dicts with keys: id, name, description, sql.
    """
    if not analysis_ids:
        return []

    query = f"""
    UNWIND $ids AS analysis_id
    MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: analysis_id}})
    OPTIONAL MATCH (ca)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
    RETURN ca.id AS ca_id, ca.name AS name, ca.description AS description,
           sql.sql_full_query AS sql_text
    """
    try:
        rows = graph().query_read(query, {"ids": analysis_ids})
    except Exception:
        logger.warning(
            "fetch_custom_analyses_with_sql: Neo4j query failed", exc_info=True
        )
        return []

    seen_ids: set[str] = set()
    result: list[dict[str, str]] = []
    for row in rows:
        ca_id = row.get("ca_id") or ""
        if ca_id in seen_ids:
            continue
        seen_ids.add(ca_id)
        result.append(
            {
                "id": ca_id,
                "name": (row.get("name") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "sql": (row.get("sql_text") or "").strip(),
            }
        )
    return result


def get_custom_analysis_by_id(analysis_id: str) -> str | None:
    """Return the id of the CustomAnalysis, or None if it doesn't exist."""
    rows = graph().query_read(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
        RETURN ca.id AS id
        LIMIT 1
        """,
        {"analysis_id": analysis_id},
    )
    return rows[0]["id"] if rows else None


def delete_custom_analysis_node(analysis_id: str) -> None:
    """DETACH DELETE the CustomAnalysis and its linked Sql node."""
    graph().query_write(
        f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: $analysis_id}})
              -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
        DETACH DELETE ca, sql
        """,
        {"analysis_id": analysis_id},
    )


def fetch_tables_from_custom_analyses(analysis_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch Tables referenced by CustomAnalysis nodes via HAS_SQL -> Sql -> SQL -> Table."""
    if not analysis_ids:
        return []
    query = f"""
    UNWIND $ids AS analysis_id
    MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: analysis_id}})
          -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
          -[:{Edges.SQL}]->(tbl:{Labels.TABLE})
    MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
          -[:{Edges.CONTAINS}]->(tbl)
    MATCH (tbl)-[:CONTAINS]->(col:Column)
    WITH db, tbl, sch, collect({{name: col.name, data_type: col.data_type,
                             description: {column_description_expr("col")}}}) AS cols
    RETURN tbl.id AS id, tbl.name AS name, tbl.description AS description,
           db.name AS database_name, sch.name AS schema_name, tbl.pk AS pk, cols
    """
    try:
        rows = graph().query_read(query, {"ids": analysis_ids})
    except Exception:
        logger.warning(
            "fetch_tables_from_custom_analyses: Neo4j query failed", exc_info=True
        )
        return []
    tables = []
    seen: set[str] = set()
    for row in rows:
        tid = row.get("id")
        if not tid or str(tid) in seen:
            continue
        seen.add(str(tid))
        cols = [c for c in (row.get("cols") or []) if c.get("name")]
        tables.append(
            {
                "id": tid,
                "name": row.get("name") or "",
                "description": row.get("description") or "",
                "database_name": row.get("database_name") or "",
                "schema_name": row.get("schema_name") or "",
                "label": Labels.TABLE,
                # Keyless tables are unusable as a prediction entity, so this has
                # to survive every path that reaches ``relevant_tables``.
                "pk": row.get("pk") or [],
                "columns": cols,
            }
        )
    return tables
