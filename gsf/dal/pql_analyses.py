# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for ``PqlAnalysis`` nodes (the predictive twin of CustomAnalysis).

A PqlAnalysis is a verified natural-language question paired with a KumoRFM PQL
query. Unlike a CustomAnalysis it does not parse into an ``Sql``/``Table`` subgraph
(PQL resolves its tables against the prediction graph at predict time), so the PQL
text is stored directly as a property on the node — no child node or edge.

These nodes live under their own label (:data:`LABEL_PQL_ANALYSIS`) so they never
enter the SQL text-to-SQL retrieval; they are embedded to the vector store and
retrieved only as few-shot examples for PQL generation.

Contains only functions that call ``get_neo4j_conn()`` directly.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from gsf.catalog.store.neo4j.connection import get_neo4j_conn

from gsf.semantic.constants import LABEL_PQL_ANALYSIS

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

_LABEL = LABEL_PQL_ANALYSIS


# ---------------------------------------------------------------------------
# Domain errors (raised by write helpers; surfaced as HTTP 409 by the router)
# ---------------------------------------------------------------------------


class PqlAnalysisNameConflict(Exception):
    """Raised when a write would collide with another PqlAnalysis name."""


class PqlAnalysisPqlConflict(Exception):
    """Raised when the submitted PQL is already linked to a different analysis."""


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_pql_analyses() -> list[dict[str, Any]]:
    """Return all ``PqlAnalysis`` nodes as ``{id, name, description, pql}``."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (pa:{_LABEL})
        WITH pa
        ORDER BY pa.name
        RETURN collect({{
            id: pa.id,
            name: pa.name,
            description: pa.description,
            pql: pa.pql
        }}) AS analyses
        """,
        {},
    )
    return rows[0]["analyses"] if rows else []


def find_pql_analysis_by_name(
    name: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (other:{_LABEL} {{name: $name}})
        WHERE $exclude_id IS NULL OR other.id <> $exclude_id
        RETURN other.id AS id, other.name AS name
        LIMIT 1
        """,
        {"name": name, "exclude_id": exclude_id},
    )
    if not rows:
        return None
    return {"id": rows[0]["id"], "name": rows[0]["name"]}


def find_pql_analysis_by_pql(
    pql: str,
    exclude_id: str | None,
) -> dict[str, str] | None:
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (other:{_LABEL} {{pql: $pql}})
        WHERE $exclude_id IS NULL OR other.id <> $exclude_id
        RETURN other.id AS id, other.name AS name
        LIMIT 1
        """,
        {"pql": pql, "exclude_id": exclude_id},
    )
    if not rows:
        return None
    return {"id": rows[0]["id"], "name": rows[0]["name"]}


def get_pql_analysis_by_id(analysis_id: str) -> str | None:
    """Return the id of the PqlAnalysis, or None if it doesn't exist."""
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (pa:{_LABEL} {{id: $analysis_id}})
        RETURN pa.id AS id
        LIMIT 1
        """,
        {"analysis_id": analysis_id},
    )
    return rows[0]["id"] if rows else None


def fetch_pql_analyses_by_ids(analysis_ids: list[str]) -> dict[str, dict[str, str]]:
    """Fetch ``{id: {id, name, description, pql}}`` for the given PqlAnalysis ids."""
    if not analysis_ids:
        return {}
    query = f"""
    UNWIND $ids AS analysis_id
    MATCH (pa:{_LABEL} {{id: analysis_id}})
    RETURN pa.id AS id, pa.name AS name, pa.description AS description, pa.pql AS pql
    """
    try:
        rows = get_neo4j_conn().query_read(query, {"ids": analysis_ids})
    except Exception:
        logger.warning("fetch_pql_analyses_by_ids: Neo4j query failed", exc_info=True)
        return {}

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = row.get("id") or ""
        if not pid:
            continue
        out[pid] = {
            "id": pid,
            "name": (row.get("name") or "").strip(),
            "description": (row.get("description") or "").strip(),
            "pql": (row.get("pql") or "").strip(),
        }
    return out


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def upsert_pql_analysis_node(
    analysis_id: str,
    name: str,
    description: str,
    pql: str,
) -> None:
    """MERGE a ``PqlAnalysis`` node by id, setting name/description/pql."""
    get_neo4j_conn().query_write(
        f"""
        MERGE (pa:{_LABEL} {{id: $analysis_id}})
        SET pa.name = $name, pa.description = $description, pa.pql = $pql
        """,
        {
            "analysis_id": analysis_id,
            "name": name,
            "description": description,
            "pql": pql,
        },
    )


def delete_pql_analysis_node(analysis_id: str) -> None:
    """DETACH DELETE the PqlAnalysis node."""
    get_neo4j_conn().query_write(
        f"""
        MATCH (pa:{_LABEL} {{id: $analysis_id}})
        DETACH DELETE pa
        """,
        {"analysis_id": analysis_id},
    )


# ---------------------------------------------------------------------------
# Embedding helper (called by the write path)
# ---------------------------------------------------------------------------


def embed_pql_analyses(
    embed_params: "EmbedParams",
    vdb: "VDB",
    analysis_id: str | None = None,
    database_name: str | None = None,
) -> None:
    """Fetch ``PqlAnalysis`` docs from Neo4j, embed them, and append to *vdb*."""
    import pandas as pd
    from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
    from nemo_retriever.operators.vdb import IngestVdbOperator

    # Embed the QUESTION text only (name + description), not the PQL body:
    # retrieval is question-to-question similarity, so the PQL is carried as
    # payload (fetched by id at retrieval time), never embedded.
    query = f"""
        MATCH (pa:{_LABEL})
        WHERE $analysis_id IS NULL OR pa.id = $analysis_id
        WITH DISTINCT pa,
             CASE
                 WHEN pa.description IS NOT NULL AND trim(toString(pa.description)) <> ''
                 THEN pa.description
                 ELSE ''
             END AS desc
        RETURN collect({{
            text: pa.name +
                  CASE WHEN desc <> '' THEN ': ' + desc ELSE '' END,
            name: pa.name,
            id: pa.id
        }}) AS docs
    """
    result = get_neo4j_conn().query_read(query, parameters={"analysis_id": analysis_id})
    docs = result[0].get("docs") if result else None
    if not docs:
        logger.info(
            "No PqlAnalysis rows found for analysis_id=%r; skipping VDB upsert.",
            analysis_id,
        )
        return

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
        tabular_fields = {
            "id": node_id,
            "label": _LABEL,
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
            f"Embedding step produced 0/{len(embedded)} PqlAnalysis rows with "
            f"embeddings; check upstream embed errors (often a transient "
            f"{embed_params.embed_invoke_url} 5xx)."
        )

    IngestVdbOperator(vdb=vdb)(with_embeddings)
    logger.info(
        "Embedded and appended %d/%d PqlAnalysis row(s) via %s in %.2fs.",
        len(with_embeddings),
        len(embedded),
        type(vdb).__name__,
        time.time() - before,
    )
