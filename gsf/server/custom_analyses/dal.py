# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — CustomAnalysis write orchestration.

All direct Neo4j calls live in gsf/neo4j/custom_analyses.py.
This module only keeps orchestration: SQL validation, Neo4j node
persistence, and VDB embedding — the three concerns that can't be
cleanly separated into a pure-graph layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Labels,
    Props,
)
from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single

from gsf.connectors import get_connectors
from gsf.neo4j.custom_analyses import (
    CustomAnalysisNameConflict,
    CustomAnalysisSqlConflict,
    CustomAnalysisSqlError,
    delete_custom_analysis_node,
    detach_existing_sql_edges,
    embed_custom_analyses,
    find_analysis_by_name,
    find_analysis_by_sql,
    get_custom_analysis_by_id,
    list_custom_analyses,
)
from gsf.retrieval.data_access.graph_schemas import (
    get_all_schemas_ids,
    get_schemas_by_ids,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "CustomAnalysisNameConflict",
    "CustomAnalysisSqlConflict",
    "CustomAnalysisSqlError",
    "list_custom_analyses",
    "create_custom_analysis",
    "update_custom_analysis",
    "delete_custom_analysis",
]


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _get_dialects() -> list[str]:
    """Return SQL dialects from active connectors (NeMo multi-connector order)."""
    connectors = get_connectors()
    dialects = [c.dialect for c in connectors if getattr(c, "dialect", None)]
    if not dialects:
        return ["generic", "ansi", "postgres"]
    return dialects


def _get_schemas() -> dict:
    """Return the full catalog snapshot for ``parse_query_single``."""
    schemas_ids = get_all_schemas_ids()
    return get_schemas_by_ids(schemas_ids)


def _validate_sql(sql: str, dialects: list[str], schemas: dict) -> Any:
    """Validate ``sql`` against *schemas*, returning a query object.

    Pure validation step: no graph writes happen here. Callers MUST run
    this before any mutating call (``detach_existing_sql_edges``,
    ``_persist_analysis_with_sql``) so a parse failure can't leave the
    graph in a half-updated state.
    """
    try:
        query_obj = parse_query_single(sql=sql, dialects=dialects, schemas=schemas)
    except Exception as exc:
        raise CustomAnalysisSqlError(
            f"SQL parse error: {exc}",
        ) from exc

    if query_obj is None:
        raise CustomAnalysisSqlError(
            "SQL doesn't reference any table known to the catalog; "
            "ingest the schema first or check the query",
        )

    return query_obj


def _persist_analysis_with_sql(
    analysis_node: Neo4jNode,
    sql: str,
    query_obj: Any,
) -> dict[str, Any]:
    """Link a pre-parsed ``query_obj`` to ``analysis_node`` via HAS_SQL."""
    query_obj.sql_node.match_props = {"sql_full_query": sql}

    edge_props = {Props.ANALYSIS_ID: analysis_node.get_id()}
    query_obj.edges.append((analysis_node, query_obj.sql_node, edge_props))

    add_query(query_obj.get_edges())

    props = analysis_node.get_properties()
    return {
        "id": analysis_node.get_id(),
        "name": props["name"],
        "description": props["description"],
        "sql": sql,
    }


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def create_custom_analysis(
    name: str,
    description: str,
    sql: str,
) -> dict[str, Any]:
    """Create a fresh ``CustomAnalysis`` linked to its ``Sql`` node.

    Raises :class:`CustomAnalysisNameConflict` when ``name`` is already used.
    Raises :class:`CustomAnalysisSqlConflict` when ``sql`` is already attached.
    Raises :class:`CustomAnalysisSqlError` when the SQL can't be resolved.
    Returns ``{id, name, description, sql}``.
    """
    name_conflict = find_analysis_by_name(name, exclude_id=None)
    if name_conflict is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    sql_conflict = find_analysis_by_sql(sql, exclude_id=None)
    if sql_conflict is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_conflict['name']!r} "
            f"(id={sql_conflict['id']!r})",
        )

    query_obj = _validate_sql(sql, _get_dialects(), _get_schemas())
    analysis_node = Neo4jNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={"name": name, "description": description},
        match_props={"name": name},
    )

    row = _persist_analysis_with_sql(analysis_node, sql, query_obj)

    from gsf.utils import get_embed_params
    from gsf.vdb import get_semantic_vdb

    vdb = get_semantic_vdb()
    embed_custom_analyses(
        embed_params=get_embed_params(),
        vdb=vdb,
        analysis_id=row["id"],
    )

    return row


def update_custom_analysis(
    analysis_id: str,
    name: str,
    description: str,
    sql: str,
) -> dict[str, Any] | None:
    """Replace name/description/sql of an existing ``CustomAnalysis`` by id.

    Returns the updated row or ``None`` when no analysis with ``analysis_id`` exists.
    Raises :class:`CustomAnalysisNameConflict`, :class:`CustomAnalysisSqlConflict`,
    or :class:`CustomAnalysisSqlError` on constraint violations.
    """
    if get_custom_analysis_by_id(analysis_id) is None:
        return None

    name_conflict = find_analysis_by_name(name, exclude_id=analysis_id)
    if name_conflict is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    sql_conflict = find_analysis_by_sql(sql, exclude_id=analysis_id)
    if sql_conflict is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_conflict['name']!r} "
            f"(id={sql_conflict['id']!r})",
        )

    # Validate BEFORE touching the graph so a parse failure can't orphan
    # the analysis from its Sql node.
    query_obj = _validate_sql(sql, _get_dialects(), _get_schemas())

    detach_existing_sql_edges(analysis_id)

    analysis_node = Neo4jNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={"name": name, "description": description},
        match_props={"id": analysis_id},
        existing_id=analysis_id,
        override_existing_props={"name": name, "description": description},
    )

    row = _persist_analysis_with_sql(analysis_node, sql, query_obj)

    from gsf.utils import get_embed_params
    from gsf.vdb import get_semantic_vdb

    vdb = get_semantic_vdb()
    vdb.delete_by_id(analysis_id)
    embed_custom_analyses(
        embed_params=get_embed_params(),
        vdb=vdb,
        analysis_id=analysis_id,
    )

    return row


def delete_custom_analysis(analysis_id: str) -> dict[str, str] | None:
    """Remove a CustomAnalysis, its Sql node, and its VDB embedding.

    Returns ``{"id": analysis_id}`` on success, or ``None`` when not found.
    """
    if get_custom_analysis_by_id(analysis_id) is None:
        return None

    delete_custom_analysis_node(analysis_id)

    from gsf.vdb import get_semantic_vdb

    get_semantic_vdb().delete_by_id(analysis_id)

    return {"id": analysis_id}
