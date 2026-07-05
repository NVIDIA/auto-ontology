# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SqlAttribute write orchestration.

All direct Neo4j calls live in gsf/dal/sql_attributes.py.
This module keeps orchestration: connector resolution, SQL validation,
Neo4j node persistence via add_query, and VDB embedding lifecycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Props

from gsf.connectors import get_connectors
from gsf.dal.sql_attributes import (
    SqlAttributeNameConflict,
    SqlAttributeSqlError,
    delete_sql_attribute_node,
    detach_existing_sql_edges,
    fetch_sql_attribute_docs,
    find_attr_by_name,
    get_sql_attribute,
    get_sql_attribute_by_id,
    link_to_term,
    list_sql_attributes,
    update_sql_attribute_props,
)
from gsf.dal.terms import get_slim_term_by_id
from gsf.semantic.constants import LABEL_SQL_ATTRIBUTE
from gsf.server.sql_utils import get_dialects, get_schemas, validate_sql
from gsf.utils import get_embed_params
from gsf.utils.embedding import embed_docs_into_vdb
from gsf.vdb import get_semantic_vdb

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

__all__ = [
    "SqlAttributeNameConflict",
    "SqlAttributeSqlError",
    "list_sql_attributes",
    "get_sql_attribute",
    "create_sql_attribute",
    "update_sql_attribute",
    "delete_sql_attribute",
]


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _resolve_connector(connector: str) -> str:
    """Look up a loaded connector by name and return its ``database_name``."""
    for c in get_connectors():
        if getattr(c, "database_name", None) == connector:
            return c.database_name
    raise ValueError(f"Connector {connector!r} not found among loaded connectors")


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


def _embed_sql_attribute(
    embed_params: "EmbedParams",
    vdb: "VDB",
    attr_id: str,
    database_name: str | None = None,
) -> None:
    """Fetch docs from Neo4j, embed them, and upsert into *vdb*."""
    docs = fetch_sql_attribute_docs(attr_id)
    if not docs:
        logger.info(
            "No SqlAttribute rows found for attr_id=%r; skipping VDB upsert.",
            attr_id,
        )
        return
    embed_docs_into_vdb(docs, embed_params, vdb, database_name)


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
    """Create a SqlAttribute, its Sql node, link to a Term, and embed.

    Raises :class:`SqlAttributeNameConflict` when ``name`` is already used.
    Raises :class:`SqlAttributeSqlError` when the SQL can't be resolved.
    Raises ``ValueError`` when the Term or connector doesn't exist.
    """
    database_name = _resolve_connector(connector)

    conflict = find_attr_by_name(name, exclude_id=None)
    if conflict is not None:
        raise SqlAttributeNameConflict(
            f"SqlAttribute with name {name!r} already exists (id={conflict['id']!r})"
        )

    term = get_slim_term_by_id(term_id)
    if not term:
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
    link_to_term(row["id"], term_id)
    row["term_name"] = term["name"]
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
    """Replace a SqlAttribute, re-parse SQL, re-link Term, and re-embed.

    Returns the updated row or ``None`` when no SqlAttribute with
    ``attr_id`` exists.
    """
    database_name = _resolve_connector(connector)

    if get_sql_attribute_by_id(attr_id) is None:
        return None

    conflict = find_attr_by_name(name, exclude_id=attr_id)
    if conflict is not None:
        raise SqlAttributeNameConflict(
            f"SqlAttribute with name {name!r} already exists (id={conflict['id']!r})"
        )

    term = get_slim_term_by_id(term_id)
    if not term:
        raise ValueError(f"Term with id {term_id!r} not found")

    query_obj = validate_sql(
        expression,
        get_dialects(database_name),
        get_schemas(database_name),
    )

    detach_existing_sql_edges(attr_id)
    update_sql_attribute_props(
        attr_id,
        name=name,
        description=description,
        expression=expression,
        source=source,
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
    link_to_term(attr_id, term_id)

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
        "term_name": term["name"],
        "term_id": term_id,
        "sql": expression,
    }


def delete_sql_attribute(attr_id: str) -> dict[str, Any] | None:
    """Delete a SqlAttribute, its edges, and its VDB embedding.

    Returns ``{"id": attr_id}`` on success, or ``None`` when not found.
    """
    if get_sql_attribute_by_id(attr_id) is None:
        return None

    delete_sql_attribute_node(attr_id)
    get_semantic_vdb().delete_by_id(attr_id)

    return {"id": attr_id}
