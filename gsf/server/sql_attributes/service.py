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
from gsf.dal import sql_attributes as sql_attr_dal
from gsf.dal.sql_attributes import (
    SqlAttributeExpressionConflict,
    SqlAttributeNameConflict,
    SqlAttributeSqlError,
    clear_sql_attribute_description_suggestion,
    delete_sql_attribute_node,
    detach_existing_sql_edges,
    fetch_sql_attribute_docs,
    find_attr_by_expression,
    find_attr_by_name,
    get_full_sql_attribute_by_id,
    get_sql_attribute_by_id,
    link_to_term,
    list_sql_attributes,
)
from gsf.dal.terms import get_slim_term_by_id
from gsf.semantic.constants import LABEL_SQL_ATTRIBUTE
from gsf.server.sql_attributes.description_suggester import (
    suggest_sql_attribute_description,
)
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
    "SqlAttributeExpressionConflict",
    "SqlAttributeSqlError",
    "list_sql_attributes",
    "get_full_sql_attribute_by_id",
    "get_sql_attribute_by_id",
    "validate_sql_attribute",
    "create_sql_attribute",
    "update_sql_attribute",
    "delete_sql_attribute",
    "suggest_sql_attribute_description",
]


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _resolve_connector(connector: str) -> str:
    """Look up a loaded connector by name and return its ``database_name``."""
    key = connector.casefold()
    for c in get_connectors():
        db = getattr(c, "database_name", None)
        if db is not None and db.casefold() == key:
            return c.database_name
    raise ValueError(f"Connector {connector!r} not found among loaded connectors")


def _resolve_database_name(connector: str | None) -> str | None:
    """Resolve which database to validate SQL against.

    The public create/update/validate API never asks the client for a
    connector (see ``gsf/connectors/registry.py`` — connectors are a
    server-side concept). When *connector* is omitted we resolve it
    automatically: with exactly one configured connector there's no
    ambiguity, so validation is scoped to its dialect/catalog; with zero
    or multiple connectors we fall back to validating against the full
    catalog (all dialects, all schemas), same as CustomAnalysis.

    Internal callers that already know the target database (e.g. the
    semantic pipeline, which processes one database at a time) may still
    pass *connector* explicitly to keep that precision.
    """
    if connector is not None:
        return _resolve_connector(connector)
    connectors = get_connectors()
    if len(connectors) == 1:
        return connectors[0].database_name
    return None


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
        "sql_id": query_obj.sql_node.get_id(),
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
# Validation API
# ---------------------------------------------------------------------------


def validate_sql_attribute(
    *,
    expression: str,
    term_id: str | None = None,
    attribute_id: str | None = None,
    connector: str | None = None,
) -> dict[str, Any]:
    """Validate a SQL expression against the catalog.

    Does not persist anything — used by the "Validate SQL" step before a
    SqlAttribute is created or updated. The connector is resolved
    server-side (see :func:`_resolve_database_name`) — callers never need
    to pass one.

    When *term_id* is given, also checks that no other SqlAttribute of the
    same term already uses an equivalent expression (*attribute_id*, when
    editing an existing attribute, excludes it from that check).

    Raises :class:`SqlAttributeExpressionConflict` on a duplicate SQL
    snippet. Raises :class:`SqlAttributeSqlError` when the SQL can't be
    resolved.
    """
    if term_id is not None:
        conflict = find_attr_by_expression(
            term_id=term_id,
            expression=expression,
            exclude_id=attribute_id,
        )
        if conflict is not None:
            raise SqlAttributeExpressionConflict("SQL snippet already exists")

    database_name = _resolve_database_name(connector)
    validate_sql(
        expression,
        get_dialects(database_name),
        get_schemas(database_name),
    )
    return {"valid": True, "expression": expression}


def _reembed_sql_attribute(attr_id: str, database_name: str | None) -> None:
    vdb = get_semantic_vdb()
    vdb.delete_by_id(attr_id)
    _embed_sql_attribute(
        embed_params=get_embed_params(),
        vdb=vdb,
        attr_id=attr_id,
        database_name=database_name,
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
    connector: str | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    """Create a SqlAttribute, its Sql node, link to a Term, and embed.

    The connector is resolved server-side (see
    :func:`_resolve_database_name`) — callers never need to pass one.

    Raises :class:`SqlAttributeNameConflict` when ``name`` is already used.
    Raises :class:`SqlAttributeSqlError` when the SQL can't be resolved.
    Raises ``ValueError`` when the Term doesn't exist.
    """
    database_name = _resolve_database_name(connector)

    conflict = find_attr_by_name(name, exclude_id=None)
    if conflict is not None:
        raise SqlAttributeNameConflict(
            f"SqlAttribute with name {name!r} already exists (id={conflict['id']!r})"
        )

    term = get_slim_term_by_id(term_id)
    if not term:
        raise ValueError(f"Term with id {term_id!r} not found")

    expression_conflict = find_attr_by_expression(
        term_id=term_id,
        expression=expression,
        exclude_id=None,
    )
    if expression_conflict is not None:
        raise SqlAttributeExpressionConflict("SQL snippet already exists")

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
    name: str | None = None,
    description: str | None = None,
    expression: str | None = None,
    term_id: str | None = None,
    connector: str | None = None,
    source: str = "manual",
) -> dict[str, Any] | None:
    """Update a SqlAttribute.

    When *expression* is given, replaces the full record: re-parses SQL,
    re-links the Term, and re-embeds. *name* and *term_id* are required in
    that case.

    When *expression* is omitted, patches name/description only.

    The connector is resolved server-side (see
    :func:`_resolve_database_name`) — callers never need to pass one.

    Returns the updated row or ``None`` when no SqlAttribute with
    ``attr_id`` exists.
    """
    existing = get_full_sql_attribute_by_id(attr_id)
    if existing is None:
        return None

    if expression is not None:
        if name is None or term_id is None:
            raise ValueError(
                "name and term_id are required when updating SQL expression"
            )

        next_name = name.strip()
        next_description = (
            description if description is not None else existing.get("description", "")
        )
        database_name = _resolve_database_name(connector)

        name_changed = (existing.get("name") or "").strip() != next_name
        expression_changed = (
            existing.get("expression") or ""
        ).strip() != expression.strip()

        conflict = find_attr_by_name(next_name, exclude_id=attr_id)
        if conflict is not None:
            raise SqlAttributeNameConflict(
                f"SqlAttribute with name {next_name!r} already exists "
                f"(id={conflict['id']!r})"
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
        if name_changed or expression_changed:
            clear_sql_attribute_description_suggestion(attr_id)
        sql_attr_dal.update_sql_attribute(
            attr_id,
            name=next_name,
            description=next_description,
            expression=expression,
            source=source,
        )

        attr_node = Neo4jNode(
            name=next_name,
            label=LABEL_SQL_ATTRIBUTE,
            props={
                "name": next_name,
                "description": next_description,
                "expression": expression,
                "source": source,
            },
            match_props={"id": attr_id},
            existing_id=attr_id,
        )

        _persist_attr_with_sql(attr_node, expression, query_obj)
        link_to_term(attr_id, term_id)
        _reembed_sql_attribute(attr_id, database_name)

        return {
            "id": attr_id,
            "name": next_name,
            "description": next_description,
            "expression": expression,
            "source": source,
            "database_name": database_name,
            "term_name": term["name"],
            "term_id": term_id,
            "sql": expression,
        }

    next_name = name.strip() if name is not None else existing.get("name", "")
    if not next_name:
        raise ValueError("SQL attribute name cannot be blank")

    next_description = (
        description if description is not None else existing.get("description", "")
    )

    name_changed = (existing.get("name") or "").strip() != next_name
    description_changed = (existing.get("description") or "") != next_description
    if not name_changed and not description_changed:
        return existing

    if name_changed:
        conflict = find_attr_by_name(next_name, exclude_id=attr_id)
        if conflict is not None:
            raise SqlAttributeNameConflict(
                f"SqlAttribute with name {next_name!r} already exists "
                f"(id={conflict['id']!r})"
            )
        clear_sql_attribute_description_suggestion(attr_id)

    sql_attr_dal.update_sql_attribute(
        attr_id,
        name=next_name if name_changed else None,
        description=next_description if description_changed else None,
    )
    _reembed_sql_attribute(attr_id, existing.get("database_name"))

    return get_full_sql_attribute_by_id(attr_id)


def delete_sql_attribute(attr_id: str) -> dict[str, Any] | None:
    """Delete a SqlAttribute, its edges, and its VDB embedding.

    Returns ``{"id": attr_id}`` on success, or ``None`` when not found.
    """
    if get_sql_attribute_by_id(attr_id) is None:
        return None

    delete_sql_attribute_node(attr_id)
    get_semantic_vdb().delete_by_id(attr_id)

    return {"id": attr_id}
