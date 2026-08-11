# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres implementation of the generic node/edge helpers.

Public surface matches ``gsf.catalog.store.neo4j.edges`` function for function —
that equivalence is what lets the selector swap one for the other without any
caller noticing.

The interesting part is :func:`add_edges`. It receives edges generically, but
``CONTAINS`` is not stored as an edge at all: it is the child's parent foreign
key, so writing one is an ``UPDATE`` of the child row while every other
relationship type is an ``INSERT``. That asymmetry is the price of the schema
decision that turned ``reset.py``'s APOC subgraph cascade into
``ON DELETE CASCADE``, and it is paid here, once.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.constants import Edges, Labels, Props
from gsf.catalog.model.node import CatalogNode
from gsf.catalog.store.pg import registry
from gsf.catalog.store.pg.nodes import upsert_node
from gsf.dal.pg.session import store

logger = logging.getLogger(__name__)


def is_flat_dict(properties: dict):
    """Reject nested property values.

    Kept identical to the Neo4j implementation even though Postgres would
    happily take nested JSON: callers rely on the *rejection*, and quietly
    accepting a shape one backend refuses would make the two stores disagree
    about what a valid write is.
    """
    for key, value in properties.items():
        if isinstance(value, list):
            if value:
                if any(isinstance(item, (list, dict)) for item in value):
                    raise ValueError(
                        f"Invalid property name: {key}\nThe property value: {value}"
                    )
        if isinstance(value, dict):
            raise ValueError(
                f"Invalid property name: {key}\nThe property value: {value}"
            )


def check_properties_compatibility_with_neo4j(
    node_from: CatalogNode, node_to: CatalogNode, edge_props: dict
):
    """Name kept for call-site compatibility; the check is store-agnostic.

    Renaming it would touch every caller for no behaviour change. Phase 11 can
    rename it once the Neo4j path is gone.
    """
    is_flat_dict(node_from.get_properties())
    if node_from.get_override_existing_props():
        is_flat_dict(node_from.get_override_existing_props())
    is_flat_dict(node_to.get_properties())
    if node_to.get_override_existing_props():
        is_flat_dict(node_to.get_override_existing_props())
    is_flat_dict(edge_props)


def _edge_label(edge) -> str:
    if Props.JOIN in edge[2]:
        return Edges.JOIN
    if Props.UNION in edge[2]:
        return Edges.UNION
    if Props.SQL_ID in edge[2]:
        return Edges.SQL
    if Props.ANALYSIS_ID in edge[2]:
        return Edges.HAS_SQL
    return next(iter(edge[2]))


def prepare_node(node: CatalogNode):
    """Unchanged from the Neo4j implementation: it only reshapes the node."""
    label = node.get_label()
    props = node.get_properties()
    identity_props = node.get_match_props()
    on_create_props = props.copy()
    override_props = node.get_override_existing_props() or {}
    return [label], identity_props, on_create_props, override_props


def prepare_edge(edge):
    """Flatten an edge tuple into the dict :func:`add_edges` consumes."""
    node_from, node_to = edge[0], edge[1]
    e_label = _edge_label(edge)
    v1_label, v1_identity, v1_on_create, v1_on_match = prepare_node(node_from)
    v2_label, v2_identity, v2_on_create, v2_on_match = prepare_node(node_to)
    edge_props = edge[2].copy()

    check_properties_compatibility_with_neo4j(node_from, node_to, edge[2])

    if e_label in (Edges.JOIN, Edges.UNION):
        edge_identity_props: dict[str, Any] = {}
    elif edge_props.get("child_idx") is not None:
        edge_identity_props = {"child_idx": edge_props["child_idx"]}
    else:
        edge_identity_props = {}

    return {
        "v1_label": v1_label,
        "v1_identity_props": v1_identity,
        "v1_on_create_props": v1_on_create,
        "v1_on_match_props": v1_on_match,
        "v2_label": v2_label,
        "v2_identity_props": v2_identity,
        "v2_on_create_props": v2_on_create,
        "v2_on_match_props": v2_on_match,
        "edge_props": edge_props,
        "edge_label": e_label,
        "edge_identity_props": edge_identity_props,
    }


def add_edges(edges_data):
    """Upsert both endpoints and the relationship between them."""
    for data in edges_data:
        from_label = data["v1_label"][0]
        to_label = data["v2_label"][0]
        edge_label = data["edge_label"]

        spec = registry.edge_spec(edge_label, from_label, to_label)

        source_id = upsert_node(
            from_label,
            data["v1_identity_props"],
            data["v1_on_create_props"],
            on_match=data["v1_on_match_props"],
        )

        # For a parent link the child's row *is* the edge, so the parent id goes
        # in as part of upserting the child rather than as a separate write.
        target_id = upsert_node(
            to_label,
            data["v2_identity_props"],
            data["v2_on_create_props"],
            parent_id=source_id if spec.is_parent_link else None,
            on_match=data["v2_on_match_props"],
        )

        if spec.is_parent_link:
            continue

        _upsert_edge_row(spec, source_id, target_id, data["edge_props"])


def _upsert_edge_row(
    spec: registry.EdgeSpec, source_id: str, target_id: str, edge_props: dict
) -> None:
    values = {spec.source_column: source_id, spec.target_column: target_id}
    payload = {k: v for k, v in edge_props.items() if k in spec.property_columns}
    values.update(payload)

    statement = insert(spec.table).values(**values)
    conflict = [spec.source_column, spec.target_column]
    statement = (
        statement.on_conflict_do_update(index_elements=conflict, set_=payload)
        if payload
        else statement.on_conflict_do_nothing(index_elements=conflict)
    )
    store().query_write(statement)


def get_node_properties_by_id(id, label: str | list[str]):
    """Return one node's columns plus its label, or None.

    The Cypher returned ``properties(n)`` with a ``label`` key bolted on, and
    could search several labels at once because a lookup by id needed no table.
    Relationally each candidate label is a separate table, so this tries them in
    turn and returns the first hit.
    """
    labels = label if isinstance(label, list) else [label]
    for candidate in labels:
        spec = registry.label_spec(candidate)
        rows = store().query_read(select(spec.table).where(spec.table.c.id == id))
        if rows:
            props = dict(rows[0])
            props["label"] = candidate
            return props
    return None


def delete_bulk_of_nodes(ids, labels):
    """Delete by id across several labels.

    No ``DETACH`` step: every relationship is either a foreign key with
    ``ON DELETE CASCADE`` or a parent column on a row that cascades with it, so
    the edges go when the row does.
    """
    if not ids:
        return
    for label in labels:
        spec = registry.label_spec(label)
        store().query_write(spec.table.delete().where(spec.table.c.id.in_(list(ids))))


def detach_bulk_of_nodes(ids):
    """No-op.

    The Neo4j version deletes ``depends_on`` relationships between ``field``
    nodes — a label and relationship type that exist nowhere in GSF's schema or
    vocabulary. It is dead code inherited from the library, kept only so the
    two implementations expose the same surface, and does nothing here rather
    than pretending to.
    """
    logger.debug("detach_bulk_of_nodes is a no-op on Postgres (dead upstream code)")


def get_node_id_by_name_and_label(name: str, label: Labels):
    spec = registry.label_spec(str(label))
    rows = store().query_read(select(spec.table.c.id).where(spec.table.c.name == name))
    return rows[0]["id"] if rows else None
