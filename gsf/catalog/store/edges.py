# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The write surface the ingestion pipeline calls: nodes and edges in, rows out.

``write.py``, ``queries.py`` and ``schemas_parser.py`` build entities and the
relationships between them and hand them here as ``add_edges`` /
``prepare_edge`` / ``prepare_node``. Everything behind this module —
:mod:`gsf.catalog.store.registry` and :mod:`gsf.catalog.store.rows` — speaks
only of tables, rows, columns and links. The translation happens here and
nowhere else.

The substance is :func:`add_edges`. It takes relationships generically, but
``CONTAINS`` is a foreign key column rather than an association table, so
writing one is an ``UPDATE`` of the child row while every other relationship is
an ``INSERT``. That asymmetry is the price of the schema shape that makes
``reset.py``'s delete a single statement, and it is paid here, once.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.constants import Edges, Labels, Props
from gsf.catalog.model.node import CatalogNode
from gsf.catalog.store import registry
from gsf.catalog.store.rows import upsert_row
from gsf.dal.session import store

logger = logging.getLogger(__name__)


def is_flat_dict(properties: dict):
    """Reject nested property values.

    Postgres would happily take nested JSON in a jsonb column, but these values
    become scalar columns. Rejecting here gives a far better message than the
    insert would.
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


def check_properties_are_flat(
    node_from: CatalogNode, node_to: CatalogNode, edge_props: dict
):
    """Reject nested property values on either endpoint or the edge.

    These values become table columns, and a dict arriving where a scalar is
    expected fails at the insert with a far less useful message than this one.
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
    """Reshape a node into the identity/create/match property triple."""
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

    check_properties_are_flat(node_from, node_to, edge[2])

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
    """Upsert both endpoint rows and the link between them."""
    for data in edges_data:
        from_label = data["v1_label"][0]
        to_label = data["v2_label"][0]
        edge_label = data["edge_label"]

        spec = registry.link_spec(edge_label, from_label, to_label)

        source_id = upsert_row(
            from_label,
            data["v1_identity_props"],
            data["v1_on_create_props"],
            on_match=data["v1_on_match_props"],
        )

        # For a foreign key link the child's row *carries* the relationship, so
        # the parent id goes in with the child rather than as a separate write.
        target_id = upsert_row(
            to_label,
            data["v2_identity_props"],
            data["v2_on_create_props"],
            parent_id=source_id if spec.is_parent_link else None,
            on_match=data["v2_on_match_props"],
        )

        if spec.is_parent_link:
            continue

        _upsert_edge_row(spec, source_id, target_id, data["edge_props"])


#: Edge properties that **accumulate** on conflict instead of overwriting.
#:
#: This is the whole point of the column: each statement that joins two columns
#: adds its own reference, so the edge records *how often and where* the join
#: was observed. Overwriting would leave only the most recent statement and make
#: the history meaningless.
_ACCUMULATING = frozenset({"refs"})


def _upsert_edge_row(
    spec: registry.LinkSpec, source_id: str, target_id: str, edge_props: dict
) -> None:
    values = {spec.source_column: source_id, spec.target_column: target_id}
    payload = {k: v for k, v in edge_props.items() if k in spec.property_columns}
    values.update(payload)

    statement = insert(spec.table).values(**values)
    conflict = [spec.source_column, spec.target_column]
    if not payload:
        statement = statement.on_conflict_do_nothing(index_elements=conflict)
    else:
        updates = {}
        for key, value in payload.items():
            column = spec.table.c[key]
            if key in _ACCUMULATING:
                # Append, then de-duplicate: re-ingesting the same statement
                # must not grow the array without bound.
                updates[key] = _dedupe(column + getattr(statement.excluded, key))
            else:
                updates[key] = getattr(statement.excluded, key)
        statement = statement.on_conflict_do_update(
            index_elements=conflict, set_=updates
        )
    store().query_write(statement)


def _dedupe(array):
    """Distinct elements of a text[], order not preserved.

    ``ARRAY(SELECT DISTINCT unnest(...))``. The graph did not de-duplicate and
    would append the same reference on every re-ingest of the same statement;
    that is a leak rather than a behaviour worth reproducing, and the readers
    treat the array as a set.
    """
    return select(func.array_agg(distinct(func.unnest(array)))).scalar_subquery()


def get_node_properties_by_id(id, label: str | list[str]):
    """Return one row's columns plus the label it came from, or None.

    Each label is a separate table, so this tries them in turn and returns the
    first hit. The ``label`` key is in the result because callers read it.
    """
    labels = label if isinstance(label, list) else [label]
    for candidate in labels:
        spec = registry.entity_spec(candidate)
        rows = store().query_read(select(spec.table).where(spec.table.c.id == id))
        if rows:
            props = dict(rows[0])
            props["label"] = candidate
            return props
    return None


def delete_bulk_of_nodes(ids, labels):
    """Delete rows by id across several tables.

    No ``DETACH`` step: every relationship is either an association row with
    ``ON DELETE CASCADE`` or a foreign key on a row that cascades with it, so
    the links go when the row does.
    """
    if not ids:
        return
    for label in labels:
        spec = registry.entity_spec(label)
        store().query_write(spec.table.delete().where(spec.table.c.id.in_(list(ids))))


def detach_bulk_of_nodes(ids):
    """No-op.

    Deletes ``depends_on`` relationships between ``field`` nodes — a label and
    relationship type that exist nowhere in GSF's schema. Dead code inherited
    from the ingestion library, kept because callers still reference it.
    """
    logger.debug("detach_bulk_of_nodes is a no-op on Postgres (dead upstream code)")


def get_node_id_by_name_and_label(name: str, label: Labels):
    spec = registry.entity_spec(str(label))
    rows = store().query_read(select(spec.table.c.id).where(spec.table.c.name == name))
    return rows[0]["id"] if rows else None
