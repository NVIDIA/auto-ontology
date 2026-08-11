# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database-level writes and the incremental re-ingest diff.

Public surface matches ``gsf.catalog.store.neo4j.db`` function for function.

The diff itself — ``update_diff_from_existing_schema`` and the two
``accumulate_*`` helpers — is **not** reimplemented here. It is pandas over two
frames plus calls back into the store, so it is storage-agnostic, and it lives
in :mod:`gsf.catalog.diff` with both backends re-exporting it. Duplicating it
would mean maintaining two copies of the subtlest code in the write path — the
code that was already silently broken for every re-ingest (DECISION-006) — and
a future fix would have to be made twice or drift.

What *is* here is the set of primitives that diff calls, and they are where the
schema shows through: deleting a table no longer needs to name its columns,
because they cascade.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select, update

from gsf.catalog.constants import Labels

# Re-exported so this module presents the same surface as its Neo4j
# counterpart. The diff is shared; only the primitives below differ.
from gsf.catalog.diff import (  # noqa: F401
    accumulate_added_column_props,
    accumulate_updated_column,
    update_diff_from_existing_schema,
)
from gsf.catalog.store.pg.registry import entity_spec
from gsf.catalog.store.pg.rows import upsert_row
from gsf.dal.pg import schema as s
from gsf.dal.pg.session import store

logger = logging.getLogger(__name__)


def db_exists(db_node):
    """Return ``(id, loaded)`` for a database, or ``(None, None)``.

    ``loaded`` answers "does this database have anything hanging off it", which
    decides whether ``populate_db`` does a first-time load or a diff. The Cypher
    counted *relationships* on the node; the relational equivalent is whether it
    has any schemas, since that is the only thing that can hang off a database.
    """
    rows = store().query_read(
        select(
            s.catalog_database.c.id,
            select(func.count())
            .select_from(s.catalog_schema)
            .where(s.catalog_schema.c.database_id == s.catalog_database.c.id)
            .scalar_subquery()
            .label("children"),
        ).where(s.catalog_database.c.name == db_node.get_name())
    )
    if not rows:
        return None, None
    return rows[0]["id"], rows[0]["children"] > 0


def update_node_property(label, node_id, update_properties):
    """Set properties on one row.

    ``write.py`` calls this with the literal label ``"db"`` rather than
    ``Labels.DB``, so the lookup is case-insensitive over the known labels
    instead of an exact match — an unmapped label here would silently skip the
    ``pulled`` timestamp that marks an ingest complete.
    """
    spec = _spec_for(label)
    values = {k: v for k, v in update_properties.items() if k in spec.columns}
    if not values:
        return
    store().query_write(
        update(spec.table).where(spec.table.c.id == node_id).values(**values)
    )


def _spec_for(label: str):
    for candidate in (label, str(label).capitalize(), str(label).upper()):
        try:
            return entity_spec(candidate)
        except KeyError:
            continue
    # "db" -> Labels.DB is the one write.py actually relies on.
    aliases = {"db": Labels.DB, "database": Labels.DB}
    return entity_spec(aliases.get(str(label).lower(), label))


def delete_schema(schema_node_id):
    """Delete a schema. Its tables and columns go with it, by cascade."""
    store().query_write(
        s.catalog_schema.delete().where(s.catalog_schema.c.id == schema_node_id)
    )


def delete_table(table_id):
    """Delete a table. Its columns cascade; the Cypher had to name them."""
    store().query_write(
        s.catalog_table.delete().where(s.catalog_table.c.id == table_id)
    )


def delete_column(column_id):
    store().query_write(
        s.catalog_column.delete().where(s.catalog_column.c.id == column_id)
    )


def delete_columns_batch(column_ids):
    if not column_ids:
        return
    store().query_write(
        s.catalog_column.delete().where(s.catalog_column.c.id.in_(list(column_ids)))
    )


def add_schemas_edge_batch(edges, created):
    """Upsert parent/child pairs, linking each child to its parent.

    *created* is ignored: the column carries a server default, where the Cypher
    had to set it explicitly with ``coalesce``.
    """
    for edge in edges:
        try:
            parent_id = upsert_row(
                edge["from_label"], edge["from_identProps"], edge["v_props"]
            )
            upsert_row(
                edge["to_label"],
                edge["to_identProps"],
                edge["u_props"],
                parent_id=parent_id,
            )
        except Exception as err:
            raise Exception(f'Error in "add_schemas_edge_batch": {err}') from err


def update_properties_in_graph_batch(items):
    """Update rows by id, **preserving any description already stored**.

    The ``coalesce`` is the point and is easy to lose in translation: a curated
    description must survive a re-ingest, which would otherwise overwrite it
    with whatever the source database reports (usually nothing).
    """
    for item in items:
        spec = _spec_for(item["label"])
        props = {k: v for k, v in item["props"].items() if k in spec.columns}
        description = props.pop("description", None)
        props.pop("id", None)

        values = dict(props)
        if description is not None:
            values["description"] = func.coalesce(spec.table.c.description, description)

        if not values:
            continue
        store().query_write(
            update(spec.table).where(spec.table.c.id == item["id"]).values(**values)
        )
