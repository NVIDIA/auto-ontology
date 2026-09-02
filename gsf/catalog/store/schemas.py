# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading and writing one schema's tables and columns.

The frames returned by the readers are consumed by
``update_diff_from_existing_schema``, which merges them against freshly parsed
ones — so their **column names have to match exactly**, including the ``database``
column that only the stored side carries. Getting that wrong is not a type
error; it is the diff silently doing nothing.
"""

from __future__ import annotations

import logging
from threading import Lock

import pandas as pd
from sqlalchemy import ARRAY, Text, and_, cast, column, delete, func, select, update
from sqlalchemy import values as sa_values

# The dialect insert, not sqlalchemy.insert: only this one has
# on_conflict_do_update.
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.constants import Labels
from gsf.catalog.model.node import CatalogNode
from gsf.catalog.model.schema import Schema
from gsf.catalog.normalize import normalize_columns, normalize_tables
from gsf.catalog.store.registry import entity_spec
from gsf.catalog.store.rows import upsert_row
from gsf.dal import schema as s
from gsf.dal.session import store, write_transaction

logger = logging.getLogger(__name__)


def _catalog_join():
    """Column → Table → Schema → Database, the walk every reader needs."""
    return (
        s.catalog_column.join(
            s.catalog_table, s.catalog_column.c.table_id == s.catalog_table.c.id
        )
        .join(s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
        .join(
            s.catalog_database,
            s.catalog_schema.c.database_id == s.catalog_database.c.id,
        )
    )


def get_schema_columns(database_name, schema_name):
    rows = store().query_read(
        select(
            s.catalog_database.c.name.label("database"),
            s.catalog_schema.c.name.label("table_schema"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.id.label("id"),
            s.catalog_column.c.data_type,
            s.catalog_column.c.is_nullable,
        )
        .select_from(_catalog_join())
        .where(
            and_(
                s.catalog_database.c.name == database_name,
                s.catalog_schema.c.name == schema_name,
            )
        )
    )
    return normalize_columns(pd.DataFrame(rows))


def get_schema_tables(database_name, schema_name):
    rows = store().query_read(
        select(
            s.catalog_database.c.name.label("database"),
            s.catalog_schema.c.name.label("table_schema"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.id.label("id"),
            s.catalog_table.c.description,
            s.catalog_table.c.table_type,
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
            ).join(
                s.catalog_database,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
        )
        .where(
            and_(
                s.catalog_database.c.name == database_name,
                s.catalog_schema.c.name == schema_name,
            )
        )
    )
    frame = pd.DataFrame(rows)
    # Nothing reads this, but the
    # column has to exist or normalize_tables sees a different frame shape.
    if not frame.empty and "created" not in frame.columns:
        frame["created"] = None
    return normalize_tables(frame)


def load_schema_from_graph(database_name, schema_name, database_node=None):
    """Rebuild a stored schema, or ``None`` when there is nothing stored.

    Returning ``None`` rather than a ``Schema`` whose frames are ``None`` is
    what the caller in :mod:`gsf.catalog.diff` already tests for. Handing back
    the half-built object instead made that guard dead code and turned the next
    line -- ``existing_schema.tables_df.table_name`` -- into an
    ``AttributeError`` that aborted the whole database's re-ingest, so a schema
    stored with tables but no columns could never re-ingest and never
    self-healed.
    """
    tables_df = get_schema_tables(database_name, schema_name)
    columns_df = get_schema_columns(database_name, schema_name)
    if tables_df.empty or columns_df.empty:
        return None

    if database_node is None:
        database_node = CatalogNode(
            name=database_name, label=Labels.DB, props={"name": database_name}
        )

    schema = Schema(database_node, tables_df, columns_df)
    schema.create_schema_node(schema_name)
    return schema


def get_schemas_ids_and_names(database_id: str = None, database_name: str = None):
    statement = select(
        s.catalog_schema.c.name.label("schema_name"),
        s.catalog_schema.c.id.label("schema_id"),
    ).select_from(
        s.catalog_schema.join(
            s.catalog_database,
            s.catalog_schema.c.database_id == s.catalog_database.c.id,
        )
    )
    if database_id:
        statement = statement.where(s.catalog_database.c.id == database_id)
    elif database_name:
        statement = statement.where(s.catalog_database.c.name == database_name)
    return store().query_read(statement)


def get_table_ids(tables_df: pd.DataFrame, database_name: str) -> pd.DataFrame:
    """Add an ``id`` column carrying each table's stored id, or None.

    This is what makes ids stable across re-ingests: the parser reads them back
    out here and writes the same values, so nothing referencing a table by id
    dangles when the table is updated.
    """
    if tables_df is None or tables_df.empty:
        return tables_df

    rows = store().query_read(
        select(
            s.catalog_schema.c.name.label("table_schema"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.id.label("id"),
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
            ).join(
                s.catalog_database,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
    )
    if not rows:
        return tables_df
    return tables_df.merge(
        pd.DataFrame(rows), on=["table_name", "table_schema"], how="left"
    )


def get_column_ids(columns_df: pd.DataFrame, database_name: str) -> pd.DataFrame:
    """Add an ``id`` column carrying each column's stored id, or None."""
    if columns_df is None or columns_df.empty:
        return columns_df

    rows = store().query_read(
        select(
            s.catalog_schema.c.name.label("table_schema"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.id.label("id"),
        )
        .select_from(_catalog_join())
        .where(s.catalog_database.c.name == database_name)
    )
    if not rows:
        return columns_df
    return columns_df.merge(
        pd.DataFrame(rows),
        on=["table_name", "table_schema", "column_name"],
        how="left",
    )


def add_schemas_edge(edge, created):
    """Upsert a parent and child, linking them by the child's parent column."""
    node_from, node_to = edge[0], edge[1]
    try:
        # One unit, for the same reason `merge_column_attribute` is (see
        # gsf/dal/attributes.py): committed separately, a failure on the child
        # leaves a parent row that nothing points at, and the next ingest
        # matches it and carries on as though the edge had been written.
        with write_transaction():
            parent_id = upsert_row(
                node_from.get_label(),
                node_from.get_match_props(),
                node_from.get_properties(),
            )
            upsert_row(
                node_to.get_label(),
                node_to.get_match_props(),
                node_to.get_properties(),
                parent_id=parent_id,
            )
    except Exception as err:
        logger.exception(err)
        raise Exception(f'Error in "add_schemas_edge" when adding edge: {edge}')


# Rows whose properties have arrived but whose parent has not. See
# merge_schema_nodes.
class PendingRows:
    """Rows held between :func:`merge_schema_nodes` and :func:`merge_schema_edges`.

    One of these belongs to one ``add_schema`` call. It used to be a module
    global, which was wrong in two ways that only show up in a long-lived
    process:

    * a node no edge ever claims is never popped, so the dict grows for the
      lifetime of the process -- and an ingest that raises between the node
      pass and the edge pass strands every node it had staged;
    * two ingests running at once shared one buffer, so a stranded entry from
      an earlier, failed run could be popped by a later run's edge and written
      in place of the properties that run had just parsed.

    Scoping it to the caller makes both impossible: the buffer is unreachable
    once ``add_schema`` returns, however it returns.

    The lock stays because the edge pass is fanned out across a
    ``ThreadPoolExecutor``; the buffer is per-ingest, not per-thread.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}
        self._lock = Lock()

    def hold(self, node_id: str, node: dict) -> None:
        with self._lock:
            self._rows[node_id] = node

    def claim(self, node_id: str) -> dict | None:
        """Take the row, if it is still waiting. Each row is claimed once."""
        with self._lock:
            return self._rows.pop(node_id, None)


def merge_schema_nodes(nodes, created, pending: PendingRows):
    """Hold each row's properties until its parent is known.

    ``add_schema`` creates all the Table and Column nodes first and *then* links
    them, because a property graph lets a node exist before any relationship
    does. Relationally it cannot: ``schema_id`` and ``table_id`` are
    ``NOT NULL``, so a table with no schema is not a row that can be written.

    So this records the properties, and :func:`merge_schema_edges` — which is
    where the parent id finally shows up — does the insert. The alternative was
    to make the parent columns nullable, which would trade a real integrity
    guarantee for the convenience of matching the old call order.

    *created* is ignored: the column has a server default.
    """
    for node in nodes:
        node_id = node["props"].get("id") or node["match_props"].get("id")
        if node_id is None:
            # Nothing to key it by, so it cannot be claimed by an edge later.
            # Write it now and let the parent column complain if it was
            # actually required.
            upsert_row(
                node["label"],
                node["match_props"],
                node["props"],
                on_match=_source_owned(node),
            )
            continue
        pending.hold(node_id, node)


#: Fields the *source database* owns, refreshed on every re-ingest.
#:
#: `upsert_row` writes `properties` on insert and only `on_match` on update, and
#: no caller here used to pass `on_match` — so a row's structural facts were
#: frozen at first ingest. Dropping a view and re-creating it as a base table
#: left `table_type` reading 'view' forever, and the text-to-SQL agent was told
#: a queryable table was a view.
#:
#: `description` is deliberately absent: it is curated in the UI and must
#: survive a re-ingest, which is the same rule `update_properties_in_graph_batch`
#: enforces with its coalesce.
_SOURCE_OWNED = {
    Labels.TABLE: ("table_type", "pk"),
    Labels.COLUMN: ("data_type", "is_nullable", "ordinal_position", "sample_values"),
}


def _source_owned(node) -> dict:
    """The subset of *node*'s properties the source is authoritative for.

    ``label`` arrives as a string from some callers and a one-element list from
    others, which is why it is normalised here rather than at each call site.
    """
    label = node["label"]
    if isinstance(label, (list, tuple)):
        label = label[0] if label else None
    fields = _SOURCE_OWNED.get(label, ())
    return {k: node["props"][k] for k in fields if k in node["props"]}


def merge_schema_edges(edges, from_label, to_label, pending: PendingRows):
    """Write each child row, now that its parent id is known.

    Links two rows that already exist. Here the relationship *is* the child's foreign key, so this is where
    the child row gets written — or updated, if it already existed.
    """
    child = entity_spec(to_label)
    if child.parent_column is None:
        raise ValueError(f"{to_label} has no parent column")

    for edge in edges:
        child_id, parent_id = edge["uid"], edge["vid"]
        node = pending.claim(child_id)

        if node is not None:
            upsert_row(
                node["label"],
                node["match_props"],
                node["props"],
                parent_id=parent_id,
                on_match=_source_owned(node),
            )
            continue

        # Already written by an earlier pass; just re-point it.
        store().query_write(
            update(child.table)
            .where(child.table.c.id == child_id)
            .values(**{child.parent_column: parent_id})
        )


def add_fks(fks_df, last_seen, database_name: str):
    """Record each foreign key, stamping *last_seen* so stale ones can be found.

    One read to resolve every column id, then one insert for the whole set.
    Per row it was two id lookups plus an autocommitted upsert -- three round
    trips each, and each id lookup a four-table join. A warehouse schema has
    thousands of foreign keys, so that was the ingest's cost, not a detail of
    it.
    """
    if fks_df is None or fks_df.empty:
        return

    ids = _column_ids_by_name(database_name)

    rows = []
    seen = set()
    for row in fks_df.to_dict(orient="records"):
        source = ids.get((row["table_schema"], row["table_name"], row["column_name"]))
        target = ids.get(
            (
                row["referenced_schema"],
                row["referenced_table"],
                row["referenced_column"],
            )
        )
        if source is None or target is None:
            # Nothing to link, so move on:
            # a foreign key onto a table outside the ingested set is normal.
            continue
        # Postgres refuses to let one INSERT ... ON CONFLICT DO UPDATE touch the
        # same row twice ("cannot affect row a second time"), so a pair repeated
        # in the source data has to be collapsed here rather than left to the
        # database. The row-at-a-time version simply upserted it twice.
        if (source, target) in seen:
            continue
        seen.add((source, target))
        rows.append(
            {
                "source_column_id": source,
                "target_column_id": target,
                "last_seen": last_seen,
            }
        )

    if not rows:
        return

    statement = insert(s.column__foreign_key).values(rows)
    store().query_write(
        statement.on_conflict_do_update(
            index_elements=["source_column_id", "target_column_id"],
            set_={"last_seen": statement.excluded.last_seen},
        )
    )


def delete_old_fks(last_seen, database_name: str):
    """Remove foreign keys this ingest did not see.

    Scoped to the database being ingested — a global delete would drop another
    database's keys whenever two ingests overlap.
    """
    source = s.catalog_column.alias("source_column")
    store().query_write(
        delete(s.column__foreign_key).where(
            s.column__foreign_key.c.source_column_id.in_(
                select(source.c.id)
                .select_from(
                    source.join(
                        s.catalog_table, source.c.table_id == s.catalog_table.c.id
                    )
                    .join(
                        s.catalog_schema,
                        s.catalog_table.c.schema_id == s.catalog_schema.c.id,
                    )
                    .join(
                        s.catalog_database,
                        s.catalog_schema.c.database_id == s.catalog_database.c.id,
                    )
                )
                .where(s.catalog_database.c.name == database_name)
            ),
            # NOT `is_distinct_from`: that is true for NULL, and an imported
            # foreign key has no `last_seen`. Using it deleted every curated FK
            # from a model import on the next re-ingest. Cypher's `<>` yielded
            # NULL here, so those rows were never matched -- preserved by
            # requiring the column to be set *and* stale.
            s.column__foreign_key.c.last_seen.isnot(None),
            s.column__foreign_key.c.last_seen != last_seen,
        )
    )


def reset_pks(database_name: str):
    store().query_write(
        update(s.catalog_table)
        .where(
            s.catalog_table.c.schema_id.in_(
                select(s.catalog_schema.c.id)
                .select_from(
                    s.catalog_schema.join(
                        s.catalog_database,
                        s.catalog_schema.c.database_id == s.catalog_database.c.id,
                    )
                )
                .where(s.catalog_database.c.name == database_name)
            )
        )
        .values(pk=None)
    )


def add_pks(pks_df, database_name: str):
    """Append each primary-key column name to its table's ``pk`` array.

    Appending rather than assigning
    (``t.pk + [col.name]``) and a composite key arrives as several rows.
    ``reset_pks`` runs first, so the array starts empty each ingest.

    A composite key's parts are grouped in Python and appended as one array, so
    a table is updated once however many columns its key has -- and one
    ``UPDATE ... FROM (VALUES ...)`` covers every table. Previously each part
    cost an id lookup plus its own autocommitted update.

    Grouping preserves the order the rows arrived in, which is what makes the
    stored ``pk`` array match the key's column order.
    """
    if pks_df is None or pks_df.empty:
        return

    ids = _table_ids_by_name(database_name)

    parts: dict[str, list[str]] = {}
    for row in pks_df.to_dict(orient="records"):
        table_id = ids.get((row["table_schema"], row["table_name"]))
        if table_id is None:
            continue
        parts.setdefault(table_id, []).append(row["column_name"])

    if not parts:
        return

    incoming = sa_values(
        column("table_id", Text),
        column("key_columns", ARRAY(Text)),
        name="incoming",
    ).data(list(parts.items()))

    store().query_write(
        update(s.catalog_table)
        .where(s.catalog_table.c.id == incoming.c.table_id)
        .values(
            pk=func.coalesce(s.catalog_table.c.pk, cast([], ARRAY(Text)))
            + incoming.c.key_columns
        )
    )


def _append_to_pk(column, value):
    """Append to a possibly-null ``text[]``.

    ``array_append(NULL, x)`` yields ``{x}`` in Postgres, but coalescing first
    keeps the intent legible and survives anyone changing the column default.
    """
    return func.array_append(func.coalesce(column, cast([], ARRAY(Text))), value)


def _table_ids_by_name(database_name: str) -> dict[tuple[str, str], str]:
    """``(schema, table) -> id`` for one database, in a single read.

    A whole database's tables at once rather than one lookup per row: the
    callers are resolving a batch, and the four-table join costs the same
    whether it returns one row or all of them.
    """
    rows = store().query_read(
        select(
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.id,
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
            ).join(
                s.catalog_database,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
    )
    return {(r["schema_name"], r["table_name"]): r["id"] for r in rows}


def _column_ids_by_name(database_name: str) -> dict[tuple[str, str, str], str]:
    """``(schema, table, column) -> id`` for one database, in a single read."""
    rows = store().query_read(
        select(
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.id,
        )
        .select_from(_catalog_join())
        .where(s.catalog_database.c.name == database_name)
    )
    return {
        (r["schema_name"], r["table_name"], r["column_name"]): r["id"] for r in rows
    }
