# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row upserts — the one write primitive the catalog pipeline uses.

Every writer needs the same thing: *find a row by its identity columns, create
it with these values if absent, update it if present*.

The upsert is keyed on the **natural** key, not on ``id`` — two runs of the
parser against the same source must converge on one row, and only the natural
key can decide that. ``Schema``, ``Table`` and ``Column`` are unique *within
their parent*, which is why :func:`upsert_row` takes ``parent_id``.

**One deliberate divergence: the stored id is never overwritten.** In a property
graph, ``id`` is an ordinary property and relationships bind to internal nodes,
so replacing it costs nothing. Here it is the primary key with other rows
referencing it, and overwriting it violates their foreign keys — a real
``IntegrityError``, not a theoretical one.

Nothing is lost by not reproducing it. The parser resolves ids out of the store
before writing (``get_table_ids`` / ``get_column_ids``), so for ``Table`` and
``Column`` the incoming id already *is* the stored one. Only a node matched on
something else — a ``Sql`` matched by its statement text — arrives with a freshly
generated id, and there keeping the stored id is the correct answer rather than
merely the safe one.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.store import registry
from gsf.dal.session import store

logger = logging.getLogger(__name__)


def _natural_key_predicate(
    spec: registry.EntitySpec, values: dict[str, Any], parent_id: str | None
):
    """Build the WHERE that decides whether this row already exists."""
    clauses = []
    for part in spec.natural_key:
        if part == "parent":
            if spec.parent_column is None:
                raise ValueError(f"{spec.label} has no parent column to match on")
            clauses.append(spec.table.c[spec.parent_column] == parent_id)
        else:
            clauses.append(spec.table.c[part] == values.get(part))
    return and_(*clauses)


def resolve_id(
    label: str,
    match_props: dict[str, Any],
    *,
    parent_id: str | None = None,
) -> str | None:
    """Find an existing row's id from whatever the caller matched on.

    ``match_props`` is not uniform across labels — ``Table`` and ``Column``
    carry a pre-generated ``id``, ``Schema`` carries ``(database_name, name)``
    where ``database_name`` is not a column on the row at all, and ``Database``
    carries ``name``. Each shape is handled explicitly rather than guessed at.
    """
    spec = registry.entity_spec(label)

    if "id" in match_props and match_props["id"]:
        rows = store().query_read(
            select(spec.table.c.id).where(spec.table.c.id == match_props["id"])
        )
        if rows:
            return rows[0]["id"]
        # Fall through: a parser-generated id that is not in the store yet is
        # the normal case for a new node, so identity falls back to the
        # natural key rather than declaring the row absent.

    if "database_name" in match_props and spec.parent_label == registry.Labels.DB:
        parent_id = resolve_id(
            registry.Labels.DB, {"name": match_props["database_name"]}
        )
        if parent_id is None:
            return None

    if not spec.natural_key:
        return None

    rows = store().query_read(
        select(spec.table.c.id).where(
            _natural_key_predicate(spec, match_props, parent_id)
        )
    )
    return rows[0]["id"] if rows else None


def upsert_row(
    label: str,
    match_props: dict[str, Any],
    properties: dict[str, Any],
    *,
    parent_id: str | None = None,
    on_match: dict[str, Any] | None = None,
) -> str:
    """Create or update one row, returning its id.

    Find-or-create, then update:
    properties are written on insert, and *on_match* is applied when the row
    already existed — except for ``id``, which is never overwritten. See the
    module docstring for why that divergence is both necessary and free.
    """
    spec = registry.entity_spec(label)
    values = registry.projected(label, properties)

    if spec.parent_column and parent_id is not None:
        values[spec.parent_column] = parent_id

    existing_id = resolve_id(label, match_props, parent_id=parent_id)

    if existing_id is None:
        # Let the natural key arbitrate a race rather than checking first and
        # inserting second: two ingestion threads can reach here at once.
        conflict_target = _conflict_target(spec)
        statement = insert(spec.table).values(**values)
        if conflict_target:
            skip = (
                set(spec.natural_key)
                | {spec.parent_column if p == "parent" else p for p in spec.natural_key}
                # `id` is the primary key. Updating it on conflict rewrites the
                # existing row's identity, dangling any id the caller already
                # holds -- the same reason the match branch below refuses to
                # overwrite it.
                | {"id"}
            )
            update = {k: v for k, v in values.items() if k not in skip}
            statement = (
                statement.on_conflict_do_update(
                    index_elements=conflict_target, set_=update
                )
                if update
                else statement.on_conflict_do_nothing(index_elements=conflict_target)
            )
        rows = store().query_write(statement.returning(spec.table.c.id))
        if rows:
            return rows[0]["id"]
        return resolve_id(label, match_props, parent_id=parent_id)

    incoming_id = properties.get("id")
    if incoming_id and incoming_id != existing_id:
        # An incoming id is never written over an existing row's. That is safe in a
        # property graph, where relationships bind to internal nodes and `id` is
        # just a property — but here `id` is the primary key and other rows
        # reference it, so overwriting it violates their foreign keys.
        #
        # Not reproducing it loses nothing: the parser resolves ids out of the
        # store before writing (get_table_ids / get_column_ids), so for Table
        # and Column the id it carries is already the one on the row. Only
        # nodes matched on something else — a Sql matched by its statement text
        # — arrive with a freshly generated id, and for those keeping the
        # stored id is the correct answer, not merely the safe one.
        logger.debug(
            "keeping stored id %s for %s rather than the incoming %s",
            existing_id,
            label,
            incoming_id,
        )

    update = registry.projected(label, dict(on_match or {}))
    update.pop("id", None)
    if spec.parent_column and parent_id is not None:
        update[spec.parent_column] = parent_id

    if update:
        store().query_write(
            spec.table.update().where(spec.table.c.id == existing_id).values(**update)
        )
        return existing_id
    return existing_id


def _conflict_target(spec: registry.EntitySpec) -> list:
    """What ``ON CONFLICT`` should name for this label.

    Usually the natural key's columns, but ``sql_query`` is unique on
    ``md5(sql_full_query)`` rather than the column itself — naming the column
    there raises "no unique or exclusion constraint matching the ON CONFLICT
    specification", because Postgres matches the target against an index.
    """
    if spec.conflict_expression is not None:
        return [spec.conflict_expression]
    if not spec.natural_key:
        return []
    columns = [
        spec.parent_column if part == "parent" else part for part in spec.natural_key
    ]
    return [c for c in columns if c]
