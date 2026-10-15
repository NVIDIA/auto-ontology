# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which row an edit is about, and the owner id stored on it.

An owner id names either an account or a group. This module only reads and
writes that column. Deciding whether the caller may edit, and whether a
chosen id is allowed to own the row, lives in ``auto_ontology.server.owners``
because both answers need the role's permissions.

The rows whose stored owner the edit check reads are a database, a schema,
a table, and a term. A column, a column attribute, and a sql attribute are
still editable. The page shows the table's owner for a column and the term's
owner for either attribute. The edit check does not read an owner on those
rows, including the owner column a sql attribute still keeps.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, update

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store

#: ``resource:action`` an owner of one of these rows must hold, and the edit must too.
CATALOG_EDIT = "catalog:edit"

_OWNED = (
    (s.catalog_database, CATALOG_EDIT),
    (s.catalog_schema, CATALOG_EDIT),
    (s.catalog_table, CATALOG_EDIT),
    (s.term, CATALOG_EDIT),
)

#: Editable. The page shows the parent's owner. This check does not read one,
#: so a viewer in that owner's group is not admitted. A sql attribute keeps an
#: owner column; it is not what this check reads.
_EDITABLE = (
    (s.catalog_column, CATALOG_EDIT),
    (s.column_attribute, CATALOG_EDIT),
    (s.sql_attribute, CATALOG_EDIT),
)


class _Missing:
    """The id is not a row of the table that was asked."""


_MISSING = _Missing()


@dataclass(frozen=True)
class EditSubject:
    """The permission an edit needs, and the owner already stored."""

    permission: str
    owner_id: str | None


def _stored_owner(table, entity_id: str) -> str | None | _Missing:
    rows = store().query_read(
        select(table.c.owner_id).where(table.c.id == entity_id).limit(1)
    )
    if not rows:
        return _MISSING
    return rows[0]["owner_id"]


def _row_exists(table, entity_id: str) -> bool:
    rows = store().query_read(
        select(table.c.id).where(table.c.id == entity_id).limit(1)
    )
    return bool(rows)


def edit_subject(entity_id: str) -> EditSubject | None:
    """The row *entity_id* names, or ``None`` when it names none of these.

    A zone and either kind of analysis are not among them. A column, a column
    attribute, and a sql attribute are, with no owner consulted here: the page
    shows the table's or the term's.
    """
    for table, permission in _OWNED:
        stored = _stored_owner(table, entity_id)
        if stored is not _MISSING:
            return EditSubject(permission=permission, owner_id=stored)
    for table, permission in _EDITABLE:
        if _row_exists(table, entity_id):
            return EditSubject(permission=permission, owner_id=None)
    return None


def current_owner(table, entity_id: str) -> str | None:
    """The stored owner, or null when the row is missing or unassigned."""
    stored = _stored_owner(table, entity_id)
    if stored is _MISSING:
        return None
    return stored


def assign_owner(table, entity_id: str, owner_id: str | None) -> None:
    """Set or clear ``owner_id``. ``None`` clears it."""
    store().query_write(
        update(table).where(table.c.id == entity_id).values(owner_id=owner_id)
    )
