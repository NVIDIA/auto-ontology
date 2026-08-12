# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone CRUD.

Public surface matches ``gsf.dal.neo4j.zones`` function for function.

Two things are less obvious than they look, and both are load-bearing:

**Zone targets are polymorphic.** A zone can point at a database, a schema or a
table, and callers hand over a flat list of ids without saying which is which —
the graph did not need to know. Here each id has to be classified before it can
be written to the right column of ``zone_target``, so
:func:`_classify_items` resolves them in one query per tier.

**Zone names are unique in a way no column constraint expresses**:
case-insensitively on the trimmed name, and only among zones sharing a database
with the zone's own items. Enforced here in application code, exactly as the
Cypher does, because a ``UNIQUE`` column would be wrong in both directions at
once — see the comment on ``zone.name`` in ``gsf/dal/pg/schema.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import distinct, func, or_, select

from gsf.dal import schema as s
from gsf.dal.session import store
from gsf.server.zones.utils import format_zone

logger = logging.getLogger(__name__)

#: API label for each tier, matching what the Cypher resolved from node labels.
_API_LABEL = {"database": "database", "schema": "schema", "table": "table"}


def _classify_items(item_ids: list[str]) -> dict[str, list[str]]:
    """Sort a flat list of ids into the tier each belongs to.

    Ids that match nothing are dropped, which is what the Cypher's ``MATCH``
    did: linking a zone to something that no longer exists is a no-op, not an
    error.
    """
    if not item_ids:
        return {"database": [], "schema": [], "table": []}

    ids = list(item_ids)
    found: dict[str, list[str]] = {}
    for tier, table in (
        ("database", s.catalog_database),
        ("schema", s.catalog_schema),
        ("table", s.catalog_table),
    ):
        rows = store().query_read(select(table.c.id).where(table.c.id.in_(ids)))
        found[tier] = [r["id"] for r in rows]
    return found


def _databases_for_items(item_ids: list[str]) -> list[str]:
    """The databases the given items belong to, at whatever tier they sit."""
    if not item_ids:
        return []

    rows = store().query_read(
        select(distinct(s.catalog_database.c.id).label("id"))
        .select_from(
            s.catalog_database.outerjoin(
                s.catalog_schema,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            ).outerjoin(
                s.catalog_table,
                s.catalog_table.c.schema_id == s.catalog_schema.c.id,
            )
        )
        .where(
            or_(
                s.catalog_database.c.id.in_(item_ids),
                s.catalog_schema.c.id.in_(item_ids),
                s.catalog_table.c.id.in_(item_ids),
            )
        )
    )
    return [r["id"] for r in rows]


def _zone_name_exists(
    name: str,
    *,
    exclude_id: str | None = None,
    item_ids: list[str] | None = None,
) -> bool:
    """Whether a conflicting zone name already exists.

    Case-insensitive on the trimmed name. When *item_ids* is given, the check is
    narrowed to zones that share a database with those items — two unrelated
    databases may each have a zone called "Sales".
    """
    statement = select(s.zone.c.id).where(
        func.lower(func.trim(s.zone.c.name)) == name.strip().lower()
    )
    if exclude_id is not None:
        statement = statement.where(s.zone.c.id != exclude_id)

    database_ids = _databases_for_items(item_ids or [])
    if database_ids:
        statement = statement.where(s.zone.c.id.in_(_zones_touching(database_ids)))

    return bool(store().query_read(statement))


def _zones_touching(database_ids: list[str]) -> list[str]:
    """Zones with a target in any of *database_ids*, at any tier.

    Three plain lookups rather than one join with two aliases of
    ``catalog_schema`` — a table reached once directly and once through
    ``catalog_table``. The join is easy to write and easy to get silently wrong,
    and this runs on a handful of zones.
    """
    zone_ids: set[str] = set()

    zone_ids.update(
        r["zone_id"]
        for r in store().query_read(
            select(s.zone_target.c.zone_id).where(
                s.zone_target.c.database_id.in_(database_ids)
            )
        )
    )
    zone_ids.update(
        r["zone_id"]
        for r in store().query_read(
            select(s.zone_target.c.zone_id)
            .select_from(
                s.zone_target.join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.zone_target.c.schema_id,
                )
            )
            .where(s.catalog_schema.c.database_id.in_(database_ids))
        )
    )
    zone_ids.update(
        r["zone_id"]
        for r in store().query_read(
            select(s.zone_target.c.zone_id)
            .select_from(
                s.zone_target.join(
                    s.catalog_table,
                    s.catalog_table.c.id == s.zone_target.c.table_id,
                ).join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                )
            )
            .where(s.catalog_schema.c.database_id.in_(database_ids))
        )
    )
    return list(zone_ids)


def _zone_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "color": row["color"],
        "enabled": row["enabled"],
    }


def list_zones() -> list[dict[str, Any]]:
    """Every zone, enabled or not, regardless of the caller's role."""
    rows = store().query_read(
        select(
            s.zone.c.id,
            s.zone.c.name,
            s.zone.c.description,
            s.zone.c.color,
            s.zone.c.enabled,
        ).order_by(s.zone.c.name)
    )
    return [format_zone(_zone_row(r)) for r in rows]


def _zone_items(zone_id: str) -> list[dict[str, Any]]:
    """The catalog items a zone points at, with the tier each came from."""
    items: list[dict[str, Any]] = []
    for tier, table, column in (
        ("database", s.catalog_database, s.zone_target.c.database_id),
        ("schema", s.catalog_schema, s.zone_target.c.schema_id),
        ("table", s.catalog_table, s.zone_target.c.table_id),
    ):
        rows = store().query_read(
            select(table.c.id, table.c.name)
            .select_from(s.zone_target.join(table, column == table.c.id))
            .where(s.zone_target.c.zone_id == zone_id)
        )
        items.extend(
            {"id": r["id"], "name": r["name"], "label": _API_LABEL[tier]} for r in rows
        )
    return items


def get_zone_by_id(zone_id: str) -> dict[str, Any] | None:
    rows = store().query_read(
        select(
            s.zone.c.id,
            s.zone.c.name,
            s.zone.c.description,
            s.zone.c.color,
            s.zone.c.enabled,
        ).where(s.zone.c.id == zone_id)
    )
    if not rows:
        return None
    return format_zone(_zone_row(rows[0]), items=_zone_items(zone_id))


def _link_items(zone_id: str, item_ids: list[str]) -> list[str]:
    """Point a zone at each item, returning the ids actually linked, in order.

    Order is preserved from *item_ids* because the API returns it and the UI
    renders it; a set would lose it.
    """
    classified = _classify_items(item_ids)
    column_for = {
        "database": "database_id",
        "schema": "schema_id",
        "table": "table_id",
    }
    linked: set[str] = set()
    for tier, ids in classified.items():
        for item_id in ids:
            store().query_write(
                s.zone_target.insert().values(
                    zone_id=zone_id, **{column_for[tier]: item_id}
                )
            )
            linked.add(item_id)

    ordered: list[str] = []
    seen: set[str] = set()
    for item_id in item_ids:
        if item_id in linked and item_id not in seen:
            seen.add(item_id)
            ordered.append(item_id)
    return ordered


def create_zone(
    *,
    name: str,
    description: str | None,
    color: str,
    item_ids: list[str],
) -> dict[str, Any]:
    """Create a zone and point it at catalog items.

    Raises ``ValueError`` when a zone of the same name already exists in the
    same databases as those items.
    """
    if _zone_name_exists(name, item_ids=item_ids):
        raise ValueError(f"Zone with name {name!r} already exists")

    rows = store().query_write(
        s.zone.insert()
        .values(name=name, description=description, color=color)
        .returning(
            s.zone.c.id,
            s.zone.c.name,
            s.zone.c.description,
            s.zone.c.color,
            s.zone.c.enabled,
        )
    )
    zone = format_zone(_zone_row(rows[0]), items=[], enabled=True)

    if item_ids:
        zone["items"] = _link_items(zone["id"], item_ids)
    return zone


def update_zone(
    *,
    zone_id: str,
    updates: dict[str, Any],
    item_ids: list[str] | None,
) -> dict[str, Any] | None:
    """Update a zone's fields and/or replace its items. ``None`` if absent."""
    existing = store().query_read(select(s.zone.c.id).where(s.zone.c.id == zone_id))
    if not existing:
        return None

    if "name" in updates and updates["name"] is not None:
        if _zone_name_exists(updates["name"], exclude_id=zone_id, item_ids=item_ids):
            raise ValueError(f"Zone with name {updates['name']!r} already exists")

    fields = {
        key: value
        for key, value in updates.items()
        if key in {"name", "description", "color"}
    }
    if fields:
        store().query_write(
            s.zone.update().where(s.zone.c.id == zone_id).values(**fields)
        )

    # None leaves membership alone; an empty list clears it. The Cypher draws
    # the same distinction, and collapsing the two would silently strip a zone's
    # items on any metadata-only edit.
    if item_ids is not None:
        store().query_write(
            s.zone_target.delete().where(s.zone_target.c.zone_id == zone_id)
        )
        _link_items(zone_id, item_ids)

    return get_zone_by_id(zone_id)


def delete_zone(zone_id: str) -> bool:
    """Delete a zone. Its memberships go by cascade."""
    rows = store().query_write(
        s.zone.delete().where(s.zone.c.id == zone_id).returning(s.zone.c.id)
    )
    return bool(rows)


def set_zone_enabled(zone_id: str, enabled: bool) -> dict[str, Any] | None:
    """Enable or disable a zone.

    The Cypher swapped the node's label between ``Zone`` and ``disableZone``.
    Here it is a boolean column, which is the same fact without the relabelling
    that every read site had to spell as ``:Zone|disableZone``.
    """
    rows = store().query_write(
        s.zone.update()
        .where(s.zone.c.id == zone_id)
        .values(enabled=enabled)
        .returning(s.zone.c.id)
    )
    if not rows:
        return None
    return get_zone_by_id(zone_id)


# ---------------------------------------------------------------------------
# Table -> zones, shared with the semantic reads
# ---------------------------------------------------------------------------


def zone_covers_table():
    """A zone target covering a table, at any of the three grains it can name.

    The Cypher walked ``(item)-[:CONTAINS*0..2]->(t)`` — zero hops meaning the
    zone names the table itself, one its schema, two its database. Containment
    is FK columns here, so the walk becomes three explicit branches.

    Expects ``catalog_table`` and ``catalog_schema`` in the enclosing query, so
    a caller joining differently still gets one definition of "covered".
    """
    return or_(
        s.zone_target.c.table_id == s.catalog_table.c.id,
        s.zone_target.c.schema_id == s.catalog_table.c.schema_id,
        s.zone_target.c.database_id == s.catalog_schema.c.database_id,
    )


def fetch_table_zones_map(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    table_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """``{table_id: [zone, ...]}`` for every visible table.

    Lives here rather than in ``exploration`` — where the Cypher version does —
    because the semantic reads need it a phase earlier and two copies of a
    zone-resolution rule is exactly how a viewer ends up seeing a chip on one
    screen and not another. Phase 9 should re-export it, not rewrite it.

    **A viewer never sees a disabled zone's chip**, even when its id is in
    *zone_ids* — access may have been granted before the zone was retired.
    Admins (``zone_ids=None``) do see them, with ``enabled: False``, so they can
    be managed; that grants nothing, since *zone_ids* is itself computed from
    enabled zones only.

    *table_ids* narrows the scan and is **intersected** with what the zones
    already allow, so it can never widen access.
    """
    from gsf.dal.users import resolve_accessible_catalog_ids

    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)

    filter_ids: set[str] | None = None
    if resolved is not None:
        filter_ids = set(resolved["table_ids"])
    if table_ids is not None:
        filter_ids = (
            set(table_ids) if filter_ids is None else filter_ids & set(table_ids)
        )
    if filter_ids is not None and not filter_ids:
        return {}

    statement = (
        select(
            s.catalog_table.c.id.label("table_id"),
            s.zone.c.id,
            s.zone.c.name,
            s.zone.c.color,
            s.zone.c.enabled,
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(s.zone_target, zone_covers_table())
            .join(s.zone, s.zone.c.id == s.zone_target.c.zone_id)
        )
        .distinct()
        .order_by(s.catalog_table.c.id, s.zone.c.name)
    )
    if filter_ids is not None:
        statement = statement.where(s.catalog_table.c.id.in_(list(filter_ids)))
    if zone_ids is not None:
        statement = statement.where(
            s.zone.c.id.in_(list(zone_ids)), s.zone.c.enabled.is_(True)
        )

    result: dict[str, list[dict[str, Any]]] = {}
    for row in store().query_read(statement):
        result.setdefault(row["table_id"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "enabled": row["enabled"],
            }
        )
    return result
