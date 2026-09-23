# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone scoping: which catalog a set of zones lets you see.

This is the access-control boundary. Nearly every read in the DAL takes a
``zone_ids`` argument and narrows itself through here, so a mistake does not
show up as a wrong answer — it shows up as one user seeing another's data.
Treated accordingly: the expansion rules are spelled out, and the tests cover
each one separately rather than in aggregate.

Two entry points: :func:`resolve_accessible_catalog_ids` returns the id sets a
caller can see, and :func:`resolve_table_filter` turns those into a predicate to
hand to ``.where()``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from auto_ontology.dal.session import store

#: One query in place of four round trips.
#:
#: Expansion runs **one level from each direct grant**, and not transitively —
#: granting a table admits its schema, but not that schema's other tables.
#: Every branch reads from ``granted`` rather than from another CTE, which is
#: what keeps it non-recursive.
_ACCESSIBLE_SQL = """
WITH granted AS (
    SELECT database_id, schema_id, table_id
      FROM zone_target
     WHERE zone_id = ANY(:zone_ids)
),
tabs AS (
    -- granted directly
    SELECT table_id AS id FROM granted WHERE table_id IS NOT NULL
    UNION
    -- under a granted schema
    SELECT t.id FROM catalog_table t
      JOIN granted g ON g.schema_id = t.schema_id
    UNION
    -- under a granted database
    SELECT t.id FROM catalog_table t
      JOIN catalog_schema s ON s.id = t.schema_id
      JOIN granted g ON g.database_id = s.database_id
),
schemas AS (
    SELECT schema_id AS id FROM granted WHERE schema_id IS NOT NULL
    UNION
    SELECT s.id FROM catalog_schema s
      JOIN granted g ON g.database_id = s.database_id
    UNION
    -- the parent of a directly granted table
    SELECT t.schema_id FROM catalog_table t
      JOIN granted g ON g.table_id = t.id
),
dbs AS (
    SELECT database_id AS id FROM granted WHERE database_id IS NOT NULL
    UNION
    -- the parent of a directly granted schema
    SELECT s.database_id FROM catalog_schema s
      JOIN granted g ON g.schema_id = s.id
    UNION
    -- the grandparent of a directly granted table
    SELECT s.database_id FROM catalog_schema s
      JOIN catalog_table t ON t.schema_id = s.id
      JOIN granted g ON g.table_id = t.id
)
SELECT
    (SELECT coalesce(array_agg(id), '{}') FROM dbs)     AS db_ids,
    (SELECT coalesce(array_agg(id), '{}') FROM schemas) AS schema_ids,
    (SELECT coalesce(array_agg(id), '{}') FROM tabs)    AS table_ids
"""


def get_accessible_catalog_ids_for_zones(
    zone_ids: list[str],
) -> dict[str, set[str]]:
    """Return the catalog ids reachable through *zone_ids*.

    Receives zones already scoped to the requesting user; it does not consult
    the user itself.

    Parents are included so the catalog tree can be rendered: a zone granting a
    table also admits that table's schema and database. Children are included
    the other way: a zone granting a database admits its schemas and their
    tables. Neither direction is transitive — see :data:`_ACCESSIBLE_SQL`.
    """
    if not zone_ids:
        return {"db_ids": set(), "schema_ids": set(), "table_ids": set()}

    rows = store().query_read(text(_ACCESSIBLE_SQL), {"zone_ids": list(zone_ids)})
    row = rows[0]
    return {
        "db_ids": {i for i in (row["db_ids"] or []) if i},
        "schema_ids": {i for i in (row["schema_ids"] or []) if i},
        "table_ids": {i for i in (row["table_ids"] or []) if i},
    }


def resolve_accessible_catalog_ids(
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> dict[str, set[str]] | None:
    """Resolve *zone_ids* to accessible catalog ids, reusing a prior result.

    ``None`` means no zone scoping at all — an admin or internal caller seeing
    the whole catalog. That is emphatically not the same as an empty list, which
    grants nothing, and conflating the two is how an access-control bug gets
    written.

    Callers needing the ids more than once in a request (the exploration graph
    builders combine node, zone-map and edge queries) resolve once and thread
    the result through as *data_ids_by_zone*.
    """
    if zone_ids is None:
        return None
    if data_ids_by_zone is not None:
        return data_ids_by_zone
    return get_accessible_catalog_ids_for_zones(zone_ids)


def resolve_table_filter(
    zone_ids: list[str] | None,
    column_ref: Any,
    *,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> Any:
    """A predicate restricting *column_ref* to the tables *zone_ids* can see.

    *column_ref* is a SQLAlchemy column — ``catalog_table.c.id``,
    ``column_attribute.c.table_id`` — and the result goes to ``.where()``.

    ``zone_ids is None`` yields ``None``: no filter, full catalog. An **empty
    list** yields a predicate matching nothing, because "scoped to no zones" has
    to deny rather than permit. Conflating the two is how an access-control bug
    gets written.
    """
    if zone_ids is None:
        return None
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    return column_ref.in_(list(resolved["table_ids"]))
