# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone scoping: which catalog a set of zones lets you see.

This is the access-control boundary. Nearly every read in the DAL takes a
``zone_ids`` argument and narrows itself through here, so a mistake does not
show up as a wrong answer — it shows up as one user seeing another's data.
Treated accordingly: the expansion rules are spelled out, and the tests cover
each one separately rather than in aggregate.

**The filter contract changes here**, and PLAN.md flags this as the decision
every later read phase depends on. The Neo4j version returns a Cypher ``WHERE``
string for callers to interpolate into an f-string. This returns a SQLAlchemy
predicate to be passed to ``.where()``. Every caller of ``resolve_table_filter``
lives inside ``gsf/dal`` and is itself ported per-phase, so the two never meet;
``resolve_accessible_catalog_ids``, which *is* called from the service layer,
keeps its shape exactly because it returns plain id sets.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from gsf.dal.pg.session import SCHEMA, store

#: One query in place of four round trips.
#:
#: Expansion runs **one level from each direct grant**, and not transitively —
#: granting a table admits its schema, but not that schema's other tables. Each
#: branch below corresponds to one of the Neo4j implementation's four passes,
#: and every branch reads from ``granted`` rather than from another CTE, which
#: is what keeps it non-recursive.
_ACCESSIBLE_SQL = f"""
WITH granted AS (
    SELECT database_id, schema_id, table_id
      FROM {SCHEMA}.zone_target
     WHERE zone_id = ANY(:zone_ids)
),
tabs AS (
    -- granted directly
    SELECT table_id AS id FROM granted WHERE table_id IS NOT NULL
    UNION
    -- under a granted schema
    SELECT t.id FROM {SCHEMA}.catalog_table t
      JOIN granted g ON g.schema_id = t.schema_id
    UNION
    -- under a granted database
    SELECT t.id FROM {SCHEMA}.catalog_table t
      JOIN {SCHEMA}.catalog_schema s ON s.id = t.schema_id
      JOIN granted g ON g.database_id = s.database_id
),
schemas AS (
    SELECT schema_id AS id FROM granted WHERE schema_id IS NOT NULL
    UNION
    SELECT s.id FROM {SCHEMA}.catalog_schema s
      JOIN granted g ON g.database_id = s.database_id
    UNION
    -- the parent of a directly granted table
    SELECT t.schema_id FROM {SCHEMA}.catalog_table t
      JOIN granted g ON g.table_id = t.id
),
dbs AS (
    SELECT database_id AS id FROM granted WHERE database_id IS NOT NULL
    UNION
    -- the parent of a directly granted schema
    SELECT s.database_id FROM {SCHEMA}.catalog_schema s
      JOIN granted g ON g.schema_id = s.id
    UNION
    -- the grandparent of a directly granted table
    SELECT s.database_id FROM {SCHEMA}.catalog_schema s
      JOIN {SCHEMA}.catalog_table t ON t.schema_id = s.id
      JOIN granted g ON g.table_id = t.id
)
SELECT
    (SELECT coalesce(array_agg(id), '{{}}') FROM dbs)     AS db_ids,
    (SELECT coalesce(array_agg(id), '{{}}') FROM schemas) AS schema_ids,
    (SELECT coalesce(array_agg(id), '{{}}') FROM tabs)    AS table_ids
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
    extra_params: dict[str, Any] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(predicate, params)`` restricting *column_ref* to visible tables.

    *column_ref* is a SQLAlchemy column — ``catalog_table.c.id``,
    ``column_attribute.c.table_id`` — not the string expression the Cypher
    version took. The predicate goes to ``.where()``; ``None`` means no
    restriction.

    *params* is carried through unchanged for signature compatibility with the
    Neo4j implementation. Core binds its own parameters, so nothing is added to
    it here; it exists so a caller that threads *extra_params* around does not
    have to care which backend it is talking to.

    ``zone_ids is None`` yields ``None`` — no filter, full catalog. An **empty
    list** yields a predicate matching nothing, because "scoped to no zones" has
    to deny rather than permit. The Neo4j version reaches the same place via an
    empty ``table_ids`` list.
    """
    params = dict(extra_params or {})
    if zone_ids is None:
        return None, params

    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    return column_ref.in_(list(resolved["table_ids"])), params
