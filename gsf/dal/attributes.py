# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ColumnAttribute and SEMANTIC_FK reads and writes, plus ``find_join_path``.

``find_join_path`` is level-at-a-time BFS driven from Python over the
:data:`~gsf.dal.schema.JOIN_PATH_EDGE_VIEW_SQL` view, deliberately not a
recursive CTE — see its docstring for why. The CTE version lives in the tests as
a cross-check.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Text, all_, and_, any_, bindparam, func, literal, select, update
from sqlalchemy.dialects.postgresql import ARRAY, insert

from gsf.dal import schema as s
from gsf.dal.datasources import fetch_col_table_contexts
from gsf.dal.session import store, write_transaction
from gsf.semantic.constants import SEMANTIC_SOURCE

logger = logging.getLogger(__name__)

#: Path length ceiling, in **edges**.
#:
#: It is tempting to lower this on the grounds that real join paths are 2-4
#: hops. That reasoning is wrong, and the depth-bound test catches it: a *hop*
#: here is four edges, not one. Getting from one FK column to the next runs
#: Column -SEMANTIC_FK-> ColumnAttribute -HAS_ATTRIBUTE-> Column -CONTAINS->
#: Table -CONTAINS-> Column, so an n-hop join path is ``4n - 2`` edges. A
#: perfectly ordinary 4-hop path is 14, and a ceiling of 10 would have silently
#: returned "no path" for it.
MAX_PATH_DEPTH = 30


# ---------------------------------------------------------------------------
# ColumnAttribute
# ---------------------------------------------------------------------------


def merge_column_attribute(
    *,
    term_name: str,
    table_id: str,
    source_column: str,
    attr_name: str,
    datatype: str,
    description: str | None,
) -> str | None:
    """Upsert a ColumnAttribute and link it to its column and Term.

    Returns ``None`` when the column or the Term does not exist — the guards
    below refuse rather than create an attribute that dangles.

    ``description`` is coalesced, not assigned: a merge that has nothing to say
    about the description must not erase a curated one. ``datatype`` *is*
    assigned, as before — it describes the column, not the human's opinion of it.
    """
    column = store().query_read(
        select(s.catalog_column.c.id).where(
            s.catalog_column.c.table_id == table_id,
            s.catalog_column.c.name == source_column,
        )
    )
    term = store().query_read(
        select(s.term.c.id).where(
            s.term.c.name == term_name, s.term.c.source == SEMANTIC_SOURCE
        )
    )
    if not column or not term:
        return None
    column_id, term_id = column[0]["id"], term[0]["id"]

    statement = insert(s.column_attribute).values(
        name=attr_name,
        source_column=source_column,
        term_name=term_name,
        table_id=table_id,
        source=SEMANTIC_SOURCE,
        datatype=datatype,
        description=description,
    )
    # One transaction: the attribute and its two links are a single fact. Without
    # this each statement autocommits, so a failure on the second leaves a
    # ColumnAttribute with no link to its column and no link to its Term --
    # visible in the Term's list, uncounted by the certification rollup, and
    # permanent. This runs inside the ingestion thread fan-out against a bounded
    # pool, so a pool timeout mid-way is the realistic trigger, not a crash.
    with write_transaction():
        rows = store().query_write(
            statement.on_conflict_do_update(
                constraint="uq_column_attribute_merge_key",
                set_={
                    "datatype": statement.excluded.datatype,
                    "description": func.coalesce(
                        statement.excluded.description,
                        s.column_attribute.c.description,
                    ),
                },
            ).returning(s.column_attribute.c.id)
        )
        attribute_id = rows[0]["id"]

        for table, values in (
            (
                s.column_has_attribute,
                {"column_id": column_id, "attribute_id": attribute_id},
            ),
            (
                s.column_attribute_term,
                {"attribute_id": attribute_id, "term_id": term_id},
            ),
        ):
            store().query_write(insert(table).values(**values).on_conflict_do_nothing())

    return attribute_id


def update_column_attribute(
    attr_id: str,
    term_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    certified: bool | None = None,
) -> dict[str, Any] | None:
    """Update an attribute's metadata and return the context an embed needs.

    ``None`` when the attribute does not exist *or* is not a property of
    *term_id*. The pairing is the point: the caller has both ids from a URL, and
    accepting a mismatched pair would let one term's request edit another's
    attribute.

    Every field is coalesced, so omitting one leaves it alone rather than
    nulling it — this is a PATCH, not a PUT.

    **A rename can fail.** ``name`` is part of the five-part merge key, which
    is a unique constraint, so renaming onto an existing key raises
    ``IntegrityError``. Left to surface rather than caught: the constraint is
    describing a genuine conflict between two attributes, and swallowing it
    would leave a duplicate no read can choose between.
    """
    owned = (
        select(literal(1))
        .where(
            s.column_attribute_term.c.attribute_id == attr_id,
            s.column_attribute_term.c.term_id == term_id,
        )
        .exists()
    )
    updated = store().query_write(
        update(s.column_attribute)
        .where(s.column_attribute.c.id == attr_id, owned)
        .values(
            name=func.coalesce(name, s.column_attribute.c.name),
            description=func.coalesce(description, s.column_attribute.c.description),
            certified=func.coalesce(certified, s.column_attribute.c.certified),
        )
        .returning(s.column_attribute.c.id)
    )
    if not updated:
        return None

    # An attribute reaches at most one column, but order the pick anyway so
    # repeated calls agree with each other.
    rows = store().query_read(
        select(
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.column_attribute.c.certified,
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.sample_values,
            s.term.c.id.label("term_id"),
            s.term.c.synonyms.label("term_synonyms"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.column_attribute.join(
                s.column_attribute_term,
                s.column_attribute_term.c.attribute_id == s.column_attribute.c.id,
            )
            .join(s.term, s.term.c.id == s.column_attribute_term.c.term_id)
            .outerjoin(
                s.column_has_attribute,
                s.column_has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
            .outerjoin(
                s.catalog_column,
                s.catalog_column.c.id == s.column_has_attribute.c.column_id,
            )
            .outerjoin(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
            .outerjoin(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .outerjoin(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.column_attribute.c.id == attr_id, s.term.c.id == term_id)
        .order_by(s.catalog_column.c.id)
        .limit(1)
    )
    return dict(rows[0]) if rows else None


def find_column_attribute_by_column_id(column_id: str) -> str | None:
    """The semantic ColumnAttribute a column carries, if any.

    ``LIMIT 1`` over a genuinely one-to-many link, as before. Ordered by id so
    the arbitrary choice is at least a consistent one.
    """
    rows = store().query_read(
        select(s.column_attribute.c.id)
        .select_from(
            s.column_has_attribute.join(
                s.column_attribute,
                s.column_attribute.c.id == s.column_has_attribute.c.attribute_id,
            )
        )
        .where(
            s.column_has_attribute.c.column_id == column_id,
            s.column_attribute.c.source == SEMANTIC_SOURCE,
        )
        .order_by(s.column_attribute.c.id)
        .limit(1)
    )
    return rows[0]["id"] if rows else None


def fetch_attr_column_contexts(
    attr_ids: list[str],
    *,
    database_name: str | None,
) -> dict[str, dict]:
    """Catalog context for each attribute id, keyed by that id.

    **Reads SEMANTIC_FK backwards**, and is the reason the stored direction and
    the traversal direction have to be kept apart: it binds an attribute and
    finds the columns pointing *at* it. ``join_path_edge`` deliberately omits
    that row, so this queries :data:`~gsf.dal.schema.column_semantic_fk`
    directly.

    Returns ``{}`` on failure rather than raising — this decorates results, and
    losing the decoration beats losing the result.

    One entry per attribute even though an attribute may be referenced by many
    columns: **the owning column wins** — the one linked by ``HAS_ATTRIBUTE``
    rather than one merely pointing at the attribute through ``SEMANTIC_FK``.

    That distinction matters. The attribute *describes* its owning column, so
    returning a referencing column's name and table as the attribute's context
    is simply wrong.
    """
    if not attr_ids:
        return {}

    # The database filter lives in the JOIN condition, not the WHERE. In the
    # WHERE it would delete the attribute's row outright when its column belongs
    # to another database. On the join it leaves the attribute in place with a
    # null database, and the empty-string defaults below take over. Callers read
    # these keys directly, so the difference would show up as a missing entry
    # rather than an error.
    database_join = s.catalog_database.c.id == s.catalog_schema.c.database_id
    if database_name is not None:
        database_join = and_(database_join, s.catalog_database.c.name == database_name)

    # 0 for the owning link, 1 for a referencing one -- the sort key that makes
    # "which column describes this attribute" a decision rather than an accident.
    link = (
        select(
            s.column_has_attribute.c.attribute_id,
            s.column_has_attribute.c.column_id,
            literal(0).label("rank"),
        )
        .union(
            select(
                s.column_semantic_fk.c.attribute_id,
                s.column_semantic_fk.c.column_id,
                literal(1).label("rank"),
            )
        )
        .subquery("link")
    )
    statement = (
        select(
            s.column_attribute.c.id.label("attr_id"),
            s.column_attribute.c.name.label("attr_name"),
            s.column_attribute.c.description.label("attr_description"),
            s.catalog_column.c.id.label("col_id"),
            s.catalog_column.c.name.label("col_name"),
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_database.c.name.label("database_name"),
            s.term.c.name.label("term_name"),
        )
        .select_from(
            s.column_attribute.outerjoin(
                link, link.c.attribute_id == s.column_attribute.c.id
            )
            .outerjoin(s.catalog_column, s.catalog_column.c.id == link.c.column_id)
            .outerjoin(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
            .outerjoin(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .outerjoin(s.catalog_database, database_join)
            .outerjoin(
                s.column_attribute_term,
                s.column_attribute_term.c.attribute_id == s.column_attribute.c.id,
            )
            .outerjoin(s.term, s.term.c.id == s.column_attribute_term.c.term_id)
        )
        .where(s.column_attribute.c.id.in_(list(attr_ids)))
        .order_by(s.column_attribute.c.id, link.c.rank, s.catalog_column.c.id)
    )
    try:
        rows = store().query_read(statement)
    except Exception:
        logger.warning("fetch_attr_column_contexts: query failed", exc_info=True)
        return {}

    # First row per attribute wins, and the ORDER BY above put the owning
    # column first. `setdefault` rather than a comprehension, which would keep
    # the last.
    result: dict[str, dict] = {}
    for row in rows:
        if not row["attr_id"]:
            continue
        result.setdefault(
            row["attr_id"],
            {
                "attr_name": row["attr_name"] or "",
                "attr_description": row["attr_description"] or "",
                "col_id": row["col_id"],
                "col_name": row["col_name"] or "",
                "table_id": row["table_id"],
                "table_name": row["table_name"] or "",
                "schema_name": row["schema_name"] or "",
                "database_name": row["database_name"] or "",
                "term_name": row["term_name"] or "",
            },
        )
    return result


# ---------------------------------------------------------------------------
# SEMANTIC_FK
# ---------------------------------------------------------------------------


#: The FK target side of the join. Aliased because `catalog_column` and
#: `catalog_table` already appear as the *source* side of the same statement.
_fk_target_column = s.catalog_column.alias("fk_target_column")
_fk_target_table = s.catalog_table.alias("fk_target_table")


def find_unlinked_fk_columns(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Columns carrying neither a SEMANTIC_FK nor a HAS_ATTRIBUTE link.

    The work list for FK resolution. Scoped to one database when asked, for the
    same reason as ``fetch_all_tables_without_term``: several databases share a
    store, and a compile pass tags its output with one database's name.

    ``fk_target_col_id`` comes from an outer join, so a column with no declared
    foreign key still appears with ``None`` — those are exactly the ones the
    semantic resolver is for.
    """
    statement = (
        select(
            s.catalog_column.c.id,
            s.catalog_column.c.name,
            s.catalog_column.c.description,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.column_foreign_key.c.target_column_id.label("fk_target_col_id"),
            # The FK target's *table*, not just its column: `resolve_semantic_fks`
            # skips a candidate whose target table is the source table, which it
            # cannot tell without this.
            _fk_target_table.c.id.label("fk_target_table_id"),
        )
        .select_from(
            s.catalog_column.join(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
            .outerjoin(
                s.column_foreign_key,
                s.column_foreign_key.c.source_column_id == s.catalog_column.c.id,
            )
            .outerjoin(
                _fk_target_column,
                _fk_target_column.c.id == s.column_foreign_key.c.target_column_id,
            )
            .outerjoin(
                _fk_target_table,
                _fk_target_table.c.id == _fk_target_column.c.table_id,
            )
        )
        .where(
            ~select(s.column_semantic_fk.c.attribute_id)
            .where(s.column_semantic_fk.c.column_id == s.catalog_column.c.id)
            .exists(),
            ~select(s.column_has_attribute.c.attribute_id)
            .where(s.column_has_attribute.c.column_id == s.catalog_column.c.id)
            .exists(),
        )
    )
    if database_name is not None:
        statement = statement.select_from(
            s.catalog_schema.join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        ).where(
            s.catalog_table.c.schema_id == s.catalog_schema.c.id,
            s.catalog_database.c.name == database_name,
        )

    return [dict(r) for r in store().query_read(statement)]


def merge_semantic_fk(src_column_id: str, tgt_attr_id: str) -> None:
    """Point a column at the attribute it references. Idempotent."""
    store().query_write(
        insert(s.column_semantic_fk)
        .values(column_id=src_column_id, attribute_id=tgt_attr_id)
        .on_conflict_do_nothing()
    )


def _column_path_select(link_table):
    """Column with its full catalog path, joined through *link_table*."""
    return select(
        link_table.c.attribute_id.label("attr_id"),
        s.catalog_column.c.id,
        s.catalog_column.c.name.label("column_name"),
        s.catalog_table.c.id.label("table_id"),
        s.catalog_table.c.name.label("table_name"),
        s.catalog_schema.c.id.label("schema_id"),
        s.catalog_database.c.id.label("db_id"),
    ).select_from(
        link_table.join(
            s.catalog_column, s.catalog_column.c.id == link_table.c.column_id
        )
        .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
        .join(s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id)
        .join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
    )


def _column_path_dict(row: dict[str, Any]) -> dict[str, Any]:
    """The API's column-ref shape: path ids for navigation, names for display."""
    return {
        "id": row["id"],
        "column_name": row["column_name"],
        "table_id": row["table_id"],
        "table_name": row["table_name"],
        "schema_id": row["schema_id"],
        "db_id": row["db_id"],
    }


def fetch_column_attribute_columns_map(
    attr_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """``attr_id -> {primary_column, referenced_columns}``.

    The primary column *has* the attribute; the referenced columns point at it.
    Every requested id gets an entry, defaulted rather than omitted, so callers
    need no ``.get`` dance.

    ``primary_column`` keeps the **first** row and ignores later ones, ordered
    by column id so "first" is the same first each time.
    """
    if not attr_ids:
        return {}

    ids = list(attr_ids)
    primary = store().query_read(
        _column_path_select(s.column_has_attribute)
        .where(s.column_has_attribute.c.attribute_id.in_(ids))
        .order_by(s.catalog_column.c.id)
    )
    referenced = store().query_read(
        _column_path_select(s.column_semantic_fk)
        .where(s.column_semantic_fk.c.attribute_id.in_(ids))
        .order_by(s.catalog_table.c.name, s.catalog_column.c.name)
    )

    result: dict[str, dict[str, Any]] = {
        attr_id: {"primary_column": None, "referenced_columns": []} for attr_id in ids
    }
    for row in primary:
        entry = result.get(row["attr_id"])
        if entry is not None and entry["primary_column"] is None:
            entry["primary_column"] = _column_path_dict(row)
    for row in referenced:
        entry = result.get(row["attr_id"])
        if entry is not None:
            entry["referenced_columns"].append(_column_path_dict(row))
    return result


# ---------------------------------------------------------------------------
# Join path traversal
# ---------------------------------------------------------------------------

#: One BFS level: every node reachable in one edge from the frontier, minus the
#: nodes already seen. Two round trips per level would be needed to do the
#: exclusion in Python, so both sets are sent.
#:
#: They go as **two array parameters**, ``= ANY(:frontier)`` and
#: ``<> ALL(:visited)``, rather than as ``IN``/``NOT IN`` over expanding
#: bindparams. Expanding bindparams render one placeholder per element, so the
#: parameter count grows with the reachable component -- and the visited set
#: only ever grows, one level feeding the next. On a hub attribute like
#: ``customer id``, fanning out across hundreds of columns, that means a SQL
#: string of a different shape on every level (so a fresh parse and plan each
#: time, and the statement cache never hits) and, far enough out, Postgres'
#: 65535-parameter ceiling. Passing arrays makes the statement one fixed shape
#: with two parameters, whatever the search costs.
#:
#: ``= ANY(array)`` is still an index lookup on ``src_id``; this trades nothing
#: away for it.
_EXPAND_LEVEL = (
    select(
        s.join_path_edge.c.src_kind,
        s.join_path_edge.c.src_id,
        s.join_path_edge.c.dst_kind,
        s.join_path_edge.c.dst_id,
    )
    .where(
        s.join_path_edge.c.src_id == any_(bindparam("frontier", type_=ARRAY(Text))),
        s.join_path_edge.c.dst_id != all_(bindparam("visited", type_=ARRAY(Text))),
    )
    .distinct()
)


def _bfs_path(anchor_col_id: str, dest_col_id: str) -> list[dict[str, Any]] | None:
    """Shortest path as a node list, or ``None`` when there is none.

    Level-at-a-time BFS with the visited set held here rather than in SQL, and
    that is not a stylistic choice. A recursive CTE tracks visited nodes *per
    path*, so every distinct route to a node is expanded separately — fine on a
    fixture, exponential on a catalog where a hub attribute like ``customer id``
    fans out across hundreds of columns. A shared visited set bounds the cost by
    the size of the reachable component, however many paths run through it.

    Real paths are 2-4 hops, so this is typically 3-5 indexed queries.
    """
    visited: dict[str, tuple[str, str | None]] = {anchor_col_id: ("column", None)}
    frontier = [anchor_col_id]

    for _ in range(MAX_PATH_DEPTH):
        if not frontier:
            return None
        rows = store().query_read(
            _EXPAND_LEVEL,
            {"frontier": frontier, "visited": list(visited)},
        )
        next_frontier: list[str] = []
        for row in rows:
            node = row["dst_id"]
            if node in visited:
                # Two edges into the same node within one level: the first wins,
                # which is what BFS means. Without this the parent pointer could
                # be overwritten by a longer route discovered in the same batch.
                continue
            visited[node] = (row["dst_kind"], row["src_id"])
            next_frontier.append(node)
            if node == dest_col_id:
                return _walk_back(visited, dest_col_id)
        frontier = next_frontier

    logger.warning(
        "find_join_path: gave up after %s levels for %s -> %s",
        MAX_PATH_DEPTH,
        anchor_col_id,
        dest_col_id,
    )
    return None


def _walk_back(
    visited: dict[str, tuple[str, str | None]], dest: str
) -> list[dict[str, Any]]:
    """Follow parent pointers back to the anchor and return the path forwards."""
    path: list[dict[str, Any]] = []
    node: str | None = dest
    while node is not None:
        kind, parent = visited[node]
        path.append({"id": node, "kind": kind})
        node = parent
    path.reverse()
    return path


def _name_columns(node_ids: list[str]) -> dict[str, str]:
    if not node_ids:
        return {}
    return {
        r["id"]: r["name"]
        for r in store().query_read(
            select(s.catalog_column.c.id, s.catalog_column.c.name).where(
                s.catalog_column.c.id.in_(node_ids)
            )
        )
    }


def find_join_path(anchor_col_id: str, dest_col_id: str) -> list[dict]:
    """The shortest semantic join route between two columns, as hop dicts.

    ``[]`` when the columns are the same, when no path exists, or when the only
    path crosses databases.

    The traversal rules live in the ``join_path_edge`` view, and the one that
    matters is that **SEMANTIC_FK is emitted outgoing only**. Traversed
    backwards, a path hops from one FK column up to a shared attribute and back
    down a *different* FK column, inventing a join between two columns that
    merely reference the same thing — two ``customer_id`` columns joined to each
    other. That is a wrong answer that looks entirely reasonable, which is why
    it gets a test of its own.

    After the traversal: keep the column nodes, pair them ``(0,1), (2,3), …``,
    and reject the path if it spans more than one database. The pairing is
    subtle — the intermediate table and attribute nodes are dropped first,
    leaving columns in join-partner order.
    """
    if anchor_col_id == dest_col_id:
        return []

    try:
        path = _bfs_path(anchor_col_id, dest_col_id)
    except Exception:
        logger.warning(
            "find_join_path: query failed for %s -> %s",
            anchor_col_id,
            dest_col_id,
            exc_info=True,
        )
        return []

    if not path:
        return []

    col_nodes = [n for n in path if n["kind"] == "column"]
    if len(col_nodes) < 2:
        return []

    col_ids = [n["id"] for n in col_nodes]
    names = _name_columns(col_ids)
    contexts = fetch_col_table_contexts(col_ids)

    databases = {
        context.get("database_name")
        for context in contexts.values()
        if context.get("database_name")
    }
    if len(databases) > 1:
        # Only reachable through a shared ColumnAttribute -- Database is not a
        # node in the view, so there is no other way across.
        logger.warning(
            "find_join_path: rejected cross-database path %s -> %s (%s)",
            anchor_col_id,
            dest_col_id,
            ", ".join(sorted(databases)),
        )
        return []

    hops: list[dict] = []
    for i in range(0, len(col_nodes) - 1, 2):
        source, target = col_nodes[i], col_nodes[i + 1]
        source_context = contexts.get(source["id"], {})
        target_context = contexts.get(target["id"], {})
        hops.append(
            {
                "source_database": source_context.get("database_name", ""),
                "source_schema": source_context.get("schema_name", ""),
                "source_table": source_context.get("table_name", ""),
                "source_column": names.get(source["id"], ""),
                "target_database": target_context.get("database_name", ""),
                "target_schema": target_context.get("schema_name", ""),
                "target_table": target_context.get("table_name", ""),
                "target_column": names.get(target["id"], ""),
            }
        )
    return hops


#: The same traversal as a recursive CTE, used by the tests as a cross-check.
#:
#: Correct, and *not* what :func:`_bfs_path` does -- its ``path`` array is a
#: per-path visited set, so it re-expands every distinct route to a node rather
#: than visiting each once, which is the scaling problem described there. Kept
#: beside the BFS rather than in the test file so the two sit together and a
#: change to the edge rules is obviously a change to both.
JOIN_PATH_CTE_SQL = """
WITH RECURSIVE walk(node_id, node_kind, path, depth) AS (
    SELECT CAST(:anchor AS text), 'column'::text, ARRAY[CAST(:anchor AS text)], 0
  UNION ALL
    SELECT e.dst_id, e.dst_kind, w.path || e.dst_id, w.depth + 1
      FROM walk w
      JOIN join_path_edge e ON e.src_id = w.node_id
     WHERE NOT e.dst_id = ANY(w.path)
       AND w.depth < :max_depth
       AND w.node_id <> CAST(:dest AS text)
)
SELECT path, depth FROM walk
 WHERE node_id = CAST(:dest AS text)
 ORDER BY depth, path
 LIMIT 1
"""
