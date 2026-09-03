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

import itertools
import logging
from typing import Any

from sqlalchemy import (
    Text,
    all_,
    and_,
    any_,
    bindparam,
    func,
    literal,
    or_,
    select,
    update,
)
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
                s.column__has_attribute,
                {"column_id": column_id, "attribute_id": attribute_id},
            ),
            (
                s.column_attribute__term,
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
            s.column_attribute__term.c.attribute_id == attr_id,
            s.column_attribute__term.c.term_id == term_id,
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
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id == s.column_attribute.c.id,
            )
            .join(s.term, s.term.c.id == s.column_attribute__term.c.term_id)
            .outerjoin(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
            .outerjoin(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
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
            s.column__has_attribute.join(
                s.column_attribute,
                s.column_attribute.c.id == s.column__has_attribute.c.attribute_id,
            )
        )
        .where(
            s.column__has_attribute.c.column_id == column_id,
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
    that row, so this queries :data:`~gsf.dal.schema.column__semantic_fk`
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
            s.column__has_attribute.c.attribute_id,
            s.column__has_attribute.c.column_id,
            literal(0).label("rank"),
        )
        .union(
            select(
                s.column__semantic_fk.c.attribute_id,
                s.column__semantic_fk.c.column_id,
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
            s.column_attribute.c.datatype.label("datatype"),
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
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id == s.column_attribute.c.id,
            )
            .outerjoin(s.term, s.term.c.id == s.column_attribute__term.c.term_id)
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
                "datatype": row["datatype"] or "",
            },
        )
    return result


# ---------------------------------------------------------------------------
# Column lookup by name (for verifying a join predicate already written in SQL)
# ---------------------------------------------------------------------------


def column_participates_in_semantic_fk(col_id: str) -> bool:
    """Whether *col_id* is already known to the FK graph, on either side.

    True if the column is itself an FK-holder (outgoing ``SEMANTIC_FK``) or is
    the referenced/identity side of one (its own ``ColumnAttribute``, via
    ``HAS_ATTRIBUTE``, is the target of some other column's ``SEMANTIC_FK``).
    Used to scope the join-path check to columns ingestion already treats as
    FK-shaped, rather than flagging arbitrary equality joins (date ranges,
    status matches, business logic) the catalog was never meant to model.
    """
    if not col_id:
        return False
    direct = (
        select(literal(1)).where(s.column__semantic_fk.c.column_id == col_id).exists()
    )
    via_attribute = (
        select(literal(1))
        .select_from(
            s.column__has_attribute.join(
                s.column__semantic_fk,
                s.column__semantic_fk.c.attribute_id
                == s.column__has_attribute.c.attribute_id,
            )
        )
        .where(s.column__has_attribute.c.column_id == col_id)
        .exists()
    )
    rows = store().query_read(select(or_(direct, via_attribute).label("participates")))
    return bool(rows and rows[0]["participates"])


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

    When *database_name* is provided, only columns belonging to that database
    are returned. Multiple databases can be co-resident in the same store, so
    scoping keeps each compile pass' FK-resolution isolated to a single
    database. When omitted, every unlinked FK column is returned.

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
            s.column__foreign_key.c.target_column_id.label("fk_target_col_id"),
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
                s.column__foreign_key,
                s.column__foreign_key.c.source_column_id == s.catalog_column.c.id,
            )
            .outerjoin(
                _fk_target_column,
                _fk_target_column.c.id == s.column__foreign_key.c.target_column_id,
            )
            .outerjoin(
                _fk_target_table,
                _fk_target_table.c.id == _fk_target_column.c.table_id,
            )
        )
        .where(
            ~select(s.column__semantic_fk.c.attribute_id)
            .where(s.column__semantic_fk.c.column_id == s.catalog_column.c.id)
            .exists(),
            ~select(s.column__has_attribute.c.attribute_id)
            .where(s.column__has_attribute.c.column_id == s.catalog_column.c.id)
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
        insert(s.column__semantic_fk)
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
        _column_path_select(s.column__has_attribute)
        .where(s.column__has_attribute.c.attribute_id.in_(ids))
        .order_by(s.catalog_column.c.id)
    )
    referenced = store().query_read(
        _column_path_select(s.column__semantic_fk)
        .where(s.column__semantic_fk.c.attribute_id.in_(ids))
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


def _bfs_path(
    anchor_col_id: str,
    dest_col_id: str,
    *,
    anchor_kind: str = "column",
    log_label: str = "find_join_path",
) -> list[dict[str, Any]] | None:
    """Shortest path as a node list, or ``None`` when there is none.

    Level-at-a-time BFS with the visited set held here rather than in SQL, and
    that is not a stylistic choice. A recursive CTE tracks visited nodes *per
    path*, so every distinct route to a node is expanded separately — fine on a
    fixture, exponential on a catalog where a hub attribute like ``customer id``
    fans out across hundreds of columns. A shared visited set bounds the cost by
    the size of the reachable component, however many paths run through it.

    Real paths are 2-4 hops, so this is typically 3-5 indexed queries.

    *anchor_kind* defaults to ``"column"`` for :func:`find_join_path`'s
    column-to-column search. :func:`find_table_bridge` passes ``"table"`` to
    reuse the same BFS engine (and the same ``join_path_edge`` traversal
    rules) starting from a Table node instead.
    """
    visited: dict[str, tuple[str, str | None]] = {anchor_col_id: (anchor_kind, None)}
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
        "%s: gave up after %s levels for %s -> %s",
        log_label,
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


def _extract_column_hops(path: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Reconstruct real crossings (column -> column_attribute -> column) from a BFS path.

    A BFS path over ``join_path_edge`` can interleave ``column <-> table`` edges
    (purely structural containment — a table has many columns) with the actual
    semantic crossings (``column <-> column_attribute``, via HAS_ATTRIBUTE /
    SEMANTIC_FK). Filtering the path down to column-kind nodes and pairing them
    by position (0&1, 2&3, ...) silently discards what actually connects each
    pair — so two columns that merely share a table (column -> table -> column,
    no attribute in between at all) would be asserted as a join. Confirmed
    live in the Neo4j-era code this replaces: two columns sharing a table with
    zero FK relationship between them were reported as a "1 hop" join.

    This walks the path in order and only emits a hop where a column reaches a
    column_attribute, and that same attribute is reached by another column —
    i.e. an actual FK crossing, not co-location. A ``table`` node is still
    needed and still used, just never as the crossing itself: it's how the
    path steps from a crossing's landing column, through its table, to a
    *different* column that continues the next crossing (e.g. bridging two
    hops through a shared pivot table).

    Pre-existing on main, not branch-specific: the Postgres rewrite of
    find_join_path/find_table_bridge dropped this guard when it moved off
    Cypher, independent of anything in this branch's own history. Worth
    surfacing to whoever owns main, since any other branch built on top of
    it inherits the same false positive.

    Returns a list of (source_column_id, target_column_id) pairs, in path order.
    """
    hops: list[tuple[str, str]] = []
    pending_src: str | None = None
    for i in range(len(path) - 1):
        cur, nxt = path[i], path[i + 1]
        if cur["kind"] == "column" and nxt["kind"] == "column_attribute":
            pending_src = cur["id"]
        elif (
            pending_src is not None
            and cur["kind"] == "column_attribute"
            and nxt["kind"] == "column"
        ):
            hops.append((pending_src, nxt["id"]))
            pending_src = None
    return hops


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


def _name_tables(node_ids: list[str]) -> dict[str, str]:
    if not node_ids:
        return {}
    return {
        r["id"]: r["name"]
        for r in store().query_read(
            select(s.catalog_table.c.id, s.catalog_table.c.name).where(
                s.catalog_table.c.id.in_(node_ids)
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

    After the traversal, hops are extracted by :func:`_extract_column_hops`,
    which only counts a pair of columns as joined when they're actually linked
    through a column_attribute node — not merely by both belonging to the same
    table. See its docstring for why: naively pairing every column-kind node on
    the path by position would report two unrelated columns that happen to
    share a table as a false "1 hop" join.
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

    hop_pairs = _extract_column_hops(path)
    if not hop_pairs:
        return []

    col_ids = list(dict.fromkeys(cid for pair in hop_pairs for cid in pair))
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
    for source_id, target_id in hop_pairs:
        source_context = contexts.get(source_id, {})
        target_context = contexts.get(target_id, {})
        hops.append(
            {
                "source_database": source_context.get("database_name", ""),
                "source_schema": source_context.get("schema_name", ""),
                "source_table": source_context.get("table_name", ""),
                "source_column": names.get(source_id, ""),
                "target_database": target_context.get("database_name", ""),
                "target_schema": target_context.get("schema_name", ""),
                "target_table": target_context.get("table_name", ""),
                "target_column": names.get(target_id, ""),
            }
        )
    return hops


def find_shared_hub_bridge(col_a_id: str, col_b_id: str) -> dict:
    """Find a genuine shared-identity hub connecting two FK columns that
    :func:`find_join_path` cannot reach.

    ``find_join_path``'s forward-only ``SEMANTIC_FK`` traversal is a
    deliberate guard: two unrelated FK columns that merely reference the
    same target (e.g. two ``person_id`` columns for different roles) must
    never be reported as directly joined by walking the edge backwards.
    But that guard also blocks a *legitimate* case with the identical shape
    — two spokes of the same real identity hub (e.g. ``robot_details`` and
    ``actuation_data`` both referencing ``robot_record`` via its own primary
    key) — from ever being discovered, since neither spoke has anything
    pointing *out* toward the other.

    This is intentionally much narrower than lifting the forward-only guard
    in general: it only reports a bridge when BOTH columns independently
    hold a *forward* ``SEMANTIC_FK`` edge to the exact same
    ``ColumnAttribute``, AND that attribute's defining column is the real,
    declared primary key of its own table (``hubCol.name IN hubTable.pk``)
    — the same discipline :func:`find_anchor_hub_siblings` already applies.
    Requiring the shared target to be a genuine identity column (not just
    any attribute two FK columns happen to share) is what keeps this from
    reopening the "two unrelated FK columns, coincidentally shared target"
    fabrication risk the forward-only design exists to prevent.

    Returns ``{"hub_table", "hub_column"}`` (schema-unqualified — callers
    already have the written table/column names and schema from the SQL
    itself, so this only needs to supply the hub side), or ``{}`` if no such
    shared, PK-anchored hub exists.
    """
    if col_a_id == col_b_id:
        return {}

    fk_a = s.column__semantic_fk.alias("fk_a")
    fk_b = s.column__semantic_fk.alias("fk_b")
    statement = (
        select(
            s.catalog_column.c.name.label("hub_column"),
            s.catalog_table.c.name.label("hub_table"),
        )
        .select_from(
            fk_a.join(fk_b, fk_b.c.attribute_id == fk_a.c.attribute_id)
            .join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == fk_a.c.attribute_id,
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
        )
        .where(
            fk_a.c.column_id == col_a_id,
            fk_b.c.column_id == col_b_id,
            # The shared target must be the hub table's own declared PK, not
            # just any attribute the two columns happen to share -- see the
            # docstring above.
            s.catalog_column.c.name == any_(s.catalog_table.c.pk),
        )
        .limit(1)
    )
    try:
        rows = store().query_read(statement)
    except Exception:
        logger.warning(
            "find_shared_hub_bridge: query failed for %s / %s",
            col_a_id,
            col_b_id,
            exc_info=True,
        )
        return {}
    if not rows:
        return {}
    return {
        "hub_table": rows[0]["hub_table"] or "",
        "hub_column": rows[0]["hub_column"] or "",
    }


def find_anchor_hub_siblings(
    anchor_table_id: str, max_siblings: int | None = 6
) -> tuple[list[dict], int]:
    """Find tables that share a hub table with *anchor_table_id* via FK.

    Looser than :func:`find_join_path` on purpose: it takes ONE step backward
    through a hub table's real, DB-declared primary key (verified against
    ``Table.pk``, not just any ColumnAttribute) — starting only from the
    anchor table's own outgoing SEMANTIC_FK edges, never from every
    candidate table's PK. This finds sibling tables (e.g. two tables that
    both reference the same parent) that :func:`find_join_path`'s
    forward-only traversal structurally cannot reach, without risking the
    false-positive fan-out find_join_path guards against.

    Includes the hub table itself (not just its siblings) — a question can
    need the hub as its own base/pivot table (e.g. to count every row even
    when a sibling has no matching data), and the hub is cheap and safe to
    add: exactly one verified, single-hop forward FK per distinct hub, not
    proportional to how many siblings reference it.

    Sibling count is capped at *max_siblings* per hub, applied here in
    Neo4j-return order (arbitrary — no relevance signal). *max_siblings=None*
    skips capping entirely and returns every sibling; callers that want a
    relevance-ranked cap (e.g. embedding similarity to the question) should
    pass ``None`` here and rank+truncate themselves — see
    ``CandidatePreparationAgent._rank_and_cap_hub_siblings`` for the ranked
    version used in production, which replaced a blind order-based cap after
    an audit found it was silently dropping the one relevant sibling in
    ~30% of truncation events. This function's own cap stays as a safety-net
    default for any other/future caller that doesn't rank — measured against
    the live graph, most hubs have few siblings (median 2), but some are
    genuine mega-hubs (up to 20), and an uncapped expansion would dump a
    large, low-precision batch of tables into the candidate set for those.

    This is a discovery/relevance signal, not a verified join path — it
    does NOT claim the anchor and sibling should be joined directly (they
    usually should each join the shared hub instead). Callers must not feed
    this into SQL-generation join instructions; it exists only to stop the
    relevance filter from dropping a structurally-connected table it has no
    other way to recognize.

    Returns ``(results, truncated_count)``. Each result dict has
    ``{source_table, target_table, id, hub_table, is_hub}`` — the first two
    keys match :func:`find_join_path`'s hop shape for prompt rendering: for
    hub entries ``target_table`` is the hub name, for sibling entries it's
    the sibling name. ``truncated_count`` is how many sibling tables were
    dropped by the cap (0 if none were).
    """
    fk_col = s.catalog_column.alias("fk_col")
    hub_col = s.catalog_column.alias("hub_col")
    hub_table = s.catalog_table.alias("hub_table")
    sibling_fk = s.column__semantic_fk.alias("sibling_fk")
    sibling_col = s.catalog_column.alias("sibling_col")
    sibling_table = s.catalog_table.alias("sibling_table")

    statement = (
        select(
            sibling_table.c.id.label("sibling_id"),
            sibling_table.c.name.label("sibling_table"),
            hub_table.c.id.label("hub_id"),
            hub_table.c.name.label("hub_table"),
            s.catalog_table.c.name.label("anchor_table"),
        )
        .distinct()
        .select_from(
            s.catalog_table.join(fk_col, fk_col.c.table_id == s.catalog_table.c.id)
            .join(
                s.column__semantic_fk,
                s.column__semantic_fk.c.column_id == fk_col.c.id,
            )
            .join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id
                == s.column__semantic_fk.c.attribute_id,
            )
            .join(hub_col, hub_col.c.id == s.column__has_attribute.c.column_id)
            .join(hub_table, hub_table.c.id == hub_col.c.table_id)
            .join(
                sibling_fk,
                sibling_fk.c.attribute_id == s.column__semantic_fk.c.attribute_id,
            )
            .join(sibling_col, sibling_col.c.id == sibling_fk.c.column_id)
            .join(sibling_table, sibling_table.c.id == sibling_col.c.table_id)
        )
        .where(
            s.catalog_table.c.id == anchor_table_id,
            # The hub column must be the hub table's own declared PK -- the
            # same discipline find_shared_hub_bridge applies.
            hub_col.c.name == any_(hub_table.c.pk),
            sibling_table.c.id != s.catalog_table.c.id,
        )
    )
    try:
        rows = store().query_read(statement)
    except Exception:
        logger.warning(
            "find_anchor_hub_siblings: query failed for table %s",
            anchor_table_id,
            exc_info=True,
        )
        return [], 0

    anchor_name = ""
    hubs: dict[str, str] = {}  # hub_id -> hub_name
    siblings_by_hub: dict[str, list[dict]] = {}
    for row in rows:
        anchor_name = row["anchor_table"] or anchor_name
        hub_id, hub_name = row["hub_id"], row["hub_table"]
        if not hub_id:
            continue
        hubs[hub_id] = hub_name or ""
        sib_id, sib_name = row["sibling_id"], row["sibling_table"]
        if sib_id:
            siblings_by_hub.setdefault(hub_id, []).append(
                {"id": sib_id, "name": sib_name or ""}
            )

    results: list[dict] = []
    truncated = 0
    for hub_id, hub_name in hubs.items():
        results.append(
            {
                "source_table": anchor_name,
                "target_table": hub_name,
                "id": hub_id,
                "hub_table": hub_name,
                "is_hub": True,
            }
        )
        sibs = siblings_by_hub.get(hub_id, [])
        if max_siblings is not None and len(sibs) > max_siblings:
            truncated += len(sibs) - max_siblings
            sibs = sibs[:max_siblings]
        for sib in sibs:
            results.append(
                {
                    "source_table": anchor_name,
                    "target_table": sib["name"],
                    "id": sib["id"],
                    "hub_table": hub_name,
                    "is_hub": False,
                }
            )
    return results, truncated


def find_table_bridge(
    table_a_id: str,
    table_b_id: str,
    allowed_table_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Find bridge tables AND their join hops (if any) needed to connect two tables.

    Runs the same forward-only traversal as :func:`find_join_path`, but
    starts from table_a as a whole rather than a single representative
    column — a table's own primary key is usually a pure identity column
    with no outgoing FK of its own (the real FK, e.g. ``member_fan_pivot``,
    is a *different* column on the same table), so anchoring on the PK
    alone would miss the real connection or find an unrelated coincidental
    one. Starting from the Table node and letting CONTAINS (undirected, same
    as find_join_path) walk into every one of its own columns finds the real
    path regardless of which specific column carries the FK. Tried in both
    directions since SEMANTIC_FK is forward-only.

    IMPORTANT: an unrestricted traversal can find a real but coincidental
    forward path through a table that was never presented to (or judged by)
    the relevance filter at all — e.g. two tables that both happen to FK
    into an unrelated "preferences" table for reasons having nothing to do
    with the actual question. That's not "the filter missed a bridge it
    should have kept", it's introducing information the filter never had a
    chance to accept or reject. So when *allowed_table_ids* is given, a
    found path is only returned if every intermediate table on it is a
    member of that set (typically: the tables that were candidates *before*
    the relevance filter ran) — this keeps the function to "restore a
    bridge the filter had and dropped", never "discover a new one".

    Returns ``(bridge_tables, hops)``:
      - ``bridge_tables``: ``{id, name}`` dicts for every distinct table
        strictly between table_a and table_b on the shortest path found —
        excludes the two endpoint tables themselves.
      - ``hops``: join hop dicts in the same shape :func:`find_join_path`
        produces (``{source_schema, source_table, source_column,
        target_schema, target_table, target_column}``), describing the
        actual FK columns connecting table_a -> ... -> table_b. Without
        this, callers only learn a bridge table's *name*, not how to join
        it — earlier versions of this function returned table names alone
        and downstream SQL-gen had to guess the join, which produced a
        fabricated join between two unrelated PK columns.

    Both empty if no path exists in either direction, or the only path
    found steps outside *allowed_table_ids*.
    """
    for src, dst in ((table_a_id, table_b_id), (table_b_id, table_a_id)):
        try:
            path = _bfs_path(
                src, dst, anchor_kind="table", log_label="find_table_bridge"
            )
        except Exception:
            logger.warning(
                "find_table_bridge: query failed for %s -> %s",
                src,
                dst,
                exc_info=True,
            )
            continue
        if not path:
            continue

        bridge_ids = [
            n["id"]
            for n in path
            if n["kind"] == "table" and n["id"] not in (table_a_id, table_b_id)
        ]
        if not bridge_ids:
            continue
        if allowed_table_ids is not None and not set(bridge_ids).issubset(
            allowed_table_ids
        ):
            logger.info(
                "find_table_bridge: discarding path %s -> %s — bridge table(s) "
                "were never a candidate, not just restoring a dropped one",
                src,
                dst,
            )
            continue

        # Only the real SEMANTIC_FK/HAS_ATTRIBUTE crossings count as join hops
        # -- a bridge table with no column reaching another column through a
        # column_attribute node has no real crossing at all, just co-location
        # (see find_join_path and _extract_column_hops, which this reuses),
        # and is rejected rather than handed back as a table we can't
        # actually explain how to join.
        hop_pairs = _extract_column_hops(path)
        if not hop_pairs:
            logger.warning(
                "find_table_bridge: path %s -> %s found bridge table(s) but "
                "no real FK crossing — discarding (co-location, not a join)",
                src,
                dst,
            )
            continue

        col_ids = list(dict.fromkeys(cid for pair in hop_pairs for cid in pair))
        names = _name_columns(col_ids)
        col_ctx = fetch_col_table_contexts(col_ids)
        databases = {
            context.get("database_name")
            for context in col_ctx.values()
            if context.get("database_name")
        }
        if len(databases) > 1:
            logger.warning(
                "find_table_bridge: rejected cross-database path %s -> %s (%s)",
                src,
                dst,
                ", ".join(sorted(databases)),
            )
            continue

        hops: list[dict] = []
        for source_id, target_id in hop_pairs:
            source_context = col_ctx.get(source_id, {})
            target_context = col_ctx.get(target_id, {})
            hops.append(
                {
                    "source_schema": source_context.get("schema_name", ""),
                    "source_table": source_context.get("table_name", ""),
                    "source_column": names.get(source_id, ""),
                    "target_schema": target_context.get("schema_name", ""),
                    "target_table": target_context.get("table_name", ""),
                    "target_column": names.get(target_id, ""),
                }
            )
        table_names = _name_tables(bridge_ids)
        bridge_tables = [
            {"id": tid, "name": table_names.get(tid, "")} for tid in bridge_ids
        ]
        return bridge_tables, hops
    return [], []


def find_kept_table_bridges(
    table_ids: list[str],
    allowed_table_ids: set[str] | None = None,
    max_bridge_tables: int = 5,
) -> tuple[list[dict], list[list[dict]], int]:
    """Find bridge tables AND their join hops needed to connect pairs of
    already-kept tables.

    Runs *after* the relevance filter has already narrowed candidates down
    — kept sets are consistently small in practice (2-4 tables), so
    checking every pair is cheap by construction. Stops early once
    *max_bridge_tables* distinct bridge tables have been found regardless,
    as a hard safety cap for the rare case a kept set is larger than usual.

    *allowed_table_ids*, if given, is passed through to
    :func:`find_table_bridge` to restrict results to tables that were
    already candidates before the relevance filter ran — see its docstring
    for why this matters.

    If the LLM already kept a genuine bridge table C alongside A and B, the
    A-C and C-B pairs would otherwise "rediscover" C as if it were new,
    burning cap budget on a table that was never missing — so any table
    already present in *table_ids* is excluded from counting as a found
    bridge (it needs no restoring; it's already there). Join hops for such
    a pair are still collected — even an already-kept table needs its join
    condition surfaced, or SQL-gen has the table but not the join, same
    failure this whole function exists to prevent.

    Returns ``(bridge_tables, bridge_paths, skipped_pairs)``:
      - ``bridge_tables``: ``{id, name}`` dicts (deduped across all pairs,
        excluding tables already in *table_ids*).
      - ``bridge_paths``: one hop-list per pair that found a bridge (each
        hop shaped like :func:`find_join_path`'s output) — meant to be
        appended to ``attribute_join_paths`` as ``{"path": hops}`` entries
        so SQL-gen actually learns how to join the bridge table in, not
        just that it exists.
      - ``skipped_pairs``: how many remaining pairs were never checked
        because the cap was already hit.
    """
    if len(table_ids) < 2:
        return [], [], 0

    already_kept = set(table_ids)
    pairs = list(itertools.combinations(table_ids, 2))
    found: dict[str, dict] = {}
    bridge_paths: list[list[dict]] = []
    checked = 0
    for a_id, b_id in pairs:
        if len(found) >= max_bridge_tables:
            break
        checked += 1
        bridge_tables, hops = find_table_bridge(a_id, b_id, allowed_table_ids)
        for t in bridge_tables:
            if t["id"] in already_kept:
                continue
            found.setdefault(t["id"], t)
        if bridge_tables and hops:
            bridge_paths.append(hops)

    skipped = len(pairs) - checked
    if skipped:
        logger.info(
            "find_kept_table_bridges: cap of %d bridge table(s) hit — "
            "skipped %d/%d remaining pair(s)",
            max_bridge_tables,
            skipped,
            len(pairs),
        )
    return list(found.values()), bridge_paths, skipped


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
