# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ColumnAttribute reads and writes, semantic FK lookups, and join-path traversal.

Semantic FKs are stored on concepts (:mod:`auto_ontology.dal.concepts`).
``find_join_path`` and ``find_table_bridge`` both run BFS over the concepts'
table edges (``_find_table_join_hops``), keeping the exact join columns --
every pair of a composite key -- on every edge.
"""

from __future__ import annotations

from collections import defaultdict, deque
import itertools
import logging
from typing import Any

from sqlalchemy import (
    and_,
    func,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.dal import schema as s
from auto_ontology.dal.concepts import (
    SEMANTIC_FK_LINKS,
    column_has_semantic_fk,
    link_column_to_attribute,
    semantic_table_edges,
)
from auto_ontology.dal.datasources import fetch_col_table_contexts
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.semantic.constants import SEMANTIC_SOURCE

logger = logging.getLogger(__name__)

#: Path length ceiling, in **edges** of the original column-level graph.
#:
#: It is tempting to lower this on the grounds that real join paths are 2-4
#: hops. That reasoning is wrong, and the depth-bound test catches it: a *hop*
#: was four edges, not one -- Column -> key -> Column -> Table -> Column -- so
#: an n-hop join path is ``4n - 2`` edges and the table BFS allows
#: ``(MAX_PATH_DEPTH + 2) // 4`` hops. A perfectly ordinary 4-hop path is 14,
#: and a ceiling of 10 would have silently returned "no path" for it.
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


def fetch_column_attribute_column(attr_id: str) -> dict[str, Any] | None:
    """Return ``{id, data_type}`` of the Column owning a ColumnAttribute.

    ``None`` when the attribute is missing or has no owning Column. Read on
    its own, ahead of ``update_column_attribute``, so a sample-value edit can
    be refused against the Column's declared type before any metadata is
    written (see ``auto_ontology.server.terms.service.update_column_attribute``).
    """
    rows = store().query_read(
        select(s.catalog_column.c.id, s.catalog_column.c.data_type)
        .select_from(
            s.catalog_column.join(
                s.column__has_attribute,
                s.column__has_attribute.c.column_id == s.catalog_column.c.id,
            )
        )
        .where(s.column__has_attribute.c.attribute_id == attr_id)
        .limit(1)
    )
    return dict(rows[0]) if rows else None


def update_column_attribute(
    attr_id: str,
    term_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    certified: bool | None = None,
    modified_by: str | None = None,
    touch: bool = False,
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
    values: dict[str, Any] = {
        "name": func.coalesce(name, s.column_attribute.c.name),
        "description": func.coalesce(description, s.column_attribute.c.description),
        "certified": func.coalesce(certified, s.column_attribute.c.certified),
    }
    if touch:
        values["modified"] = func.now()
        values["modified_by"] = modified_by
    updated = store().query_write(
        update(s.column_attribute)
        .where(s.column_attribute.c.id == attr_id, owned)
        .values(**values)
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

    **Reads semantic FKs backwards**: it binds an attribute and also finds the
    columns referencing it -- the columns whose semantic FK joins the
    attribute's column, read from
    :data:`~auto_ontology.dal.concepts.SEMANTIC_FK_LINKS`.

    Returns ``{}`` on failure rather than raising — this decorates results, and
    losing the decoration beats losing the result.

    One entry per attribute even though an attribute may be referenced by many
    columns: **the owning column wins** — the one linked by ``HAS_ATTRIBUTE``
    rather than one merely referencing the attribute's column.

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
                SEMANTIC_FK_LINKS.c.attribute_id,
                SEMANTIC_FK_LINKS.c.column_id,
                literal(1).label("rank"),
            ).where(SEMANTIC_FK_LINKS.c.attribute_id.is_not(None))
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

    True if the column is part of a semantic FK, or of a concept key that a
    semantic FK references or that a second member table shares. Used to
    scope the join-path check to columns ingestion already treats as
    FK-shaped, rather than flagging arbitrary equality joins (date ranges,
    status matches, business logic) the catalog was never meant to model.
    """
    if not col_id:
        return False
    direct = column_has_semantic_fk(col_id)
    referenced = (
        select(literal(1))
        .where(SEMANTIC_FK_LINKS.c.target_column_id == col_id)
        .exists()
    )
    other_member = s.concept__pk.alias("other_member_key")
    shared_key = (
        select(literal(1))
        .select_from(
            s.concept__pk.join(
                other_member,
                and_(
                    other_member.c.concept_id == s.concept__pk.c.concept_id,
                    other_member.c.table_id != s.concept__pk.c.table_id,
                ),
            )
        )
        .where(s.concept__pk.c.column_id == col_id)
        .exists()
    )
    rows = store().query_read(
        select(or_(direct, referenced, shared_key).label("participates"))
    )
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
            ~s.catalog_table.c.is_legacy,
            ~column_has_semantic_fk(s.catalog_column.c.id),
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


def merge_semantic_fk(src_column_id: str, tgt_attr_id: str) -> bool:
    """Point a column at the concept keyed on the column owning *tgt_attr_id*.

    A thin wrapper over :func:`auto_ontology.dal.concepts.link_column_to_attribute`,
    which enforces every rule a reference has to satisfy (same database, no
    legacy table, matching key length). Returns whether a reference exists
    afterwards.
    """
    return link_column_to_attribute(src_column_id, tgt_attr_id) is not None


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
        _column_path_select(SEMANTIC_FK_LINKS)
        .where(SEMANTIC_FK_LINKS.c.attribute_id.in_(ids))
        .distinct()
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

#: One table crossing: the ``(source_column_id, target_column_id)`` pairs that
#: must all be equal, in key order. One pair unless the key is composite.
HopPairs = tuple[tuple[str, str], ...]


def _find_table_join_hops(
    anchor_table_id: str,
    dest_table_id: str,
    database_id: str,
) -> list[HopPairs]:
    """Find table-to-table joins, oriented from anchor table to destination.

    The edges are the concepts' (:func:`~auto_ontology.dal.concepts.semantic_table_edges`):
    a reference joins its table to each member table of the concept, in
    both directions because the key is the member table's identity, and the
    member tables join each other on their keys. Two tables that merely
    reference the same concept get no edge between them -- they point at the
    same thing, not at each other; :func:`find_shared_hub_bridge` covers the
    case where both endpoints are such references.
    """
    adjacency: dict[str, list[tuple[str, HopPairs]]] = defaultdict(list)
    for edge in semantic_table_edges(database_id):
        if edge["src_table_id"] == edge["dst_table_id"]:
            continue
        adjacency[edge["src_table_id"]].append((edge["dst_table_id"], edge["pairs"]))
    for edges in adjacency.values():
        edges.sort()

    max_hops = (MAX_PATH_DEPTH + 2) // 4
    parents: dict[str, tuple[str, HopPairs] | None] = {anchor_table_id: None}
    frontier: deque[tuple[str, int]] = deque([(anchor_table_id, 0)])
    while frontier:
        table_id, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for next_table_id, pairs in adjacency.get(table_id, []):
            if next_table_id in parents:
                continue
            parents[next_table_id] = (table_id, pairs)
            if next_table_id == dest_table_id:
                hops: list[HopPairs] = []
                current = dest_table_id
                while parents[current] is not None:
                    previous, hop_pairs = parents[current]
                    hops.append(hop_pairs)
                    current = previous
                hops.reverse()
                return hops
            frontier.append((next_table_id, depth + 1))
    return []


def _hop_dicts(
    hop_pairs: list[HopPairs], *, with_database: bool = True
) -> list[dict[str, Any]]:
    """Render hops in the shape :func:`find_join_path` returns.

    ``source_column`` / ``target_column`` name the first pair. A composite
    crossing adds ``column_pairs``: every ``{source_column, target_column}``
    in key order, all of which the join must equate.
    """
    col_ids = list(
        dict.fromkeys(c for pairs in hop_pairs for pair in pairs for c in pair)
    )
    names = _name_columns(col_ids)
    contexts = fetch_col_table_contexts(col_ids)
    hops: list[dict[str, Any]] = []
    for pairs in hop_pairs:
        source_id, target_id = pairs[0]
        source = contexts.get(source_id, {})
        target = contexts.get(target_id, {})
        hop: dict[str, Any] = {}
        if with_database:
            hop["source_database"] = source.get("database_name", "")
        hop.update(
            source_schema=source.get("schema_name", ""),
            source_table=source.get("table_name", ""),
            source_column=names.get(source_id, ""),
        )
        if with_database:
            hop["target_database"] = target.get("database_name", "")
        hop.update(
            target_schema=target.get("schema_name", ""),
            target_table=target.get("table_name", ""),
            target_column=names.get(target_id, ""),
        )
        if len(pairs) > 1:
            hop["column_pairs"] = [
                {"source_column": names.get(a, ""), "target_column": names.get(b, "")}
                for a, b in pairs
            ]
        hops.append(hop)
    return hops


def _semantic_edges_among(table_ids: list[str]) -> list[dict[str, Any]]:
    """Concept edges in the databases of *table_ids* (one query per database)."""
    rows = store().query_read(
        select(s.catalog_schema.c.database_id)
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.catalog_table.c.id.in_(table_ids))
        .distinct()
    )
    edges: list[dict[str, Any]] = []
    for row in rows:
        edges.extend(semantic_table_edges(row["database_id"]))
    return edges


def _table_database_id(table_id: str) -> str | None:
    """The database a table belongs to, for scoping :func:`_find_table_join_hops`."""
    rows = store().query_read(
        select(s.catalog_schema.c.database_id)
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id
            )
        )
        .where(s.catalog_table.c.id == table_id)
    )
    return rows[0]["database_id"] if rows else None


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


def find_connected_junction_tables(
    table_ids: list[str],
) -> tuple[list[dict], list[list[dict]]]:
    """Return junction tables with a one-hop FK into any supplied table.

    Both declared ``FOREIGN_KEY`` edges and inferred ``SEMANTIC_FK`` edges
    count. Each returned hop contains the exact source and target columns so
    callers can safely expose the added table to SQL generation.
    """
    kept_ids = list(dict.fromkeys(table_id for table_id in table_ids if table_id))
    if not kept_ids:
        return [], []

    junction_column = s.catalog_column.alias("connected_junction_column")
    junction_table = s.catalog_table.alias("connected_junction_table")
    junction_schema = s.catalog_schema.alias("connected_junction_schema")
    target_column = s.catalog_column.alias("connected_target_column")
    target_table = s.catalog_table.alias("connected_target_table")
    target_schema = s.catalog_schema.alias("connected_target_schema")

    declared = (
        select(
            junction_table.c.id.label("junction_table_id"),
            junction_table.c.name.label("junction_table_name"),
            junction_column.c.id.label("source_column_id"),
            target_column.c.id.label("target_column_id"),
        )
        .select_from(
            s.column__foreign_key.join(
                junction_column,
                junction_column.c.id == s.column__foreign_key.c.source_column_id,
            )
            .join(junction_table, junction_table.c.id == junction_column.c.table_id)
            .join(junction_schema, junction_schema.c.id == junction_table.c.schema_id)
            .join(
                target_column,
                target_column.c.id == s.column__foreign_key.c.target_column_id,
            )
            .join(target_table, target_table.c.id == target_column.c.table_id)
            .join(target_schema, target_schema.c.id == target_table.c.schema_id)
        )
        .where(
            junction_table.c.is_junction_table,
            target_table.c.id.in_(kept_ids),
            junction_table.c.id != target_table.c.id,
            junction_schema.c.database_id == target_schema.c.database_id,
        )
    )

    try:
        declared_rows = store().query_read(declared)
        kept = set(kept_ids)
        fk_edges = [
            edge
            for edge in _semantic_edges_among(kept_ids)
            if edge["kind"] == "fk"
            and not edge["reverse"]
            and edge["dst_table_id"] in kept
        ]
        junction_flags = {
            r["id"]: r["name"]
            for r in store().query_read(
                select(s.catalog_table.c.id, s.catalog_table.c.name).where(
                    s.catalog_table.c.id.in_(
                        list({e["src_table_id"] for e in fk_edges}) or [""]
                    ),
                    s.catalog_table.c.is_junction_table,
                )
            )
        }
    except Exception:
        logger.warning(
            "find_connected_junction_tables: query failed for %d table(s)",
            len(kept_ids),
            exc_info=True,
        )
        return [], []

    entries: dict[HopPairs, tuple[str, str]] = {}
    for row in declared_rows:
        pairs = ((row["source_column_id"], row["target_column_id"]),)
        entries.setdefault(
            pairs, (row["junction_table_id"], row["junction_table_name"] or "")
        )
    for edge in fk_edges:
        name = junction_flags.get(edge["src_table_id"])
        if name is not None and edge["src_table_id"] != edge["dst_table_id"]:
            entries.setdefault(edge["pairs"], (edge["src_table_id"], name))

    ordered = sorted(
        entries.items(), key=lambda item: (item[1][1], item[1][0], item[0])
    )
    rendered = _hop_dicts([pairs for pairs, _ in ordered])
    junctions: dict[str, dict] = {}
    paths: list[list[dict]] = []
    for (_, (junction_id, junction_name)), hop in zip(ordered, rendered):
        if not junction_id or not hop["source_table"] or not hop["target_table"]:
            continue
        junctions.setdefault(junction_id, {"id": junction_id, "name": junction_name})
        paths.append([hop])
    return list(junctions.values()), paths


def find_direct_table_joins(table_ids: list[str]) -> list[list[dict]]:
    """Every one-hop join between two of *table_ids*, as single-hop paths.

    Declared ``FOREIGN_KEY`` edges and ``SEMANTIC_FK`` edges marked
    ``use_in_join_paths`` both count. Two tables joined by several column
    pairs get one path per pair, since each can be a different role. A pair
    recorded in both directions, or by both edge kinds, appears once. Edges
    within one table or across databases are left out.
    """
    ids = list(dict.fromkeys(table_id for table_id in table_ids if table_id))
    if len(ids) < 2:
        return []

    source_column = s.catalog_column.alias("direct_source_column")
    source_table = s.catalog_table.alias("direct_source_table")
    source_schema = s.catalog_schema.alias("direct_source_schema")
    target_column = s.catalog_column.alias("direct_target_column")
    target_table = s.catalog_table.alias("direct_target_table")
    target_schema = s.catalog_schema.alias("direct_target_schema")

    def scoped(statement):
        return statement.where(
            source_table.c.id.in_(ids),
            target_table.c.id.in_(ids),
            source_table.c.id != target_table.c.id,
            source_schema.c.database_id == target_schema.c.database_id,
        )

    def ends(join_from):
        return (
            join_from.join(source_table, source_table.c.id == source_column.c.table_id)
            .join(source_schema, source_schema.c.id == source_table.c.schema_id)
            .join(target_table, target_table.c.id == target_column.c.table_id)
            .join(target_schema, target_schema.c.id == target_table.c.schema_id)
        )

    columns = (
        source_column.c.id.label("source_column_id"),
        target_column.c.id.label("target_column_id"),
    )
    declared = scoped(
        select(*columns).select_from(
            ends(
                s.column__foreign_key.join(
                    source_column,
                    source_column.c.id == s.column__foreign_key.c.source_column_id,
                ).join(
                    target_column,
                    target_column.c.id == s.column__foreign_key.c.target_column_id,
                )
            )
        )
    )
    try:
        rows = store().query_read(declared)
        id_set = set(ids)
        semantic_edges = [
            edge
            for edge in _semantic_edges_among(ids)
            if edge["src_table_id"] in id_set
            and edge["dst_table_id"] in id_set
            and edge["src_table_id"] != edge["dst_table_id"]
            and not edge.get("reverse")
        ]
    except Exception:
        logger.warning(
            "find_direct_table_joins: query failed for %d table(s)",
            len(ids),
            exc_info=True,
        )
        return []

    candidates: list[HopPairs] = [
        ((row["source_column_id"], row["target_column_id"]),) for row in rows
    ] + [edge["pairs"] for edge in semantic_edges]
    hop_pairs: list[HopPairs] = []
    seen: set[frozenset[frozenset[str]]] = set()
    for pairs in sorted(candidates):
        key = frozenset(frozenset(pair) for pair in pairs)
        if all(len(pair) == 2 for pair in key) and key not in seen:
            seen.add(key)
            hop_pairs.append(pairs)

    paths: list[list[dict]] = [
        [hop]
        for hop in _hop_dicts(hop_pairs)
        if hop["source_table"] and hop["target_table"]
    ]
    paths.sort(
        key=lambda path: (
            path[0]["source_table"],
            path[0]["target_table"],
            path[0]["source_column"],
            path[0]["target_column"],
        )
    )
    return paths


def find_join_path(anchor_col_id: str, dest_col_id: str) -> list[dict]:
    """The shortest semantic join route between the columns' tables.

    The endpoint columns identify the tables to connect; they do not have to be
    the join columns. Returned hops contain the actual columns for each
    adjacent table crossing.

    ``[]`` when both columns belong to the same table, no semantic route exists,
    either endpoint is missing, or the endpoints belong to different databases.
    """
    if anchor_col_id == dest_col_id:
        return []

    endpoint_contexts = fetch_col_table_contexts([anchor_col_id, dest_col_id])
    anchor_context = endpoint_contexts.get(anchor_col_id)
    dest_context = endpoint_contexts.get(dest_col_id)
    if not anchor_context or not dest_context:
        return []
    if anchor_context["table_id"] == dest_context["table_id"]:
        return []
    if anchor_context["database_id"] != dest_context["database_id"]:
        logger.warning(
            "find_join_path: rejected cross-database path %s -> %s",
            anchor_col_id,
            dest_col_id,
        )
        return []

    try:
        hop_pairs = _find_table_join_hops(
            anchor_context["table_id"],
            dest_context["table_id"],
            anchor_context["database_id"],
        )
    except Exception:
        logger.warning(
            "find_join_path: query failed for %s -> %s",
            anchor_col_id,
            dest_col_id,
            exc_info=True,
        )
        return []

    if not hop_pairs:
        return []
    return _hop_dicts(hop_pairs)


def find_shared_hub_bridge(col_a_id: str, col_b_id: str) -> dict:
    """Find a genuine shared-identity hub connecting two FK columns that
    :func:`find_join_path` cannot reach.

    Two references to the same concept are never joined to each other: two
    ``person_id`` columns for different roles merely point at the same thing.
    But two spokes of one real hub (``robot_details`` and ``actuation_data``
    both referencing ``robot_record``) are joinable *through* the hub, and
    neither has anything pointing out toward the other for
    :func:`find_join_path` to follow.

    Reports a bridge only when both columns are in references, used in join
    paths, to the same concept, at the same key position. The hub is the
    member table both join (the first by name) and the hub column is its key
    column at that position -- the concept's key, so a genuine identity.

    Returns ``{"hub_table", "hub_column"}`` (schema-unqualified — callers
    already have the written table/column names and schema from the SQL
    itself, so this only needs to supply the hub side), or ``{}`` if no such
    shared hub exists.
    """
    if col_a_id == col_b_id:
        return {}

    fk_a = SEMANTIC_FK_LINKS.alias("fk_a")
    fk_b = SEMANTIC_FK_LINKS.alias("fk_b")
    statement = (
        select(
            s.catalog_column.c.name.label("hub_column"),
            s.catalog_table.c.name.label("hub_table"),
        )
        .select_from(
            fk_a.join(
                fk_b,
                and_(
                    fk_b.c.concept_id == fk_a.c.concept_id,
                    fk_b.c.target_column_id == fk_a.c.target_column_id,
                ),
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.id == fk_a.c.target_column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
        )
        .where(
            fk_a.c.column_id == col_a_id,
            fk_b.c.column_id == col_b_id,
            fk_a.c.use_in_join_paths.is_(True),
            fk_b.c.use_in_join_paths.is_(True),
        )
        .order_by(s.catalog_table.c.name)
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
    through a hub -- a member table of a concept the anchor table itself
    references, joined on the concept's key -- never from every candidate
    table's key. This finds sibling tables (e.g. two tables that both
    reference the same parent) that :func:`find_join_path` structurally
    cannot reach, because two references to one concept get no edge between
    them, without risking the fan-out that rule guards against.

    Includes the hub table itself (not just its siblings) — a question can
    need the hub as its own base/pivot table (e.g. to count every row even
    when a sibling has no matching data), and the hub is cheap and safe to
    add: exactly one verified, single-hop forward FK per distinct hub, not
    proportional to how many siblings reference it.

    Sibling count is capped at *max_siblings* per hub, applied here in
    DB-return order (arbitrary — no relevance signal). *max_siblings=None*
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
    this into SQL-generation join instructions; it exists only to bring a
    structurally-connected table into scope that retrieval has no other way
    to recognize.

    Returns ``(results, truncated_count)``. Each result dict has
    ``{source_table, target_table, id, hub_table, is_hub}`` — the first two
    keys match :func:`find_join_path`'s hop shape for prompt rendering: for
    hub entries ``target_table`` is the hub name, for sibling entries it's
    the sibling name. ``truncated_count`` is how many sibling tables were
    dropped by the cap (0 if none were).
    """
    anchor_fk = SEMANTIC_FK_LINKS.alias("anchor_fk")
    sibling_fk = SEMANTIC_FK_LINKS.alias("sibling_fk")
    hub_table = s.catalog_table.alias("hub_table")
    sibling_table = s.catalog_table.alias("sibling_table")

    # Hubs are the member tables of every concept the anchor references; the
    # siblings of a hub are the other tables referencing that concept.
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
            s.catalog_table.join(
                anchor_fk, anchor_fk.c.source_table_id == s.catalog_table.c.id
            )
            .join(hub_table, hub_table.c.id == anchor_fk.c.target_table_id)
            .join(
                sibling_fk,
                and_(
                    sibling_fk.c.concept_id == anchor_fk.c.concept_id,
                    sibling_fk.c.target_table_id == anchor_fk.c.target_table_id,
                    sibling_fk.c.source_table_id != s.catalog_table.c.id,
                    sibling_fk.c.use_in_join_paths.is_(True),
                ),
            )
            .join(sibling_table, sibling_table.c.id == sibling_fk.c.source_table_id)
        )
        .where(
            s.catalog_table.c.id == anchor_table_id,
            anchor_fk.c.use_in_join_paths.is_(True),
        )
        .order_by(hub_table.c.name, sibling_table.c.name)
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

    Delegates to :func:`_find_table_join_hops`, the same table-level
    traversal :func:`find_join_path` uses — table_a and table_b are already
    table IDs, so there's no need for find_join_path's column-to-table
    resolution step.

    Tried in both directions, same as before: the traversal isn't fully
    symmetric even now. A real outgoing FK (table_a -> owner) is always a
    legitimate step; entering a table from its *owner* side only works when
    that table's own declared PK is the shared attribute (see
    ``_find_table_join_hops``'s docstring) — so a table reachable as the
    anchor isn't guaranteed reachable as the destination, or vice versa.

    IMPORTANT: an unrestricted traversal can find a real but coincidental
    path through a table that was never a candidate at all — e.g. two tables
    that both happen to FK into an unrelated "preferences" table for reasons
    having nothing to do with the actual question. That is introducing a
    table retrieval never surfaced, not restoring a needed bridge. So when
    *allowed_table_ids* is given, a found path is
    only returned if every intermediate table on it is a member of that set
    (typically: the tables that were candidates *before* the relevance
    filter ran) — this keeps the function to "restore a bridge the filter
    had and dropped", never "discover a new one".

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

    Both empty if the tables aren't in the same database, no path exists in
    either direction, or the only path found steps outside
    *allowed_table_ids*.
    """
    for src, dst in ((table_a_id, table_b_id), (table_b_id, table_a_id)):
        database_id = _table_database_id(src)
        if not database_id:
            continue
        try:
            hop_pairs = _find_table_join_hops(src, dst, database_id)
        except Exception:
            logger.warning(
                "find_table_bridge: query failed for %s -> %s",
                src,
                dst,
                exc_info=True,
            )
            continue
        if not hop_pairs:
            continue

        target_ids = [pairs[0][1] for pairs in hop_pairs[:-1]]
        col_ctx = fetch_col_table_contexts(target_ids)

        # Every hop's target table is the next table on the path; the last
        # hop's target is dst itself, so everything before it is a bridge.
        bridge_ids = list(
            dict.fromkeys(
                col_ctx.get(target_id, {}).get("table_id") for target_id in target_ids
            )
        )
        bridge_ids = [tid for tid in bridge_ids if tid]
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

        hops = _hop_dicts(hop_pairs, with_database=False)
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

    Checks every pair of kept tables. Stops early once *max_bridge_tables*
    distinct bridge tables have been found, as a hard safety cap for a large
    kept set.

    *allowed_table_ids*, if given, is passed through to
    :func:`find_table_bridge` to restrict results to tables that were
    already candidates — see its docstring for why this matters.

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
