# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term reads and writes.

ColumnAttribute and SEMANTIC_FK operations live in :mod:`gsf.dal.attributes`.

Three rules are shared by almost everything here, and each one exists because a
previous version of this code had two copies of it that drifted:

* **What counts as a Term** — a semantic Term with at least one table
  representing it (:func:`_semantic_terms`). The list, its total, and the
  single-id check all read it from one place, so they cannot disagree about
  which terms are real.
* **All-or-nothing table visibility** (:func:`_in_scope`). A term represented by
  *any* out-of-zone table is hidden entirely. Not a relevance filter — an
  authorization boundary.
* **What "related" means** (:func:`fetch_term_table_pairs`). Three paths from a
  table to a term, and the related list, the per-term badge, and the Exploration
  graph's edges all count the same rows. Spelling them out separately is what
  once let a card's badge, the "Relationships" column, and the length of the
  list behind them each report a different number.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import (
    String,
    and_,
    bindparam,
    case,
    func,
    literal,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.constants import Edges, Labels
from gsf.dal import schema as s
from gsf.dal.attributes import fetch_column_attribute_columns_map
from gsf.dal.datasources import fetch_col_table_contexts
from gsf.dal.session import store, write_transaction
from gsf.dal.users import resolve_accessible_catalog_ids
from gsf.dal.zones import fetch_table_zones_map, zone_covers_table
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
    SEMANTIC_SOURCE,
)
from gsf.utils.sample_values import stringify_sample_values

logger = logging.getLogger(__name__)


def _with_parsed_sample_values(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render ``sample_values`` on each row as the string list callers expect.

    These rows serve both the terms API and semantic embedding text, so the
    typed values stored as JSON are rendered once here rather than at each use.
    """
    for row in rows:
        row["sample_values"] = stringify_sample_values(row.get("sample_values"))
    return rows


# ---------------------------------------------------------------------------
# The shared filters
# ---------------------------------------------------------------------------


def _in_scope(table_ids: list[str] | None):
    """All-or-nothing table visibility, or ``None`` when unscoped.

    A term represented by any table outside *table_ids* is hidden entirely,
    matching the rule ``list_custom_analyses`` applies. Phrased in the negative
    — "no representing table is out of scope" — because the positive form would
    show a term that also represents something the caller cannot see.
    """
    if table_ids is None:
        return None
    return ~(
        select(literal(1))
        .select_from(s.table__term)
        .where(
            s.table__term.c.term_id == s.term.c.id,
            s.table__term.c.table_id.notin_(table_ids),
        )
        .correlate(s.term)
        .exists()
    )


def _scope_table_ids(
    zone_ids: list[str] | None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[str] | None:
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    return None if resolved is None else list(resolved["table_ids"])


def _represented():
    """The term has at least one table representing it.

    Half of what it means for a Term to exist as far as the UI is concerned;
    the other half is ``source = 'semantic'``.
    """
    return (
        select(literal(1))
        .where(s.table__term.c.term_id == s.term.c.id)
        .correlate(s.term)
        .exists()
    )


def _semantic_terms(
    zone_ids: list[str] | None = None,
    search: str | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    *,
    term_id: str | None = None,
):
    """The predicate list every read of the visible Term set applies.

    ``fetch_all_terms`` and ``count_terms`` must select exactly the same terms —
    otherwise the total does not describe the list being paged — and
    ``term_is_in_scope`` must answer for one id whatever those two would answer
    for the whole set. All three take their filter from here.
    """
    conditions = [s.term.c.source == SEMANTIC_SOURCE, _represented()]
    scope = _in_scope(_scope_table_ids(zone_ids, data_ids_by_zone))
    if scope is not None:
        conditions.append(scope)
    if term_id is not None:
        conditions.append(s.term.c.id == term_id)
    if search:
        # Escaped: the caller means a literal substring. Unescaped, `_`
        # matches any character (so `customer_id` also finds `customerXid`)
        # and a lone `%` returns the entire glossary.
        needle = (
            search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        conditions.append(s.term.c.name.ilike(f"%{needle}%", escape="\\"))
    return conditions


def _certification(table_ids: list[str] | None):
    """A Term's aggregate certification: ``certified``, ``partial`` or ``pending``.

    The single definition of the rollup — the list cards, the detail page and
    the certification writes all project this, so the three cannot disagree.

    The flags are the Term's own two booleans plus one per column and sql
    attribute. Because the Term's two are always present the set is never empty,
    and a Term with no attributes is decided by them alone.

    When *table_ids* is given, each attribute count applies the same visibility
    rule as the endpoint that lists those attributes — plain ``table_id``
    membership for ColumnAttribute, all-or-nothing over the tables a
    SqlAttribute's SQL touches — so the badge never reflects an attribute the
    caller is not allowed to see.
    """

    def counts(link_table, attribute_table, extra=None):
        conditions = [
            link_table.c.term_id == s.term.c.id,
            link_table.c.attribute_id == attribute_table.c.id,
        ]
        if extra is not None:
            conditions.append(extra)
        total = (
            select(func.count())
            .select_from(link_table.join(attribute_table, conditions[1]))
            .where(*conditions)
            .correlate(s.term)
            .scalar_subquery()
        )
        certified = (
            select(func.count())
            .select_from(link_table.join(attribute_table, conditions[1]))
            .where(*conditions, attribute_table.c.certified.is_(True))
            .correlate(s.term)
            .scalar_subquery()
        )
        return total, certified

    column_visible = None
    sql_visible = None
    if table_ids is not None:
        column_visible = s.column_attribute.c.table_id.in_(table_ids)
        sql_visible = ~(
            select(literal(1))
            .select_from(
                s.sql_attribute__sql.join(
                    s.sql_query__table,
                    s.sql_query__table.c.sql_query_id
                    == s.sql_attribute__sql.c.sql_query_id,
                )
            )
            .where(
                s.sql_attribute__sql.c.attribute_id == s.sql_attribute.c.id,
                s.sql_query__table.c.table_id.notin_(table_ids),
            )
            .correlate(s.sql_attribute)
            .exists()
        )

    column_total, column_certified = counts(
        s.column_attribute__term, s.column_attribute, column_visible
    )
    sql_total, sql_certified = counts(
        s.sql_attribute__term, s.sql_attribute, sql_visible
    )

    own_certified = case((s.term.c.name_certified, 1), else_=0) + case(
        (s.term.c.description_certified, 1), else_=0
    )
    certified = own_certified + column_certified + sql_certified
    total = literal(2) + column_total + sql_total

    return case(
        (certified == total, "certified"),
        (certified == 0, "pending"),
        else_="partial",
    )


# ---------------------------------------------------------------------------
# Single-term reads
# ---------------------------------------------------------------------------


def get_term_certification(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> str | None:
    """One Term's aggregate certification, or ``None`` when it is not visible.

    Certification writes call this so the list card can be updated without the
    frontend duplicating the rollup rule or refetching the whole list.
    """
    table_ids = _scope_table_ids(zone_ids)
    statement = select(_certification(table_ids).label("certification")).where(
        s.term.c.id == term_id
    )
    scope = _in_scope(table_ids)
    if scope is not None:
        statement = statement.where(scope)

    rows = store().query_read(statement.limit(1))
    return rows[0]["certification"] if rows else None


def term_is_in_scope(
    term_id: str,
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> bool:
    """Would ``fetch_all_terms`` return this term?

    Scope is the glossary set, not whichever nodes an Exploration payload
    happens to carry: a term dropped by graph truncation is still in scope, and
    has to be, since the relationship counts are computed untruncated.
    """
    rows = store().query_read(
        select(s.term.c.id)
        .where(
            *_semantic_terms(
                zone_ids, data_ids_by_zone=data_ids_by_zone, term_id=term_id
            )
        )
        .limit(1)
    )
    return bool(rows)


def semantic_layer_calculated() -> bool:
    """Has the semantic layer been built at all?"""
    return bool(
        store().query_read(
            select(s.term.c.id).where(s.term.c.source == SEMANTIC_SOURCE).limit(1)
        )
    )


def get_term_record_for_table(table_id: str) -> dict[str, str] | None:
    """``{id, name, description}`` for the Term a table represents."""
    rows = store().query_read(
        select(
            s.term.c.id,
            s.term.c.name,
            func.coalesce(s.term.c.description, "").label("description"),
        )
        .select_from(s.table__term.join(s.term, s.term.c.id == s.table__term.c.term_id))
        .where(
            s.table__term.c.table_id == table_id,
            s.term.c.source == SEMANTIC_SOURCE,
        )
        .order_by(s.term.c.id)
        .limit(1)
    )
    return dict(rows[0]) if rows else None


def get_slim_term_by_id(term_id: str) -> dict[str, str] | None:
    """``{id, name}``, for callers that only need to name a term."""
    rows = store().query_read(
        select(s.term.c.id, s.term.c.name).where(s.term.c.id == term_id).limit(1)
    )
    return dict(rows[0]) if rows else None


def update_term(
    term_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    name_certified: bool | None = None,
    description_certified: bool | None = None,
) -> dict[str, Any] | None:
    """Update a Term, returning old and new values, or ``None`` when missing.

    ``old_name`` is in the return because the caller needs it: renaming a Term
    invalidates its embeddings, which are keyed on the old text.

    A rename also rewrites ``term_name`` on every ColumnAttribute of the Term.
    That is a denormalised copy, and keeping it in step here is not optional —
    ``fetch_column_attributes_by_term_id`` matches attributes to their term
    *through it*, so a stale copy silently empties a term's attribute list.
    """
    # One transaction: the rename and the denormalised copy on every
    # ColumnAttribute are the same fact. If the second statement fails on its
    # own, `fetch_column_attributes_by_term_id` matches through the stale copy
    # and the Term's attribute list reads as empty from then on.
    with write_transaction():
        # Read the old name first. RETURNING gives the *new* one, and the caller
        # compares the two to decide whether a rename happened -- with it always
        # None, `name_changed` was true for every edit, so a description-only
        # PATCH cleared description_suggestion on every SqlAttribute of the term
        # and re-embedded all of them.
        previous = store().query_read(
            select(s.term.c.name).where(s.term.c.id == term_id)
        )
        old_name = previous[0]["name"] if previous else None

        rows = store().query_write(
            update(s.term)
            .where(s.term.c.id == term_id)
            .values(
                name=func.coalesce(literal(name, String), s.term.c.name),
                description=func.coalesce(
                    literal(description, String), s.term.c.description
                ),
                name_certified=func.coalesce(name_certified, s.term.c.name_certified),
                description_certified=func.coalesce(
                    description_certified, s.term.c.description_certified
                ),
            )
            .returning(
                s.term.c.id,
                s.term.c.name,
                s.term.c.description,
                s.term.c.name_certified,
                s.term.c.description_certified,
            )
        )
        if not rows:
            return None
        result = dict(rows[0])
        result["old_name"] = old_name

        # Read before the UPDATE would have been simpler, but RETURNING gives
        # the new name and the old one is only knowable beforehand -- so it is
        # fetched first, above, by way of this second statement being the
        # *rename*.
        store().query_write(
            update(s.column_attribute)
            .where(
                s.column_attribute.c.id.in_(
                    select(s.column_attribute__term.c.attribute_id).where(
                        s.column_attribute__term.c.term_id == term_id
                    )
                )
            )
            .values(term_name=result["name"])
        )
    return result


def merge_term(
    name: str,
    description: str,
    table_id: str,
    synonyms: list[str] | None = None,
) -> str | None:
    """Upsert a Term and link the table to it. ``None`` if the table is missing.

    Unlike ``merge_column_attribute``, ``description`` is **assigned** rather
    than coalesced, so a re-run of the semantic build overwrites a hand-edited
    description. Deliberate, and worth knowing before changing it: a rebuild is
    meant to be authoritative.
    """
    if not store().query_read(
        select(s.catalog_table.c.id).where(s.catalog_table.c.id == table_id)
    ):
        return None

    statement = insert(s.term).values(
        name=name,
        source=SEMANTIC_SOURCE,
        description=description,
        synonyms=synonyms or [],
    )
    # One transaction, as `update_term` above. A term with no `table__term` link
    # is excluded by `_represented()`, so it is invisible to every read while
    # still occupying `uq_term_name_source` -- it blocks its own name forever.
    with write_transaction():
        rows = store().query_write(
            statement.on_conflict_do_update(
                constraint="uq_term_name_source",
                set_={
                    "description": statement.excluded.description,
                    "synonyms": statement.excluded.synonyms,
                },
            ).returning(s.term.c.id)
        )
        term_id = rows[0]["id"]
        store().query_write(
            insert(s.table__term)
            .values(table_id=table_id, term_id=term_id)
            .on_conflict_do_nothing()
        )
    return term_id


def fetch_term_synonyms(attr_ids: list[str]) -> dict[str, list[str]]:
    """``{term_name: [synonym, ...]}`` for the terms behind these attributes.

    Keyed by *name*, not id, because the caller matches synonyms against text.
    Terms with no synonyms are omitted rather than mapped to an empty list.
    """
    if not attr_ids:
        return {}
    try:
        rows = store().query_read(
            select(s.term.c.name, s.term.c.synonyms)
            .select_from(
                s.column_attribute__term.join(
                    s.term, s.term.c.id == s.column_attribute__term.c.term_id
                )
            )
            .where(
                s.column_attribute__term.c.attribute_id.in_(list(attr_ids)),
                s.term.c.synonyms.isnot(None),
                func.array_length(s.term.c.synonyms, 1) > 0,
            )
            .distinct()
        )
    except Exception:
        logger.warning("fetch_term_synonyms: query failed", exc_info=True)
        return {}

    result: dict[str, list[str]] = {}
    for row in rows:
        synonyms = [synonym for synonym in (row["synonyms"] or []) if synonym]
        if row["name"] and synonyms:
            result[row["name"]] = synonyms
    return result


# ---------------------------------------------------------------------------
# The Term list
# ---------------------------------------------------------------------------


def fetch_all_terms(
    zone_ids: list[str] | None = None,
    search: str | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """The glossary, paged.

    *zone_ids* is a hard authorization boundary. A Term can be represented by
    more than one table, so it is returned only when **every** representing
    table is reachable — a term also representing an out-of-zone table is
    excluded outright. ``None`` returns everything, for admin callers.

    Each row carries its ``zones`` and its aggregate ``certification``, both
    scoped to the same boundary, so the list renders chips and badges from one
    response rather than a request per card.

    Ordered by name case-insensitively then by id — the id breaks ties between
    same-named terms, without which a page boundary could repeat one and skip
    another. Pair with ``count_terms`` for the total.
    """
    table_ids = _scope_table_ids(zone_ids, data_ids_by_zone)
    schema_names = (
        select(func.array_agg(func.distinct(s.catalog_schema.c.name)))
        .select_from(
            s.table__term.join(
                s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id
            ).join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.table__term.c.term_id == s.term.c.id)
        .correlate(s.term)
        .scalar_subquery()
    )

    statement = (
        select(
            s.term.c.name,
            s.term.c.description,
            s.term.c.synonyms,
            s.term.c.id,
            schema_names.label("schema_names"),
            s.term.c.name_certified,
            s.term.c.description_certified,
            _certification(table_ids).label("certification"),
        )
        .where(*_semantic_terms(zone_ids, search, data_ids_by_zone))
        .order_by(func.lower(s.term.c.name), s.term.c.id)
        .offset(skip)
    )
    if limit is not None:
        statement = statement.limit(limit)

    terms = [dict(r) for r in store().query_read(statement)]
    # Scoped to the rows just read, so paging does not resolve zones for the
    # rest of the glossary on every page.
    zones_by_term = fetch_term_zones_map(
        zone_ids, term_ids=[row["id"] for row in terms]
    )
    for row in terms:
        row["schema_names"] = row["schema_names"] or []
        row["zones"] = zones_by_term.get(row["id"], [])
    return terms


def count_terms(
    zone_ids: list[str] | None = None,
    search: str | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> int:
    """The unpaged size of :func:`fetch_all_terms`, from the same filter."""
    rows = store().query_read(
        select(func.count(func.distinct(s.term.c.id)).label("total")).where(
            *_semantic_terms(zone_ids, search, data_ids_by_zone)
        )
    )
    return int(rows[0]["total"]) if rows else 0


def fetch_all_terms_and_attributes(
    zone_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every Term and every ColumnAttribute, for a one-shot bulk re-embed.

    Only ``embed_all_semantic_nodes`` wants this. The second scan is global and
    expensive; callers that need Term rows alone — the ``/terms`` list, the
    Exploration graph — should call :func:`fetch_all_terms` and skip it.
    """
    terms = fetch_all_terms(zone_ids=zone_ids)

    statement = (
        select(
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.column_attribute.c.id,
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(
            s.column_attribute.join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.column_attribute.c.source == SEMANTIC_SOURCE)
    )
    table_ids = _scope_table_ids(zone_ids)
    if table_ids is not None:
        statement = statement.where(s.catalog_table.c.id.in_(table_ids))

    return terms, [dict(r) for r in store().query_read(statement)]


def get_full_term_by_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """One Term with its tables, table count, zones and certification.

    Applies the same all-or-nothing rule as :func:`fetch_all_terms`, so a viewer
    cannot bypass list-level scoping by requesting a term directly by id —
    ``None`` here becomes a 404.

    ``zones`` is resolved through **both** paths a term carries content by:
    ColumnAttribute → Column → Table, and SqlAttribute → Sql → Table. A term's
    SqlAttribute can reference tables no ColumnAttribute touches (a cross-table
    formula, say), and the term participates in a zone when *either* path
    reaches it.
    """
    table_ids = _scope_table_ids(zone_ids)
    statement = select(
        s.term.c.name,
        s.term.c.description,
        s.term.c.synonyms,
        s.term.c.id,
        s.term.c.name_certified,
        s.term.c.description_certified,
        _certification(table_ids).label("certification"),
    ).where(s.term.c.id == term_id)
    scope = _in_scope(table_ids)
    if scope is not None:
        statement = statement.where(scope)

    rows = store().query_read(statement.limit(1))
    if not rows:
        return None
    result = dict(rows[0])

    tables = store().query_read(
        select(
            s.catalog_table.c.id,
            s.catalog_table.c.name,
            s.catalog_schema.c.id.label("schema_id"),
            s.catalog_database.c.id.label("db_id"),
        )
        .select_from(
            s.table__term.join(
                s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.table__term.c.term_id == term_id)
        .order_by(s.catalog_table.c.id)
    )
    result["tables"] = [dict(r) for r in tables]
    result["table_count"] = len(result["tables"])
    result["zones"] = fetch_term_zones_map(zone_ids, term_ids=[term_id]).get(
        term_id, []
    )
    return result


def _term_zone_rows(path_join, term_filter):
    """Zone rows reachable from a Term along one attribute path."""
    return (
        select(
            s.term.c.id.label("term_id"),
            s.zone.c.id,
            s.zone.c.name,
            s.zone.c.color,
            s.zone.c.enabled,
        )
        .select_from(path_join)
        .where(*term_filter)
        .distinct()
    )


def fetch_term_zones_map(
    zone_ids: list[str] | None = None,
    term_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """``{term_id: [zone, ...]}``, resolving many terms in one pass.

    The union of the two paths :func:`get_full_term_by_id` describes, so the
    Terms list and the Exploration graph render chips without a request per
    node — and, more to the point, agree with the detail page.

    *term_ids* narrows the scan; it never widens access, which *zone_ids* alone
    decides. An empty list means no terms, not all of them.
    """
    if term_ids is not None and not term_ids:
        return {}

    column_path = (
        s.term.join(
            s.column_attribute__term,
            s.column_attribute__term.c.term_id == s.term.c.id,
        )
        .join(
            s.column__has_attribute,
            s.column__has_attribute.c.attribute_id
            == s.column_attribute__term.c.attribute_id,
        )
        .join(
            s.catalog_column,
            s.catalog_column.c.id == s.column__has_attribute.c.column_id,
        )
        .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
        .join(s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id)
        .join(s.zone_target, zone_covers_table())
        .join(s.zone, s.zone.c.id == s.zone_target.c.zone_id)
    )
    sql_path = (
        s.term.join(
            s.sql_attribute__term, s.sql_attribute__term.c.term_id == s.term.c.id
        )
        .join(
            s.sql_attribute__sql,
            s.sql_attribute__sql.c.attribute_id == s.sql_attribute__term.c.attribute_id,
        )
        .join(
            s.sql_query__table,
            s.sql_query__table.c.sql_query_id == s.sql_attribute__sql.c.sql_query_id,
        )
        .join(s.catalog_table, s.catalog_table.c.id == s.sql_query__table.c.table_id)
        .join(s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id)
        .join(s.zone_target, zone_covers_table())
        .join(s.zone, s.zone.c.id == s.zone_target.c.zone_id)
    )

    term_filter: list[Any] = []
    if term_ids is not None:
        term_filter.append(s.term.c.id.in_(list(term_ids)))
    if zone_ids is not None:
        # A viewer never sees a disabled zone's chip, even when its id is in
        # zone_ids -- access may have been granted before it was retired.
        term_filter.append(s.zone.c.id.in_(list(zone_ids)))
        term_filter.append(s.zone.c.enabled.is_(True))

    rows = store().query_read(
        _term_zone_rows(column_path, term_filter).union(
            _term_zone_rows(sql_path, term_filter)
        )
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["term_id"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "enabled": row["enabled"],
            }
        )
    for term_id, zones in result.items():
        result[term_id] = sorted(zones, key=lambda zone: zone["name"])
    return result


# ---------------------------------------------------------------------------
# Suggester and embedding inputs
# ---------------------------------------------------------------------------


def fetch_table_schema_map(database_name: str) -> dict[str, str]:
    """``{table_name_lower: schema_name}`` for one database.

    The SqlAttribute suggester uses it to qualify bare table names in generated
    SELECTs. Lower-cased keys, and last write wins on a collision — two schemas
    with the same table name resolve arbitrarily, as before.
    """
    rows = store().query_read(
        select(
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
        .order_by(s.catalog_table.c.id)
    )
    return {
        row["table_name"].lower(): row["schema_name"]
        for row in rows
        if row["table_name"] and row["schema_name"]
    }


def fetch_terms_with_sqls() -> list[dict[str, Any]]:
    """Every semantic Term with the **ingestion** SQL of its tables.

    Statements owned by a SqlAttribute or CustomAnalysis are excluded, so
    generated semantic SQL cannot feed the next round of suggestions — a
    feedback loop where the suggester learns from itself.

    Terms with no ingestion query are omitted.

    ``props`` is the statement row itself. There are no per-month counters —
    they were measured as unread by anything, and the schema does not carry
    them.
    """
    owned = (
        select(literal(1))
        .where(s.sql_attribute__sql.c.sql_query_id == s.sql_query.c.id)
        .correlate(s.sql_query)
        .exists()
    ) | (
        select(literal(1))
        .where(s.custom_analysis__sql.c.sql_query_id == s.sql_query.c.id)
        .correlate(s.sql_query)
        .exists()
    )

    rows = store().query_read(
        select(
            s.term.c.id.label("term_id"),
            s.term.c.name.label("term_name"),
            s.term.c.description.label("term_description"),
            s.sql_query,
        )
        .select_from(
            s.term.join(s.table__term, s.table__term.c.term_id == s.term.c.id)
            .join(
                s.sql_query__table,
                s.sql_query__table.c.table_id == s.table__term.c.table_id,
            )
            .join(s.sql_query, s.sql_query.c.id == s.sql_query__table.c.sql_query_id)
        )
        .where(s.term.c.source == SEMANTIC_SOURCE, ~owned)
        .distinct()
        .order_by(s.term.c.id, s.sql_query.c.id)
    )

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = grouped.setdefault(
            row["term_id"],
            {
                "term_id": row["term_id"],
                "term_name": row["term_name"],
                "term_description": row["term_description"],
                "sqls": [],
            },
        )
        props = {
            key: value
            for key, value in row.items()
            if key not in {"term_id", "term_name", "term_description"}
        }
        entry["sqls"].append(
            {
                "sql_text": props.get("sql_full_query"),
                "sql_id": props.get("id"),
                "props": props,
            }
        )
    return list(grouped.values())


def _embedding_attrs_for_table(table_id: str):
    return (
        select(
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.column_attribute.c.id,
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(
            s.column_attribute.join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(
            s.catalog_table.c.id == table_id,
            s.column_attribute.c.source == SEMANTIC_SOURCE,
        )
    )


def fetch_terms_and_attributes_for_table(
    table_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(terms, attrs)`` written for one table — the inline embedding input."""
    terms = store().query_read(
        select(
            s.term.c.name,
            s.term.c.description,
            s.term.c.synonyms,
            s.term.c.id,
            func.array_agg(func.distinct(s.catalog_schema.c.name)).label(
                "schema_names"
            ),
        )
        .select_from(
            s.table__term.join(s.term, s.term.c.id == s.table__term.c.term_id)
            .join(s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(
            s.table__term.c.table_id == table_id,
            s.term.c.source == SEMANTIC_SOURCE,
        )
        .group_by(s.term.c.id)
    )
    attrs = store().query_read(_embedding_attrs_for_table(table_id))
    return [dict(r) for r in terms], [dict(r) for r in attrs]


def fetch_term_and_column_attributes_for_embedding(
    term_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """One Term and its ColumnAttributes, for a semantic VDB re-embed."""
    terms = store().query_read(
        select(
            s.term.c.name,
            s.term.c.description,
            s.term.c.synonyms,
            s.term.c.id,
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.term.outerjoin(s.table__term, s.table__term.c.term_id == s.term.c.id)
            .outerjoin(
                s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id
            )
            .outerjoin(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .outerjoin(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.term.c.id == term_id)
        # Ordered so a term spanning two databases reports the same one on
        # every call.
        .order_by(s.catalog_database.c.id)
        .limit(1)
    )
    if not terms:
        return None, []

    attrs = store().query_read(
        select(
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.column_attribute.c.id,
        )
        .select_from(
            s.column_attribute__term.join(
                s.column_attribute,
                s.column_attribute.c.id == s.column_attribute__term.c.attribute_id,
            )
            .outerjoin(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
            .outerjoin(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            # Outer, like the join above it: an attribute need not reach a
            # column, and an inner join here would drop those attributes
            # entirely rather than leaving their table fields null.
            .outerjoin(
                s.catalog_table,
                s.catalog_table.c.id == s.catalog_column.c.table_id,
            )
        )
        .where(s.column_attribute__term.c.term_id == term_id)
        .order_by(s.column_attribute.c.id)
    )
    return dict(terms[0]), [dict(r) for r in attrs]


def fetch_column_attribute_embedding_contexts_by_column_id(
    column_id: str,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The Term contexts a column's sample-value update invalidates.

    Sample values are part of a ColumnAttribute's embedding text, so profiling a
    column stales every attribute on it and every term behind those.
    """
    rows = store().query_read(
        select(
            s.term.c.name.label("term_name"),
            s.term.c.description.label("term_description"),
            s.term.c.synonyms.label("term_synonyms"),
            s.term.c.id.label("term_id"),
            s.catalog_database.c.name.label("database_name"),
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.column_attribute.c.id,
        )
        .select_from(
            s.catalog_column.join(
                s.column__has_attribute,
                s.column__has_attribute.c.column_id == s.catalog_column.c.id,
            )
            .join(
                s.column_attribute,
                s.column_attribute.c.id == s.column__has_attribute.c.attribute_id,
            )
            .join(
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id == s.column_attribute.c.id,
            )
            .join(s.term, s.term.c.id == s.column_attribute__term.c.term_id)
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
        .where(s.catalog_column.c.id == column_id)
        .order_by(s.term.c.id, s.column_attribute.c.id)
    )

    contexts: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for row in rows:
        term_id = row["term_id"]
        if not term_id:
            continue
        if term_id not in contexts:
            contexts[term_id] = (
                {
                    "name": row["term_name"] or "",
                    "description": row["term_description"] or "",
                    "synonyms": row["term_synonyms"] or [],
                    "id": term_id,
                    "database_name": row["database_name"] or "",
                },
                [],
            )
        contexts[term_id][1].append(
            {
                "name": row["name"],
                "description": row["description"],
                "term_name": row["term_name"],
                "source_column": row["source_column"],
                "sample_values": row["sample_values"],
                "id": row["id"],
            }
        )
    return list(contexts.values())


# ---------------------------------------------------------------------------
# ColumnAttributes of a Term
# ---------------------------------------------------------------------------


def _column_attribute_scope(term_id: str, zone_ids: list[str] | None):
    """A ColumnAttribute is owned by exactly one table, via ``attr.table_id``.

    So a plain membership filter is enough here — no all-or-nothing check, which
    is a genuine difference from the Term rule and not an oversight.
    """
    conditions = [
        s.column_attribute.c.source == SEMANTIC_SOURCE,
        s.column_attribute.c.term_name
        == select(s.term.c.name).where(s.term.c.id == term_id).scalar_subquery(),
    ]
    table_ids = _scope_table_ids(zone_ids)
    if table_ids is not None:
        conditions.append(s.column_attribute.c.table_id.in_(table_ids))
    return conditions


def fetch_column_attribute_counts(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    term_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """``[{term_id, count}]`` per Term. Terms with none are omitted."""
    statement = (
        select(
            s.column_attribute__term.c.term_id,
            func.count(func.distinct(s.column_attribute.c.id)).label("count"),
        )
        .select_from(
            s.column_attribute.join(
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id == s.column_attribute.c.id,
            )
        )
        .where(s.column_attribute.c.source == SEMANTIC_SOURCE)
        .group_by(s.column_attribute__term.c.term_id)
    )
    table_ids = _scope_table_ids(zone_ids, data_ids_by_zone)
    if table_ids is not None:
        statement = statement.where(s.column_attribute.c.table_id.in_(table_ids))
    if term_ids is not None:
        statement = statement.where(
            s.column_attribute__term.c.term_id.in_(list(term_ids))
        )

    return [
        {"term_id": r["term_id"], "count": int(r["count"])}
        for r in store().query_read(statement)
    ]


def fetch_column_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """One Term's ColumnAttributes, with their columns and zones.

    One row per attribute even when several columns declare it: the sample
    values come from the lowest-id one. Without that collapse the rows outnumber
    the total the pager was handed, and the last attributes become unreachable.

    Ordered by name then id, so *skip* and *limit* page it stably; pair with
    ``count_column_attributes_by_term_id``.
    """
    sample_values = (
        select(s.catalog_column.c.sample_values)
        .select_from(
            s.column__has_attribute.join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
        )
        .where(s.column__has_attribute.c.attribute_id == s.column_attribute.c.id)
        .order_by(s.catalog_column.c.id)
        .limit(1)
        .correlate(s.column_attribute)
        .scalar_subquery()
    )
    statement = (
        select(
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_attribute.c.term_name,
            s.column_attribute.c.source_column,
            s.column_attribute.c.datatype,
            s.column_attribute.c.table_id,
            sample_values.label("sample_values"),
            s.column_attribute.c.certified,
        )
        .where(*_column_attribute_scope(term_id, zone_ids))
        .order_by(s.column_attribute.c.name, s.column_attribute.c.id)
        .offset(skip)
    )
    if limit is not None:
        statement = statement.limit(limit)

    rows = [dict(r) for r in store().query_read(statement)]
    table_ids = [row["table_id"] for row in rows if row["table_id"]]
    zones_by_table = (
        fetch_table_zones_map(zone_ids=zone_ids, table_ids=list(set(table_ids)))
        if table_ids
        else {}
    )
    columns_by_attr = fetch_column_attribute_columns_map([row["id"] for row in rows])
    empty = {"primary_column": None, "referenced_columns": []}
    return [
        {
            **row,
            "zones": zones_by_table.get(row["table_id"], []),
            **columns_by_attr.get(row["id"], empty),
        }
        for row in _with_parsed_sample_values(rows)
    ]


def count_column_attributes_by_term_id(
    term_id: str,
    zone_ids: list[str] | None = None,
) -> int:
    """The unpaged size of :func:`fetch_column_attributes_by_term_id`."""
    rows = store().query_read(
        select(func.count(func.distinct(s.column_attribute.c.id)).label("total")).where(
            *_column_attribute_scope(term_id, zone_ids)
        )
    )
    return int(rows[0]["total"]) if rows else 0


def find_column_attribute_by_column_id(column_id: str) -> str | None:
    """The semantic ColumnAttribute a column carries.

    Re-exported from :mod:`gsf.dal.attributes`: callers import it from both
    modules, so both keep it.
    """
    from gsf.dal.attributes import find_column_attribute_by_column_id as delegate

    return delegate(column_id)


# ---------------------------------------------------------------------------
# Related terms
# ---------------------------------------------------------------------------


def fetch_term_table_pairs(
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    *,
    term_ids: list[str] | None = None,
    table_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """``{term_id, table_id}`` — the one definition of "related".

    A term is linked to a table by REPRESENTS, or through a ColumnAttribute
    reached by HAS_ATTRIBUTE **or** SEMANTIC_FK. That third path is what links,
    say, ``Request`` to ``User`` when ``requests.creator_id`` references
    ``users.id``.

    Shared by the related list, the per-term count and the Exploration graph, so
    a term's badge, the "Relationships" column, and the length of the list
    behind them cannot report three different numbers.

    Both branches require ``source = 'semantic'`` on the Term, matching
    ``fetch_all_terms``: a non-semantic Term never appears in the semantic
    graph, so counting it as a neighbour would put a number on a card no list
    can reach.

    ``path`` names which of those three links actually connected this
    particular pair (``REPRESENTS``, ``HAS_ATTRIBUTE`` or ``SEMANTIC_FK``), so a
    caller that cares *why* a term and a table are linked -- not just *that*
    they are -- does not have to re-run the match itself;
    :func:`gsf.dal.exploration.fetch_semantic_exploration_graph` uses it to
    label term↔term edges with the real relationship type(s) behind them.

    *term_ids* and *table_ids* narrow the scan; neither widens access, both
    being intersected with what *zone_ids* already allows.
    """
    scope_ids = _scope_table_ids(zone_ids, data_ids_by_zone)

    def restrict(statement, table_column):
        conditions: list[Any] = [s.term.c.source == SEMANTIC_SOURCE]
        if scope_ids is not None:
            conditions.append(table_column.in_(scope_ids))
            conditions.append(_in_scope(scope_ids))
        if term_ids is not None:
            conditions.append(s.term.c.id.in_(list(term_ids)))
        if table_ids is not None:
            conditions.append(table_column.in_(list(table_ids)))
        return statement.where(*conditions)

    represents = restrict(
        select(
            s.term.c.id.label("term_id"),
            s.table__term.c.table_id.label("table_id"),
            literal(REL_REPRESENTS).label("path"),
        ).select_from(
            s.table__term.join(s.term, s.term.c.id == s.table__term.c.term_id)
        ),
        s.table__term.c.table_id,
    )

    # The link's own type is carried along so ``path`` can name it below -- the
    # two legs are otherwise identical, and unioning them first would lose which
    # one matched.
    link = (
        select(
            s.column__has_attribute.c.column_id,
            s.column__has_attribute.c.attribute_id,
            literal(REL_HAS_ATTRIBUTE).label("rel_type"),
        )
        .union(
            select(
                s.column__semantic_fk.c.column_id,
                s.column__semantic_fk.c.attribute_id,
                literal(REL_SEMANTIC_FK).label("rel_type"),
            )
        )
        .subquery("attribute_link")
    )
    via_attribute = restrict(
        select(
            s.term.c.id.label("term_id"),
            s.catalog_column.c.table_id.label("table_id"),
            link.c.rel_type.label("path"),
        ).select_from(
            s.catalog_column.join(link, link.c.column_id == s.catalog_column.c.id)
            .join(
                s.column_attribute,
                and_(
                    s.column_attribute.c.id == link.c.attribute_id,
                    s.column_attribute.c.source == SEMANTIC_SOURCE,
                ),
            )
            .join(
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id == s.column_attribute.c.id,
            )
            .join(s.term, s.term.c.id == s.column_attribute__term.c.term_id)
        ),
        s.catalog_column.c.table_id,
    )

    return [dict(r) for r in store().query_read(represents.union(via_attribute))]


def build_term_table_maps(
    pairs: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Split ``{term_id, table_id}`` rows into term→tables and table→terms."""
    term_tables: dict[str, set[str]] = {}
    table_terms: dict[str, set[str]] = {}
    for row in pairs:
        term_id, table_id = row.get("term_id"), row.get("table_id")
        if term_id and table_id:
            term_tables.setdefault(term_id, set()).add(table_id)
            table_terms.setdefault(table_id, set()).add(term_id)
    return term_tables, table_terms


def fetch_related_term_ids(
    term_id: str,
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[str]:
    """Ids of every Term sharing a table with *term_id*.

    Split from :func:`fetch_related_terms` because the two steps page
    differently: which terms are related is a property of the whole accessible
    graph, and is what a pager's total counts, while one page of rows is a
    bounded read. The sort is only for stability — the order a reader sees is
    the name order :func:`fetch_terms_by_ids` imposes.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    own_pairs = fetch_term_table_pairs(zone_ids, resolved, term_ids=[term_id])
    shared = sorted({r["table_id"] for r in own_pairs if r.get("table_id")})
    if not shared:
        return []

    _, table_terms = build_term_table_maps(
        fetch_term_table_pairs(zone_ids, resolved, table_ids=shared)
    )
    related: set[str] = set()
    for table_id in shared:
        related.update(table_terms.get(table_id, set()))
    related.discard(term_id)
    return sorted(related)


def fetch_terms_by_ids(
    term_ids: list[str],
    *,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """``{id, name, description}`` for *term_ids*, ordered and paged in SQL.

    *term_ids* is a plain id list — whatever produced it owns the zone scoping.
    """
    if not term_ids:
        return []
    statement = (
        select(s.term.c.id, s.term.c.name, s.term.c.description)
        .where(s.term.c.id.in_(list(term_ids)))
        .order_by(func.lower(s.term.c.name), s.term.c.id)
        .offset(skip)
    )
    if limit is not None:
        statement = statement.limit(limit)
    return [dict(r) for r in store().query_read(statement)]


def fetch_related_terms(
    term_id: str,
    zone_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Terms related to *term_id* by sharing a table, paged by name.

    Related means co-located: both terms connect to the same table, by any of
    the three paths :func:`fetch_term_table_pairs` defines.

    When *zone_ids* is supplied both steps apply the all-or-nothing rule, so a
    related term the caller could not actually open — it would 404 through
    ``get_full_term_by_id`` — is never shown as a chip.
    """
    related_ids = fetch_related_term_ids(term_id, zone_ids, data_ids_by_zone)
    return fetch_terms_by_ids(related_ids, skip=skip, limit=limit)


def fetch_related_terms_counts(
    zone_ids: list[str] | None = None,
    term_ids: list[str] | None = None,
    data_ids_by_zone: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """``[{term_id, count}]`` — how many terms each one shares a table with.

    *term_ids* restricts which terms get an **entry**, not what counts as
    related: neighbours are resolved across the whole accessible graph, so a
    term's count does not shrink because the terms it relates to landed on
    another page.
    """
    resolved = resolve_accessible_catalog_ids(zone_ids, data_ids_by_zone)
    if term_ids is None:
        term_tables, table_terms = build_term_table_maps(
            fetch_term_table_pairs(zone_ids, resolved)
        )
        wanted = list(term_tables)
    else:
        if not term_ids:
            return []
        own_pairs = fetch_term_table_pairs(zone_ids, resolved, term_ids=term_ids)
        term_tables, _ = build_term_table_maps(own_pairs)
        touched = sorted({r["table_id"] for r in own_pairs})
        _, table_terms = build_term_table_maps(
            fetch_term_table_pairs(zone_ids, resolved, table_ids=touched)
            if touched
            else []
        )
        wanted = term_ids

    result: list[dict[str, Any]] = []
    for term_id in wanted:
        related: set[str] = set()
        for table_id in term_tables.get(term_id, set()):
            related.update(table_terms.get(table_id, set()))
        related.discard(term_id)
        result.append({"term_id": term_id, "count": len(related)})
    return result


# ---------------------------------------------------------------------------
# Term-to-term link paths
# ---------------------------------------------------------------------------

#: Path length ceiling, in **edges**. The longest chain worth showing is
#: ``Term -REPRESENTS- Table -CONTAINS- Column -SEMANTIC_FK- ColumnAttribute
#: -PROPERTY_OF- Term`` at four; ten leaves room for the join-table shapes
#: without letting a pathological catalog walk the whole graph.
_MAX_LINK_PATH_DEPTH = 10

#: Node kind (as carried through the traversal below) → the label the API reports.
_LINK_PATH_LABELS = {
    "term": LABEL_TERM,
    "table": Labels.TABLE,
    "column": Labels.COLUMN,
    "column_attribute": LABEL_COLUMN_ATTRIBUTE,
}


def _term_link_edges():
    """Every edge a term↔term path may walk, **undirected**, tagged with its type.

    Unlike :data:`~gsf.dal.schema.JOIN_PATH_EDGE_VIEW_SQL`, SEMANTIC_FK is
    emitted in *both* directions here. A term↔term edge already means the two
    terms genuinely share a table under :func:`fetch_term_table_pairs`'s rules,
    so tracing some real path between them cannot fabricate a link the way
    ``find_join_path``'s column-to-column search could -- it can only surface a
    chain that already exists. It has to be undirected for the join-table shape:
    when both terms reach their shared table only through SEMANTIC_FK, the path
    must step *against* one of the two stored (column -> attribute) directions
    whichever end it starts from.

    Not a database view, deliberately: one function reads it, and a view would
    have to be kept in step through a migration for no gain.
    """

    def leg(src_kind, src_id, dst_kind, dst_id, rel_type, source):
        return select(
            literal(src_kind).label("src_kind"),
            src_id.label("src_id"),
            literal(dst_kind).label("dst_kind"),
            dst_id.label("dst_id"),
            literal(rel_type).label("rel_type"),
        ).select_from(source)

    tt, ct, ca, cf, ha = (
        s.table__term,
        s.catalog_column,
        s.column_attribute__term,
        s.column__semantic_fk,
        s.column__has_attribute,
    )
    return (
        leg("table", tt.c.table_id, "term", tt.c.term_id, REL_REPRESENTS, tt)
        .union_all(
            leg("term", tt.c.term_id, "table", tt.c.table_id, REL_REPRESENTS, tt),
            leg("table", ct.c.table_id, "column", ct.c.id, Edges.CONTAINS, ct),
            leg("column", ct.c.id, "table", ct.c.table_id, Edges.CONTAINS, ct),
            leg(
                "column",
                ha.c.column_id,
                "column_attribute",
                ha.c.attribute_id,
                REL_HAS_ATTRIBUTE,
                ha,
            ),
            leg(
                "column_attribute",
                ha.c.attribute_id,
                "column",
                ha.c.column_id,
                REL_HAS_ATTRIBUTE,
                ha,
            ),
            leg(
                "column",
                cf.c.column_id,
                "column_attribute",
                cf.c.attribute_id,
                REL_SEMANTIC_FK,
                cf,
            ),
            leg(
                "column_attribute",
                cf.c.attribute_id,
                "column",
                cf.c.column_id,
                REL_SEMANTIC_FK,
                cf,
            ),
            leg(
                "column_attribute",
                ca.c.attribute_id,
                "term",
                ca.c.term_id,
                REL_PROPERTY_OF,
                ca,
            ),
            leg(
                "term",
                ca.c.term_id,
                "column_attribute",
                ca.c.attribute_id,
                REL_PROPERTY_OF,
                ca,
            ),
        )
        .subquery("term_link_edge")
    )


def _link_path_bfs(anchor_id: str, dest_id: str) -> list[dict[str, Any]] | None:
    """Shortest term-to-term path as a node list, or ``None`` when there is none.

    Level-at-a-time BFS with the visited set held in Python, for the same reason
    :func:`gsf.dal.attributes._bfs_path` does it that way: a recursive CTE tracks
    visited nodes per *path*, so a hub attribute shared by hundreds of columns
    fans out exponentially. A shared visited set bounds the work by the size of
    the reachable component instead.

    Each entry of the returned list is ``{id, kind, relationship}``, where
    ``relationship`` is the edge type walked to *reach* that node -- ``None`` on
    the anchor, which nothing was walked to reach.
    """
    edges = _term_link_edges()
    expand = (
        select(edges.c.src_id, edges.c.dst_kind, edges.c.dst_id, edges.c.rel_type)
        .where(
            edges.c.src_id.in_(bindparam("frontier", expanding=True)),
            edges.c.dst_id.notin_(bindparam("visited", expanding=True)),
        )
        .distinct()
    )

    #: node id -> (kind, parent id, relationship walked from that parent)
    visited: dict[str, tuple[str, str | None, str | None]] = {
        anchor_id: ("term", None, None)
    }
    frontier = [anchor_id]

    for _ in range(_MAX_LINK_PATH_DEPTH):
        if not frontier:
            return None
        rows = store().query_read(
            expand, {"frontier": frontier, "visited": list(visited)}
        )
        next_frontier: list[str] = []
        for row in rows:
            node = row["dst_id"]
            if node in visited:
                # Two edges into the same node within one level: the first wins,
                # which is what BFS means. Without this the parent pointer could
                # be overwritten by a longer route found in the same batch.
                continue
            visited[node] = (row["dst_kind"], row["src_id"], row["rel_type"])
            next_frontier.append(node)
            if node == dest_id:
                path: list[dict[str, Any]] = []
                current: str | None = node
                while current is not None:
                    kind, parent, relationship = visited[current]
                    path.append(
                        {"id": current, "kind": kind, "relationship": relationship}
                    )
                    current = parent
                path.reverse()
                return path
        frontier = next_frontier

    logger.warning(
        "find_term_link_path: gave up after %s levels for %s -> %s",
        _MAX_LINK_PATH_DEPTH,
        anchor_id,
        dest_id,
    )
    return None


def _name_link_path_nodes(path: list[dict[str, Any]]) -> dict[str, str]:
    """``node id -> name`` for every node on the path, one query per kind."""
    by_kind = {
        "term": (s.term.c.id, s.term.c.name),
        "table": (s.catalog_table.c.id, s.catalog_table.c.name),
        "column": (s.catalog_column.c.id, s.catalog_column.c.name),
        "column_attribute": (s.column_attribute.c.id, s.column_attribute.c.name),
    }
    names: dict[str, str] = {}
    for kind, (id_column, name_column) in by_kind.items():
        ids = [node["id"] for node in path if node["kind"] == kind]
        if not ids:
            continue
        names.update(
            {
                row["id"]: row["name"]
                for row in store().query_read(
                    select(id_column.label("id"), name_column.label("name")).where(
                        id_column.in_(ids)
                    )
                )
            }
        )
    return names


def _enrich_catalog_path_nodes(path_nodes: list[dict[str, Any]]) -> None:
    """Attach catalog ids/names onto each Table/Column node, in place.

    The traversal never walks through a Schema or Database, so a Table/Column
    hop otherwise carries nothing beyond a bare id and name -- not enough for a
    client to expand it any further the way every other Table/Column expansion
    does (``expandTableNode``/``expandColumnNode`` in ``ExplorationView.tsx``
    need a Table's own database/schema ids, and a Column's own owning Table id).
    Best-effort: a failed lookup leaves those nodes without the extra fields,
    the same as if this were never called.
    """
    table_ids = list(
        {n["id"] for n in path_nodes if n.get("label") == Labels.TABLE and n.get("id")}
    )
    if table_ids:
        try:
            rows = store().query_read(
                select(
                    s.catalog_table.c.id.label("table_id"),
                    s.catalog_database.c.id.label("database_id"),
                    s.catalog_database.c.name.label("database_name"),
                    s.catalog_schema.c.id.label("schema_id"),
                    s.catalog_schema.c.name.label("schema_name"),
                )
                .select_from(
                    s.catalog_table.join(
                        s.catalog_schema,
                        s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                    ).join(
                        s.catalog_database,
                        s.catalog_database.c.id == s.catalog_schema.c.database_id,
                    )
                )
                .where(s.catalog_table.c.id.in_(table_ids))
            )
        except Exception:
            logger.warning(
                "find_term_link_path: table catalog lookup failed for %s",
                table_ids,
                exc_info=True,
            )
            rows = []
        by_table_id = {row["table_id"]: row for row in rows}
        for node in path_nodes:
            if node.get("label") != Labels.TABLE:
                continue
            info = by_table_id.get(node["id"])
            if info is not None:
                node["database_id"] = info["database_id"]
                node["database_name"] = info["database_name"]
                node["schema_id"] = info["schema_id"]
                node["schema_name"] = info["schema_name"]

    column_ids = list(
        {n["id"] for n in path_nodes if n.get("label") == Labels.COLUMN and n.get("id")}
    )
    if column_ids:
        contexts = fetch_col_table_contexts(column_ids)
        for node in path_nodes:
            if node.get("label") != Labels.COLUMN:
                continue
            info = contexts.get(node["id"])
            if info is not None:
                node.update(
                    {
                        key: info.get(key)
                        for key in (
                            "table_id",
                            "table_name",
                            "database_id",
                            "database_name",
                            "schema_id",
                            "schema_name",
                        )
                    }
                )


def find_term_link_path(term_a_id: str, term_b_id: str) -> list[dict[str, Any]]:
    """The shortest semantic path connecting two Terms, as ordered hop dicts.

    One level up from :func:`gsf.dal.attributes.find_join_path`'s
    column-to-column traversal: this walks term to term across whichever of
    REPRESENTS, CONTAINS, HAS_ATTRIBUTE, SEMANTIC_FK and PROPERTY_OF actually
    connects them -- the same links ``fetch_semantic_exploration_graph`` folds
    into one term↔term edge's collapsed ``relationship_types`` label, but
    returned here as the real ordered chain a client can graft and highlight,
    e.g. ``Term1 <-REPRESENTS- Table -CONTAINS-> Column -SEMANTIC_FK->
    ColumnAttribute -PROPERTY_OF-> Term2``.

    Returns, ordered from *term_a_id* to *term_b_id*::

        [{relationship, source: {id, name, label, ...},
                        target: {id, name, label, ...}}, ...]

    ``[]`` when the two ids are equal, either Term is missing, or nothing
    connects them -- which should not happen for a real Exploration graph edge,
    but guards a stale or hand-crafted request.
    """
    if not term_a_id or not term_b_id or term_a_id == term_b_id:
        return []

    try:
        path = _link_path_bfs(term_a_id, term_b_id)
    except Exception:
        logger.warning(
            "find_term_link_path: query failed for %s -> %s",
            term_a_id,
            term_b_id,
            exc_info=True,
        )
        return []
    if not path or len(path) < 2:
        return []

    names = _name_link_path_nodes(path)
    nodes = [
        {
            "id": node["id"],
            "name": names.get(node["id"], ""),
            "label": _LINK_PATH_LABELS[node["kind"]],
        }
        for node in path
    ]
    _enrich_catalog_path_nodes(nodes)
    return [
        {
            "relationship": path[index + 1]["relationship"],
            "source": nodes[index],
            "target": nodes[index + 1],
        }
        for index in range(len(nodes) - 1)
    ]
