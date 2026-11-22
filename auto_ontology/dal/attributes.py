# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ColumnAttribute and SEMANTIC_FK reads and writes, plus join-path traversal.

A SEMANTIC_FK targets a Concept key (``auto_ontology.dal.concepts``), not an
attribute: the key has one member column per member table of the Concept.
``find_join_path`` projects FKs and Concept identities into undirected table
edges (``_find_table_join_hops``), then runs BFS over tables while retaining
the exact join columns on every edge.
"""

from __future__ import annotations

from collections import defaultdict, deque
import logging
from typing import Any

from sqlalchemy import (
    and_,
    bindparam,
    case,
    delete,
    func,
    literal,
    or_,
    union,
    select,
    true,
    update,
)
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.dal import schema as s
from auto_ontology.dal.concepts import (
    ensure_column_key,
    fetch_column_key_id,
    fetch_key_concept_id,
    fetch_table_concept_id,
    semantic_fk_targets,
)
from auto_ontology.dal.datasources import fetch_col_table_contexts
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.dal.tags import PII_TAG_NAME
from auto_ontology.dal.term_families import attribute_family_prefix
from auto_ontology.semantic.constants import SEMANTIC_SOURCE

logger = logging.getLogger(__name__)

#: Path length ceiling, in **edges**.
#:
#: It is tempting to lower this on the grounds that real join paths are 2-4
#: hops. That reasoning is wrong, and the depth-bound test catches it: a *hop*
#: here is four edges, not one. Getting from one FK column to the next runs
#: Column -SEMANTIC_FK-> ConceptKey -KEY_COLUMN-> Column -CONTAINS->
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
    description: str | None = None,
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
            s.catalog_column.c.description.label("column_description"),
            s.catalog_column.c.sample_values,
            attribute_family_prefix(s.column_attribute.c.id).label("family_prefix"),
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

    **Reads SEMANTIC_FK backwards**: it binds an attribute and finds the
    columns pointing *at* it — the FK columns whose Concept key has the
    attribute's own column as a member column.

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
    targets = semantic_fk_targets("context_fk_target")
    link = (
        select(
            s.column__has_attribute.c.attribute_id,
            s.column__has_attribute.c.column_id,
            literal(0).label("rank"),
        )
        .union(
            select(
                s.column__has_attribute.c.attribute_id,
                targets.c.fk_column_id,
                literal(1).label("rank"),
            ).join(
                targets,
                targets.c.target_column_id == s.column__has_attribute.c.column_id,
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
# SEMANTIC_FK
# ---------------------------------------------------------------------------


def find_unlinked_fk_columns(
    database_name: str | None = None,
) -> list[dict[str, Any]]:
    """Columns carrying no SEMANTIC_FK, SEMANTIC_FK_DIRECT or HAS_ATTRIBUTE link.

    A Concept key column is the entity's identity, not a reference to another
    entity, so it is a candidate only with a declared foreign key — a table
    whose key also references another entity's, such as a 1:1 extension.

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
            s.catalog_column.c.distinct_values,
            s.catalog_column.c.data_type,
            s.catalog_column.c.is_unique,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
            s.column__foreign_key.c.target_column_id.label("fk_target_col_id"),
        )
        .select_from(
            s.catalog_column.join(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
            .join(
                s.catalog_schema,
                s.catalog_schema.c.id == s.catalog_table.c.schema_id,
            )
            .outerjoin(
                s.column__foreign_key,
                s.column__foreign_key.c.source_column_id == s.catalog_column.c.id,
            )
        )
        .where(
            ~select(s.column__semantic_fk.c.key_id)
            .where(s.column__semantic_fk.c.column_id == s.catalog_column.c.id)
            .exists(),
            ~select(s.column__has_attribute.c.attribute_id)
            .where(s.column__has_attribute.c.column_id == s.catalog_column.c.id)
            .exists(),
            ~select(s.column__semantic_fk_direct.c.group_id)
            .where(s.column__semantic_fk_direct.c.column_id == s.catalog_column.c.id)
            .exists(),
            or_(
                s.column__foreign_key.c.target_column_id.is_not(None),
                ~select(s.concept_key__column.c.key_id)
                .where(s.concept_key__column.c.column_id == s.catalog_column.c.id)
                .exists(),
            ),
        )
    )
    if database_name is not None:
        statement = statement.where(
            s.catalog_schema.c.database_id.in_(
                select(s.catalog_database.c.id).where(
                    s.catalog_database.c.name == database_name
                )
            )
        )

    return [dict(r) for r in store().query_read(statement)]


def fetch_value_key_candidates(database_name: str) -> list[dict[str, Any]]:
    """Every column of *database_name* a semantic FK may point at.

    The same targets the vector search allows: a column that owns a
    ColumnAttribute and was profiled unique. One row per column, carrying the
    attribute to link to (the lowest id when it owns several) and its stored
    ``distinct_values``, which is ``None`` until a probe has written them.

    A column tagged ``PII`` is never a candidate, whatever its profile says: a
    candidate without stored values gets them probed from the warehouse.

    Rows carry ``attribute_id``, ``column_id``, ``column_name``, ``data_type``,
    ``distinct_values``, ``table_id``, ``table_name`` and ``schema_name``.
    """
    pii_tagged = (
        select(literal(1))
        .select_from(s.tag_target.join(s.tag, s.tag.c.id == s.tag_target.c.tag_id))
        .where(
            s.tag_target.c.column_id == s.catalog_column.c.id,
            func.lower(func.trim(s.tag.c.name)) == PII_TAG_NAME.lower(),
        )
        .exists()
    )
    statement = (
        select(
            func.min(s.column__has_attribute.c.attribute_id).label("attribute_id"),
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.data_type,
            s.catalog_column.c.distinct_values,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
        )
        .select_from(
            s.column__has_attribute.join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(
            s.catalog_database.c.name == database_name,
            s.catalog_column.c.is_unique.is_(True),
            ~pii_tagged,
        )
        .group_by(
            s.catalog_column.c.id,
            s.catalog_table.c.id,
            s.catalog_schema.c.name,
        )
        .order_by(s.catalog_table.c.name, s.catalog_column.c.name)
    )
    return [dict(row) for row in store().query_read(statement)]


def fetch_attributes_to_describe(database_name: str) -> list[dict[str, Any]]:
    """Every semantic ColumnAttribute of *database_name* with no description yet.

    One row per attribute, joined to its owning column, table and Term, with
    everything a description is written from: ``attribute_id``,
    ``attribute_name``, ``term_name``, ``term_description``, ``table_id``,
    ``table_name``, ``table_description``, ``column_id``, ``column_name``,
    ``column_description``, ``data_type``, ``sample_values``, ``format`` and
    ``family_prefix``.
    """
    blank = func.trim(func.coalesce(s.column_attribute.c.description, "")) == ""
    statement = (
        select(
            s.column_attribute.c.id.label("attribute_id"),
            s.column_attribute.c.name.label("attribute_name"),
            s.term.c.name.label("term_name"),
            s.term.c.description.label("term_description"),
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.description.label("table_description"),
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.description.label("column_description"),
            s.catalog_column.c.data_type,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.format,
            attribute_family_prefix(s.column_attribute.c.id).label("family_prefix"),
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
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
            .join(
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id == s.column_attribute.c.id,
            )
            .join(s.term, s.term.c.id == s.column_attribute__term.c.term_id)
        )
        .where(
            s.catalog_database.c.name == database_name,
            s.column_attribute.c.source == SEMANTIC_SOURCE,
            blank,
        )
        .order_by(s.catalog_table.c.name, s.catalog_column.c.ordinal_position)
    )
    return [dict(row) for row in store().query_read(statement)]


def fetch_semantic_fk_ends(database_name: str) -> list[dict[str, Any]]:
    """Both ends of every SEMANTIC_FK whose source column is in *database_name*.

    One row per FK and member column of the key it targets. Rows carry
    ``source_column_id``, ``source_table_name``, ``source_column_name``,
    ``target_concept_id``, ``target_attribute_id`` (``None`` when the member
    column owns no attribute), ``target_table_name`` and ``target_column_name``.
    """
    targets = semantic_fk_targets("fk_end_target")
    source_column = s.catalog_column.alias("fk_end_source_column")
    source_table = s.catalog_table.alias("fk_end_source_table")
    source_schema = s.catalog_schema.alias("fk_end_source_schema")
    target_column = s.catalog_column.alias("fk_end_target_column")
    target_table = s.catalog_table.alias("fk_end_target_table")
    statement = (
        select(
            source_column.c.id.label("source_column_id"),
            source_table.c.name.label("source_table_name"),
            source_column.c.name.label("source_column_name"),
            targets.c.concept_id.label("target_concept_id"),
            targets.c.target_attribute_id,
            target_table.c.name.label("target_table_name"),
            target_column.c.name.label("target_column_name"),
        )
        .select_from(
            targets.join(source_column, source_column.c.id == targets.c.fk_column_id)
            .join(source_table, source_table.c.id == source_column.c.table_id)
            .join(source_schema, source_schema.c.id == source_table.c.schema_id)
            .join(
                s.catalog_database,
                s.catalog_database.c.id == source_schema.c.database_id,
            )
            .join(target_column, target_column.c.id == targets.c.target_column_id)
            .join(target_table, target_table.c.id == target_column.c.table_id)
        )
        .where(s.catalog_database.c.name == database_name)
        .order_by(source_table.c.name, source_column.c.name, target_table.c.name)
    )
    return [dict(row) for row in store().query_read(statement)]


def store_attribute_descriptions(descriptions: dict[str, str]) -> int:
    """Write descriptions onto attributes that still have none.

    Keyed by attribute id. An attribute described in the meantime — by a person
    editing it while a compile runs — keeps its description. Returns the number
    of attributes written.
    """
    if not descriptions:
        return 0
    attribute = s.column_attribute
    rows = store().query_write(
        update(attribute)
        .where(
            attribute.c.id.in_(list(descriptions)),
            func.trim(func.coalesce(attribute.c.description, "")) == "",
        )
        .values(
            description=case(
                {
                    attr_id: literal(text, attribute.c.description.type)
                    for attr_id, text in descriptions.items()
                },
                value=attribute.c.id,
                else_=attribute.c.description,
            )
        )
        .returning(attribute.c.id)
    )
    return len(rows)


def _same_database(column_a_id: str, column_b_id: str) -> bool:
    """Whether the two columns belong to the same database."""
    contexts = fetch_col_table_contexts([column_a_id, column_b_id])
    database_a = contexts.get(column_a_id, {}).get("database_name")
    database_b = contexts.get(column_b_id, {}).get("database_name")
    return bool(database_a) and database_a == database_b


def attribute_owner_column_id(attr_id: str) -> str | None:
    """The lowest-id column *attr_id* describes, or ``None``."""
    rows = store().query_read(
        select(func.min(s.column__has_attribute.c.column_id).label("column_id")).where(
            s.column__has_attribute.c.attribute_id == attr_id
        )
    )
    return rows[0]["column_id"] if rows else None


def merge_semantic_fk_to_column(
    src_column_id: str, target_column_id: str, *, declared: bool = False
) -> bool:
    """Point a column at the Concept key *target_column_id* stands for. Idempotent.

    The target column becomes a key of its own table's Concept when it is not
    one already (:func:`~auto_ontology.dal.concepts.ensure_column_key`), and
    the FK targets that key — so a join can later reach any member of the
    Concept. A column holds at most one FK; an existing one is kept.

    Refuses — writes nothing, returns ``False`` — when the two columns are the
    same, belong to different ingested databases, when the source column is a
    Concept key column itself (an identity, not a reference) — unless the FK
    is *declared* and targets a different Concept — or when the target's table
    has no Concept. This is the one place a ``SEMANTIC_FK`` edge
    is created, so it is the one place that has to enforce that it never
    crosses a database boundary: a path built from such an edge would later
    name a table that doesn't exist in the query's target database.
    """
    if src_column_id == target_column_id:
        return False
    if not _same_database(src_column_id, target_column_id):
        logger.warning(
            "merge_semantic_fk: refusing cross-database edge %s -> %s",
            src_column_id,
            target_column_id,
        )
        return False
    src_key_id = fetch_column_key_id(src_column_id)
    if src_key_id is not None and not declared:
        logger.info(
            "merge_semantic_fk: %s is a Concept key column; not an FK",
            src_column_id,
        )
        return False
    with write_transaction():
        if src_key_id is not None:
            target_key_id = fetch_column_key_id(target_column_id)
            if target_key_id is not None:
                target_concept_id = fetch_key_concept_id(target_key_id)
            else:
                target_table_id = (
                    fetch_col_table_contexts([target_column_id])
                    .get(target_column_id, {})
                    .get("table_id")
                )
                target_concept_id = (
                    fetch_table_concept_id(target_table_id) if target_table_id else None
                )
            if target_concept_id == fetch_key_concept_id(src_key_id):
                return False
        key_id = ensure_column_key(target_column_id)
        if key_id is None:
            return False
        store().query_write(
            insert(s.column__semantic_fk)
            .values(column_id=src_column_id, key_id=key_id)
            .on_conflict_do_nothing()
        )
        rows = store().query_read(
            select(s.column__semantic_fk.c.key_id).where(
                s.column__semantic_fk.c.column_id == src_column_id
            )
        )
    return bool(rows) and rows[0]["key_id"] == key_id


def merge_semantic_fk(src_column_id: str, tgt_attr_id: str) -> bool:
    """Point a column at the key of the column owning *tgt_attr_id*.

    The attribute-addressed form of :func:`merge_semantic_fk_to_column`, for
    the resolvers that search ColumnAttributes. ``False`` when the attribute
    owns no column.
    """
    owner_column_id = attribute_owner_column_id(tgt_attr_id)
    if owner_column_id is None:
        return False
    return merge_semantic_fk_to_column(src_column_id, owner_column_id)


# ---------------------------------------------------------------------------
# SEMANTIC_FK_DIRECT
# ---------------------------------------------------------------------------


def fetch_direct_fk_peers(database_name: str) -> list[dict[str, Any]]:
    """Every FK column of *database_name* an unlinked FK column may join directly.

    The columns already linked: those holding a SEMANTIC_FK, and members of a
    SEMANTIC_FK_DIRECT group. Rows carry ``id``, ``name``, ``table_id``,
    ``table_name``, ``schema_name`` and ``group_id`` (``None`` for a column
    in no group).
    """
    linked = or_(
        select(s.column__semantic_fk.c.key_id)
        .where(s.column__semantic_fk.c.column_id == s.catalog_column.c.id)
        .exists(),
        s.column__semantic_fk_direct.c.group_id.is_not(None),
    )
    statement = (
        select(
            s.catalog_column.c.id,
            s.catalog_column.c.name,
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
            s.column__semantic_fk_direct.c.group_id,
        )
        .select_from(
            s.catalog_column.join(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
            .outerjoin(
                s.column__semantic_fk_direct,
                s.column__semantic_fk_direct.c.column_id == s.catalog_column.c.id,
            )
        )
        .where(s.catalog_database.c.name == database_name, linked)
        .order_by(s.catalog_table.c.name, s.catalog_column.c.name)
    )
    return [dict(row) for row in store().query_read(statement)]


def link_direct_fk_columns(column_ids: list[str]) -> str | None:
    """Put *column_ids* in one SEMANTIC_FK_DIRECT group. Idempotent.

    A group one of the columns already belongs to is reused — the lowest id
    when they belong to several, and the others are merged into it — so a
    recompile grows the existing groups rather than duplicating them. Returns
    the group id, or ``None`` without writing when fewer than two distinct
    columns are given or they span databases: a join between them would name
    a table missing from the query's database.
    """
    ids = sorted(set(column_ids))
    if len(ids) < 2:
        return None
    contexts = fetch_col_table_contexts(ids)
    database_ids = {contexts.get(column_id, {}).get("database_id") for column_id in ids}
    if len(database_ids) != 1 or None in database_ids:
        logger.warning(
            "link_direct_fk_columns: refusing a group across databases or with "
            "unknown columns: %s",
            ids,
        )
        return None
    (database_id,) = database_ids

    member = s.column__semantic_fk_direct
    with write_transaction():
        existing = sorted(
            {
                row["group_id"]
                for row in store().query_read(
                    select(member.c.group_id).where(member.c.column_id.in_(ids))
                )
            }
        )
        if existing:
            group_id, absorbed = existing[0], existing[1:]
        else:
            group_id = store().query_write(
                insert(s.semantic_fk_direct)
                .values(database_id=database_id)
                .returning(s.semantic_fk_direct.c.id)
            )[0]["id"]
            absorbed = []
        if absorbed:
            store().query_write(
                update(member)
                .where(member.c.group_id.in_(absorbed))
                .values(group_id=group_id)
            )
            store().query_write(
                delete(s.semantic_fk_direct).where(
                    s.semantic_fk_direct.c.id.in_(absorbed)
                )
            )
        store().query_write(
            insert(member)
            .values(
                [{"column_id": column_id, "group_id": group_id} for column_id in ids]
            )
            .on_conflict_do_nothing()
        )
    return group_id


def fetch_semantic_fk_join_groups(database_name: str) -> list[dict[str, Any]]:
    """Return semantic FKs with the context needed to choose one join default.

    One row per FK. Its group is ``(source_table_id, target_concept_id)``: every
    FK of one table into one Concept competes for the default, whichever member
    each was resolved against. The target columns describe the FK's first
    member table by name.
    """
    targets = semantic_fk_targets("join_default_target")
    source_column = s.catalog_column.alias("join_default_source_column")
    source_table = s.catalog_table.alias("join_default_source_table")
    source_schema = s.catalog_schema.alias("join_default_source_schema")
    source_database = s.catalog_database.alias("join_default_source_database")
    target_column = s.catalog_column.alias("join_default_target_column")
    target_table = s.catalog_table.alias("join_default_target_table")
    target_schema = s.catalog_schema.alias("join_default_target_schema")

    statement = (
        select(
            source_table.c.id.label("source_table_id"),
            source_table.c.name.label("source_table_name"),
            source_table.c.description.label("source_table_description"),
            source_schema.c.name.label("source_schema_name"),
            source_column.c.id.label("source_column_id"),
            source_column.c.name.label("source_column_name"),
            source_column.c.description.label("source_column_description"),
            targets.c.concept_id.label("target_concept_id"),
            targets.c.key_id.label("target_key_id"),
            targets.c.target_attribute_id,
            s.column_attribute.c.name.label("target_attribute_name"),
            s.column_attribute.c.description.label("target_attribute_description"),
            target_table.c.id.label("target_table_id"),
            target_table.c.name.label("target_table_name"),
            target_table.c.description.label("target_table_description"),
            target_schema.c.name.label("target_schema_name"),
            target_column.c.id.label("target_column_id"),
            target_column.c.name.label("target_column_name"),
            target_column.c.description.label("target_column_description"),
            targets.c.use_in_join_paths,
        )
        .distinct(source_column.c.id)
        .select_from(
            targets.join(
                source_column,
                source_column.c.id == targets.c.fk_column_id,
            )
            .join(source_table, source_table.c.id == source_column.c.table_id)
            .join(source_schema, source_schema.c.id == source_table.c.schema_id)
            .join(
                source_database,
                source_database.c.id == source_schema.c.database_id,
            )
            .outerjoin(
                s.column_attribute,
                s.column_attribute.c.id == targets.c.target_attribute_id,
            )
            .join(
                target_column,
                target_column.c.id == targets.c.target_column_id,
            )
            .join(target_table, target_table.c.id == target_column.c.table_id)
            .join(target_schema, target_schema.c.id == target_table.c.schema_id)
        )
        .where(source_database.c.name == database_name)
        .order_by(source_column.c.id, target_table.c.name, target_column.c.id)
    )
    rows = [dict(row) for row in store().query_read(statement)]
    rows.sort(
        key=lambda row: (
            row["source_table_id"],
            row["target_concept_id"],
            str(row["source_column_name"] or "").lower(),
            row["source_column_id"],
        )
    )
    return rows


def set_semantic_fk_join_default(
    source_table_id: str,
    target_concept_id: str,
    selected_column_id: str,
) -> None:
    """Atomically enable one semantic FK in a source-table/target-Concept group."""
    source_column_ids = select(s.catalog_column.c.id).where(
        s.catalog_column.c.table_id == source_table_id
    )
    concept_key_ids = select(s.concept_key.c.id).where(
        s.concept_key.c.concept_id == target_concept_id
    )
    statement = (
        update(s.column__semantic_fk)
        .where(
            s.column__semantic_fk.c.key_id.in_(concept_key_ids),
            s.column__semantic_fk.c.column_id.in_(source_column_ids),
        )
        .values(
            use_in_join_paths=case(
                (
                    s.column__semantic_fk.c.column_id == selected_column_id,
                    True,
                ),
                else_=False,
            )
        )
        .returning(s.column__semantic_fk.c.column_id)
    )
    with write_transaction():
        updated = store().query_write(statement)
        if selected_column_id not in {row["column_id"] for row in updated}:
            raise ValueError(
                "Selected semantic FK column does not belong to the requested group"
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

    The primary column *has* the attribute; the referenced columns point at it
    — they hold a semantic FK whose Concept key has the attribute's column as a
    member column.
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
    targets = semantic_fk_targets("referenced_fk_target")
    referencing = (
        select(
            s.column__has_attribute.c.attribute_id,
            targets.c.fk_column_id.label("column_id"),
        )
        .join(
            targets, targets.c.target_column_id == s.column__has_attribute.c.column_id
        )
        .distinct()
        .subquery("referencing")
    )
    referenced = store().query_read(
        _column_path_select(referencing)
        .where(referencing.c.attribute_id.in_(ids))
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

_fk_column = s.catalog_column.alias("join_fk_column")
_fk_table = s.catalog_table.alias("join_fk_table")
_fk_schema = s.catalog_schema.alias("join_fk_schema")
_member_table = s.catalog_table.alias("join_member_table")
_member_schema = s.catalog_schema.alias("join_member_schema")
_fk_targets = semantic_fk_targets("join_fk_target")
_identity_from = s.concept_key__column.alias("join_identity_from")
_identity_to = s.concept_key__column.alias("join_identity_to")
_direct_from = s.column__semantic_fk_direct.alias("join_direct_from")
_direct_to = s.column__semantic_fk_direct.alias("join_direct_to")
_direct_from_column = s.catalog_column.alias("join_direct_from_column")
_direct_to_column = s.catalog_column.alias("join_direct_to_column")

#: Every table-crossing a join may take, as ``(from_table_id, from_column_id,
#: to_table_id, to_column_id, key_id, use_in_join_paths)``. Three kinds, all
#: walkable either way:
#:
#: * a semantic FK, from its column to **each** member column of the key it
#:   targets — so the path can enter whichever member table holds the field it
#:   needs, on that table's own key column;
#: * an identity hop between two member columns of one key, which joins the
#:   parts of a split entity to each other;
#: * a direct FK hop between two members of one SEMANTIC_FK_DIRECT group on
#:   different tables. It goes through no Concept, so its ``key_id`` is ``''``.
#:
#: FKs the join-default step turned off are included with
#: ``use_in_join_paths`` false: they are offered as direct alternatives but
#: never walked by a multi-hop route.
_SEMANTIC_TABLE_EDGES = union(
    select(
        _fk_column.c.table_id.label("from_table_id"),
        _fk_targets.c.fk_column_id.label("from_column_id"),
        _fk_targets.c.target_table_id.label("to_table_id"),
        _fk_targets.c.target_column_id.label("to_column_id"),
        _fk_targets.c.key_id,
        _fk_targets.c.use_in_join_paths,
    )
    .select_from(
        _fk_targets.join(_fk_column, _fk_column.c.id == _fk_targets.c.fk_column_id)
        .join(_fk_table, _fk_table.c.id == _fk_column.c.table_id)
        .join(_fk_schema, _fk_schema.c.id == _fk_table.c.schema_id)
        .join(_member_table, _member_table.c.id == _fk_targets.c.target_table_id)
        .join(_member_schema, _member_schema.c.id == _member_table.c.schema_id)
    )
    .where(
        _fk_schema.c.database_id == bindparam("database_id"),
        _member_schema.c.database_id == bindparam("database_id"),
    ),
    select(
        _identity_from.c.table_id,
        _identity_from.c.column_id,
        _identity_to.c.table_id,
        _identity_to.c.column_id,
        _identity_from.c.key_id,
        true(),
    )
    .select_from(
        _identity_from.join(
            _identity_to,
            (_identity_to.c.key_id == _identity_from.c.key_id)
            & (_identity_to.c.table_id != _identity_from.c.table_id),
        )
        .join(s.concept_key, s.concept_key.c.id == _identity_from.c.key_id)
        .join(s.concept, s.concept.c.id == s.concept_key.c.concept_id)
    )
    .where(s.concept.c.database_id == bindparam("database_id")),
    select(
        _direct_from_column.c.table_id,
        _direct_from.c.column_id,
        _direct_to_column.c.table_id,
        _direct_to.c.column_id,
        literal("", s.concept_key.c.id.type),
        true(),
    )
    .select_from(
        _direct_from.join(
            _direct_to,
            (_direct_to.c.group_id == _direct_from.c.group_id)
            & (_direct_to.c.column_id > _direct_from.c.column_id),
        )
        .join(_direct_from_column, _direct_from_column.c.id == _direct_from.c.column_id)
        .join(_direct_to_column, _direct_to_column.c.id == _direct_to.c.column_id)
        .join(
            s.semantic_fk_direct, s.semantic_fk_direct.c.id == _direct_from.c.group_id
        )
    )
    .where(
        s.semantic_fk_direct.c.database_id == bindparam("database_id"),
        _direct_to_column.c.table_id != _direct_from_column.c.table_id,
    ),
)


#: One table crossing: ``(source_column_id, target_column_id, key_id)``.
JoinHop = tuple[str, str, str]


def _find_table_join_hops(
    anchor_table_id: str,
    dest_table_id: str,
    database_id: str,
) -> list[tuple[list[JoinHop], bool]]:
    """Find table-to-table join routes, oriented from anchor table to destination.

    Returns ``(hops, is_default)`` routes. When the two tables are directly
    connected, every direct column pair is its own one-hop route — FKs the
    join-default step turned off included, as non-default — default routes
    first. Otherwise the single shortest route over default edges, or ``[]``.

    Each edge of :data:`_SEMANTIC_TABLE_EDGES` is a concrete column pair, so
    every hop names the two physical columns it joins. Both ends of a semantic
    FK are walkable: its target is always a Concept key column — an identity —
    so two FK columns referencing one key meet at that key's table, never at
    each other. Only a SEMANTIC_FK_DIRECT group joins FK columns to each other.
    """
    rows = store().query_read(
        _SEMANTIC_TABLE_EDGES,
        {"database_id": database_id},
    )
    adjacency: dict[str, set[tuple[str, str, str, str, bool]]] = defaultdict(set)
    for row in rows:
        from_table_id = row["from_table_id"]
        to_table_id = row["to_table_id"]
        if from_table_id == to_table_id:
            continue
        default = bool(row["use_in_join_paths"])
        adjacency[from_table_id].add(
            (
                to_table_id,
                row["from_column_id"],
                row["to_column_id"],
                row["key_id"],
                default,
            )
        )
        adjacency[to_table_id].add(
            (
                from_table_id,
                row["to_column_id"],
                row["from_column_id"],
                row["key_id"],
                default,
            )
        )

    direct = sorted(
        (not default, source_col_id, target_col_id, key_id)
        for next_table_id, source_col_id, target_col_id, key_id, default in (
            adjacency.get(anchor_table_id, ())
        )
        if next_table_id == dest_table_id
    )
    if direct:
        return [
            ([(source_col_id, target_col_id, key_id)], not non_default)
            for non_default, source_col_id, target_col_id, key_id in direct
        ]

    ordered = {
        table_id: sorted(edge[:4] for edge in edges if edge[4])
        for table_id, edges in adjacency.items()
    }
    max_hops = (MAX_PATH_DEPTH + 2) // 4
    parents: dict[str, tuple[str, str, str, str] | None] = {anchor_table_id: None}
    frontier: deque[tuple[str, int]] = deque([(anchor_table_id, 0)])
    while frontier:
        table_id, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for next_table_id, source_col_id, target_col_id, key_id in ordered.get(
            table_id, []
        ):
            if next_table_id in parents:
                continue
            parents[next_table_id] = (table_id, source_col_id, target_col_id, key_id)
            if next_table_id == dest_table_id:
                hops: list[JoinHop] = []
                current = dest_table_id
                while parents[current] is not None:
                    previous, source_col_id, target_col_id, key_id = parents[current]
                    hops.append((source_col_id, target_col_id, key_id))
                    current = previous
                hops.reverse()
                return [(hops, True)]
            frontier.append((next_table_id, depth + 1))
    return []


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


def find_join_routes(anchor_col_id: str, dest_col_id: str) -> list[dict]:
    """The semantic join routes between the columns' tables.

    The endpoint columns identify the tables to connect; they do not have to be
    the join columns. Each route is ``{"hops": [...], "default": bool}``: every
    direct column pair when the tables are one hop apart, default routes first,
    else the one shortest route. Each hop names both columns and ``key``, the
    ``"Concept.key"`` it joins on — ``""`` for a SEMANTIC_FK_DIRECT hop, which
    joins two FK columns through no Concept.

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
            "find_join_routes: rejected cross-database path %s -> %s",
            anchor_col_id,
            dest_col_id,
        )
        return []

    try:
        routes = _find_table_join_hops(
            anchor_context["table_id"],
            dest_context["table_id"],
            anchor_context["database_id"],
        )
    except Exception:
        logger.warning(
            "find_join_routes: query failed for %s -> %s",
            anchor_col_id,
            dest_col_id,
            exc_info=True,
        )
        return []

    if not routes:
        return []

    all_hops = [hop for hops, _ in routes for hop in hops]
    col_ids = list(dict.fromkeys(c for hop in all_hops for c in hop[:2]))
    names = _name_columns(col_ids)
    contexts = fetch_col_table_contexts(col_ids)
    try:
        keys = _key_labels({hop[2] for hop in all_hops})
    except Exception:
        logger.warning(
            "find_join_routes: key label lookup failed for %s -> %s",
            anchor_col_id,
            dest_col_id,
            exc_info=True,
        )
        keys = {}

    def describe(hop: JoinHop) -> dict:
        source_id, target_id, key_id = hop
        source_context = contexts.get(source_id, {})
        target_context = contexts.get(target_id, {})
        return {
            "source_database": source_context.get("database_name", ""),
            "source_schema": source_context.get("schema_name", ""),
            "source_table": source_context.get("table_name", ""),
            "source_column": names.get(source_id, ""),
            "target_database": target_context.get("database_name", ""),
            "target_schema": target_context.get("schema_name", ""),
            "target_table": target_context.get("table_name", ""),
            "target_column": names.get(target_id, ""),
            "key": keys.get(key_id, ""),
        }

    return [
        {"hops": [describe(hop) for hop in hops], "default": default}
        for hops, default in routes
    ]


def find_join_path(anchor_col_id: str, dest_col_id: str) -> list[dict]:
    """The first route :func:`find_join_routes` returns, as its hops.

    ``[]`` in the same cases as :func:`find_join_routes`.
    """
    routes = find_join_routes(anchor_col_id, dest_col_id)
    return routes[0]["hops"] if routes else []


def _key_labels(key_ids: set[str]) -> dict[str, str]:
    """``"Concept.key"`` for each key id."""
    key_ids = {key_id for key_id in key_ids if key_id}
    if not key_ids:
        return {}
    return {
        row["id"]: f"{row['concept_name']}.{row['name']}"
        for row in store().query_read(
            select(
                s.concept_key.c.id,
                s.concept_key.c.name,
                s.concept.c.name.label("concept_name"),
            )
            .join(s.concept, s.concept.c.id == s.concept_key.c.concept_id)
            .where(s.concept_key.c.id.in_(list(key_ids)))
        )
    }
