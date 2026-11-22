# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concepts: the real-world entities Terms are parts of, and their keys.

A Term that represents a table belongs to at most one Concept, and has one
only once the entity has a key: a keyless Concept is never a join target. Most
Concepts have one member; a vertically split entity (``customer`` and
``customer_ext`` sharing ``customer_id``) has one per part. A Concept key names
the entity's identity and maps it to one column on each member table, and a
semantic FK targets a key rather than a single table's column.

Deciding which Concepts to merge is ``auto_ontology.semantic.concepts``; this
module only stores the outcome.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
from typing import Any

from sqlalchemy import ColumnElement, Subquery, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.semantic.date_format import DATE_TYPE_TOKENS

logger = logging.getLogger(__name__)


def is_date_column(column=s.catalog_column) -> ColumnElement[bool]:
    """Whether *column* holds dates: a date/time type, or a profiled date format.

    The format catches dates stored as text, as SQLite stores them. Two
    columns of dates share values without referencing each other, so an
    inferred key or FK never lands on one.
    """
    data_type = func.lower(func.coalesce(column.c.data_type, ""))
    return or_(
        column.c.format.is_not(None),
        *(data_type.contains(token) for token in DATE_TYPE_TOKENS),
    )


def semantic_fk_targets(
    name: str = "semantic_fk_target", *, one_per_fk: bool = False
) -> Subquery:
    """Every semantic FK, once per member column of the key it targets.

    Columns: ``fk_column_id``, ``key_id``, ``concept_id``,
    ``use_in_join_paths``, ``target_column_id``, ``target_table_id`` and
    ``target_attribute_id`` — the lowest-id ColumnAttribute the target column
    owns, ``NULL`` when it owns none. A key with one member yields one row per
    FK, exactly the shape the FK had when it targeted an attribute.

    *one_per_fk* keeps a single member per FK — one whose column owns an
    attribute when there is one, then the lowest table id — for readers that
    need one concrete column to join, and any member is that.
    """
    target_attribute = (
        select(func.min(s.column__has_attribute.c.attribute_id))
        .where(s.column__has_attribute.c.column_id == s.concept_key__column.c.column_id)
        .scalar_subquery()
    )
    statement = select(
        s.column__semantic_fk.c.column_id.label("fk_column_id"),
        s.column__semantic_fk.c.key_id,
        s.concept_key.c.concept_id,
        s.column__semantic_fk.c.use_in_join_paths,
        s.concept_key__column.c.column_id.label("target_column_id"),
        s.concept_key__column.c.table_id.label("target_table_id"),
        target_attribute.label("target_attribute_id"),
    ).select_from(
        s.column__semantic_fk.join(
            s.concept_key, s.concept_key.c.id == s.column__semantic_fk.c.key_id
        ).join(
            s.concept_key__column,
            s.concept_key__column.c.key_id == s.column__semantic_fk.c.key_id,
        )
    )
    if one_per_fk:
        statement = statement.distinct(s.column__semantic_fk.c.column_id).order_by(
            s.column__semantic_fk.c.column_id,
            target_attribute.is_(None),
            s.concept_key__column.c.table_id,
        )
    return statement.subquery(name)


def semantic_fk_attribute_links(name: str = "semantic_fk_attribute") -> Subquery:
    """``(column_id, attribute_id)`` for every attribute an FK column references.

    The referenced attributes are those owned by the member columns of the key
    the FK targets — the link a SEMANTIC_FK used to store directly, one per
    member now.
    """
    targets = semantic_fk_targets(f"{name}_target")
    return (
        select(
            targets.c.fk_column_id.label("column_id"),
            s.column__has_attribute.c.attribute_id,
        )
        .join(
            s.column__has_attribute,
            s.column__has_attribute.c.column_id == targets.c.target_column_id,
        )
        .distinct()
        .subquery(name)
    )


def _table_concept(table_id: Any):
    """The Concept of the Term representing *table_id*, as a scalar subquery."""
    return (
        select(s.term__concept.c.concept_id)
        .join(s.table__term, s.table__term.c.term_id == s.term__concept.c.term_id)
        .where(s.table__term.c.table_id == table_id)
        .order_by(s.term__concept.c.concept_id)
        .limit(1)
        .scalar_subquery()
    )


def fetch_table_concept_id(table_id: str) -> str | None:
    """The Concept of the Term representing *table_id*, or ``None``."""
    rows = store().query_read(select(_table_concept(table_id).label("concept_id")))
    return rows[0]["concept_id"] if rows else None


def ensure_term_concept(term_id: str, *, require_key: bool = False) -> str | None:
    """Return the Term's Concept, creating a one-member Concept when it has none.

    The new Concept is scoped to the database of the table the Term represents
    and named after the Term. When that table's declared primary key is a
    single column, the Concept is given a key mapped to it. Returns ``None``
    for a Term that represents no table: there is no database to scope it to.

    With *require_key*, a Concept is only created when it gets that key: a
    Concept without one is never a join target, so a Term without a primary key
    gets its Concept later, when an FK first resolves to one of its columns.
    """
    with write_transaction():
        # Concurrent callers for one Term wait here, and the later one then
        # finds the Concept the first created.
        store().query_read(
            select(s.term.c.id).where(s.term.c.id == term_id).with_for_update()
        )
        existing = store().query_read(
            select(s.term__concept.c.concept_id).where(
                s.term__concept.c.term_id == term_id
            )
        )
        if existing:
            return existing[0]["concept_id"]

        rows = store().query_read(
            select(
                s.term.c.name,
                s.term.c.description,
                s.catalog_table.c.id.label("table_id"),
                s.catalog_table.c.pk,
                s.catalog_schema.c.database_id,
            )
            .select_from(
                s.term.join(s.table__term, s.table__term.c.term_id == s.term.c.id)
                .join(s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id)
                .join(
                    s.catalog_schema,
                    s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                )
            )
            .where(s.term.c.id == term_id)
            .order_by(s.catalog_table.c.id)
            .limit(1)
        )
        if not rows:
            return None
        row = rows[0]

        pk = list(row["pk"] or [])
        pk_column_id = None
        if len(pk) == 1:
            columns = store().query_read(
                select(s.catalog_column.c.id).where(
                    s.catalog_column.c.table_id == row["table_id"],
                    s.catalog_column.c.name == pk[0],
                )
            )
            pk_column_id = columns[0]["id"] if columns else None
        if require_key and pk_column_id is None:
            return None

        concept_id = None
        for name in (row["name"], f"{row['name']} ({term_id})"):
            inserted = store().query_write(
                insert(s.concept)
                .values(
                    database_id=row["database_id"],
                    name=name,
                    description=row["description"],
                )
                .on_conflict_do_nothing(constraint="uq_concept_database_name")
                .returning(s.concept.c.id)
            )
            if inserted:
                concept_id = inserted[0]["id"]
                break
        if concept_id is None:
            raise RuntimeError(f"Could not allocate a Concept name for Term {term_id}")
        store().query_write(
            insert(s.term__concept).values(term_id=term_id, concept_id=concept_id)
        )

        key_id = ensure_concept_key(concept_id, pk_column_id) if pk_column_id else None
        if require_key and key_id is None:
            store().query_write(delete(s.concept).where(s.concept.c.id == concept_id))
            return None
        return concept_id


def fetch_column_key_id(column_id: str) -> str | None:
    """The key *column_id* is a member column of, or ``None``."""
    rows = store().query_read(
        select(s.concept_key__column.c.key_id).where(
            s.concept_key__column.c.column_id == column_id
        )
    )
    return rows[0]["key_id"] if rows else None


def ensure_concept_key(concept_id: str, column_id: str) -> str | None:
    """Return the key *column_id* stands for, creating a one-column key if needed.

    A column already serving as a key column keeps its key, which is returned
    only when it belongs to *concept_id*. Otherwise the
    column must belong to one of the Concept's member tables; a new key named
    after the column is created with the column as its member on that table.
    Returns ``None`` when the column is on no member table, or is not the
    table's single-column primary key and was profiled non-unique: a key is an
    identity, and joining two references through a column that repeats would
    relate rows that merely share a value. A date column (:func:`is_date_column`)
    is a key only when declared: the primary key, or a declared FK's target.
    """
    with write_transaction():
        existing = fetch_column_key_id(column_id)
        if existing is not None:
            return existing if fetch_key_concept_id(existing) == concept_id else None
        is_pk = (func.cardinality(s.catalog_table.c.pk) == 1) & (
            s.catalog_table.c.pk[1] == s.catalog_column.c.name
        )
        is_declared_target = (
            select(s.column__foreign_key.c.target_column_id)
            .where(s.column__foreign_key.c.target_column_id == s.catalog_column.c.id)
            .exists()
        )
        columns = store().query_read(
            select(s.catalog_column.c.table_id, s.catalog_column.c.name)
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
            .where(
                s.catalog_column.c.id == column_id,
                _table_concept(s.catalog_column.c.table_id) == concept_id,
                is_pk
                | (
                    s.catalog_column.c.is_unique.isnot(False)
                    & (~is_date_column() | is_declared_target)
                ),
            )
        )
        if not columns:
            return None
        column = columns[0]
        key_id = store().query_write(
            insert(s.concept_key)
            .values(concept_id=concept_id, name=column["name"])
            .returning(s.concept_key.c.id)
        )[0]["id"]
        added = store().query_write(
            insert(s.concept_key__column)
            .values(key_id=key_id, column_id=column_id, table_id=column["table_id"])
            .on_conflict_do_nothing(index_elements=["column_id"])
            .returning(s.concept_key__column.c.key_id)
        )
        if added:
            return key_id
        # Another writer made this column a key column after the read above.
        store().query_write(delete(s.concept_key).where(s.concept_key.c.id == key_id))
        existing = fetch_column_key_id(column_id)
        return existing if fetch_key_concept_id(existing) == concept_id else None


def ensure_column_key(column_id: str) -> str | None:
    """The key for *column_id* on its own table's Concept, created if needed.

    ``None`` when the column's table is represented by no Term, so has no
    Concept for the key to belong to.
    """
    with write_transaction():
        existing = fetch_column_key_id(column_id)
        if existing is not None:
            return existing
        rows = store().query_read(
            select(s.table__term.c.term_id)
            .join(
                s.catalog_column,
                s.catalog_column.c.table_id == s.table__term.c.table_id,
            )
            .where(s.catalog_column.c.id == column_id)
            .order_by(s.table__term.c.term_id)
            .limit(1)
        )
        if not rows:
            return None
        concept_id = ensure_term_concept(rows[0]["term_id"])
        if concept_id is None:
            return None
        return ensure_concept_key(concept_id, column_id)


def _key_tables(key_id: str) -> set[str]:
    return {
        row["table_id"]
        for row in store().query_read(
            select(s.concept_key__column.c.table_id).where(
                s.concept_key__column.c.key_id == key_id
            )
        )
    }


def fetch_key_concept_id(key_id: str) -> str | None:
    """The Concept *key_id* identifies, or ``None`` when there is no such key."""
    rows = store().query_read(
        select(s.concept_key.c.concept_id).where(s.concept_key.c.id == key_id)
    )
    return rows[0]["concept_id"] if rows else None


def merge_concepts(
    survivor_id: str,
    absorbed_id: str,
    *,
    key_pairs: Mapping[str, str] | None = None,
    name: str | None = None,
) -> bool:
    """Fold *absorbed_id* into *survivor_id*, moving rows rather than recreating.

    *key_pairs* maps an absorbed key to the survivor key it is the same
    identity as; the absorbed key's member columns join the survivor key and
    every FK targeting it is re-pointed with an UPDATE, so no cascade can drop a
    resolved FK. A pair is skipped when the survivor key already has a column
    on one of the absorbed key's tables. Unpaired keys move to the survivor as
    they are. *absorbed_id* may equal *survivor_id*, to unify two keys of one
    Concept.

    A member key column is identity, not a reference, so a semantic FK held by
    one and targeting a key of the merged Concept is removed — the satellite
    pattern, where ``customer_ext.customer_id`` was resolved as an FK to
    ``customer`` before the two were known to be one entity.

    *name* renames the survivor unless another Concept of its database holds
    the name, or the survivor is certified. Returns whether the merge was
    applied; ``False`` when either Concept is gone or they span databases.
    """
    with write_transaction():
        concepts = {
            row["id"]: row
            for row in store().query_read(
                select(s.concept.c.id, s.concept.c.database_id, s.concept.c.certified)
                .where(s.concept.c.id.in_({survivor_id, absorbed_id}))
                .with_for_update()
            )
        }
        survivor = concepts.get(survivor_id)
        absorbed = concepts.get(absorbed_id)
        if survivor is None or absorbed is None:
            return False
        if survivor["database_id"] != absorbed["database_id"]:
            logger.warning(
                "merge_concepts: refusing cross-database merge %s <- %s",
                survivor_id,
                absorbed_id,
            )
            return False

        for absorbed_key, survivor_key in (key_pairs or {}).items():
            if absorbed_key == survivor_key:
                continue
            if fetch_key_concept_id(absorbed_key) != absorbed_id:
                continue
            if fetch_key_concept_id(survivor_key) != survivor_id:
                continue
            if _key_tables(survivor_key) & _key_tables(absorbed_key):
                continue
            store().query_write(
                update(s.concept_key__column)
                .where(s.concept_key__column.c.key_id == absorbed_key)
                .values(key_id=survivor_key)
            )
            store().query_write(
                update(s.column__semantic_fk)
                .where(s.column__semantic_fk.c.key_id == absorbed_key)
                .values(key_id=survivor_key)
            )
            store().query_write(
                delete(s.concept_key).where(s.concept_key.c.id == absorbed_key)
            )

        if absorbed_id != survivor_id:
            store().query_write(
                update(s.concept_key)
                .where(s.concept_key.c.concept_id == absorbed_id)
                .values(concept_id=survivor_id)
            )
            store().query_write(
                update(s.term__concept)
                .where(s.term__concept.c.concept_id == absorbed_id)
                .values(concept_id=survivor_id)
            )
            store().query_write(delete(s.concept).where(s.concept.c.id == absorbed_id))

        survivor_keys = select(s.concept_key.c.id).where(
            s.concept_key.c.concept_id == survivor_id
        )
        store().query_write(
            delete(s.column__semantic_fk).where(
                s.column__semantic_fk.c.key_id.in_(survivor_keys),
                s.column__semantic_fk.c.column_id.in_(
                    select(s.concept_key__column.c.column_id).where(
                        s.concept_key__column.c.key_id.in_(survivor_keys)
                    )
                ),
            )
        )

        name = (name or "").strip()
        if name and not survivor["certified"]:
            taken = store().query_read(
                select(s.concept.c.id).where(
                    s.concept.c.database_id == survivor["database_id"],
                    s.concept.c.name == name,
                    s.concept.c.id != survivor_id,
                )
            )
            if not taken:
                store().query_write(
                    update(s.concept)
                    .where(s.concept.c.id == survivor_id)
                    .values(name=name)
                )
        return True


def fetch_term_concepts(term_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The Concept of each Term in *term_ids*, keyed by Term id.

    Each value is ``{id, name, term_ids}``; ``term_ids`` lists every member
    Term, *term_ids* or not, so a reader can tell a one-member Concept from a
    split one. A Term without a Concept is absent.
    """
    ids = list(dict.fromkeys(term_ids))
    if not ids:
        return {}
    member = s.term__concept.alias("member")
    rows = store().query_read(
        select(
            s.term__concept.c.term_id,
            s.concept.c.id,
            s.concept.c.name,
            member.c.term_id.label("member_term_id"),
        )
        .join(s.concept, s.concept.c.id == s.term__concept.c.concept_id)
        .join(member, member.c.concept_id == s.concept.c.id)
        .where(s.term__concept.c.term_id.in_(ids))
        .order_by(s.term__concept.c.term_id, member.c.term_id)
    )
    concepts: dict[str, dict[str, Any]] = {}
    for row in rows:
        concept = concepts.setdefault(
            row["term_id"], {"id": row["id"], "name": row["name"], "term_ids": []}
        )
        concept["term_ids"].append(row["member_term_id"])
    return concepts


def delete_orphan_concepts(database_ids: Iterable[str]) -> int:
    """Delete the Concepts of *database_ids* that no Term belongs to any more.

    Deleting a Term removes its membership but not the Concept, whose name and
    keys would otherwise outlive every Term they described. An FK targeting
    one of its keys goes with it. Returns how many were deleted.
    """
    ids = list(dict.fromkeys(database_ids))
    if not ids:
        return 0
    deleted = store().query_write(
        delete(s.concept)
        .where(
            s.concept.c.database_id.in_(ids),
            ~select(s.term__concept.c.term_id)
            .where(s.term__concept.c.concept_id == s.concept.c.id)
            .exists(),
        )
        .returning(s.concept.c.id)
    )
    return len(deleted)


def delete_keyless_concepts(database_names: Iterable[str]) -> int:
    """Delete the Concepts of *database_names* that have no key.

    Nothing can join to a Concept without a key, so one is only worth keeping
    once a key exists. Concepts are created on demand while a compile looks
    for keys, and this removes the ones it found none for. Returns how many
    were deleted.
    """
    names = list(dict.fromkeys(database_names))
    if not names:
        return 0
    deleted = store().query_write(
        delete(s.concept)
        .where(
            s.concept.c.database_id.in_(
                select(s.catalog_database.c.id).where(
                    s.catalog_database.c.name.in_(names)
                )
            ),
            ~select(s.concept_key.c.id)
            .where(s.concept_key.c.concept_id == s.concept.c.id)
            .exists(),
        )
        .returning(s.concept.c.id)
    )
    return len(deleted)


def fetch_concept_merge_candidates(database_name: str) -> list[dict[str, Any]]:
    """Every single-column key candidate of every Term-represented table.

    One row per table and column that could identify the table's entity: a
    single-column declared primary key, or a column profiled unique.
    ``concept_id`` and ``concept_name`` are ``None`` for a Term that has no
    Concept yet. Rows carry
    ``concept_id``, ``concept_name``, ``term_id``, ``term_name``,
    ``term_description``, ``table_id``, ``table_name``, ``table_description``,
    ``schema_name``, ``column_id``, ``column_name``, ``column_description``,
    ``data_type``, ``distinct_values``, ``is_pk``, ``fk_target_column_id`` (the
    column's declared FK target, if any) and ``family_id`` (``NULL`` outside
    every family).
    """
    is_pk = (func.cardinality(s.catalog_table.c.pk) == 1) & (
        s.catalog_table.c.pk[1] == s.catalog_column.c.name
    )
    family = (
        select(s.term__term_family.c.family_id)
        .where(s.term__term_family.c.term_id == s.term.c.id)
        .limit(1)
        .scalar_subquery()
    )
    declared = (
        select(s.column__foreign_key.c.target_column_id)
        .where(s.column__foreign_key.c.source_column_id == s.catalog_column.c.id)
        .order_by(s.column__foreign_key.c.target_column_id)
        .limit(1)
        .scalar_subquery()
    )
    statement = (
        select(
            s.concept.c.id.label("concept_id"),
            s.concept.c.name.label("concept_name"),
            s.term.c.id.label("term_id"),
            s.term.c.name.label("term_name"),
            s.term.c.description.label("term_description"),
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.description.label("table_description"),
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.description.label("column_description"),
            s.catalog_column.c.data_type,
            s.catalog_column.c.distinct_values,
            is_pk.label("is_pk"),
            declared.label("fk_target_column_id"),
            family.label("family_id"),
        )
        .select_from(
            s.term.outerjoin(s.term__concept, s.term__concept.c.term_id == s.term.c.id)
            .outerjoin(s.concept, s.concept.c.id == s.term__concept.c.concept_id)
            .join(s.table__term, s.table__term.c.term_id == s.term.c.id)
            .join(s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.table_id == s.catalog_table.c.id,
            )
        )
        .where(
            s.catalog_database.c.name == database_name,
            is_pk | s.catalog_column.c.is_unique.is_(True),
        )
        .order_by(s.catalog_table.c.name, s.catalog_column.c.name)
    )
    return [dict(row) for row in store().query_read(statement)]


def fetch_concept_members(table_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The Concept of each table in *table_ids*, keyed by table id.

    Each entry is ``{"concept_id", "concept_name", "members", "keys"}``:
    ``members`` lists every member table of the Concept as ``table_id``,
    ``schema_name`` and ``table_name``; ``keys`` lists every key as ``key_id``,
    ``key_name`` and ``columns`` (``table_id``, ``schema_name``,
    ``table_name``, ``column_name``). A table with no Concept has no entry.
    """
    ids = list(dict.fromkeys(table_ids))
    if not ids:
        return {}
    concept_of = {
        row["table_id"]: row["concept_id"]
        for row in store().query_read(
            select(
                s.table__term.c.table_id,
                func.min(s.term__concept.c.concept_id).label("concept_id"),
            )
            .join(s.term__concept, s.term__concept.c.term_id == s.table__term.c.term_id)
            .where(s.table__term.c.table_id.in_(ids))
            .group_by(s.table__term.c.table_id)
        )
    }
    concept_ids = sorted(set(concept_of.values()))
    if not concept_ids:
        return {}

    member_rows = store().query_read(
        select(
            s.concept.c.id.label("concept_id"),
            s.concept.c.name.label("concept_name"),
            s.catalog_table.c.id.label("table_id"),
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_table.c.name.label("table_name"),
        )
        .select_from(
            s.concept.join(
                s.term__concept, s.term__concept.c.concept_id == s.concept.c.id
            )
            .join(s.table__term, s.table__term.c.term_id == s.term__concept.c.term_id)
            .join(s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.concept.c.id.in_(concept_ids))
        .order_by(s.catalog_schema.c.name, s.catalog_table.c.name)
    )
    key_rows = store().query_read(
        select(
            s.concept_key.c.concept_id,
            s.concept_key.c.id.label("key_id"),
            s.concept_key.c.name.label("key_name"),
            s.concept_key__column.c.table_id,
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_column.c.name.label("column_name"),
        )
        .select_from(
            s.concept_key.join(
                s.concept_key__column,
                s.concept_key__column.c.key_id == s.concept_key.c.id,
            )
            .join(
                s.catalog_column,
                s.catalog_column.c.id == s.concept_key__column.c.column_id,
            )
            .join(
                s.catalog_table,
                s.catalog_table.c.id == s.concept_key__column.c.table_id,
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.concept_key.c.concept_id.in_(concept_ids))
        .order_by(
            s.concept_key.c.name,
            s.concept_key.c.id,
            s.catalog_schema.c.name,
            s.catalog_table.c.name,
        )
    )

    concepts: dict[str, dict[str, Any]] = {}
    for row in member_rows:
        entry = concepts.setdefault(
            row["concept_id"],
            {
                "concept_id": row["concept_id"],
                "concept_name": row["concept_name"],
                "members": [],
                "keys": [],
            },
        )
        entry["members"].append(
            {
                "table_id": row["table_id"],
                "schema_name": row["schema_name"],
                "table_name": row["table_name"],
            }
        )
    keys: dict[str, dict[str, Any]] = {}
    for row in key_rows:
        entry = concepts.get(row["concept_id"])
        if entry is None:
            continue
        key = keys.get(row["key_id"])
        if key is None:
            key = {"key_id": row["key_id"], "key_name": row["key_name"], "columns": []}
            keys[row["key_id"]] = key
            entry["keys"].append(key)
        key["columns"].append(
            {
                "table_id": row["table_id"],
                "schema_name": row["schema_name"],
                "table_name": row["table_name"],
                "column_name": row["column_name"],
            }
        )
    return {
        table_id: concepts[concept_id]
        for table_id, concept_id in concept_of.items()
        if concept_id in concepts
    }


def fetch_concept_fields(concept_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Every field of the given Concepts, with where it comes from.

    A Concept's fields are its member Terms' ColumnAttributes. Each row names
    the Term, the attribute and the physical column it describes —
    ``concept_id``, ``concept_name``, ``term_id``, ``term_name``,
    ``attribute_id``, ``attribute_name``, ``column_id``, ``column_name``,
    ``table_id``, ``table_name``, ``schema_name`` and ``database_name`` — so a
    query using the field addresses ``schema.table.column`` of that member and
    no other.
    """
    ids = list(dict.fromkeys(concept_ids))
    if not ids:
        return []
    statement = (
        select(
            s.concept.c.id.label("concept_id"),
            s.concept.c.name.label("concept_name"),
            s.term.c.id.label("term_id"),
            s.term.c.name.label("term_name"),
            s.column_attribute.c.id.label("attribute_id"),
            s.column_attribute.c.name.label("attribute_name"),
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.concept.join(
                s.term__concept, s.term__concept.c.concept_id == s.concept.c.id
            )
            .join(s.term, s.term.c.id == s.term__concept.c.term_id)
            .join(
                s.column_attribute__term,
                s.column_attribute__term.c.term_id == s.term.c.id,
            )
            .join(
                s.column_attribute,
                s.column_attribute.c.id == s.column_attribute__term.c.attribute_id,
            )
            .join(
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
        )
        .where(s.concept.c.id.in_(ids))
        .order_by(
            s.concept.c.name,
            s.term.c.name,
            s.catalog_column.c.ordinal_position,
            s.column_attribute.c.id,
        )
    )
    return [dict(row) for row in store().query_read(statement)]
