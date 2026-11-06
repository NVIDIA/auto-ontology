# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concepts: the things Terms describe, their keys, and the semantic FKs to them.

A Concept groups one or more Terms that describe the same thing. Its key is
recorded once per member table (``concept__pk``), as an ordered list of
columns of the same length on every table. A semantic FK is a reference from
a table to a concept (``concept__fk``), its columns ordered to match the key
(``concept__fk_column``). A reference therefore joins its table to *every*
member table in the same database, and member tables join each other on
their keys.

Every write that creates a reference goes through :func:`merge_concept_fk`,
which is the one place the invariants are enforced: same key length, one
source table, no legacy table, at least one member table in the source
table's database, and at most one reference per column.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

from sqlalchemy import (
    and_,
    bindparam,
    case,
    delete,
    func,
    literal,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.semantic.constants import SEMANTIC_SOURCE

logger = logging.getLogger(__name__)

_src_table = s.catalog_table.alias("concept_link_src_table")
_src_schema = s.catalog_schema.alias("concept_link_src_schema")
_tgt_table = s.catalog_table.alias("concept_link_tgt_table")
_tgt_schema = s.catalog_schema.alias("concept_link_tgt_schema")

#: Every semantic FK flattened to one row per referencing column and member
#: table it joins: the referencing ``column_id``, the key column it equals on
#: that table (``target_column_id``) and the attribute that key column owns
#: (``attribute_id``, null when it owns none). Only member tables in the
#: source table's own database appear, and never the source table itself.
#:
#: This is the old ``column__semantic_fk`` shape, plus the target. Readers that
#: only ask "which columns reference what" select from it; path finding reads
#: :func:`semantic_table_edges`, which keeps a composite key's columns together.
SEMANTIC_FK_LINKS = (
    select(
        s.concept__fk_column.c.column_id,
        s.concept__fk.c.id.label("fk_id"),
        s.concept__fk.c.concept_id,
        s.concept__fk.c.source_table_id,
        s.concept__fk.c.use_in_join_paths,
        s.concept__fk_column.c.ordinal,
        s.concept__pk.c.column_id.label("target_column_id"),
        s.concept__pk.c.table_id.label("target_table_id"),
        s.column__has_attribute.c.attribute_id,
    )
    .select_from(
        s.concept__fk.join(
            s.concept__fk_column,
            s.concept__fk_column.c.fk_id == s.concept__fk.c.id,
        )
        .join(
            s.concept__pk,
            and_(
                s.concept__pk.c.concept_id == s.concept__fk.c.concept_id,
                s.concept__pk.c.ordinal == s.concept__fk_column.c.ordinal,
                s.concept__pk.c.table_id != s.concept__fk.c.source_table_id,
            ),
        )
        .join(_src_table, _src_table.c.id == s.concept__fk.c.source_table_id)
        .join(_src_schema, _src_schema.c.id == _src_table.c.schema_id)
        .join(_tgt_table, _tgt_table.c.id == s.concept__pk.c.table_id)
        .join(
            _tgt_schema,
            and_(
                _tgt_schema.c.id == _tgt_table.c.schema_id,
                _tgt_schema.c.database_id == _src_schema.c.database_id,
            ),
        )
        .outerjoin(
            s.column__has_attribute,
            s.column__has_attribute.c.column_id == s.concept__pk.c.column_id,
        )
    )
    .where(~_tgt_table.c.is_legacy)
    .subquery("semantic_fk_link")
)


#: :data:`SEMANTIC_FK_LINKS` reduced to ``(column_id, attribute_id,
#: use_in_join_paths)``: a referencing column and the attribute owned by a key
#: column it joins. The column -> attribute shape readers walk when they treat
#: a reference as linking a column to the attribute it points at.
SEMANTIC_FK_ATTRIBUTES = (
    select(
        SEMANTIC_FK_LINKS.c.column_id,
        SEMANTIC_FK_LINKS.c.attribute_id,
        SEMANTIC_FK_LINKS.c.use_in_join_paths,
    )
    .where(SEMANTIC_FK_LINKS.c.attribute_id.is_not(None))
    .distinct()
    .subquery("semantic_fk_attribute")
)


def column_has_semantic_fk(column_id_expr: Any) -> Any:
    """``EXISTS`` clause: the column is part of some semantic FK."""
    return (
        select(literal(1))
        .where(s.concept__fk_column.c.column_id == column_id_expr)
        .exists()
    )


# ---------------------------------------------------------------------------
# Concepts and their terms
# ---------------------------------------------------------------------------


def concept_of_term(term_id: str) -> str | None:
    rows = store().query_read(
        select(s.term__concept.c.concept_id).where(s.term__concept.c.term_id == term_id)
    )
    return rows[0]["concept_id"] if rows else None


def ensure_term_concept(term_id: str) -> str | None:
    """The term's concept, creating a single-term one named after it if needed.

    ``None`` when the term does not exist.
    """
    existing = concept_of_term(term_id)
    if existing:
        return existing
    term_rows = store().query_read(
        select(s.term.c.name, s.term.c.description, s.term.c.source).where(
            s.term.c.id == term_id
        )
    )
    if not term_rows:
        return None
    term = term_rows[0]
    statement = insert(s.concept).values(
        name=term["name"],
        description=term["description"],
        source=term["source"] or SEMANTIC_SOURCE,
    )
    with write_transaction():
        concept_id = store().query_write(
            statement.on_conflict_do_update(
                constraint="uq_concept_name_source",
                set_={
                    "description": func.coalesce(
                        s.concept.c.description, statement.excluded.description
                    )
                },
            ).returning(s.concept.c.id)
        )[0]["id"]
        store().query_write(
            insert(s.term__concept)
            .values(term_id=term_id, concept_id=concept_id)
            .on_conflict_do_nothing()
        )
    return concept_of_term(term_id)


def fetch_concepts(database_name: str | None = None) -> list[dict[str, Any]]:
    """Concepts with their terms and keys, ordered by name.

    Scoped to *database_name* when given: a concept is listed when one of its
    terms describes a table there, and only that database's keys are returned.
    Each entry: ``id``, ``name``, ``description``, ``terms`` (``[{id, name,
    description}]``) and ``keys`` -- ``{table_id: {"table_name",
    "schema_name", "columns": [{"id", "name", "data_type"}]}}`` in key order.
    """
    term_rows = store().query_read(
        select(
            s.concept.c.id,
            s.concept.c.name,
            s.concept.c.description,
            s.term.c.id.label("term_id"),
            s.term.c.name.label("term_name"),
            s.term.c.description.label("term_description"),
        )
        .select_from(
            s.concept.join(
                s.term__concept, s.term__concept.c.concept_id == s.concept.c.id
            ).join(s.term, s.term.c.id == s.term__concept.c.term_id)
        )
        .order_by(s.concept.c.name, s.term.c.name)
    )
    key_statement = (
        select(
            s.concept__pk.c.concept_id,
            s.concept__pk.c.table_id,
            s.concept__pk.c.ordinal,
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.data_type,
            s.catalog_table.c.name.label("table_name"),
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.concept__pk.join(
                s.catalog_column, s.catalog_column.c.id == s.concept__pk.c.column_id
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.concept__pk.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .order_by(s.concept__pk.c.table_id, s.concept__pk.c.ordinal)
    )
    if database_name is not None:
        key_statement = key_statement.where(s.catalog_database.c.name == database_name)
    keys: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in store().query_read(key_statement):
        table = keys[row["concept_id"]].setdefault(
            row["table_id"],
            {
                "table_name": row["table_name"],
                "schema_name": row["schema_name"],
                "columns": [],
            },
        )
        table["columns"].append(
            {
                "id": row["column_id"],
                "name": row["column_name"],
                "data_type": row["data_type"],
            }
        )

    in_scope: set[str] | None = None
    if database_name is not None:
        in_scope = {
            r["concept_id"]
            for r in store().query_read(
                select(s.term__concept.c.concept_id)
                .select_from(
                    s.term__concept.join(
                        s.table__term,
                        s.table__term.c.term_id == s.term__concept.c.term_id,
                    )
                    .join(
                        s.catalog_table,
                        s.catalog_table.c.id == s.table__term.c.table_id,
                    )
                    .join(
                        s.catalog_schema,
                        s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                    )
                    .join(
                        s.catalog_database,
                        s.catalog_database.c.id == s.catalog_schema.c.database_id,
                    )
                )
                .where(s.catalog_database.c.name == database_name)
            )
        }

    concepts: dict[str, dict[str, Any]] = {}
    for row in term_rows:
        if in_scope is not None and row["id"] not in in_scope:
            continue
        concept = concepts.setdefault(
            row["id"],
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "terms": [],
                "keys": keys.get(row["id"], {}),
            },
        )
        concept["terms"].append(
            {
                "id": row["term_id"],
                "name": row["term_name"],
                "description": row["term_description"],
            }
        )
    return list(concepts.values())


def fetch_concept_keys(concept_id: str) -> dict[str, list[str]]:
    """``{table_id: [column_id, ...]}`` in key order."""
    rows = store().query_read(
        select(
            s.concept__pk.c.table_id,
            s.concept__pk.c.column_id,
        )
        .where(s.concept__pk.c.concept_id == concept_id)
        .order_by(s.concept__pk.c.table_id, s.concept__pk.c.ordinal)
    )
    keys: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        keys[row["table_id"]].append(row["column_id"])
    return dict(keys)


def fetch_concepts_for_tables(table_ids: list[str]) -> list[dict[str, Any]]:
    """The concepts that join two or more of *table_ids*, ordered by name.

    Only the keys and references on *table_ids* are returned. A concept is
    listed when it has a key on one of them and touches at least two of them:
    two references alone do not join their tables. Each entry: ``id``,
    ``name``, ``description``, ``keys`` and ``references``, each a list of
    ``{table_id, database_name, schema_name, table_name, columns}`` (column
    names in key order) ordered by schema and table; a reference also carries
    ``use_in_join_paths``.
    """
    ids = list(dict.fromkeys(t for t in table_ids if t))
    if len(ids) < 2:
        return []
    location = (
        s.catalog_table.c.name.label("table_name"),
        s.catalog_schema.c.name.label("schema_name"),
        s.catalog_database.c.name.label("database_name"),
    )

    def located(join: Any, table_id: Any) -> Any:
        return (
            join.join(s.catalog_table, s.catalog_table.c.id == table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )

    key_rows = store().query_read(
        select(
            s.concept__pk.c.concept_id,
            s.concept__pk.c.table_id,
            s.catalog_column.c.name.label("column_name"),
            *location,
        )
        .select_from(
            located(
                s.concept__pk.join(
                    s.catalog_column,
                    s.catalog_column.c.id == s.concept__pk.c.column_id,
                ),
                s.concept__pk.c.table_id,
            )
        )
        .where(s.concept__pk.c.table_id.in_(ids), ~s.catalog_table.c.is_legacy)
        .order_by(s.concept__pk.c.table_id, s.concept__pk.c.ordinal)
    )
    if not key_rows:
        return []
    reference_rows = store().query_read(
        select(
            s.concept__fk.c.id.label("fk_id"),
            s.concept__fk.c.concept_id,
            s.concept__fk.c.source_table_id.label("table_id"),
            s.concept__fk.c.use_in_join_paths,
            s.catalog_column.c.name.label("column_name"),
            *location,
        )
        .select_from(
            located(
                s.concept__fk.join(
                    s.concept__fk_column,
                    s.concept__fk_column.c.fk_id == s.concept__fk.c.id,
                ).join(
                    s.catalog_column,
                    s.catalog_column.c.id == s.concept__fk_column.c.column_id,
                ),
                s.concept__fk.c.source_table_id,
            )
        )
        .where(
            s.concept__fk.c.source_table_id.in_(ids),
            s.concept__fk.c.concept_id.in_({r["concept_id"] for r in key_rows}),
            ~s.catalog_table.c.is_legacy,
        )
        .order_by(s.concept__fk.c.id, s.concept__fk_column.c.ordinal)
    )

    def grouped(rows: list[Any], group_key: str) -> dict[str, list[dict[str, Any]]]:
        entries: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = entries.setdefault(
                row[group_key],
                {
                    "concept_id": row["concept_id"],
                    "table_id": row["table_id"],
                    "database_name": row["database_name"],
                    "schema_name": row["schema_name"],
                    "table_name": row["table_name"],
                    "columns": [],
                    **(
                        {"use_in_join_paths": row["use_in_join_paths"]}
                        if "use_in_join_paths" in row
                        else {}
                    ),
                },
            )
            entry["columns"].append(row["column_name"])
        by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in sorted(
            entries.values(),
            key=lambda e: (e["schema_name"], e["table_name"], e["columns"]),
        ):
            by_concept[entry.pop("concept_id")].append(entry)
        return by_concept

    keys = grouped(
        [{**dict(r), "key_id": (r["concept_id"], r["table_id"])} for r in key_rows],
        "key_id",
    )
    references = grouped([dict(r) for r in reference_rows], "fk_id")
    joining = [
        concept_id
        for concept_id, members in keys.items()
        if len({e["table_id"] for e in members + references.get(concept_id, [])}) >= 2
    ]
    if not joining:
        return []
    concepts = store().query_read(
        select(s.concept.c.id, s.concept.c.name, s.concept.c.description)
        .where(s.concept.c.id.in_(joining))
        .order_by(s.concept.c.name, s.concept.c.id)
    )
    return [
        {
            **dict(concept),
            "keys": keys[concept["id"]],
            "references": references.get(concept["id"], []),
        }
        for concept in concepts
    ]


def _key_length(concept_id: str, *, except_table_id: str | None = None) -> int | None:
    lengths = {
        len(columns)
        for table_id, columns in fetch_concept_keys(concept_id).items()
        if table_id != except_table_id
    }
    return lengths.pop() if lengths else None


def set_concept_key(concept_id: str, table_id: str, column_ids: list[str]) -> bool:
    """Replace the concept's key on *table_id* with *column_ids*, in order.

    Refuses -- writes nothing, returns ``False`` -- when the columns are not
    all on *table_id*, or when the length differs from the concept's key on
    its other tables, or when the concept already has references whose
    length would no longer match.
    """
    if not column_ids or len(set(column_ids)) != len(column_ids):
        return False
    owners = {
        r["id"]: r["table_id"]
        for r in store().query_read(
            select(s.catalog_column.c.id, s.catalog_column.c.table_id).where(
                s.catalog_column.c.id.in_(column_ids)
            )
        )
    }
    if any(owners.get(column_id) != table_id for column_id in column_ids):
        return False
    expected = _key_length(concept_id, except_table_id=table_id)
    if expected is not None and expected != len(column_ids):
        logger.info(
            "set_concept_key: %d column(s) on %s, but concept %s is keyed on %d",
            len(column_ids),
            table_id,
            concept_id,
            expected,
        )
        return False
    if expected is None and _reference_lengths(concept_id) - {len(column_ids)}:
        return False
    with write_transaction():
        store().query_write(
            delete(s.concept__pk).where(
                s.concept__pk.c.concept_id == concept_id,
                s.concept__pk.c.table_id == table_id,
            )
        )
        for ordinal, column_id in enumerate(column_ids):
            store().query_write(
                insert(s.concept__pk).values(
                    concept_id=concept_id,
                    column_id=column_id,
                    table_id=table_id,
                    ordinal=ordinal,
                )
            )
    return True


def _reference_lengths(concept_id: str) -> set[int]:
    rows = store().query_read(
        select(func.count(s.concept__fk_column.c.ordinal).label("n"))
        .select_from(
            s.concept__fk.join(
                s.concept__fk_column,
                s.concept__fk_column.c.fk_id == s.concept__fk.c.id,
            )
        )
        .where(s.concept__fk.c.concept_id == concept_id)
        .group_by(s.concept__fk.c.id)
    )
    return {int(r["n"]) for r in rows}


def merge_concepts(
    keep_id: str,
    drop_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> bool:
    """Fold *drop_id* into *keep_id*: its terms, keys and references move over.

    Refuses when the two keys differ in length, or when both are keyed on the
    same table. The dropped concept is deleted.
    """
    if keep_id == drop_id:
        return False
    keep_keys = fetch_concept_keys(keep_id)
    drop_keys = fetch_concept_keys(drop_id)
    lengths = {len(c) for c in keep_keys.values()} | {
        len(c) for c in drop_keys.values()
    }
    if len(lengths) > 1 or set(keep_keys) & set(drop_keys):
        return False
    with write_transaction():
        store().query_write(
            update(s.term__concept)
            .where(s.term__concept.c.concept_id == drop_id)
            .values(concept_id=keep_id)
        )
        store().query_write(
            update(s.concept__pk)
            .where(s.concept__pk.c.concept_id == drop_id)
            .values(concept_id=keep_id)
        )
        store().query_write(
            update(s.concept__fk)
            .where(s.concept__fk.c.concept_id == drop_id)
            .values(concept_id=keep_id)
        )
        store().query_write(delete(s.concept).where(s.concept.c.id == drop_id))
        if name or description:
            values: dict[str, Any] = {}
            if name:
                values["name"] = name
            if description:
                values["description"] = description
            clash = store().query_read(
                select(s.concept.c.id).where(
                    s.concept.c.name == (name or ""),
                    s.concept.c.id != keep_id,
                )
            )
            if clash:
                values.pop("name", None)
            if values:
                store().query_write(
                    update(s.concept).where(s.concept.c.id == keep_id).values(**values)
                )
    return True


def concept_keys_containing(column_id: str) -> list[dict[str, Any]]:
    """Every concept key *column_id* is part of: ``concept_id``, ``table_id``,
    ``ordinal`` and the whole ``key`` (column ids in order)."""
    rows = store().query_read(
        select(
            s.concept__pk.c.concept_id,
            s.concept__pk.c.table_id,
            s.concept__pk.c.ordinal,
        ).where(s.concept__pk.c.column_id == column_id)
    )
    result = []
    for row in rows:
        key = fetch_concept_keys(row["concept_id"]).get(row["table_id"], [])
        result.append({**dict(row), "key": key})
    return result


# ---------------------------------------------------------------------------
# Semantic FKs
# ---------------------------------------------------------------------------


def _column_context(column_ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = store().query_read(
        select(
            s.catalog_column.c.id,
            s.catalog_column.c.name,
            s.catalog_column.c.table_id,
            s.catalog_table.c.is_legacy,
            s.catalog_schema.c.database_id,
        )
        .select_from(
            s.catalog_column.join(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            ).join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.catalog_column.c.id.in_(list(column_ids)))
    )
    return {r["id"]: dict(r) for r in rows}


def _member_databases(concept_id: str) -> set[str]:
    rows = store().query_read(
        select(s.catalog_schema.c.database_id)
        .select_from(
            s.concept__pk.join(
                s.catalog_table, s.catalog_table.c.id == s.concept__pk.c.table_id
            ).join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.concept__pk.c.concept_id == concept_id, ~s.catalog_table.c.is_legacy)
        .distinct()
    )
    return {r["database_id"] for r in rows}


def merge_concept_fk(concept_id: str, source_column_ids: list[str]) -> str | None:
    """Record that *source_column_ids* reference *concept_id*. Idempotent.

    The columns are in key order. Returns the reference id, or ``None`` when
    it is refused: the length differs from the concept's key, the columns are
    not on one table, that table is legacy, the concept has no member table
    in the same database, or a column already references something else.
    An identical existing reference is returned as is.
    """
    if not source_column_ids or len(set(source_column_ids)) != len(source_column_ids):
        return None
    if _key_length(concept_id) != len(source_column_ids):
        return None
    context = _column_context(source_column_ids)
    if len(context) != len(source_column_ids):
        return None
    tables = {c["table_id"] for c in context.values()}
    if len(tables) != 1:
        return None
    source_table_id = tables.pop()
    first = context[source_column_ids[0]]
    if first["is_legacy"]:
        logger.info("merge_concept_fk: refusing reference from legacy table")
        return None
    if first["database_id"] not in _member_databases(concept_id):
        logger.warning(
            "merge_concept_fk: refusing cross-database reference %s -> concept %s",
            source_column_ids,
            concept_id,
        )
        return None

    existing = store().query_read(
        select(
            s.concept__fk_column.c.column_id,
            s.concept__fk_column.c.ordinal,
            s.concept__fk.c.id,
            s.concept__fk.c.concept_id,
        )
        .select_from(
            s.concept__fk_column.join(
                s.concept__fk, s.concept__fk.c.id == s.concept__fk_column.c.fk_id
            )
        )
        .where(s.concept__fk_column.c.column_id.in_(source_column_ids))
    )
    if existing:
        fk_ids = {r["id"] for r in existing}
        same = (
            len(fk_ids) == 1
            and existing[0]["concept_id"] == concept_id
            and len(existing) == len(source_column_ids)
            and all(source_column_ids[r["ordinal"]] == r["column_id"] for r in existing)
        )
        return fk_ids.pop() if same else None

    with write_transaction():
        fk_id = store().query_write(
            insert(s.concept__fk)
            .values(concept_id=concept_id, source_table_id=source_table_id)
            .returning(s.concept__fk.c.id)
        )[0]["id"]
        for ordinal, column_id in enumerate(source_column_ids):
            store().query_write(
                insert(s.concept__fk_column).values(
                    fk_id=fk_id, ordinal=ordinal, column_id=column_id
                )
            )
    return fk_id


def delete_concept_fk(fk_id: str) -> None:
    """Remove one reference; a no-op when it is not there."""
    store().query_write(delete(s.concept__fk).where(s.concept__fk.c.id == fk_id))


def _candidate_concepts(column_id: str, table_id: str) -> list[str]:
    """Concepts the column could key: its table's terms', then its attributes'."""
    table_terms = select(s.table__term.c.term_id.label("term_id")).where(
        s.table__term.c.table_id == table_id
    )
    attribute_terms = (
        select(s.column_attribute__term.c.term_id.label("term_id"))
        .select_from(
            s.column__has_attribute.join(
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id
                == s.column__has_attribute.c.attribute_id,
            )
        )
        .where(s.column__has_attribute.c.column_id == column_id)
    )
    term_ids = [r["term_id"] for r in store().query_read(table_terms)]
    term_ids += [
        r["term_id"]
        for r in store().query_read(attribute_terms)
        if r["term_id"] not in term_ids
    ]
    concepts: list[str] = []
    for term_id in term_ids:
        concept_id = ensure_term_concept(term_id)
        if concept_id and concept_id not in concepts:
            concepts.append(concept_id)
    return concepts


def link_column_to_key_column(
    source_column_id: str, target_column_id: str
) -> str | None:
    """Reference the concept whose key *target_column_id* is part of.

    The concept key is the columns references point at. A target outside
    every key is accepted only when a concept of its table's term (or of an
    attribute the target owns) has no other member table and no references
    yet: that table is re-keyed onto the target first. For a key of several
    columns, the other columns of the reference are taken from the source
    table -- the column with a declared foreign key to that key column, else
    the one with the same name -- and the reference is refused when any of
    them is missing.
    """
    keys = concept_keys_containing(target_column_id)
    if not keys:
        # A table keyed by its declared primary key keeps that identity: a
        # reference to another of its columns -- a shared status code, say --
        # would otherwise re-key it and join every referrer through it.
        context = _column_context([target_column_id]).get(target_column_id)
        if not context:
            return None
        declared = _declared_key(context["table_id"])
        for concept_id in _candidate_concepts(target_column_id, context["table_id"]):
            members = fetch_concept_keys(concept_id)
            if set(members) - {context["table_id"]} or _reference_lengths(concept_id):
                continue
            if declared and members.get(context["table_id"]) == declared:
                continue
            if set_concept_key(concept_id, context["table_id"], [target_column_id]):
                keys = concept_keys_containing(target_column_id)
                break
    for entry in keys:
        columns = _source_columns_for_key(
            source_column_id, entry["key"], entry["ordinal"]
        )
        if columns is None:
            continue
        fk_id = merge_concept_fk(entry["concept_id"], columns)
        if fk_id:
            return fk_id
    return None


def _declared_key(table_id: str) -> list[str]:
    """Column ids of the table's declared primary key, in order; ``[]`` if none."""
    rows = store().query_read(
        select(s.catalog_table.c.pk).where(s.catalog_table.c.id == table_id)
    )
    names = list((rows[0]["pk"] if rows else None) or [])
    if not names:
        return []
    by_name = {
        r["name"].casefold(): r["id"]
        for r in store().query_read(
            select(s.catalog_column.c.id, s.catalog_column.c.name).where(
                s.catalog_column.c.table_id == table_id
            )
        )
    }
    key = [by_name.get(name.casefold()) for name in names]
    return [c for c in key if c] if all(key) else []


def link_column_to_attribute(source_column_id: str, attribute_id: str) -> str | None:
    """:func:`link_column_to_key_column` for the column owning *attribute_id*."""
    rows = store().query_read(
        select(s.column__has_attribute.c.column_id)
        .where(s.column__has_attribute.c.attribute_id == attribute_id)
        .order_by(s.column__has_attribute.c.column_id)
        .limit(1)
    )
    if not rows:
        return None
    return link_column_to_key_column(source_column_id, rows[0]["column_id"])


def _source_columns_for_key(
    source_column_id: str, key: list[str], ordinal: int
) -> list[str] | None:
    if len(key) == 1:
        return [source_column_id]
    context = _column_context([source_column_id, *key])
    source = context.get(source_column_id)
    if not source:
        return None
    source_columns = store().query_read(
        select(s.catalog_column.c.id, s.catalog_column.c.name).where(
            s.catalog_column.c.table_id == source["table_id"]
        )
    )
    by_name = {r["name"].casefold(): r["id"] for r in source_columns}
    declared = {
        r["target_column_id"]: r["source_column_id"]
        for r in store().query_read(
            select(
                s.column__foreign_key.c.source_column_id,
                s.column__foreign_key.c.target_column_id,
            ).where(
                s.column__foreign_key.c.source_column_id.in_(
                    [r["id"] for r in source_columns]
                ),
                s.column__foreign_key.c.target_column_id.in_(key),
            )
        )
    }
    columns: list[str] = []
    for position, key_column_id in enumerate(key):
        if position == ordinal:
            columns.append(source_column_id)
            continue
        name = (context.get(key_column_id) or {}).get("name") or ""
        match = declared.get(key_column_id) or by_name.get(name.casefold())
        if not match:
            return None
        columns.append(match)
    return columns if len(set(columns)) == len(columns) else None


def fetch_semantic_fk_join_groups(database_name: str) -> list[dict[str, Any]]:
    """Every reference in *database_name*, with the context to choose a default.

    One row per reference. ``source_column_*`` describe its first column and
    ``source_column_names`` all of them; ``target_*`` describe the key on the
    first member table (by schema and table name) in the same database. A
    group is the references from one source table to one concept.
    """
    source_table = s.catalog_table.alias("fk_group_source_table")
    source_schema = s.catalog_schema.alias("fk_group_source_schema")
    source_database = s.catalog_database.alias("fk_group_source_database")
    target_table = s.catalog_table.alias("fk_group_target_table")
    target_schema = s.catalog_schema.alias("fk_group_target_schema")

    references = store().query_read(
        select(
            s.concept__fk.c.id.label("fk_id"),
            s.concept__fk.c.concept_id,
            s.concept__fk.c.use_in_join_paths,
            s.concept.c.name.label("concept_name"),
            s.concept.c.description.label("concept_description"),
            source_table.c.id.label("source_table_id"),
            source_table.c.name.label("source_table_name"),
            source_table.c.description.label("source_table_description"),
            source_schema.c.name.label("source_schema_name"),
            source_schema.c.database_id,
        )
        .select_from(
            s.concept__fk.join(s.concept, s.concept.c.id == s.concept__fk.c.concept_id)
            .join(source_table, source_table.c.id == s.concept__fk.c.source_table_id)
            .join(source_schema, source_schema.c.id == source_table.c.schema_id)
            .join(source_database, source_database.c.id == source_schema.c.database_id)
        )
        .where(source_database.c.name == database_name)
    )
    if not references:
        return []
    fk_ids = [r["fk_id"] for r in references]
    fk_columns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in store().query_read(
        select(
            s.concept__fk_column.c.fk_id,
            s.catalog_column.c.id,
            s.catalog_column.c.name,
            s.catalog_column.c.description,
        )
        .select_from(
            s.concept__fk_column.join(
                s.catalog_column,
                s.catalog_column.c.id == s.concept__fk_column.c.column_id,
            )
        )
        .where(s.concept__fk_column.c.fk_id.in_(fk_ids))
        .order_by(s.concept__fk_column.c.fk_id, s.concept__fk_column.c.ordinal)
    ):
        fk_columns[row["fk_id"]].append(dict(row))

    concept_ids = list({r["concept_id"] for r in references})
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in store().query_read(
        select(
            s.concept__pk.c.concept_id,
            s.concept__pk.c.ordinal,
            target_table.c.id.label("table_id"),
            target_table.c.name.label("table_name"),
            target_table.c.description.label("table_description"),
            target_schema.c.name.label("schema_name"),
            target_schema.c.database_id,
            s.catalog_column.c.id.label("column_id"),
            s.catalog_column.c.name.label("column_name"),
            s.catalog_column.c.description.label("column_description"),
        )
        .select_from(
            s.concept__pk.join(
                target_table, target_table.c.id == s.concept__pk.c.table_id
            )
            .join(target_schema, target_schema.c.id == target_table.c.schema_id)
            .join(s.catalog_column, s.catalog_column.c.id == s.concept__pk.c.column_id)
        )
        .where(s.concept__pk.c.concept_id.in_(concept_ids), ~target_table.c.is_legacy)
        .order_by(
            s.concept__pk.c.concept_id,
            target_schema.c.name,
            target_table.c.name,
            s.concept__pk.c.ordinal,
        )
    ):
        target = targets.setdefault(
            (row["concept_id"], row["database_id"]),
            {
                "target_table_id": row["table_id"],
                "target_table_name": row["table_name"],
                "target_table_description": row["table_description"],
                "target_schema_name": row["schema_name"],
                "target_columns": [],
            },
        )
        if target["target_table_id"] == row["table_id"]:
            target["target_columns"].append(dict(row))

    result = []
    for reference in references:
        columns = fk_columns.get(reference["fk_id"]) or []
        target = targets.get((reference["concept_id"], reference["database_id"]))
        if not columns or not target or not target["target_columns"]:
            continue
        first_target = target["target_columns"][0]
        result.append(
            {
                **{k: v for k, v in reference.items() if k != "database_id"},
                "source_column_id": columns[0]["id"],
                "source_column_name": columns[0]["name"],
                "source_column_description": columns[0]["description"],
                "source_column_ids": [c["id"] for c in columns],
                "source_column_names": [c["name"] for c in columns],
                "target_table_id": target["target_table_id"],
                "target_table_name": target["target_table_name"],
                "target_table_description": target["target_table_description"],
                "target_schema_name": target["target_schema_name"],
                "target_column_id": first_target["column_id"],
                "target_column_name": first_target["column_name"],
                "target_column_description": first_target["column_description"],
                "target_column_names": [
                    c["column_name"] for c in target["target_columns"]
                ],
            }
        )
    result.sort(
        key=lambda r: (
            r["source_table_id"],
            r["concept_id"],
            str(r["source_column_name"]).casefold(),
            r["source_column_id"],
        )
    )
    return result


def set_semantic_fk_join_default(
    source_table_id: str, concept_id: str, selected_fk_id: str
) -> None:
    """Atomically make *selected_fk_id* the one reference used in join paths
    among the references from *source_table_id* to *concept_id*."""
    statement = (
        update(s.concept__fk)
        .where(
            s.concept__fk.c.source_table_id == source_table_id,
            s.concept__fk.c.concept_id == concept_id,
        )
        .values(
            use_in_join_paths=case(
                (s.concept__fk.c.id == selected_fk_id, True), else_=False
            )
        )
        .returning(s.concept__fk.c.id)
    )
    with write_transaction():
        updated = store().query_write(statement)
        if selected_fk_id not in {row["id"] for row in updated}:
            raise ValueError(
                "Selected semantic FK does not belong to the requested group"
            )


# ---------------------------------------------------------------------------
# Join edges
# ---------------------------------------------------------------------------

_edge_table = s.catalog_table.alias("concept_edge_table")
_edge_schema = s.catalog_schema.alias("concept_edge_schema")

_MEMBER_KEYS = (
    select(
        s.concept__pk.c.concept_id,
        s.concept__pk.c.table_id,
        s.concept__pk.c.ordinal,
        s.concept__pk.c.column_id,
    )
    .select_from(
        s.concept__pk.join(
            _edge_table, _edge_table.c.id == s.concept__pk.c.table_id
        ).join(_edge_schema, _edge_schema.c.id == _edge_table.c.schema_id)
    )
    .where(
        _edge_schema.c.database_id == bindparam("database_id"),
        ~_edge_table.c.is_legacy,
    )
    .order_by(
        s.concept__pk.c.concept_id, s.concept__pk.c.table_id, s.concept__pk.c.ordinal
    )
)

_fk_edge_table = s.catalog_table.alias("concept_fk_edge_table")
_fk_edge_schema = s.catalog_schema.alias("concept_fk_edge_schema")

_REFERENCES = (
    select(
        s.concept__fk.c.id.label("fk_id"),
        s.concept__fk.c.concept_id,
        s.concept__fk.c.source_table_id,
        s.concept__fk.c.use_in_join_paths,
        s.concept__fk_column.c.ordinal,
        s.concept__fk_column.c.column_id,
    )
    .select_from(
        s.concept__fk.join(
            s.concept__fk_column, s.concept__fk_column.c.fk_id == s.concept__fk.c.id
        )
        .join(_fk_edge_table, _fk_edge_table.c.id == s.concept__fk.c.source_table_id)
        .join(_fk_edge_schema, _fk_edge_schema.c.id == _fk_edge_table.c.schema_id)
    )
    .where(
        _fk_edge_schema.c.database_id == bindparam("database_id"),
        ~_fk_edge_table.c.is_legacy,
    )
    .order_by(s.concept__fk.c.id, s.concept__fk_column.c.ordinal)
)


def semantic_table_edges(
    database_id: str, *, include_disabled: bool = False
) -> list[dict[str, Any]]:
    """Every table-to-table join the concepts define in one database.

    Each edge: ``src_table_id``, ``dst_table_id``, ``pairs`` (a tuple of
    ``(src_column_id, dst_column_id)`` in key order), ``kind`` and
    ``concept_id``, plus ``fk_id`` for a reference. Kinds:

    * ``"fk"`` -- a reference to a member table. Emitted once in each
      direction (the reverse with ``reverse=True``), because the key is the
      member table's identity, so the join is genuine both ways.
    * ``"member"`` -- two member tables of one concept, joined on their keys.

    Two references to the same concept never produce an edge between their
    tables: they merely point at the same thing. References with
    ``use_in_join_paths`` off are left out unless *include_disabled*.
    """
    params = {"database_id": database_id}
    keys: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in store().query_read(_MEMBER_KEYS, params):
        keys[row["concept_id"]][row["table_id"]].append(row["column_id"])

    edges: list[dict[str, Any]] = []
    for concept_id, members in keys.items():
        tables = sorted(members)
        for i, left in enumerate(tables):
            for right in tables[i + 1 :]:
                if len(members[left]) != len(members[right]):
                    continue
                for src, dst in ((left, right), (right, left)):
                    edges.append(
                        {
                            "src_table_id": src,
                            "dst_table_id": dst,
                            "pairs": tuple(zip(members[src], members[dst])),
                            "kind": "member",
                            "concept_id": concept_id,
                        }
                    )

    references: dict[str, dict[str, Any]] = {}
    for row in store().query_read(_REFERENCES, params):
        if not include_disabled and not row["use_in_join_paths"]:
            continue
        reference = references.setdefault(
            row["fk_id"],
            {
                "concept_id": row["concept_id"],
                "source_table_id": row["source_table_id"],
                "use_in_join_paths": row["use_in_join_paths"],
                "columns": [],
            },
        )
        reference["columns"].append(row["column_id"])
    for fk_id, reference in references.items():
        source = reference["source_table_id"]
        for target, key in sorted(keys.get(reference["concept_id"], {}).items()):
            if target == source or len(key) != len(reference["columns"]):
                continue
            pairs = tuple(zip(reference["columns"], key))
            common = {
                "kind": "fk",
                "concept_id": reference["concept_id"],
                "fk_id": fk_id,
                "use_in_join_paths": reference["use_in_join_paths"],
            }
            edges.append(
                {
                    "src_table_id": source,
                    "dst_table_id": target,
                    "pairs": pairs,
                    "reverse": False,
                    **common,
                }
            )
            edges.append(
                {
                    "src_table_id": target,
                    "dst_table_id": source,
                    "pairs": tuple((b, a) for a, b in pairs),
                    "reverse": True,
                    **common,
                }
            )
    return edges
