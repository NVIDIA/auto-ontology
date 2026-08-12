# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSF model YAML export and import.

Conceptually the simplest of the large modules — CRUD over entities that all
exist by now — and the plan asked for the simplifications to be **banked rather
than ported**. Three were:

* **``_ensure_import_indexes`` is gone.** It created uniqueness constraints and
  ``imported_id`` indexes on every import, because a schemaless store had no
  other way to guarantee them and an unindexed ``imported_id`` turned a
  several-thousand-column import into a quadratic crawl. The schema declares
  them once; there is nothing to ensure.
* **The split transaction collapses into one.** SQL attributes and custom
  analyses used to be applied *outside* the main transaction: persisting them
  calls ``add_query``, which in the Neo4j build opened its own auto-commit
  session and would deadlock against locks the outer transaction held. The
  Postgres ``add_query`` runs on the same connection, so the whole import is one
  atomic unit — a failure part-way now leaves nothing behind, where before it
  could leave a catalog with no semantics on top.
* **Bug 3 is fixed** (see :func:`_is_nullable`).

``imported_id`` is the mechanism the whole import turns on: entities are matched
by the YAML ``id`` against ``imported_id`` *or* the live ``id``, so re-importing
a document is a no-op and importing onto an existing catalog adopts it rather
than duplicating it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import func, literal, select, update
from sqlalchemy.dialects.postgresql import insert

from gsf.catalog.constants import Props
from gsf.catalog.model import CatalogNode
from gsf.catalog.store.queries import add_query
from gsf.dal import schema as s
from gsf.dal.custom_analyses import (
    detach_existing_sql_edges as detach_ca_sql_edges,
)
from gsf.dal.session import store, write_transaction
from gsf.dal.sql_attributes import detach_existing_sql_edges, link_to_term
from gsf.semantic.constants import (
    SEMANTIC_SOURCE,
    SQL_ATTR_SOURCE_BRIDGE,
    SQL_ATTR_SOURCE_MANUAL,
    SQL_ATTR_SOURCE_SQL,
    SQL_ATTR_SOURCE_TABLE,
)
from gsf.server.model_interchange.embed import (
    ColumnCatalogMeta,
    ImportEmbedBuffer,
    build_column_attribute_semantic_rows,
    build_column_data_row,
    build_custom_analysis_semantic_row,
    build_sql_attribute_semantic_row,
    build_table_data_row,
    build_term_semantic_rows,
)
from gsf.server.model_interchange.schemas import (
    GsfModelDocument,
    ModelColumn,
    ModelColumnAttribute,
    ModelCustomAnalysis,
    ModelDatabase,
    ModelDataLayer,
    ModelForeignKey,
    ModelJoin,
    ModelSchema,
    ModelSemanticFk,
    ModelSemanticLayer,
    ModelSqlAttribute,
    ModelSqlAttributesBySource,
    ModelTable,
    ModelTerm,
)
from gsf.server.sql_utils import get_dialects, get_schemas, validate_sql
from gsf.utils.sample_values import parse_sample_values

logger = logging.getLogger(__name__)

_SOURCE_TO_YAML_KEY: dict[str, str] = {
    SQL_ATTR_SOURCE_MANUAL: "manual",
    SQL_ATTR_SOURCE_TABLE: "table",
    SQL_ATTR_SOURCE_SQL: "sql",
    SQL_ATTR_SOURCE_BRIDGE: "bridge_table",
}

_YAML_KEY_TO_SOURCE: dict[str, str] = {v: k for k, v in _SOURCE_TO_YAML_KEY.items()}

_YAML_SOURCE_KEYS = ("manual", "table", "sql", "bridge_table")


class ModelInterchangeError(Exception):
    """Base error for model interchange operations."""


class UnknownDatabaseIdsError(ModelInterchangeError):
    """Raised when one or more requested database ids do not exist."""

    def __init__(self, database_ids: list[str]) -> None:
        self.database_ids = database_ids
        super().__init__(f"Unknown database id(s): {', '.join(database_ids)}")


class ModelImportValidationError(ModelInterchangeError):
    """Raised when an import payload references ids outside the export scope."""


# ---------------------------------------------------------------------------
# Nullability — bug 3
# ---------------------------------------------------------------------------


def _is_nullable(raw: Any) -> bool:
    """Whether a column is nullable, from what the catalog actually stores.

    **This is the fix for bug 3.** The catalog stores the *strings* ``'YES'``
    and ``'NO'`` — the values ``information_schema`` reports — and the previous
    reader was ``bool(raw)``. ``bool('NO')`` is ``True``, so **every column in
    every export claimed to be nullable**: measured on the fixture, 112 of 218
    columns were wrong, and the export is what another deployment imports as
    truth.

    Nothing failed, because a bool is exactly what the schema expects and
    ``True`` is a plausible value for it. The only way to see it was to look at
    what was stored rather than at what was read.

    Absent still means nullable — the permissive default the catalog has always
    used for a column it could not determine. Booleans pass through so an
    already-parsed value round-trips, which is what an import writes back.
    """
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().upper() not in {"NO", "FALSE", "0", "F", "N"}


def _nullable_to_stored(value: bool | None) -> str:
    """The inverse of :func:`_is_nullable`, for the import path.

    Written back in ``information_schema``'s vocabulary rather than as a
    boolean, so an imported column is indistinguishable from an ingested one —
    otherwise a re-ingest diff would see every imported column as changed.
    """
    return "YES" if value is None or value else "NO"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _database_scope(database_ids: list[str]):
    """An empty id list means **every** database, not none.

    The Cypher's ``size($database_ids) = 0 OR ...``. Reading it the other way
    would silently export an empty document.
    """
    if not database_ids:
        return literal(True)
    return s.catalog_database.c.id.in_(list(database_ids))


def _catalog_join():
    return (
        s.catalog_database.join(
            s.catalog_schema, s.catalog_schema.c.database_id == s.catalog_database.c.id
        )
        .join(s.catalog_table, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
        .join(s.catalog_column, s.catalog_column.c.table_id == s.catalog_table.c.id)
    )


def validate_database_ids(database_ids: list[str]) -> None:
    """Raise :class:`UnknownDatabaseIdsError` when any id is missing."""
    if not database_ids:
        return
    known = {row["id"] for row in store().query_read(select(s.catalog_database.c.id))}
    unknown = [db_id for db_id in database_ids if db_id not in known]
    if unknown:
        raise UnknownDatabaseIdsError(unknown)


def _export_catalog(database_ids: list[str]) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in store().query_read(
            select(
                s.catalog_database.c.id.label("db_id"),
                s.catalog_database.c.name.label("db_name"),
                s.catalog_schema.c.id.label("schema_id"),
                s.catalog_schema.c.name.label("schema_name"),
                s.catalog_table.c.id.label("table_id"),
                s.catalog_table.c.name.label("table_name"),
                s.catalog_table.c.description.label("table_description"),
                s.catalog_table.c.pk,
                s.catalog_table.c.table_type,
                s.catalog_column.c.id.label("column_id"),
                s.catalog_column.c.name.label("column_name"),
                s.catalog_column.c.description.label("column_description"),
                s.catalog_column.c.data_type.label("column_type"),
                s.catalog_column.c.sample_values,
                s.catalog_column.c.is_unique,
                s.catalog_column.c.is_nullable,
                s.catalog_column.c.ordinal_position,
            )
            .select_from(_catalog_join())
            .where(_database_scope(database_ids))
            .order_by(
                s.catalog_database.c.name,
                s.catalog_schema.c.name,
                s.catalog_table.c.name,
                s.catalog_column.c.ordinal_position,
            )
        )
    ]


def _export_foreign_keys(database_ids: list[str]) -> list[dict[str, Any]]:
    target = s.catalog_column.alias("fk_target")
    return [
        dict(r)
        for r in store().query_read(
            select(
                s.catalog_column.c.id.label("source_column_id"),
                target.c.id.label("target_column_id"),
            )
            .select_from(
                _catalog_join()
                .join(
                    s.column_foreign_key,
                    s.column_foreign_key.c.source_column_id == s.catalog_column.c.id,
                )
                .join(target, target.c.id == s.column_foreign_key.c.target_column_id)
            )
            .where(_database_scope(database_ids))
            .distinct()
        )
    ]


def _export_joins(database_ids: list[str]) -> list[dict[str, Any]]:
    """Joins whose **both** ends are in scope.

    The Cypher required the target table to belong to a scoped database too, or
    an export would carry a join pointing at a table the document does not
    contain — which the importer then cannot resolve.
    """
    target_table = s.catalog_table.alias("join_target")
    target_schema = s.catalog_schema.alias("join_target_schema")
    target_database = s.catalog_database.alias("join_target_database")

    statement = (
        select(
            s.table_join.c.source_table_id,
            s.table_join.c.target_table_id,
            s.table_join.c.join_columns,
        )
        .select_from(
            s.catalog_database.join(
                s.catalog_schema,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
            .join(s.catalog_table, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
            .join(s.table_join, s.table_join.c.source_table_id == s.catalog_table.c.id)
            .join(target_table, target_table.c.id == s.table_join.c.target_table_id)
            .join(target_schema, target_schema.c.id == target_table.c.schema_id)
            .join(target_database, target_database.c.id == target_schema.c.database_id)
        )
        .where(_database_scope(database_ids))
        .distinct()
    )
    if database_ids:
        statement = statement.where(target_database.c.id.in_(list(database_ids)))
    return [dict(r) for r in store().query_read(statement)]


def _export_terms(database_ids: list[str]) -> list[dict[str, Any]]:
    """Terms represented by an in-scope table, with **all** their represents.

    A term reached through one scoped table exports every table representing it,
    scoped or not — that is what the Cypher's second, unscoped ``OPTIONAL
    MATCH`` did, and it keeps a partial export honest about a term it only
    partly owns.
    """
    in_scope = (
        select(s.table_term.c.term_id)
        .select_from(
            s.table_term.join(
                s.catalog_table, s.catalog_table.c.id == s.table_term.c.table_id
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(_database_scope(database_ids))
        .distinct()
    )
    terms = store().query_read(
        select(s.term.c.id, s.term.c.name, s.term.c.description)
        .where(s.term.c.id.in_(in_scope), s.term.c.source == SEMANTIC_SOURCE)
        .order_by(s.term.c.name)
    )
    term_ids = [row["id"] for row in terms]
    if not term_ids:
        return []

    represents: dict[str, list[str]] = {}
    for row in store().query_read(
        select(s.table_term.c.term_id, s.table_term.c.table_id)
        .where(s.table_term.c.term_id.in_(term_ids))
        .order_by(s.table_term.c.table_id)
    ):
        represents.setdefault(row["term_id"], []).append(row["table_id"])

    attributes: dict[str, list[dict[str, Any]]] = {}
    for row in store().query_read(
        select(
            s.column_attribute_term.c.term_id,
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column_has_attribute.c.column_id,
        )
        .select_from(
            s.column_attribute_term.join(
                s.column_attribute,
                s.column_attribute.c.id == s.column_attribute_term.c.attribute_id,
            ).join(
                s.column_has_attribute,
                s.column_has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
        )
        .where(s.column_attribute_term.c.term_id.in_(term_ids))
        .distinct()
        .order_by(s.column_attribute.c.id)
    ):
        attributes.setdefault(row["term_id"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"] or "",
                "column_id": row["column_id"],
            }
        )

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"] or "",
            "represents": represents.get(row["id"], []),
            "columns_attributes": attributes.get(row["id"], []),
        }
        for row in terms
    ]


def _export_semantic_fks(database_ids: list[str]) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in store().query_read(
            select(
                s.column_semantic_fk.c.column_id,
                s.column_semantic_fk.c.attribute_id.label("column_attribute_id"),
            )
            .select_from(
                _catalog_join().join(
                    s.column_semantic_fk,
                    s.column_semantic_fk.c.column_id == s.catalog_column.c.id,
                )
            )
            .where(_database_scope(database_ids))
            .distinct()
        )
    ]


def _export_sql_owners(database_ids: list[str], link_table, owner_table, owner_column):
    """The statement and database behind each SqlAttribute/CustomAnalysis."""
    return (
        select(
            owner_table.c.id,
            owner_table.c.name,
            owner_table.c.description,
            s.sql_query.c.sql_full_query.label("sql"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.catalog_database.join(
                s.catalog_schema,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
            .join(s.catalog_table, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
            .join(
                s.sql_query_table,
                s.sql_query_table.c.table_id == s.catalog_table.c.id,
            )
            .join(s.sql_query, s.sql_query.c.id == s.sql_query_table.c.sql_query_id)
            .join(link_table, link_table.c.sql_query_id == s.sql_query.c.id)
            .join(owner_table, owner_table.c.id == owner_column)
        )
        .where(_database_scope(database_ids))
        .distinct()
    )


def _export_sql_attributes(database_ids: list[str]) -> list[dict[str, Any]]:
    statement = _export_sql_owners(
        database_ids,
        s.sql_attribute_sql,
        s.sql_attribute,
        s.sql_attribute_sql.c.attribute_id,
    )
    statement = (
        statement.add_columns(
            s.sql_attribute.c.expression,
            s.sql_attribute.c.source,
            s.term.c.id.label("term_id"),
        )
        .join(
            s.sql_attribute_term,
            s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
        )
        .join(s.term, s.term.c.id == s.sql_attribute_term.c.term_id)
        .order_by(s.sql_attribute.c.name)
    )
    return [
        {
            **dict(row),
            "description": row["description"] or "",
            "expression": row["expression"] or "",
            "source": row["source"] or SQL_ATTR_SOURCE_MANUAL,
        }
        for row in store().query_read(statement)
    ]


def _export_custom_analyses(database_ids: list[str]) -> list[dict[str, Any]]:
    statement = _export_sql_owners(
        database_ids,
        s.custom_analysis_sql,
        s.custom_analysis,
        s.custom_analysis_sql.c.analysis_id,
    ).order_by(s.custom_analysis.c.name)
    return [
        {**dict(row), "description": row["description"] or ""}
        for row in store().query_read(statement)
    ]


def fetch_export_rows(database_ids: list[str]) -> dict[str, Any]:
    """The raw rows :func:`assemble_export_document` turns into a document."""
    return {
        "catalog": _export_catalog(database_ids),
        "foreign_keys": _export_foreign_keys(database_ids),
        "joins": _export_joins(database_ids),
        "terms": _export_terms(database_ids),
        "semantic_fks": _export_semantic_fks(database_ids),
        "sql_attributes": _export_sql_attributes(database_ids),
        "custom_analyses": _export_custom_analyses(database_ids),
    }


def assemble_export_document(
    rows: dict[str, Any],
    *,
    dialect_by_db_name: dict[str, str],
    sql_column_resolver: Any,
) -> GsfModelDocument:
    """Build a validated document from raw export rows."""
    databases = _assemble_databases(rows["catalog"], dialect_by_db_name)
    foreign_keys = [
        ModelForeignKey(
            source_column_id=str(row["source_column_id"]),
            target_column_id=str(row["target_column_id"]),
        )
        for row in rows["foreign_keys"]
        if row.get("source_column_id") and row.get("target_column_id")
    ]
    joins = [
        ModelJoin(
            source_table_id=str(row["source_table_id"]),
            target_table_id=str(row["target_table_id"]),
            join_columns=row.get("join_columns") or [],
        )
        for row in rows["joins"]
        if row.get("source_table_id") and row.get("target_table_id")
    ]
    terms = [
        ModelTerm(
            id=str(row["id"]),
            name=row.get("name") or "",
            description=row.get("description") or "",
            represents=[str(x) for x in (row.get("represents") or []) if x],
            columns_attributes=[
                ModelColumnAttribute(
                    id=str(attr["id"]),
                    name=attr.get("name") or "",
                    description=attr.get("description") or "",
                    column_id=str(attr.get("column_id") or ""),
                )
                for attr in (row.get("columns_attributes") or [])
                if attr.get("id")
            ],
        )
        for row in rows["terms"]
        if row.get("id")
    ]
    semantic_fks = [
        ModelSemanticFk(
            column_attribute_id=str(row["column_attribute_id"]),
            column_id=str(row["column_id"]),
        )
        for row in rows["semantic_fks"]
        if row.get("column_attribute_id") and row.get("column_id")
    ]
    sql_attributes = _assemble_sql_attributes(
        rows["sql_attributes"], sql_column_resolver
    )
    custom_analyses = [
        ModelCustomAnalysis(
            id=str(row["id"]),
            name=row.get("name") or "",
            description=row.get("description") or "",
            sql=row.get("sql") or "",
            sql_column_is=sql_column_resolver(
                row.get("sql") or "", row.get("database_name")
            ),
        )
        for row in rows["custom_analyses"]
        if row.get("id")
    ]
    return GsfModelDocument(
        data_layer=ModelDataLayer(
            databases=databases, foreign_keys=foreign_keys, joins=joins
        ),
        semantic_layer=ModelSemanticLayer(
            terms=terms,
            semantic_fks=semantic_fks,
            sql_attributes=sql_attributes,
            custom_analyses=custom_analyses,
        ),
        zones=[],
    )


def _assemble_databases(
    catalog_rows: list[dict[str, Any]],
    dialect_by_db_name: dict[str, str],
) -> list[ModelDatabase]:
    db_map: dict[str, dict[str, Any]] = {}
    for row in catalog_rows:
        db_id = str(row["db_id"])
        db_entry = db_map.setdefault(
            db_id,
            {
                "id": db_id,
                "dialect": dialect_by_db_name.get(row.get("db_name") or "", ""),
                "schemas": {},
            },
        )
        schema_id = str(row["schema_id"])
        schema_entry = db_entry["schemas"].setdefault(
            schema_id,
            {
                "id": schema_id,
                "name": row.get("schema_name") or "",
                "database_name": row.get("db_name") or "",
                "tables": {},
            },
        )
        table_id = str(row["table_id"])
        table_entry = schema_entry["tables"].setdefault(
            table_id,
            {
                "id": table_id,
                "name": row.get("table_name") or "",
                "description": row.get("table_description") or "",
                "pk": row.get("pk") or [],
                "type": row.get("table_type") or "",
                "columns": [],
            },
        )
        if row.get("column_id"):
            table_entry["columns"].append(
                ModelColumn(
                    id=str(row["column_id"]),
                    name=row.get("column_name") or "",
                    description=row.get("column_description") or "",
                    type=row.get("column_type") or "",
                    sample_values=parse_sample_values(row.get("sample_values")) or [],
                    is_nullable=_is_nullable(row.get("is_nullable")),
                    is_unique=bool(row.get("is_unique") or False),
                ),
            )

    databases: list[ModelDatabase] = []
    for db_entry in db_map.values():
        schemas = [
            ModelSchema(
                id=schema_entry["id"],
                name=schema_entry["name"],
                database_name=schema_entry["database_name"],
                tables=[
                    ModelTable(**table_entry)
                    for table_entry in schema_entry["tables"].values()
                ],
            )
            for schema_entry in db_entry["schemas"].values()
        ]
        databases.append(
            ModelDatabase(
                id=db_entry["id"], dialect=db_entry["dialect"], schemas=schemas
            )
        )
    databases.sort(key=lambda db: db.id)
    return databases


def _assemble_sql_attributes(
    rows: list[dict[str, Any]],
    sql_column_resolver: Any,
) -> ModelSqlAttributesBySource:
    grouped: dict[str, list[ModelSqlAttribute]] = {key: [] for key in _YAML_SOURCE_KEYS}
    for row in rows:
        yaml_key = _SOURCE_TO_YAML_KEY.get(row.get("source") or "", "manual")
        sql_text = row.get("sql") or row.get("expression") or ""
        grouped[yaml_key].append(
            ModelSqlAttribute(
                id=str(row["id"]),
                name=row.get("name") or "",
                description=row.get("description") or "",
                sql=sql_text,
                sql_column_is=sql_column_resolver(sql_text, row.get("database_name")),
                term_id=str(row.get("term_id") or ""),
            ),
        )
    return ModelSqlAttributesBySource(**grouped)


def resolve_sql_column_ids(sql: str, database_name: str | None) -> list[str]:
    """Parse SQL against the scoped catalog and return referenced column ids."""
    if not sql.strip():
        return []
    try:
        query_obj = validate_sql(
            sql, get_dialects(database_name), get_schemas(database_name)
        )
    except Exception:
        logger.debug("Could not resolve sql_column_is for SQL snippet", exc_info=True)
        return []
    return [str(col_id) for col_id in query_obj.get_column_ids() if col_id]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _resolve_entities_batch(
    table,
    items: list[tuple[str, dict[str, Any]]],
) -> dict[str, tuple[str, bool]]:
    """``{yaml_id: (live_id, created)}`` for a whole same-table batch.

    Matches on ``imported_id`` **or** the live ``id`` — the second is what makes
    the first import onto an existing catalog adopt it rather than duplicate it.
    A row found this way has its ``imported_id`` stamped if it had none, so the
    *next* import matches it the fast way.

    Three statements regardless of batch size. The Neo4j version needed the same
    batching to avoid two round trips per entity; here it also means one
    ``INSERT ... RETURNING`` rather than a create per row.
    """
    if not items:
        return {}

    # Duplicate YAML ids within one batch: keep the first create_props, which is
    # what repeated sequential calls did -- the first creates, the rest find.
    deduped: dict[str, dict[str, Any]] = {}
    for imported_id, create_props in items:
        deduped.setdefault(imported_id, create_props)
    imported_ids = list(deduped)

    existing: dict[str, str] = {}
    for row in store().query_read(
        select(table.c.id, table.c.imported_id).where(
            table.c.imported_id.in_(imported_ids) | table.c.id.in_(imported_ids)
        )
    ):
        # A row can match by either column. Its own id wins only when nothing
        # already claimed that payload id by imported_id, so a row explicitly
        # stamped by a previous import is never shadowed by an id collision.
        if row["imported_id"] in deduped:
            existing[row["imported_id"]] = row["id"]
        elif row["id"] in deduped:
            existing.setdefault(row["id"], row["id"])

    result: dict[str, tuple[str, bool]] = {}
    to_create: list[tuple[str, dict[str, Any]]] = []
    for imported_id, create_props in deduped.items():
        live_id = existing.get(imported_id)
        if live_id is None:
            to_create.append((imported_id, dict(create_props or {})))
            continue
        result[imported_id] = (live_id, False)
        # Stamp it so the *next* import matches by imported_id directly. Only
        # when unset -- overwriting would relabel a row another payload owns.
        store().query_write(
            update(table)
            .where(table.c.id == live_id, table.c.imported_id.is_(None))
            .values(imported_id=imported_id)
        )

    if to_create:
        rows = store().query_write(
            insert(table)
            .values(
                [
                    {**props, "imported_id": imported_id}
                    for imported_id, props in to_create
                ]
            )
            .returning(table.c.id, table.c.imported_id)
        )
        for row in rows:
            result[row["imported_id"]] = (row["id"], True)

    return result


def _link(table, rows: list[dict[str, Any]]) -> None:
    """Idempotent link-table insert; ``[]`` is a no-op, not an empty INSERT."""
    if not rows:
        return
    store().query_write(insert(table).values(rows).on_conflict_do_nothing())


def _payload_identity(table):
    """``coalesce(imported_id, id)`` — the identity an import payload names.

    A row created by an earlier import is named by its ``imported_id``; a row
    that predates any import is named by its own id. Matching on the coalesce
    covers both, which is what lets ``replace`` recognise an entity it wrote
    last time and one it merely adopted.
    """
    return func.coalesce(table.c.imported_id, table.c.id)


def _remap(id_map: dict[str, str], yaml_id: str, *, kind: str) -> str:
    live_id = id_map.get(yaml_id)
    if not live_id:
        raise ModelImportValidationError(
            f"Cannot resolve {kind} id {yaml_id!r} — missing from catalog/semantic import",
        )
    return live_id


def _database_names_for_terms(term_ids: list[str]) -> dict[str, str]:
    if not term_ids:
        return {}
    names: dict[str, str] = {}
    for row in store().query_read(
        select(
            s.table_term.c.term_id,
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.table_term.join(
                s.catalog_table, s.catalog_table.c.id == s.table_term.c.table_id
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.table_term.c.term_id.in_(term_ids))
        .order_by(s.catalog_database.c.id)
    ):
        names.setdefault(row["term_id"], row["database_name"])
    return names


def _database_names_for_columns(column_ids: list[str]) -> dict[str, str]:
    if not column_ids:
        return {}
    names: dict[str, str] = {}
    for row in store().query_read(
        select(
            s.catalog_column.c.id.label("column_id"),
            s.catalog_database.c.name.label("database_name"),
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
        )
        .where(s.catalog_column.c.id.in_(column_ids))
        .order_by(s.catalog_database.c.id)
    ):
        names.setdefault(row["column_id"], row["database_name"])
    return names


def apply_import_model(
    document: GsfModelDocument,
    *,
    replace: bool,
    embed_buffer: ImportEmbedBuffer | None = None,
) -> dict[str, Any]:
    """Apply a validated model document.

    Entities are matched by ``imported_id`` (the YAML ``id``) or by a live ``id``
    equal to it, so a second import of the same document creates nothing.
    Catalog rows are created when missing, so an import works against an empty
    store.

    **The whole import is one transaction.** The Neo4j version had to apply SQL
    attributes and custom analyses outside it — ``add_query`` opened its own
    auto-commit session there and would deadlock against locks the outer
    transaction held on rows it had just created. The Postgres ``add_query``
    runs on this connection, so that carve-out is gone and a failure part-way
    leaves nothing behind, rather than a catalog with half its semantics.
    """
    id_map: dict[str, str] = {}
    column_meta: dict[str, ColumnCatalogMeta] = {}
    created: dict[str, int] = {
        "databases": 0,
        "schemas": 0,
        "tables": 0,
        "columns": 0,
        "terms": 0,
        "column_attributes": 0,
        "sql_attributes": 0,
        "custom_analyses": 0,
    }
    skipped: dict[str, int] = {key: 0 for key in created}
    term_names = {term.id: term.name for term in document.semantic_layer.terms}
    schema_cache: dict[str | None, tuple[list[str], dict[str, Any]]] = {}

    with write_transaction():
        live_db_ids = _import_catalog(
            document, id_map, created, skipped, embed_buffer, column_meta
        )
        if replace:
            _delete_scoped_semantics_not_in_payload(document, live_db_ids)
        _import_foreign_keys(document, id_map)
        _import_joins(document, id_map)
        _import_terms(document, id_map, created, skipped, embed_buffer)
        _import_column_attributes(
            document, id_map, created, skipped, embed_buffer, column_meta
        )
        _import_semantic_fks(document, id_map)
        _import_sql_attributes(
            document, id_map, created, skipped, embed_buffer, term_names, schema_cache
        )
        _import_custom_analyses(
            document, id_map, created, skipped, embed_buffer, schema_cache
        )

    summary: dict[str, Any] = {
        "database_ids": live_db_ids,
        "created": created,
        "skipped": skipped,
        "replace": replace,
        "terms": len(document.semantic_layer.terms),
        "column_attributes": sum(
            len(term.columns_attributes) for term in document.semantic_layer.terms
        ),
        "semantic_fks": len(document.semantic_layer.semantic_fks),
        "sql_attributes": sum(
            len(getattr(document.semantic_layer.sql_attributes, key))
            for key in _YAML_SOURCE_KEYS
        ),
        "custom_analyses": len(document.semantic_layer.custom_analyses),
    }
    if embed_buffer is not None:
        summary["pending_embed"] = {
            "data_rows": len(embed_buffer.data_rows),
            "semantic_rows": len(embed_buffer.semantic_rows),
        }
    return summary


def _import_catalog(
    document: GsfModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
    column_meta: dict[str, ColumnCatalogMeta],
) -> list[str]:
    """Import databases, schemas, tables and columns, a level at a time.

    Containment is a parent FK column here, so each level is created *with* its
    parent rather than created and then wired up — the Cypher needed a second
    batched ``MERGE`` per level for the ``CONTAINS`` edges.
    """
    databases = document.data_layer.databases

    db_names = {
        db.id: (
            db.schemas[0].database_name or db.schemas[0].name or ""
            if db.schemas
            else ""
        )
        for db in databases
    }
    db_results = _resolve_entities_batch(
        s.catalog_database,
        [(db.id, {"name": db_names[db.id] or db.id}) for db in databases],
    )
    live_db_ids: list[str] = []
    for db in databases:
        live_db_id, was_created = db_results[db.id]
        id_map[db.id] = live_db_id
        live_db_ids.append(live_db_id)
        (created if was_created else skipped)["databases"] += 1

    schema_db_names: dict[str, str] = {}
    schema_items: list[tuple[str, dict[str, Any]]] = []
    for db in databases:
        for schema in db.schemas:
            schema_db_names[schema.id] = schema.database_name or db_names[db.id]
            schema_items.append(
                (schema.id, {"name": schema.name, "database_id": id_map[db.id]})
            )
    schema_results = _resolve_entities_batch(s.catalog_schema, schema_items)
    for schema_id, (live_id, was_created) in schema_results.items():
        id_map[schema_id] = live_id
        (created if was_created else skipped)["schemas"] += 1

    table_items: list[tuple[str, dict[str, Any]]] = []
    for db in databases:
        for schema in db.schemas:
            for table in schema.tables:
                table_items.append(
                    (
                        table.id,
                        {
                            "name": table.name,
                            "description": table.description,
                            "pk": table.pk,
                            "table_type": table.type,
                            "schema_id": id_map[schema.id],
                        },
                    )
                )
    table_results = _resolve_entities_batch(s.catalog_table, table_items)
    for table_id, (live_id, was_created) in table_results.items():
        id_map[table_id] = live_id
        (created if was_created else skipped)["tables"] += 1

    column_items: list[tuple[str, dict[str, Any]]] = []
    for db in databases:
        for schema in db.schemas:
            schema_db_name = schema_db_names.get(schema.id, "")
            for table in schema.tables:
                for ordinal, column in enumerate(table.columns, start=1):
                    column_meta[column.id] = ColumnCatalogMeta(
                        name=column.name,
                        description=column.description,
                        data_type=column.type,
                        sample_values=column.sample_values,
                        table_yaml_id=table.id,
                        schema_name=schema.name,
                        database_name=schema_db_name,
                    )
                    column_items.append(
                        (
                            column.id,
                            {
                                "name": column.name,
                                "description": column.description,
                                "data_type": column.type,
                                "sample_values": (
                                    json.dumps(column.sample_values)
                                    if column.sample_values
                                    else None
                                ),
                                "is_unique": column.is_unique,
                                # Written in information_schema's vocabulary so
                                # an imported column is indistinguishable from
                                # an ingested one -- see _nullable_to_stored.
                                "is_nullable": _nullable_to_stored(column.is_nullable),
                                "ordinal_position": ordinal,
                                "table_id": id_map[table.id],
                            },
                        ),
                    )
    column_results = _resolve_entities_batch(s.catalog_column, column_items)
    for column_id, (live_id, was_created) in column_results.items():
        id_map[column_id] = live_id
        (created if was_created else skipped)["columns"] += 1

    if embed_buffer is not None:
        for db in databases:
            for schema in db.schemas:
                schema_db_name = schema_db_names.get(schema.id, "")
                for table in schema.tables:
                    live_table_id, tbl_created = table_results[table.id]
                    specs: list[dict[str, Any]] = []
                    for column in table.columns:
                        live_col_id, col_created = column_results[column.id]
                        if col_created:
                            embed_buffer.data_rows.append(
                                build_column_data_row(
                                    live_id=live_col_id,
                                    column_name=column.name,
                                    column_description=column.description,
                                    data_type=column.type,
                                    sample_values=column.sample_values,
                                    table_name=table.name,
                                    schema_name=schema.name,
                                    database_name=schema_db_name,
                                ),
                            )
                        specs.append(
                            {
                                "column_name": column.name,
                                "data_type": column.type,
                                "description": column.description,
                            }
                        )
                    if tbl_created:
                        embed_buffer.data_rows.append(
                            build_table_data_row(
                                live_id=live_table_id,
                                table_name=table.name,
                                table_description=table.description,
                                schema_name=schema.name,
                                database_name=schema_db_name,
                                columns=specs,
                            ),
                        )

    return live_db_ids


def _import_foreign_keys(document: GsfModelDocument, id_map: dict[str, str]) -> None:
    _link(
        s.column_foreign_key,
        [
            {
                "source_column_id": _remap(
                    id_map, fk.source_column_id, kind="foreign-key source column"
                ),
                "target_column_id": _remap(
                    id_map, fk.target_column_id, kind="foreign-key target column"
                ),
            }
            for fk in document.data_layer.foreign_keys
        ],
    )


def _import_joins(document: GsfModelDocument, id_map: dict[str, str]) -> None:
    rows = [
        {
            "source_table_id": _remap(
                id_map, join.source_table_id, kind="join source table"
            ),
            "target_table_id": _remap(
                id_map, join.target_table_id, kind="join target table"
            ),
            "join_columns": join.join_columns,
        }
        for join in document.data_layer.joins
    ]
    if not rows:
        return
    statement = insert(s.table_join).values(rows)
    store().query_write(
        statement.on_conflict_do_update(
            index_elements=[
                s.table_join.c.source_table_id,
                s.table_join.c.target_table_id,
            ],
            set_={"join_columns": statement.excluded.join_columns},
        )
    )


def _delete_scoped_semantics_not_in_payload(
    document: GsfModelDocument,
    live_db_ids: list[str],
) -> None:
    """Drop in-scope semantics the payload does not mention.

    Keep sets are matched against ``imported_id`` **or** the live id, so an
    entity created by an earlier import and unchanged since is recognised
    either way.

    A Term is only deleted when **every** table representing it is inside the
    imported databases — the same all-or-nothing rule the reads use. A term
    shared with a database outside this import is that database's too.
    """
    keep_terms = [term.id for term in document.semantic_layer.terms]
    keep_attrs = [
        attr.id
        for term in document.semantic_layer.terms
        for attr in term.columns_attributes
    ]
    keep_sql_attrs = [
        attr.id
        for key in _YAML_SOURCE_KEYS
        for attr in getattr(document.semantic_layer.sql_attributes, key)
    ]
    keep_analyses = [ca.id for ca in document.semantic_layer.custom_analyses]

    scoped_tables = (
        select(s.catalog_table.c.id)
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
        )
        .where(s.catalog_schema.c.database_id.in_(live_db_ids))
    )

    for owner_table, link_table, owner_column, keep in (
        (
            s.sql_attribute,
            s.sql_attribute_sql,
            s.sql_attribute_sql.c.attribute_id,
            keep_sql_attrs,
        ),
        (
            s.custom_analysis,
            s.custom_analysis_sql,
            s.custom_analysis_sql.c.analysis_id,
            keep_analyses,
        ),
    ):
        in_scope = (
            select(owner_column)
            .select_from(
                link_table.join(
                    s.sql_query_table,
                    s.sql_query_table.c.sql_query_id == link_table.c.sql_query_id,
                )
            )
            .where(s.sql_query_table.c.table_id.in_(scoped_tables))
            .distinct()
        )
        store().query_write(
            owner_table.delete().where(
                owner_table.c.id.in_(in_scope),
                _payload_identity(owner_table).notin_(keep or [""]),
            )
        )

    attribute_in_scope = (
        select(s.column_attribute.c.id)
        .where(s.column_attribute.c.table_id.in_(scoped_tables))
        .distinct()
    )
    store().query_write(
        s.column_attribute.delete().where(
            s.column_attribute.c.id.in_(attribute_in_scope),
            _payload_identity(s.column_attribute).notin_(keep_attrs or [""]),
        )
    )

    # Only terms whose every representing table is inside this import.
    outside = (
        select(s.table_term.c.term_id)
        .where(s.table_term.c.table_id.notin_(scoped_tables))
        .distinct()
    )
    inside = (
        select(s.table_term.c.term_id)
        .where(s.table_term.c.table_id.in_(scoped_tables))
        .distinct()
    )
    store().query_write(
        s.term.delete().where(
            s.term.c.id.in_(inside),
            s.term.c.id.notin_(outside),
            s.term.c.source == SEMANTIC_SOURCE,
            _payload_identity(s.term).notin_(keep_terms or [""]),
        )
    )


def _import_terms(
    document: GsfModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
) -> None:
    terms = document.semantic_layer.terms
    term_results = _resolve_entities_batch(
        s.term,
        [
            (
                term.id,
                {
                    "name": term.name,
                    "description": term.description,
                    "source": SEMANTIC_SOURCE,
                },
            )
            for term in terms
        ],
    )

    represents_rows: list[dict[str, Any]] = []
    newly_created: list[str] = []
    for term in terms:
        live_term_id, was_created = term_results[term.id]
        id_map[term.id] = live_term_id
        if was_created:
            created["terms"] += 1
            newly_created.append(live_term_id)
        else:
            skipped["terms"] += 1
        # The payload is authoritative about which tables represent a term, so
        # the existing links go first -- a table dropped from the YAML must stop
        # representing it.
        store().query_write(
            s.table_term.delete().where(s.table_term.c.term_id == live_term_id)
        )
        for table_id in term.represents:
            represents_rows.append(
                {
                    "term_id": live_term_id,
                    "table_id": _remap(id_map, table_id, kind="term represents table"),
                }
            )
    _link(s.table_term, represents_rows)

    if embed_buffer is not None and newly_created:
        db_names = _database_names_for_terms(newly_created)
        for term in terms:
            live_term_id, was_created = term_results[term.id]
            if not was_created:
                continue
            embed_buffer.semantic_rows.extend(
                build_term_semantic_rows(
                    database_name=db_names.get(live_term_id, ""),
                    live_id=live_term_id,
                    name=term.name,
                    description=term.description,
                ),
            )


def _import_column_attributes(
    document: GsfModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
    column_meta: dict[str, ColumnCatalogMeta],
) -> None:
    attr_items: list[tuple[str, dict[str, Any]]] = []
    context: dict[str, dict[str, Any]] = {}

    for term in document.semantic_layer.terms:
        live_term_id = _remap(id_map, term.id, kind="term")
        for attr in term.columns_attributes:
            col_ctx = column_meta.get(attr.column_id)
            attr_items.append(
                (
                    attr.id,
                    {
                        "name": attr.name,
                        "description": attr.description,
                        "source": SEMANTIC_SOURCE,
                        "term_name": term.name,
                        "source_column": col_ctx.name if col_ctx else "",
                        # Deliberately '' when the column is unknown: the schema
                        # keeps table_id NOT NULL but not a foreign key, exactly
                        # so this import stays legal.
                        "table_id": (
                            _remap(
                                id_map,
                                col_ctx.table_yaml_id,
                                kind="column attribute table",
                            )
                            if col_ctx
                            else ""
                        ),
                    },
                ),
            )
            context[attr.id] = {
                "attr": attr,
                "term": term,
                "col_ctx": col_ctx,
                "live_term_id": live_term_id,
            }

    attr_results = _resolve_entities_batch(s.column_attribute, attr_items)

    has_attribute_rows: list[dict[str, Any]] = []
    property_of_rows: list[dict[str, Any]] = []
    for attr_id, (live_attr_id, was_created) in attr_results.items():
        ctx = context[attr_id]
        attr, term, col_ctx = ctx["attr"], ctx["term"], ctx["col_ctx"]
        id_map[attr_id] = live_attr_id
        (created if was_created else skipped)["column_attributes"] += 1

        has_attribute_rows.append(
            {
                "column_id": _remap(
                    id_map, attr.column_id, kind="column attribute column"
                ),
                "attribute_id": live_attr_id,
            }
        )
        property_of_rows.append(
            {"attribute_id": live_attr_id, "term_id": ctx["live_term_id"]}
        )
        if was_created and embed_buffer is not None and col_ctx is not None:
            embed_buffer.semantic_rows.extend(
                build_column_attribute_semantic_rows(
                    database_name=col_ctx.database_name,
                    live_id=live_attr_id,
                    name=attr.name,
                    description=attr.description,
                    term_name=term.name,
                    source_column=col_ctx.name,
                    sample_values=col_ctx.sample_values,
                    schema_name=col_ctx.schema_name,
                ),
            )

    _link(s.column_has_attribute, has_attribute_rows)
    _link(s.column_attribute_term, property_of_rows)


def _import_semantic_fks(document: GsfModelDocument, id_map: dict[str, str]) -> None:
    _link(
        s.column_semantic_fk,
        [
            {
                "column_id": _remap(id_map, fk.column_id, kind="semantic fk column"),
                "attribute_id": _remap(
                    id_map, fk.column_attribute_id, kind="semantic fk attribute"
                ),
            }
            for fk in document.semantic_layer.semantic_fks
        ],
    )


def _cached_dialects_and_schemas(
    cache: dict[str | None, tuple[list[str], dict[str, Any]]],
    database_name: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """Memoised ``(dialects, schemas)`` for the life of one import.

    ``get_schemas`` rebuilds a database's whole catalog snapshot per call, which
    is fine once and expensive across hundreds of attributes that share a
    handful of database names. Scoped to one call, so it cannot serve stale data
    across imports.
    """
    if database_name not in cache:
        cache[database_name] = (get_dialects(database_name), get_schemas(database_name))
    return cache[database_name]


def _persist_sql_object(
    *,
    node_label: str,
    node_id: str,
    name: str,
    description: str,
    sql: str,
    database_name: str | None,
    schema_cache: dict[str | None, tuple[list[str], dict[str, Any]]],
    extra_props: dict[str, Any] | None = None,
) -> None:
    """Parse the SQL and link the owner to it through the catalog write path.

    Reuses ``add_query`` rather than writing the statement here, so an imported
    statement lands with the same table and column links an ingested one gets —
    which is what makes the SQL attribute show up in the exploration graph and
    the zone checks afterwards.
    """
    dialects, schemas = _cached_dialects_and_schemas(schema_cache, database_name)
    query_obj = validate_sql(sql, dialects, schemas)
    props: dict[str, Any] = {"name": name, "description": description}
    if extra_props:
        props.update(extra_props)
    node = CatalogNode(
        name=name,
        label=node_label,
        props=props,
        existing_id=node_id,
        match_props={"id": node_id},
    )
    query_obj.sql_node.match_props = {"sql_full_query": sql}
    query_obj.edges.append((node, query_obj.sql_node, {Props.ANALYSIS_ID: node_id}))
    add_query(query_obj.get_edges())


def _import_sql_attributes(
    document: GsfModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
    term_names: dict[str, str],
    schema_cache: dict[str | None, tuple[list[str], dict[str, Any]]],
) -> None:
    grouped: list[tuple[str, ModelSqlAttribute]] = [
        (_YAML_KEY_TO_SOURCE[key], attr)
        for key in _YAML_SOURCE_KEYS
        for attr in getattr(document.semantic_layer.sql_attributes, key)
    ]
    attr_results = _resolve_entities_batch(
        s.sql_attribute,
        [
            (
                attr.id,
                {
                    "name": attr.name,
                    "description": attr.description,
                    "expression": attr.sql,
                    "source": source,
                },
            )
            for source, attr in grouped
        ],
    )
    for _source, attr in grouped:
        id_map[attr.id] = attr_results[attr.id][0]

    # SQL parsing is inherently per-item, but the resolve and the database-name
    # lookups it needs are batched up front.
    newly_created_terms = list(
        {
            _remap(id_map, attr.term_id, kind="sql attribute term")
            for _source, attr in grouped
            if attr_results[attr.id][1]
        }
    )
    db_names = _database_names_for_terms(newly_created_terms)

    for source, attr in grouped:
        live_attr_id, was_created = attr_results[attr.id]
        if not was_created:
            skipped["sql_attributes"] += 1
            continue
        created["sql_attributes"] += 1
        live_term_id = _remap(id_map, attr.term_id, kind="sql attribute term")
        database_name = db_names.get(live_term_id)
        detach_existing_sql_edges(live_attr_id)
        _persist_sql_object(
            node_label="SqlAttribute",
            node_id=live_attr_id,
            name=attr.name,
            description=attr.description,
            sql=attr.sql,
            database_name=database_name,
            schema_cache=schema_cache,
            extra_props={"expression": attr.sql, "source": source},
        )
        link_to_term(live_attr_id, live_term_id)
        if embed_buffer is not None:
            embed_buffer.semantic_rows.append(
                build_sql_attribute_semantic_row(
                    live_id=live_attr_id,
                    name=attr.name,
                    description=attr.description,
                    term_name=term_names.get(attr.term_id, ""),
                    sql=attr.sql,
                    database_name=database_name,
                ),
            )


def _import_custom_analyses(
    document: GsfModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
    schema_cache: dict[str | None, tuple[list[str], dict[str, Any]]],
) -> None:
    analyses = document.semantic_layer.custom_analyses
    results = _resolve_entities_batch(
        s.custom_analysis,
        [
            (analysis.id, {"name": analysis.name, "description": analysis.description})
            for analysis in analyses
        ],
    )
    for analysis in analyses:
        id_map[analysis.id] = results[analysis.id][0]

    live_column: dict[str, str | None] = {}
    lookup_ids: list[str] = []
    for analysis in analyses:
        _live_id, was_created = results[analysis.id]
        column_id: str | None = None
        if was_created and analysis.sql_column_is:
            try:
                column_id = _remap(
                    id_map, analysis.sql_column_is[0], kind="custom analysis column"
                )
            except ModelImportValidationError:
                column_id = None
        live_column[analysis.id] = column_id
        if column_id:
            lookup_ids.append(column_id)

    db_names = _database_names_for_columns(lookup_ids)

    for analysis in analyses:
        live_ca_id, was_created = results[analysis.id]
        if not was_created:
            skipped["custom_analyses"] += 1
            continue
        created["custom_analyses"] += 1
        column_id = live_column.get(analysis.id)
        database_name = db_names.get(column_id) if column_id else None
        detach_ca_sql_edges(live_ca_id)
        _persist_sql_object(
            node_label="CustomAnalysis",
            node_id=live_ca_id,
            name=analysis.name,
            description=analysis.description,
            sql=analysis.sql,
            database_name=database_name,
            schema_cache=schema_cache,
        )
        if embed_buffer is not None:
            embed_buffer.semantic_rows.append(
                build_custom_analysis_semantic_row(
                    live_id=live_ca_id,
                    name=analysis.name,
                    description=analysis.description,
                    sql=analysis.sql,
                    database_name=database_name,
                ),
            )
