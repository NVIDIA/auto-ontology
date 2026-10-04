# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto Ontology model YAML export and import.

CRUD over the catalog and semantic entities, driven by a YAML document.

``imported_id`` is the mechanism the whole import turns on: entities are matched
by the YAML ``id`` against ``imported_id`` *or* the live ``id``, so re-importing
a document is a no-op and importing onto an existing catalog adopts it rather
than duplicating it.

The whole import is **one transaction**, so a failure part-way leaves nothing
behind rather than a catalog with half its semantics on top.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from sqlalchemy import func, literal, select, update
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.catalog.constants import Props
from auto_ontology.catalog.model import CatalogNode
from auto_ontology.catalog.store.queries import add_query
from auto_ontology.dal import schema as s
from auto_ontology.dal.custom_analyses import (
    detach_existing_sql_edges as detach_ca_sql_edges,
)
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.dal.sql_attributes import detach_existing_sql_edges, link_to_term
from auto_ontology.semantic.constants import (
    SEMANTIC_SOURCE,
    SQL_ATTR_SOURCE_BRIDGE,
    SQL_ATTR_SOURCE_MANUAL,
    SQL_ATTR_SOURCE_SQL,
    SQL_ATTR_SOURCE_TABLE,
)
from auto_ontology.server.model_interchange.embed import (
    ColumnCatalogMeta,
    ImportEmbedBuffer,
    build_column_attribute_semantic_rows,
    build_column_data_row,
    build_custom_analysis_semantic_row,
    build_sql_attribute_semantic_row,
    build_table_data_row,
    build_term_semantic_rows,
)
from auto_ontology.server.model_interchange.schemas import (
    AutoOntologyModelDocument,
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
from auto_ontology.server.sql_utils import get_dialects, get_schemas, validate_sql
from auto_ontology.utils.sample_values import parse_sample_values

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
# Export
# ---------------------------------------------------------------------------


def _nullable_or_default(raw: Any) -> bool:
    """The stored boolean, defaulting an undetermined column to nullable.

    The catalog stores ``is_nullable`` as a real boolean but leaves it NULL
    when the connector could not determine it; the export schema types it as a
    plain ``bool``. Absent has always meant nullable — the permissive reading
    for a constraint we cannot prove — so that is what NULL becomes here.

    This is the *only* place the two vocabularies meet. It used to be a pair of
    string parsers on both sides of the store, because the column held
    ``'YES'``/``'NO'`` and ``bool('NO')`` is ``True``.
    """
    return True if raw is None else bool(raw)


def _database_scope(database_ids: list[str]):
    """An empty id list means **every** database, not none.

    Reading it the other way would silently export an empty document.
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
                # The same column twice: `column_id` is what the document is
                # keyed by, `column_live_id` is what the SQL parser resolves to.
                # They coincide today and `_column_live_id_map` translates
                # between them, so an export that ever keys the document by
                # something else (an `imported_id`, say) keeps working.
                s.catalog_column.c.id.label("column_live_id"),
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
                    s.column__foreign_key,
                    s.column__foreign_key.c.source_column_id == s.catalog_column.c.id,
                )
                .join(target, target.c.id == s.column__foreign_key.c.target_column_id)
            )
            .where(_database_scope(database_ids))
            .distinct()
        )
    ]


def _export_joins(database_ids: list[str]) -> list[dict[str, Any]]:
    """Joins whose **both** ends are in scope.

    The target table has to belong to a scoped database too, or the export
    carries a join pointing at a table the document does not contain — which
    the importer then cannot resolve.
    """
    target_table = s.catalog_table.alias("join_target")
    target_schema = s.catalog_schema.alias("join_target_schema")
    target_database = s.catalog_database.alias("join_target_database")

    statement = (
        select(
            s.table__join.c.source_table_id,
            s.table__join.c.target_table_id,
            s.table__join.c.join_columns,
        )
        .select_from(
            s.catalog_database.join(
                s.catalog_schema,
                s.catalog_schema.c.database_id == s.catalog_database.c.id,
            )
            .join(s.catalog_table, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
            .join(
                s.table__join, s.table__join.c.source_table_id == s.catalog_table.c.id
            )
            .join(target_table, target_table.c.id == s.table__join.c.target_table_id)
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

    A term reached through one scoped table exports every table representing
    it, scoped or not, which keeps a partial export honest about a term it only
    partly owns.
    """
    in_scope = (
        select(s.table__term.c.term_id)
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
        select(s.table__term.c.term_id, s.table__term.c.table_id)
        .where(s.table__term.c.term_id.in_(term_ids))
        .order_by(s.table__term.c.table_id)
    ):
        represents.setdefault(row["term_id"], []).append(row["table_id"])

    attributes: dict[str, list[dict[str, Any]]] = {}
    for row in store().query_read(
        select(
            s.column_attribute__term.c.term_id,
            s.column_attribute.c.id,
            s.column_attribute.c.name,
            s.column_attribute.c.description,
            s.column__has_attribute.c.column_id,
        )
        .select_from(
            s.column_attribute__term.join(
                s.column_attribute,
                s.column_attribute.c.id == s.column_attribute__term.c.attribute_id,
            ).join(
                s.column__has_attribute,
                s.column__has_attribute.c.attribute_id == s.column_attribute.c.id,
            )
        )
        .where(s.column_attribute__term.c.term_id.in_(term_ids))
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
                s.column__semantic_fk.c.column_id,
                s.column__semantic_fk.c.attribute_id.label("column_attribute_id"),
            )
            .select_from(
                _catalog_join().join(
                    s.column__semantic_fk,
                    s.column__semantic_fk.c.column_id == s.catalog_column.c.id,
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
                s.sql_query__table,
                s.sql_query__table.c.table_id == s.catalog_table.c.id,
            )
            .join(s.sql_query, s.sql_query.c.id == s.sql_query__table.c.sql_query_id)
            .join(link_table, link_table.c.sql_query_id == s.sql_query.c.id)
            .join(owner_table, owner_table.c.id == owner_column)
        )
        .where(_database_scope(database_ids))
        .distinct()
    )


def _export_sql_attributes(database_ids: list[str]) -> list[dict[str, Any]]:
    statement = _export_sql_owners(
        database_ids,
        s.sql_attribute__sql,
        s.sql_attribute,
        s.sql_attribute__sql.c.attribute_id,
    )
    statement = (
        statement.add_columns(
            s.sql_attribute.c.expression,
            s.sql_attribute.c.source,
            s.term.c.id.label("term_id"),
        )
        .join(
            s.sql_attribute__term,
            s.sql_attribute__term.c.attribute_id == s.sql_attribute.c.id,
        )
        .join(s.term, s.term.c.id == s.sql_attribute__term.c.term_id)
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
        s.custom_analysis__sql,
        s.custom_analysis,
        s.custom_analysis__sql.c.analysis_id,
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


def _catalog_ids(databases: list[ModelDatabase]) -> tuple[set[str], set[str]]:
    """Return the table and column ids the exported catalog actually carries."""
    tables = [
        table
        for database in databases
        for schema in database.schemas
        for table in schema.tables
    ]
    return (
        {table.id for table in tables},
        {column.id for table in tables for column in table.columns},
    )


def _scoped_ids(values: Iterable[Any], known: set[str]) -> list[str]:
    """Keep the ids present in the exported catalog, in their original order.

    A scoped export must be importable on its own. Emitting a reference to a
    table the document does not carry -- a Term also represented in another
    database, say -- makes the document unimportable anywhere, because the
    importer has nothing to resolve it against.

    The importer is *also* tolerant of unresolvable references (see
    ``_remap_optional``), so this is belt and braces: the document stays
    self-contained, and a document produced elsewhere still imports.
    """
    return [str(value) for value in values if str(value) in known]


def assemble_export_document(
    rows: dict[str, Any],
    *,
    dialect_by_db_name: dict[str, str],
    sql_column_resolver: Any,
) -> AutoOntologyModelDocument:
    """Build a validated document from raw export rows."""
    databases = _assemble_databases(rows["catalog"], dialect_by_db_name)
    table_ids, column_ids = _catalog_ids(databases)
    live_to_catalog_column_id = _column_live_id_map(rows["catalog"])
    foreign_keys = [
        ModelForeignKey(
            source_column_id=str(row["source_column_id"]),
            target_column_id=str(row["target_column_id"]),
        )
        for row in rows["foreign_keys"]
        if str(row.get("source_column_id") or "") in column_ids
        and str(row.get("target_column_id") or "") in column_ids
    ]
    joins = [
        ModelJoin(
            source_table_id=str(row["source_table_id"]),
            target_table_id=str(row["target_table_id"]),
            join_columns=row.get("join_columns") or [],
        )
        for row in rows["joins"]
        if str(row.get("source_table_id") or "") in table_ids
        and str(row.get("target_table_id") or "") in table_ids
    ]
    terms = [
        ModelTerm(
            id=str(row["id"]),
            name=row.get("name") or "",
            description=row.get("description") or "",
            represents=_scoped_ids(row.get("represents") or [], table_ids),
            columns_attributes=[
                ModelColumnAttribute(
                    id=str(attr["id"]),
                    name=attr.get("name") or "",
                    description=attr.get("description") or "",
                    column_id=str(attr.get("column_id") or ""),
                )
                for attr in (row.get("columns_attributes") or [])
                if attr.get("id") and str(attr.get("column_id") or "") in column_ids
            ],
        )
        for row in rows["terms"]
        if row.get("id")
    ]
    _term_ids = {term.id for term in terms}
    _attribute_ids = {attr.id for term in terms for attr in term.columns_attributes}
    semantic_fks = [
        ModelSemanticFk(
            column_attribute_id=str(row["column_attribute_id"]),
            column_id=str(row["column_id"]),
        )
        for row in rows["semantic_fks"]
        if str(row.get("column_attribute_id") or "") in _attribute_ids
        and str(row.get("column_id") or "") in column_ids
    ]
    sql_attributes = _assemble_sql_attributes(
        rows["sql_attributes"],
        sql_column_resolver,
        column_ids=column_ids,
        term_ids=_term_ids,
        live_to_catalog_column_id=live_to_catalog_column_id,
    )
    custom_analyses = [
        ModelCustomAnalysis(
            id=str(row["id"]),
            name=row.get("name") or "",
            description=row.get("description") or "",
            sql=row.get("sql") or "",
            sql_column_is=_scoped_column_ids(
                sql_column_resolver(row.get("sql") or "", row.get("database_name")),
                live_to_catalog_column_id,
                column_ids,
            ),
        )
        for row in rows["custom_analyses"]
        if row.get("id")
    ]
    return AutoOntologyModelDocument(
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


def _column_live_id_map(catalog_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map live column ids to the id used in the exported catalog.

    An exported catalog id is not required to be the column's live ``id`` --
    a previously-imported model can keep the id it was imported under. The
    SQL parser behind ``sql_column_resolver``
    (:func:`auto_ontology.server.sql_utils.get_schemas`) only ever resolves columns by
    the live ``id``, so the two id spaces can diverge. See
    :func:`_scoped_column_ids`.
    """
    return {
        str(row["column_live_id"]): str(row["column_id"])
        for row in catalog_rows
        if row.get("column_id") and row.get("column_live_id")
    }


def _scoped_column_ids(
    values: Iterable[Any],
    live_to_catalog_column_id: dict[str, str],
    known: set[str],
) -> list[str]:
    """Translate SQL-resolver column ids into catalog ids, then scope them.

    Without the translation, every column belonging to a previously
    imported table would fail the membership check below (its live id is
    never a key of *known*), and ``sql_column_is`` would silently come back
    empty for the vast majority of SQL attributes/custom analyses.
    """
    translated = (
        live_to_catalog_column_id.get(str(value), str(value)) for value in values
    )
    return _scoped_ids(translated, known)


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
                    # Decoded, not rendered: the exported document is
                    # re-importable, so an integer column has to leave as
                    # numbers to come back as numbers.
                    sample_values=parse_sample_values(row.get("sample_values")) or [],
                    is_nullable=_nullable_or_default(row.get("is_nullable")),
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
    *,
    term_ids: set[str],
    column_ids: set[str],
    live_to_catalog_column_id: dict[str, str],
) -> ModelSqlAttributesBySource:
    grouped: dict[str, list[ModelSqlAttribute]] = {key: [] for key in _YAML_SOURCE_KEYS}
    for row in rows:
        # An attribute whose Term did not make the document references an id the
        # importer cannot resolve, so it is dropped rather than emitted dangling.
        if term_ids is not None and str(row.get("term_id") or "") not in term_ids:
            continue
        yaml_key = _SOURCE_TO_YAML_KEY.get(row.get("source") or "", "manual")
        sql_text = row.get("sql") or row.get("expression") or ""
        grouped[yaml_key].append(
            ModelSqlAttribute(
                id=str(row["id"]),
                name=row.get("name") or "",
                description=row.get("description") or "",
                sql=sql_text,
                sql_column_is=_scoped_column_ids(
                    sql_column_resolver(sql_text, row.get("database_name")),
                    live_to_catalog_column_id,
                    column_ids,
                ),
                term_id=str(row.get("term_id") or ""),
            ),
        )
    return ModelSqlAttributesBySource(**grouped)


def _column_ids_from_sql(
    sql: str, dialects: list[str], schemas: dict[str, Any]
) -> list[str]:
    try:
        query_obj = validate_sql(sql, dialects, schemas)
    except Exception:
        logger.debug("Could not resolve sql_column_is for SQL snippet", exc_info=True)
        return []
    return [str(col_id) for col_id in query_obj.get_column_ids() if col_id]


def resolve_sql_column_ids(sql: str, database_name: str | None) -> list[str]:
    """Parse SQL against the scoped catalog and return referenced column ids.

    ``sql_column_is`` is a best-effort enrichment: a broken connector or a
    transient catalog-lookup failure for *database_name* must not raise out
    of here, or callers assembling an export/import document would fail
    entirely over what is, at worst, a missing cross-reference.
    """
    if not sql.strip():
        return []
    try:
        dialects = get_dialects(database_name)
        schemas = get_schemas(database_name)
    except Exception:
        logger.debug(
            "Could not build dialects/schemas for database %r; skipping "
            "sql_column_is for this SQL snippet",
            database_name,
            exc_info=True,
        )
        return []
    return _column_ids_from_sql(sql, dialects, schemas)


def make_cached_sql_column_resolver() -> Callable[[str, str | None], list[str]]:
    """Build a ``sql_column_resolver`` that reuses ``(dialects, schemas)`` per database.

    A single export calls this once per sql_attribute/custom_analysis, and
    those usually share a handful of database names. Unlike
    :func:`resolve_sql_column_ids`, which rebuilds the whole catalog snapshot
    on every call (see :func:`_cached_dialects_and_schemas`), the resolver
    returned here caches that snapshot for the lifetime of one export.
    """
    cache: dict[str | None, tuple[list[str], dict[str, Any]]] = {}

    def _resolve(sql: str, database_name: str | None) -> list[str]:
        if not sql.strip():
            return []
        dialects, schemas = _safe_cached_dialects_and_schemas(cache, database_name)
        return _column_ids_from_sql(sql, dialects, schemas)

    return _resolve


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

    Three statements regardless of batch size — one ``INSERT ... RETURNING``
    rather than a create per row.
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


def _remap_optional(id_map: dict[str, str], yaml_id: str) -> str | None:
    """Resolve an id that a scoped export may legitimately not carry.

    A scoped export is deliberately honest about the parts of a term it does not
    own: ``_export_terms`` emits *every* table representing a term, in scope or
    not, and ``_export_semantic_fks`` every pair touching an in-scope column.
    Those ids point outside the document by design, so requiring them -- as
    :func:`_remap` does -- made any such document abort the whole import.

    Structural references still use :func:`_remap`. The difference is what a
    missing id means: for a foreign key's endpoints it is a corrupt document,
    for these it is the expected shape of a partial export.
    """
    return id_map.get(yaml_id)


def _database_names_for_terms(term_ids: list[str]) -> dict[str, str]:
    if not term_ids:
        return {}
    names: dict[str, str] = {}
    for row in store().query_read(
        select(
            s.table__term.c.term_id,
            s.catalog_database.c.name.label("database_name"),
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
        .where(s.table__term.c.term_id.in_(term_ids))
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
    document: AutoOntologyModelDocument,
    *,
    replace: bool,
    embed_buffer: ImportEmbedBuffer | None = None,
) -> dict[str, Any]:
    """Apply a validated model document.

    Entities are matched by ``imported_id`` (the YAML ``id``) or by a live ``id``
    equal to it, so a second import of the same document creates nothing.
    Catalog rows are created when missing, so an import works against an empty
    store.

    **The whole import is one transaction**, SQL attributes and analyses
    included, so a failure part-way leaves nothing behind rather than a catalog
    with half its semantics on top.
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
    document: AutoOntologyModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
    column_meta: dict[str, ColumnCatalogMeta],
) -> list[str]:
    """Import databases, schemas, tables and columns, a level at a time.

    Containment is a parent foreign key, so each level is created *with* its
    parent rather than created and then wired up.
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
                        # Imported values have not passed this deployment's PII
                        # policy and must not reach storage or embeddings.
                        sample_values=[],
                        is_unique=column.is_unique,
                        table_yaml_id=table.id,
                        table_name=table.name,
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
                                "sample_values": None,
                                "is_unique": column.is_unique,
                                "is_nullable": column.is_nullable,
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
                                    sample_values=None,
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


def _import_foreign_keys(
    document: AutoOntologyModelDocument, id_map: dict[str, str]
) -> None:
    _link(
        s.column__foreign_key,
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


def _import_joins(document: AutoOntologyModelDocument, id_map: dict[str, str]) -> None:
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
    statement = insert(s.table__join).values(rows)
    store().query_write(
        statement.on_conflict_do_update(
            index_elements=[
                s.table__join.c.source_table_id,
                s.table__join.c.target_table_id,
            ],
            set_={"join_columns": statement.excluded.join_columns},
        )
    )


def _delete_scoped_semantics_not_in_payload(
    document: AutoOntologyModelDocument,
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
            s.sql_attribute__sql,
            s.sql_attribute__sql.c.attribute_id,
            keep_sql_attrs,
        ),
        (
            s.custom_analysis,
            s.custom_analysis__sql,
            s.custom_analysis__sql.c.analysis_id,
            keep_analyses,
        ),
    ):
        in_scope = (
            select(owner_column)
            .select_from(
                link_table.join(
                    s.sql_query__table,
                    s.sql_query__table.c.sql_query_id == link_table.c.sql_query_id,
                )
            )
            .where(s.sql_query__table.c.table_id.in_(scoped_tables))
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
        select(s.table__term.c.term_id)
        .where(s.table__term.c.table_id.notin_(scoped_tables))
        .distinct()
    )
    inside = (
        select(s.table__term.c.term_id)
        .where(s.table__term.c.table_id.in_(scoped_tables))
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
    document: AutoOntologyModelDocument,
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
        # The payload is authoritative about which tables represent a term --
        # but only for the tables it carries. Scoped to those, so a term also
        # represented in a database outside this export keeps that link: the
        # unscoped delete removed it and the re-insert could not restore it,
        # because the id does not resolve. Before out-of-scope references were
        # skipped rather than rejected this could not happen, since such a
        # document aborted the import outright.
        # `id_map.values()` is every live id this document resolved. Only table
        # ids can match `table_id`, so the wider list is harmless and avoids
        # threading the document's table set through here.
        store().query_write(
            s.table__term.delete().where(
                s.table__term.c.term_id == live_term_id,
                s.table__term.c.table_id.in_(list(id_map.values())),
            )
        )
        for table_id in term.represents:
            live_table_id = _remap_optional(id_map, table_id)
            if live_table_id is None:
                # A table from another database this term also represents. The
                # export records it on purpose; this document cannot create it.
                logger.debug(
                    "import: term %s represents out-of-scope table %s; skipping",
                    term.id,
                    table_id,
                )
                continue
            represents_rows.append({"term_id": live_term_id, "table_id": live_table_id})
    _link(s.table__term, represents_rows)

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
    document: AutoOntologyModelDocument,
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
            # Deliberately '' when the column is unknown: the schema keeps
            # table_id NOT NULL but not a foreign key, exactly so this import
            # stays legal. Bound once here because the embedding context needs
            # the same value -- computing it twice is how the two drifted apart.
            live_table_id = (
                _remap(id_map, col_ctx.table_yaml_id, kind="column attribute table")
                if col_ctx
                else ""
            )
            attr_items.append(
                (
                    attr.id,
                    {
                        "name": attr.name,
                        "description": attr.description,
                        "source": SEMANTIC_SOURCE,
                        "term_name": term.name,
                        "source_column": col_ctx.name if col_ctx else "",
                        "table_id": live_table_id,
                    },
                ),
            )
            context[attr.id] = {
                "attr": attr,
                "term": term,
                "col_ctx": col_ctx,
                "live_term_id": live_term_id,
                "live_table_id": live_table_id,
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
                    table_id=ctx["live_table_id"],
                    table_name=col_ctx.table_name,
                    is_unique=col_ctx.is_unique,
                    sample_values=col_ctx.sample_values,
                    schema_name=col_ctx.schema_name,
                ),
            )

    _link(s.column__has_attribute, has_attribute_rows)
    _link(s.column_attribute__term, property_of_rows)


def _import_semantic_fks(
    document: AutoOntologyModelDocument, id_map: dict[str, str]
) -> None:
    rows = []
    for fk in document.semantic_layer.semantic_fks:
        column_id = _remap_optional(id_map, fk.column_id)
        attribute_id = _remap_optional(id_map, fk.column_attribute_id)
        if column_id is None or attribute_id is None:
            # The pair reaches an attribute whose term fell outside this export.
            logger.debug(
                "import: semantic fk %s -> %s is out of scope; skipping",
                fk.column_id,
                fk.column_attribute_id,
            )
            continue
        rows.append({"column_id": column_id, "attribute_id": attribute_id})
    _link(
        s.column__semantic_fk,
        rows,
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


def _safe_cached_dialects_and_schemas(
    cache: dict[str | None, tuple[list[str], dict[str, Any]]],
    database_name: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """Like :func:`_cached_dialects_and_schemas`, but never raises.

    Used by the export path's ``sql_column_resolver``, where
    ``sql_column_is`` is a best-effort cross-reference: a broken connector
    or a transient catalog-lookup error for one database must not fail the
    whole export. The failure is cached too (as empty dialects/schemas), so
    a persistently broken *database_name* doesn't re-raise on every
    sql_attribute/custom_analysis that references it.
    """
    if database_name not in cache:
        try:
            cache[database_name] = (
                get_dialects(database_name),
                get_schemas(database_name),
            )
        except Exception:
            logger.debug(
                "Could not build dialects/schemas for database %r; "
                "sql_column_is will be empty for SQL referencing it",
                database_name,
                exc_info=True,
            )
            cache[database_name] = ([], {})
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
    document: AutoOntologyModelDocument,
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
    document: AutoOntologyModelDocument,
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
