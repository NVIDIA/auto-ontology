# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j read/write helpers for GSF model YAML export/import."""

from __future__ import annotations

import json
import logging
from typing import Any

from gsf.catalog.store.queries import add_query
from gsf.catalog.model import CatalogNode
from gsf.catalog.constants import (
    Edges,
    Labels,
    Props,
)

from gsf.dal.custom_analyses import detach_existing_sql_edges as detach_ca_sql_edges
from gsf.dal.neo4j_tx import graph, write_transaction
from gsf.dal.sql_attributes import detach_existing_sql_edges, link_to_term
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_HAS_ATTRIBUTE,
    REL_PROPERTY_OF,
    REL_REPRESENTS,
    REL_SEMANTIC_FK,
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


class ModelInterchangeError(Exception):
    """Base error for model interchange operations."""


class UnknownDatabaseIdsError(ModelInterchangeError):
    """Raised when one or more requested database ids do not exist."""

    def __init__(self, database_ids: list[str]) -> None:
        self.database_ids = database_ids
        super().__init__(f"Unknown database id(s): {', '.join(database_ids)}")


class ModelImportValidationError(ModelInterchangeError):
    """Raised when an import payload references ids outside the export scope."""


_EXPORT_CATALOG_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(sch:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(tbl:{Labels.TABLE})
      -[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
RETURN db.id AS db_id,
       db.name AS db_name,
       sch.id AS schema_id,
       sch.name AS schema_name,
       tbl.id AS table_id,
       tbl.name AS table_name,
       tbl.description AS table_description,
       tbl.pk AS pk,
       tbl.table_type AS table_type,
       col.id AS column_id,
       col.name AS column_name,
       col.description AS column_description,
       col.data_type AS column_type,
       col.sample_values AS sample_values,
       coalesce(col.is_unique, false) AS is_unique,
       coalesce(col.is_nullable, true) AS is_nullable,
       col.ordinal_position AS ordinal_position
ORDER BY db_name, schema_name, table_name, ordinal_position
"""

_EXPORT_FKS_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(:{Labels.TABLE})
      -[:{Edges.CONTAINS}]->(src:{Labels.COLUMN})
      -[:{Edges.FOREIGN_KEY}]->(tgt:{Labels.COLUMN})
RETURN DISTINCT src.id AS source_column_id,
                tgt.id AS target_column_id
"""

_EXPORT_JOINS_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(t1:{Labels.TABLE})
      -[j:{Edges.JOIN}]->(t2:{Labels.TABLE})
WHERE size($database_ids) = 0
   OR EXISTS {{
        MATCH (db2:{Labels.DB})-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t2)
        WHERE db2.id IN $database_ids
   }}
RETURN DISTINCT t1.id AS source_table_id,
                t2.id AS target_table_id,
                j.join_columns AS join_columns
"""

_EXPORT_TERMS_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(tbl:{Labels.TABLE})
OPTIONAL MATCH (tbl)-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{source: $source}})
WITH DISTINCT term
WHERE term IS NOT NULL
OPTIONAL MATCH (tbl2:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term)
WITH term, collect(DISTINCT tbl2.id) AS represents
OPTIONAL MATCH (col:{Labels.COLUMN})-[:{REL_HAS_ATTRIBUTE}]->
              (attr:{LABEL_COLUMN_ATTRIBUTE})-[:{REL_PROPERTY_OF}]->(term)
WITH term, represents,
     collect(DISTINCT {{
         id: attr.id,
         name: attr.name,
         description: coalesce(attr.description, ''),
         column_id: col.id
     }}) AS column_attributes
RETURN term.id AS id,
       term.name AS name,
       coalesce(term.description, '') AS description,
       [x IN represents WHERE x IS NOT NULL] AS represents,
       [x IN column_attributes WHERE x.id IS NOT NULL] AS columns_attributes
ORDER BY term.name
"""

_EXPORT_SEMANTIC_FKS_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(:{Labels.TABLE})
      -[:{Edges.CONTAINS}]->(col:{Labels.COLUMN})
      -[:{REL_SEMANTIC_FK}]->(attr:{LABEL_COLUMN_ATTRIBUTE})
RETURN DISTINCT col.id AS column_id,
                attr.id AS column_attribute_id
"""

_EXPORT_SQL_ATTRIBUTES_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(:{Labels.TABLE})<-[:{Edges.SQL}]-
      (sql:{Labels.SQL})<-[:{Edges.HAS_SQL}]-
      (attr:{LABEL_SQL_ATTRIBUTE})
MATCH (attr)-[:{REL_PROPERTY_OF}]->(term:{LABEL_TERM})
RETURN DISTINCT attr.id AS id,
                attr.name AS name,
                coalesce(attr.description, '') AS description,
                coalesce(attr.expression, '') AS expression,
                coalesce(attr.source, $default_source) AS source,
                sql.sql_full_query AS sql,
                term.id AS term_id,
                db.name AS database_name
ORDER BY attr.name
"""

_EXPORT_CUSTOM_ANALYSES_QUERY = f"""
MATCH (db:{Labels.DB})
WHERE size($database_ids) = 0 OR db.id IN $database_ids
MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
      -[:{Edges.CONTAINS}]->(:{Labels.TABLE})<-[:{Edges.SQL}]-
      (sql:{Labels.SQL})<-[:{Edges.HAS_SQL}]-
      (ca:{Labels.CUSTOM_ANALYSIS})
RETURN DISTINCT ca.id AS id,
                ca.name AS name,
                coalesce(ca.description, '') AS description,
                sql.sql_full_query AS sql,
                db.name AS database_name
ORDER BY ca.name
"""

_LIST_DATABASE_IDS_QUERY = f"""
MATCH (db:{Labels.DB})
RETURN collect(db.id) AS ids
"""


def validate_database_ids(database_ids: list[str]) -> None:
    """Raise :class:`UnknownDatabaseIdsError` when any id is missing."""
    if not database_ids:
        return
    rows = graph().query_read(_LIST_DATABASE_IDS_QUERY)
    known = set(rows[0]["ids"]) if rows else set()
    unknown = [db_id for db_id in database_ids if db_id not in known]
    if unknown:
        raise UnknownDatabaseIdsError(unknown)


def fetch_export_rows(database_ids: list[str]) -> dict[str, Any]:
    """Return raw Neo4j rows used to assemble a :class:`GsfModelDocument`."""
    params = {"database_ids": database_ids, "source": SEMANTIC_SOURCE}
    conn = graph()
    return {
        "catalog": conn.query_read(_EXPORT_CATALOG_QUERY, params),
        "foreign_keys": conn.query_read(_EXPORT_FKS_QUERY, params),
        "joins": conn.query_read(_EXPORT_JOINS_QUERY, params),
        "terms": conn.query_read(_EXPORT_TERMS_QUERY, params),
        "semantic_fks": conn.query_read(_EXPORT_SEMANTIC_FKS_QUERY, params),
        "sql_attributes": conn.query_read(
            _EXPORT_SQL_ATTRIBUTES_QUERY,
            {**params, "default_source": SQL_ATTR_SOURCE_MANUAL},
        ),
        "custom_analyses": conn.query_read(_EXPORT_CUSTOM_ANALYSES_QUERY, params),
    }


def assemble_export_document(
    rows: dict[str, Any],
    *,
    dialect_by_db_name: dict[str, str],
    sql_column_resolver: Any,
) -> GsfModelDocument:
    """Build a validated document from raw Neo4j export rows."""
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
                row.get("sql") or "",
                row.get("database_name"),
            ),
        )
        for row in rows["custom_analyses"]
        if row.get("id")
    ]
    return GsfModelDocument(
        data_layer=ModelDataLayer(
            databases=databases,
            foreign_keys=foreign_keys,
            joins=joins,
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
        schemas = db_entry["schemas"]
        schema_entry = schemas.setdefault(
            schema_id,
            {
                "id": schema_id,
                "name": row.get("schema_name") or "",
                "database_name": row.get("db_name") or "",
                "tables": {},
            },
        )
        table_id = str(row["table_id"])
        tables = schema_entry["tables"]
        table_entry = tables.setdefault(
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
            sample_values = parse_sample_values(row.get("sample_values")) or []
            table_entry["columns"].append(
                ModelColumn(
                    id=str(row["column_id"]),
                    name=row.get("column_name") or "",
                    description=row.get("column_description") or "",
                    type=row.get("column_type") or "",
                    sample_values=sample_values,
                    is_nullable=bool(row.get("is_nullable", True)),
                    is_unique=bool(row.get("is_unique", False)),
                ),
            )

    databases: list[ModelDatabase] = []
    for db_entry in db_map.values():
        schemas: list[ModelSchema] = []
        for schema_entry in db_entry["schemas"].values():
            tables = [
                ModelTable(**table_entry)
                for table_entry in schema_entry["tables"].values()
            ]
            schemas.append(
                ModelSchema(
                    id=schema_entry["id"],
                    name=schema_entry["name"],
                    database_name=schema_entry["database_name"],
                    tables=tables,
                ),
            )
        databases.append(
            ModelDatabase(
                id=db_entry["id"],
                dialect=db_entry["dialect"],
                schemas=schemas,
            ),
        )
    databases.sort(key=lambda db: db.id)
    return databases


def _assemble_sql_attributes(
    rows: list[dict[str, Any]],
    sql_column_resolver: Any,
) -> ModelSqlAttributesBySource:
    grouped: dict[str, list[ModelSqlAttribute]] = {
        "manual": [],
        "table": [],
        "sql": [],
        "bridge_table": [],
    }
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
            sql,
            get_dialects(database_name),
            get_schemas(database_name),
        )
    except Exception:
        logger.debug("Could not resolve sql_column_is for SQL snippet", exc_info=True)
        return []
    column_ids = query_obj.get_column_ids()
    return [str(col_id) for col_id in column_ids if col_id]


_IMPORTED_ID_INDEX_LABELS = (
    Labels.DB,
    Labels.SCHEMA,
    Labels.TABLE,
    Labels.COLUMN,
    Labels.CUSTOM_ANALYSIS,
    LABEL_TERM,
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
)


def _ensure_import_indexes() -> None:
    """Ensure ``id``/``imported_id`` are indexed for every label resolved by
    :func:`_resolve_entities_batch`.

    Without an index, ``MATCH (n:Label) WHERE n.imported_id = x OR n.id =
    x`` falls back to a full label scan per lookup. That's invisible on a
    handful of nodes but turns a several-thousand-column import into a
    quadratic-time crawl. ``nemo_retriever`` already indexes ``id`` for its
    own catalog labels but knows nothing about ``imported_id``, or about
    GSF's semantic labels (Term/ColumnAttribute/SqlAttribute), so this fills
    the gap. Statements are idempotent (``IF NOT EXISTS``) and near-instant
    once the indexes exist, so it's cheap to call on every import. Schema
    changes can't run inside a transaction that also writes data, so this
    must be called before :func:`write_transaction` opens one.
    """
    conn = graph()
    for label in _IMPORTED_ID_INDEX_LABELS:
        conn.query_write(
            f"""
            CREATE CONSTRAINT constraint_on_{label.lower()}_id IF NOT EXISTS
            FOR (n:{label}) REQUIRE (n.id) IS UNIQUE
            """,
        )
        conn.query_write(
            f"""
            CREATE INDEX index_on_{label.lower()}_imported_id IF NOT EXISTS
            FOR (n:{label}) ON (n.imported_id)
            """,
        )


def apply_import_model(
    document: GsfModelDocument,
    *,
    replace: bool,
    embed_buffer: ImportEmbedBuffer | None = None,
) -> dict[str, Any]:
    """Apply a validated model document to Neo4j.

    Entities are matched by ``imported_id`` (the YAML ``id``). When a node
    already carries that ``imported_id`` (or its live ``id`` equals the YAML
    id), it is skipped. Otherwise a new node is created with a fresh ``id``
    and ``imported_id`` set to the YAML id. Catalog nodes are created when
    missing, so import works against an empty Neo4j.

    When *embed_buffer* is supplied, pre-embed rows for newly created nodes
    are appended for a later :func:`flush_import_embeddings` call.

    The catalog/semantic-graph portion (databases through semantic FKs) runs
    in one transaction, so a failure part-way through leaves the graph
    untouched rather than half-populated. SQL attributes and custom analyses
    are applied afterwards, outside that transaction: persisting them calls
    into ``nemo_retriever``'s :func:`add_query`, which opens its own
    auto-commit Neo4j session. Running that from inside our transaction can
    self-deadlock — our open transaction holds locks on Table/Column nodes
    it just created, ``add_query()``'s separate session blocks waiting on
    those same locks, and we're synchronously stuck waiting for it to
    return.
    """
    _ensure_import_indexes()

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

    with write_transaction():
        live_db_ids = _import_catalog(
            document,
            id_map,
            created,
            skipped,
            embed_buffer,
            column_meta,
        )

        if replace:
            _delete_scoped_semantics_not_in_payload(document, live_db_ids)

        _import_foreign_keys(document, id_map)
        _import_joins(document, id_map)
        _import_terms(document, id_map, created, skipped, embed_buffer)
        _import_column_attributes(
            document,
            id_map,
            created,
            skipped,
            embed_buffer,
            column_meta,
        )
        _import_semantic_fks(document, id_map)

    term_names = {term.id: term.name for term in document.semantic_layer.terms}
    schema_cache: dict[str | None, tuple[list[str], dict[str, Any]]] = {}
    _import_sql_attributes(
        document,
        id_map,
        created,
        skipped,
        embed_buffer,
        term_names,
        schema_cache,
    )
    _import_custom_analyses(
        document,
        id_map,
        created,
        skipped,
        embed_buffer,
        schema_cache,
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
            for key in ("manual", "table", "sql", "bridge_table")
        ),
        "custom_analyses": len(document.semantic_layer.custom_analyses),
    }
    if embed_buffer is not None:
        summary["pending_embed"] = {
            "data_rows": len(embed_buffer.data_rows),
            "semantic_rows": len(embed_buffer.semantic_rows),
        }
    return summary


def _resolve_entity(
    label: str,
    imported_id: str,
    *,
    create_props: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Return ``(live_id, created)`` for a YAML entity id.

    Skips creation when a node already has ``imported_id`` equal to the YAML
    id, or when a node already has ``id`` equal to that value (first import
    onto an existing catalog). Newly created nodes get a fresh UUID ``id``
    and ``imported_id`` set to the YAML id.
    """
    conn = graph()
    existing = conn.query_read(
        f"""
        MATCH (n:{label})
        WHERE n.imported_id = $imported_id OR n.id = $imported_id
        RETURN n.id AS id
        LIMIT 1
        """,
        {"imported_id": imported_id},
    )
    if existing:
        live_id = str(existing[0]["id"])
        conn.query_write(
            f"""
            MATCH (n:{label} {{id: $id}})
            SET n.imported_id = coalesce(n.imported_id, $imported_id)
            """,
            {"id": live_id, "imported_id": imported_id},
        )
        return live_id, False

    props = dict(create_props or {})
    props["imported_id"] = imported_id
    rows = conn.query_write(
        f"""
        CREATE (n:{label})
        SET n.id = randomUUID(),
            n += $props
        RETURN n.id AS id
        """,
        {"props": props},
    )
    return str(rows[0]["id"]), True


def _resolve_entities_batch(
    label: str,
    items: list[tuple[str, dict[str, Any]]],
) -> dict[str, tuple[str, bool]]:
    """Batched :func:`_resolve_entity`.

    Large imports (thousands of columns) were previously doing two Neo4j
    round trips *per entity*, which dominates import time. This resolves an
    entire same-label batch of ``(imported_id, create_props)`` pairs in at
    most three round trips total: one read to find existing nodes, one write
    to touch their ``imported_id``, and one write to create the missing
    ones. Semantics match :func:`_resolve_entity` exactly.
    """
    if not items:
        return {}

    # Guard against duplicate YAML ids within a batch: keep the first
    # create_props seen, matching what repeated sequential calls would do
    # (the first call creates the node, later calls just find it).
    deduped: dict[str, dict[str, Any]] = {}
    for imported_id, create_props in items:
        deduped.setdefault(imported_id, create_props)

    conn = graph()
    imported_ids = list(deduped.keys())
    existing_rows = conn.query_read(
        f"""
        UNWIND $imported_ids AS imported_id
        MATCH (n:{label})
        WHERE n.imported_id = imported_id OR n.id = imported_id
        RETURN imported_id, n.id AS live_id
        """,
        {"imported_ids": imported_ids},
    )
    existing_map = {row["imported_id"]: str(row["live_id"]) for row in existing_rows}

    result: dict[str, tuple[str, bool]] = {}
    touch_rows: list[dict[str, Any]] = []
    create_rows: list[dict[str, Any]] = []
    for imported_id, create_props in deduped.items():
        live_id = existing_map.get(imported_id)
        if live_id is not None:
            result[imported_id] = (live_id, False)
            touch_rows.append({"id": live_id, "imported_id": imported_id})
        else:
            props = dict(create_props or {})
            props["imported_id"] = imported_id
            create_rows.append({"imported_id": imported_id, "props": props})

    if touch_rows:
        conn.query_write(
            f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{id: row.id}})
            SET n.imported_id = coalesce(n.imported_id, row.imported_id)
            """,
            {"rows": touch_rows},
        )

    if create_rows:
        created_rows = conn.query_write(
            f"""
            UNWIND $rows AS row
            CREATE (n:{label})
            SET n.id = randomUUID(),
                n += row.props
            RETURN row.imported_id AS imported_id, n.id AS live_id
            """,
            {"rows": create_rows},
        )
        for row in created_rows:
            result[row["imported_id"]] = (str(row["live_id"]), True)

    return result


def _database_names_for_terms(term_ids: list[str]) -> dict[str, str]:
    """Batched lookup of a representative database name per term id."""
    if not term_ids:
        return {}
    rows = graph().query_read(
        f"""
        UNWIND $term_ids AS term_id
        MATCH (tbl:{Labels.TABLE})-[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{id: term_id}})
        MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(tbl)
        RETURN term_id, collect(db.name)[0] AS database_name
        """,
        {"term_ids": term_ids},
    )
    return {
        row["term_id"]: row["database_name"] for row in rows if row.get("database_name")
    }


def _database_names_for_columns(column_ids: list[str]) -> dict[str, str]:
    """Batched lookup of a representative database name per column id."""
    if not column_ids:
        return {}
    rows = graph().query_read(
        f"""
        UNWIND $column_ids AS column_id
        MATCH (col:{Labels.COLUMN} {{id: column_id}})<-[:{Edges.CONTAINS}]-
              (tbl:{Labels.TABLE})<-[:{Edges.CONTAINS}]-
              (sch:{Labels.SCHEMA})<-[:{Edges.CONTAINS}]-
              (db:{Labels.DB})
        RETURN column_id, collect(db.name)[0] AS database_name
        """,
        {"column_ids": column_ids},
    )
    return {
        row["column_id"]: row["database_name"]
        for row in rows
        if row.get("database_name")
    }


def _remap(id_map: dict[str, str], yaml_id: str, *, kind: str) -> str:
    live_id = id_map.get(yaml_id)
    if not live_id:
        raise ModelImportValidationError(
            f"Cannot resolve {kind} id {yaml_id!r} — missing from catalog/semantic import",
        )
    return live_id


def _import_catalog(
    document: GsfModelDocument,
    id_map: dict[str, str],
    created: dict[str, int],
    skipped: dict[str, int],
    embed_buffer: ImportEmbedBuffer | None,
    column_meta: dict[str, ColumnCatalogMeta],
) -> list[str]:
    """Import databases/schemas/tables/columns level by level.

    Each level is resolved and wired up in a small, constant number of
    Neo4j round trips via :func:`_resolve_entities_batch` and one batched
    ``MERGE`` for the containment edges, instead of two round trips *per
    entity*. This is what makes large catalogs (thousands of columns)
    import in seconds rather than minutes.
    """
    databases = document.data_layer.databases

    db_names: dict[str, str] = {}
    for db in databases:
        db_name = ""
        if db.schemas:
            db_name = db.schemas[0].database_name or db.schemas[0].name or ""
        db_names[db.id] = db_name

    db_results = _resolve_entities_batch(
        Labels.DB,
        [
            (db.id, {"name": db_names[db.id]} if db_names[db.id] else {})
            for db in databases
        ],
    )
    live_db_ids: list[str] = []
    for db in databases:
        live_db_id, was_created = db_results[db.id]
        id_map[db.id] = live_db_id
        live_db_ids.append(live_db_id)
        if was_created:
            created["databases"] += 1
        else:
            skipped["databases"] += 1

    schema_parent_db: dict[str, str] = {}
    schema_db_names: dict[str, str] = {}
    schema_items: list[tuple[str, dict[str, Any]]] = []
    for db in databases:
        for schema in db.schemas:
            schema_parent_db[schema.id] = db.id
            schema_db_names[schema.id] = schema.database_name or db_names[db.id]
            schema_items.append((schema.id, {"name": schema.name}))

    schema_results = _resolve_entities_batch(Labels.SCHEMA, schema_items)
    for schema_id, (live_schema_id, sch_created) in schema_results.items():
        id_map[schema_id] = live_schema_id
        if sch_created:
            created["schemas"] += 1
        else:
            skipped["schemas"] += 1

    if schema_items:
        graph().query_write(
            f"""
            UNWIND $rows AS row
            MATCH (db:{Labels.DB} {{id: row.db_id}})
            MATCH (sch:{Labels.SCHEMA} {{id: row.schema_id}})
            MERGE (db)-[:{Edges.CONTAINS}]->(sch)
            """,
            {
                "rows": [
                    {
                        "db_id": id_map[schema_parent_db[schema_id]],
                        "schema_id": id_map[schema_id],
                    }
                    for schema_id, _ in schema_items
                ],
            },
        )

    table_parent_schema: dict[str, str] = {}
    table_items: list[tuple[str, dict[str, Any]]] = []
    for db in databases:
        for schema in db.schemas:
            for table in schema.tables:
                table_parent_schema[table.id] = schema.id
                table_items.append(
                    (
                        table.id,
                        {
                            "name": table.name,
                            "description": table.description,
                            "pk": table.pk,
                            "table_type": table.type,
                        },
                    ),
                )

    table_results = _resolve_entities_batch(Labels.TABLE, table_items)
    for table_id, (live_table_id, tbl_created) in table_results.items():
        id_map[table_id] = live_table_id
        if tbl_created:
            created["tables"] += 1
        else:
            skipped["tables"] += 1

    if table_items:
        graph().query_write(
            f"""
            UNWIND $rows AS row
            MATCH (sch:{Labels.SCHEMA} {{id: row.schema_id}})
            MATCH (tbl:{Labels.TABLE} {{id: row.table_id}})
            MERGE (sch)-[:{Edges.CONTAINS}]->(tbl)
            """,
            {
                "rows": [
                    {
                        "schema_id": id_map[table_parent_schema[table_id]],
                        "table_id": id_map[table_id],
                    }
                    for table_id, _ in table_items
                ],
            },
        )

    column_parent_table: dict[str, str] = {}
    column_items: list[tuple[str, dict[str, Any]]] = []
    for db in databases:
        for schema in db.schemas:
            schema_db_name = schema_db_names.get(schema.id, "")
            for table in schema.tables:
                for ordinal, column in enumerate(table.columns, start=1):
                    sample_values_json = (
                        json.dumps(column.sample_values)
                        if column.sample_values
                        else None
                    )
                    column_parent_table[column.id] = table.id
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
                                "sample_values": sample_values_json,
                                "is_unique": column.is_unique,
                                "is_nullable": column.is_nullable,
                                "ordinal_position": ordinal,
                            },
                        ),
                    )

    column_results = _resolve_entities_batch(Labels.COLUMN, column_items)
    for column_id, (live_col_id, col_created) in column_results.items():
        id_map[column_id] = live_col_id
        if col_created:
            created["columns"] += 1
        else:
            skipped["columns"] += 1

    if column_items:
        graph().query_write(
            f"""
            UNWIND $rows AS row
            MATCH (tbl:{Labels.TABLE} {{id: row.table_id}})
            MATCH (col:{Labels.COLUMN} {{id: row.column_id}})
            MERGE (tbl)-[:{Edges.CONTAINS}]->(col)
            """,
            {
                "rows": [
                    {
                        "table_id": id_map[column_parent_table[column_id]],
                        "column_id": id_map[column_id],
                    }
                    for column_id, _ in column_items
                ],
            },
        )

    if embed_buffer is not None:
        for db in databases:
            for schema in db.schemas:
                schema_db_name = schema_db_names.get(schema.id, "")
                for table in schema.tables:
                    live_table_id, tbl_created = table_results[table.id]
                    table_column_embed_specs: list[dict[str, Any]] = []
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
                        table_column_embed_specs.append(
                            {
                                "column_name": column.name,
                                "data_type": column.type,
                                "description": column.description,
                            },
                        )
                    if tbl_created:
                        embed_buffer.data_rows.append(
                            build_table_data_row(
                                live_id=live_table_id,
                                table_name=table.name,
                                table_description=table.description,
                                schema_name=schema.name,
                                database_name=schema_db_name,
                                columns=table_column_embed_specs,
                            ),
                        )

    return live_db_ids


def _import_foreign_keys(document: GsfModelDocument, id_map: dict[str, str]) -> None:
    rows = [
        {
            "source_column_id": _remap(
                id_map, fk.source_column_id, kind="foreign-key source column"
            ),
            "target_column_id": _remap(
                id_map, fk.target_column_id, kind="foreign-key target column"
            ),
        }
        for fk in document.data_layer.foreign_keys
    ]
    if not rows:
        return
    graph().query_write(
        f"""
        UNWIND $rows AS row
        MATCH (src:{Labels.COLUMN} {{id: row.source_column_id}})
        MATCH (tgt:{Labels.COLUMN} {{id: row.target_column_id}})
        MERGE (src)-[:{Edges.FOREIGN_KEY}]->(tgt)
        """,
        {"rows": rows},
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
    graph().query_write(
        f"""
        UNWIND $rows AS row
        MATCH (t1:{Labels.TABLE} {{id: row.source_table_id}})
        MATCH (t2:{Labels.TABLE} {{id: row.target_table_id}})
        MERGE (t1)-[j:{Edges.JOIN}]->(t2)
        SET j.join_columns = row.join_columns
        """,
        {"rows": rows},
    )


def _delete_scoped_semantics_not_in_payload(
    document: GsfModelDocument,
    live_db_ids: list[str],
) -> None:
    keep_term_ids = [term.id for term in document.semantic_layer.terms]
    keep_attr_ids = [
        attr.id
        for term in document.semantic_layer.terms
        for attr in term.columns_attributes
    ]
    keep_sql_attr_ids = [
        attr.id
        for attrs in (
            document.semantic_layer.sql_attributes.manual,
            document.semantic_layer.sql_attributes.table,
            document.semantic_layer.sql_attributes.sql,
            document.semantic_layer.sql_attributes.bridge_table,
        )
        for attr in attrs
    ]
    keep_ca_ids = [ca.id for ca in document.semantic_layer.custom_analyses]
    params = {
        "database_ids": live_db_ids,
        "keep_term_ids": keep_term_ids,
        "keep_attr_ids": keep_attr_ids,
        "keep_sql_attr_ids": keep_sql_attr_ids,
        "keep_ca_ids": keep_ca_ids,
        "source": SEMANTIC_SOURCE,
    }
    conn = graph()
    # Match keep sets against imported_id (YAML id) or live id.
    conn.query_write(
        f"""
        MATCH (db:{Labels.DB})
        WHERE db.id IN $database_ids
        MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(:{Labels.TABLE})<-[:{Edges.SQL}]-
              (:{Labels.SQL})<-[:{Edges.HAS_SQL}]-
              (attr:{LABEL_SQL_ATTRIBUTE})
        WHERE NOT coalesce(attr.imported_id, attr.id) IN $keep_sql_attr_ids
        DETACH DELETE attr
        """,
        params,
    )
    conn.query_write(
        f"""
        MATCH (db:{Labels.DB})
        WHERE db.id IN $database_ids
        MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(:{Labels.TABLE})<-[:{Edges.SQL}]-
              (:{Labels.SQL})<-[:{Edges.HAS_SQL}]-
              (ca:{Labels.CUSTOM_ANALYSIS})
        WHERE NOT coalesce(ca.imported_id, ca.id) IN $keep_ca_ids
        DETACH DELETE ca
        """,
        params,
    )
    conn.query_write(
        f"""
        MATCH (db:{Labels.DB})
        WHERE db.id IN $database_ids
        MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(:{Labels.COLUMN})
              -[:{REL_HAS_ATTRIBUTE}|{REL_SEMANTIC_FK}]->(attr:{LABEL_COLUMN_ATTRIBUTE})
        WHERE NOT coalesce(attr.imported_id, attr.id) IN $keep_attr_ids
        DETACH DELETE attr
        """,
        params,
    )
    conn.query_write(
        f"""
        MATCH (db:{Labels.DB})
        WHERE db.id IN $database_ids
        MATCH (db)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(tbl:{Labels.TABLE})
              -[:{REL_REPRESENTS}]->(term:{LABEL_TERM} {{source: $source}})
        WITH term, collect(DISTINCT tbl.id) AS scoped_tables
        WHERE NOT coalesce(term.imported_id, term.id) IN $keep_term_ids
          AND all(
              table_id IN scoped_tables
              WHERE EXISTS {{
                  MATCH (db2:{Labels.DB})
                  WHERE db2.id IN $database_ids
                  MATCH (db2)-[:{Edges.CONTAINS}]->(:{Labels.SCHEMA})
                        -[:{Edges.CONTAINS}]->(:{Labels.TABLE} {{id: table_id}})
              }}
          )
        DETACH DELETE term
        """,
        params,
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
        LABEL_TERM,
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
    newly_created_term_ids: list[str] = []
    for term in terms:
        live_term_id, was_created = term_results[term.id]
        id_map[term.id] = live_term_id
        if was_created:
            created["terms"] += 1
            newly_created_term_ids.append(live_term_id)
        else:
            skipped["terms"] += 1
        table_ids = [
            _remap(id_map, table_id, kind="term represents table")
            for table_id in term.represents
        ]
        represents_rows.append({"term_id": live_term_id, "table_ids": table_ids})

    if represents_rows:
        graph().query_write(
            f"""
            UNWIND $rows AS row
            MATCH (term:{LABEL_TERM} {{id: row.term_id}})
            OPTIONAL MATCH (:{Labels.TABLE})-[old:{REL_REPRESENTS}]->(term)
            DELETE old
            WITH term, row
            UNWIND row.table_ids AS table_id
            MATCH (tbl:{Labels.TABLE} {{id: table_id}})
            MERGE (tbl)-[:{REL_REPRESENTS}]->(term)
            """,
            {"rows": represents_rows},
        )

    if embed_buffer is not None and newly_created_term_ids:
        db_names_by_term = _database_names_for_terms(newly_created_term_ids)
        for term in terms:
            live_term_id, was_created = term_results[term.id]
            if not was_created:
                continue
            embed_buffer.semantic_rows.extend(
                build_term_semantic_rows(
                    database_name=db_names_by_term.get(live_term_id, ""),
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
    attr_context: dict[str, dict[str, Any]] = {}

    for term in document.semantic_layer.terms:
        live_term_id = _remap(id_map, term.id, kind="term")
        for attr in term.columns_attributes:
            col_ctx = column_meta.get(attr.column_id)
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
            attr_context[attr.id] = {
                "attr": attr,
                "term": term,
                "col_ctx": col_ctx,
                "live_term_id": live_term_id,
            }

    attr_results = _resolve_entities_batch(LABEL_COLUMN_ATTRIBUTE, attr_items)

    edge_rows: list[dict[str, Any]] = []
    for attr_id, (live_attr_id, was_created) in attr_results.items():
        ctx = attr_context[attr_id]
        attr = ctx["attr"]
        term = ctx["term"]
        col_ctx = ctx["col_ctx"]
        id_map[attr_id] = live_attr_id
        if was_created:
            created["column_attributes"] += 1
        else:
            skipped["column_attributes"] += 1
        live_col_id = _remap(id_map, attr.column_id, kind="column attribute column")
        edge_rows.append(
            {
                "attr_id": live_attr_id,
                "column_id": live_col_id,
                "term_id": ctx["live_term_id"],
            },
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

    if edge_rows:
        graph().query_write(
            f"""
            UNWIND $rows AS row
            MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: row.attr_id}})
            MATCH (col:{Labels.COLUMN} {{id: row.column_id}})
            MATCH (term:{LABEL_TERM} {{id: row.term_id}})
            MERGE (col)-[:{REL_HAS_ATTRIBUTE}]->(attr)
            MERGE (attr)-[:{REL_PROPERTY_OF}]->(term)
            """,
            {"rows": edge_rows},
        )


def _import_semantic_fks(document: GsfModelDocument, id_map: dict[str, str]) -> None:
    rows = [
        {
            "column_id": _remap(id_map, fk.column_id, kind="semantic fk column"),
            "column_attribute_id": _remap(
                id_map, fk.column_attribute_id, kind="semantic fk attribute"
            ),
        }
        for fk in document.semantic_layer.semantic_fks
    ]
    if not rows:
        return
    graph().query_write(
        f"""
        UNWIND $rows AS row
        MATCH (col:{Labels.COLUMN} {{id: row.column_id}})
        MATCH (attr:{LABEL_COLUMN_ATTRIBUTE} {{id: row.column_attribute_id}})
        MERGE (col)-[:{REL_SEMANTIC_FK}]->(attr)
        """,
        {"rows": rows},
    )


def _cached_dialects_and_schemas(
    cache: dict[str | None, tuple[list[str], dict[str, Any]]],
    database_name: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """Memoized ``(dialects, schemas)`` lookup for one import call.

    ``get_schemas`` rebuilds the whole catalog snapshot for a database on
    every call, which is fine for a single ad-hoc SQL validation but adds up
    fast across hundreds of sql_attributes/custom_analyses that usually
    share the same handful of database names. The cache lives only for the
    duration of one :func:`apply_import_model` call, so it can't serve
    stale data across separate imports.
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
    dialects, schemas = _cached_dialects_and_schemas(schema_cache, database_name)
    query_obj = validate_sql(sql, dialects, schemas)
    props = {
        "name": name,
        "description": description,
    }
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
    edge_props = {Props.ANALYSIS_ID: node_id}
    query_obj.edges.append((node, query_obj.sql_node, edge_props))
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
    grouped_attrs: list[tuple[str, ModelSqlAttribute]] = [
        (_YAML_KEY_TO_SOURCE[yaml_key], attr)
        for yaml_key in ("manual", "table", "sql", "bridge_table")
        for attr in getattr(document.semantic_layer.sql_attributes, yaml_key)
    ]

    attr_results = _resolve_entities_batch(
        LABEL_SQL_ATTRIBUTE,
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
            for source, attr in grouped_attrs
        ],
    )
    for _source, attr in grouped_attrs:
        id_map[attr.id] = attr_results[attr.id][0]

    # SQL parsing/graph linking (validate_sql, add_query) is inherently
    # per-item, but the resolve step and the database-name lookups it needs
    # are batched up front to cut round trips.
    newly_created_term_ids = list(
        {
            _remap(id_map, attr.term_id, kind="sql attribute term")
            for _source, attr in grouped_attrs
            if attr_results[attr.id][1]
        },
    )
    db_names_by_term = _database_names_for_terms(newly_created_term_ids)

    for source, attr in grouped_attrs:
        live_attr_id, was_created = attr_results[attr.id]
        if not was_created:
            skipped["sql_attributes"] += 1
            continue
        created["sql_attributes"] += 1
        live_term_id = _remap(id_map, attr.term_id, kind="sql attribute term")
        database_name = db_names_by_term.get(live_term_id)
        detach_existing_sql_edges(live_attr_id)
        _persist_sql_object(
            node_label=LABEL_SQL_ATTRIBUTE,
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
    ca_results = _resolve_entities_batch(
        Labels.CUSTOM_ANALYSIS,
        [
            (analysis.id, {"name": analysis.name, "description": analysis.description})
            for analysis in analyses
        ],
    )
    for analysis in analyses:
        id_map[analysis.id] = ca_results[analysis.id][0]

    live_col_by_analysis: dict[str, str | None] = {}
    lookup_col_ids: list[str] = []
    for analysis in analyses:
        _live_id, was_created = ca_results[analysis.id]
        live_col_id: str | None = None
        if was_created and analysis.sql_column_is:
            try:
                live_col_id = _remap(
                    id_map,
                    analysis.sql_column_is[0],
                    kind="custom analysis column",
                )
            except ModelImportValidationError:
                live_col_id = None
        live_col_by_analysis[analysis.id] = live_col_id
        if live_col_id:
            lookup_col_ids.append(live_col_id)

    db_names_by_column = _database_names_for_columns(lookup_col_ids)

    for analysis in analyses:
        live_ca_id, was_created = ca_results[analysis.id]
        if not was_created:
            skipped["custom_analyses"] += 1
            continue
        created["custom_analyses"] += 1
        live_col_id = live_col_by_analysis.get(analysis.id)
        database_name = db_names_by_column.get(live_col_id) if live_col_id else None
        detach_ca_sql_edges(live_ca_id)
        _persist_sql_object(
            node_label=Labels.CUSTOM_ANALYSIS,
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
