# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SQL validation helpers used by both CustomAnalysis and SqlAttribute DALs."""

from __future__ import annotations

from typing import Any

from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single

from gsf.connectors import get_connectors
from gsf.dal.datasources import fetch_schema_ids_for_database
from gsf.retrieval.data_access.graph_schemas import (
    fetch_all_schema_ids,
    get_schemas_by_ids,
)


class SqlParseError(Exception):
    """SQL could not be parsed or resolved against the catalog."""


def get_dialects(database_name: str | None = None) -> list[str]:
    """Return SQL dialects from active connectors.

    When *database_name* is given, only the connector whose
    ``database_name`` matches is considered.
    """
    connectors = get_connectors()
    if database_name is not None:
        connectors = [
            c for c in connectors if getattr(c, "database_name", None) == database_name
        ]
    dialects = [c.dialect for c in connectors if getattr(c, "dialect", None)]
    if not dialects:
        return ["generic", "ansi", "postgres"]
    return dialects


def get_schemas(database_name: str | None = None) -> dict:
    """Return catalog snapshot for ``parse_query_single``.

    When *database_name* is given, the catalog is built from only that
    database's schema IDs. This avoids schema-name collisions (e.g. many
    SQLite DBs all using ``main``) that would otherwise merge/overwrite
    each other in the assembled ``all_schemas`` map.
    """
    if database_name is None:
        schemas_ids = fetch_all_schema_ids()
        return get_schemas_by_ids(schemas_ids)

    scoped_ids = fetch_schema_ids_for_database(database_name)
    return get_schemas_by_ids(scoped_ids)


def validate_sql(sql: str, dialects: list[str], schemas: dict) -> Any:
    """Parse ``sql`` against the catalog. Returns query_obj or raises.

    Raises :class:`SqlParseError` when the SQL can't be parsed
    (sqlglot syntax / dialect error) or doesn't resolve to any
    table known to the catalog.
    """
    try:
        query_obj = parse_query_single(sql=sql, dialects=dialects, schemas=schemas)
    except Exception as exc:
        raise SqlParseError(f"SQL parse error: {exc}") from exc

    if query_obj is None:
        raise SqlParseError(
            "SQL doesn't reference any table known to the catalog; "
            "ingest the schema first or check the query",
        )
    return query_obj
