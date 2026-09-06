# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull schema entities out of a SQLDatabase connector.

``extract_tabular_db_data`` takes a **connector** directly: it reads nothing
else, and taking a wrapper object would leave a library dependency in code that
is otherwise storage- and library-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from gsf.catalog.normalize import normalize_columns, normalize_tables

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gsf.connectors.base import SQLDatabase


class IncompleteCatalogExtractionError(RuntimeError):
    """Raised when listed relations do not have a complete column inventory."""


def _relation_keys(frame: Any) -> set[tuple[str, str]]:
    if frame is None or frame.empty:
        return set()
    keys: set[tuple[str, str]] = set()
    for _, row in frame.iterrows():
        raw_schema = row.get("table_schema")
        raw_table = row.get("table_name")
        schema = "" if pd.isna(raw_schema) else str(raw_schema)
        table = "" if pd.isna(raw_table) else str(raw_table)
        if table:
            keys.add((schema, table))
    return keys


def _validate_relation_column_coverage(tables: Any, columns: Any) -> None:
    """Fail closed when a connector lists relations it could not describe."""
    listed = _relation_keys(tables)
    if not listed:
        return
    described = _relation_keys(columns)
    missing = sorted(listed - described)
    if not missing:
        return
    preview = ", ".join(
        f"{schema + '.' if schema else ''}{table}" for schema, table in missing[:10]
    )
    suffix = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
    raise IncompleteCatalogExtractionError(
        f"Catalog extraction listed {len(listed)} relation(s), but "
        f"{len(missing)} had no described columns: {preview}{suffix}"
    )


def create_dataframe(connector: "SQLDatabase"):
    """Extract raw schema DataFrames from any SQLDatabase connector."""
    tables = connector.get_tables()
    columns = connector.get_columns()
    views = connector.get_views()
    queries = connector.get_queries()
    pks = connector.get_pks()
    fks = connector.get_fks()
    _validate_relation_column_coverage(tables, columns)
    return tables, columns, views, queries, pks, fks


def data_for_populate_tabular(connector: "SQLDatabase") -> dict[str, Any]:
    """Build the ``data`` dict ``populate_tabular_data()`` expects."""
    tables, columns, views, queries, pks, fks = create_dataframe(connector)
    tables = normalize_tables(tables)
    columns = normalize_columns(columns)
    data = {
        "database_name": connector.database_name,
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
        "queries": queries,
    }
    return data


def extract_tabular_db_data(connector: "SQLDatabase | None" = None) -> dict[str, Any]:
    """Step 1 — pull schema entities from the relational DB into a data dict.

    Returns a data dict with keys ``database_name``, ``tables``, ``columns``,
    ``views``, ``pks``, ``fks`` and ``queries``, or ``{}`` when *connector* is
    ``None``.
    """
    if connector is None:
        return {}
    return data_for_populate_tabular(connector)
