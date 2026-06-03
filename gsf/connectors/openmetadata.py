# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenMetadata metadata connector.

Unlike the :class:`~nemo_retriever.tabular_data.sql_database.SQLDatabase`
connectors (Postgres, DuckDB), this is **not** a queryable SQL source.
OpenMetadata is a metadata catalog exposed over a REST API — it describes
tables, it is not a table you can run SQL against. So this connector does not
implement ``execute`` / ``get_tables`` / text-to-SQL; it exposes metadata-fetch
functions instead.

Current scope: table and column descriptions. The class is the natural home for
future metadata pulls (tags, PII labels, lineage) without disturbing the SQL
connector abstraction.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_PAGE_SIZE = 100

# Columns returned by :meth:`OpenMetadataConnector.get_descriptions`.
DESCRIPTION_COLUMNS = [
    "level",
    "database",
    "schema",
    "table",
    "column",
    "description",
    "fqn",
]


def _split_fqn(fqn: str) -> list[str]:
    """Split an OpenMetadata FQN on unquoted dots.

    OpenMetadata wraps any name part that itself contains a dot in double
    quotes (e.g. ``service.db.schema."odd.name"``), so a naive ``split('.')``
    would over-split. This respects the quoting.
    """
    parts: list[str] = []
    buf: list[str] = []
    in_quotes = False
    for ch in fqn:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "." and not in_quotes:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


class OpenMetadataConnector:
    """Read-only metadata connector for an OpenMetadata server.

    Parameters
    ----------
    host:
        Base URL of the OpenMetadata server, e.g. ``http://localhost:8585``.
    token:
        Bearer token (the ingestion-bot JWT). Optional, but most servers
        require it.
    timeout:
        Per-request timeout in seconds.
    """

    kind = "openmetadata"

    def __init__(
        self,
        host: str,
        token: str = "",
        *,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not host:
            raise ValueError(
                "OpenMetadataConnector requires a host, e.g. http://localhost:8585"
            )
        self._base = f"{host.rstrip('/')}/api/v1"
        self._token = token
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenMetadataConnector":
        """Build a connector from ``OPENMETADATA_HOST`` / ``OPENMETADATA_TOKEN``.

        These are the same env vars the frontend proxy uses, so there is a
        single source of truth for the OpenMetadata connection.
        """
        host = os.environ.get("OPENMETADATA_HOST", "")
        token = os.environ.get("OPENMETADATA_TOKEN", "")
        if not host:
            raise ValueError(
                "OPENMETADATA_HOST is not set. Add it to your .env, e.g.:\n\n"
                "    OPENMETADATA_HOST=http://localhost:8585\n"
                "    OPENMETADATA_TOKEN=<ingestion-bot JWT>"
            )
        return cls(host, token)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def iter_tables(
        self,
        fields: str = "columns,database,databaseSchema",
    ) -> Iterator[dict]:
        """Yield every table entity, transparently following pagination."""
        after: Optional[str] = None
        with httpx.Client(timeout=self._timeout, headers=self._headers()) as client:
            while True:
                params: dict[str, object] = {"fields": fields, "limit": _PAGE_SIZE}
                if after:
                    params["after"] = after
                resp = client.get(f"{self._base}/tables", params=params)
                resp.raise_for_status()
                body = resp.json()
                for table in body.get("data", []):
                    yield table
                after = (body.get("paging") or {}).get("after")
                if not after:
                    break

    def get_descriptions(self) -> pd.DataFrame:
        """Return all non-empty table and column descriptions.

        One row per described entity, with columns
        ``level`` (``'table'`` | ``'column'``), ``database``, ``schema``,
        ``table``, ``column`` (``None`` for table-level rows), ``description``
        and ``fqn``.
        """
        records: list[dict[str, Optional[str]]] = []
        for table in self.iter_tables():
            database, schema = self._db_schema(table)
            table_name = table.get("name")

            table_desc = (table.get("description") or "").strip()
            if table_desc:
                records.append(
                    {
                        "level": "table",
                        "database": database,
                        "schema": schema,
                        "table": table_name,
                        "column": None,
                        "description": table_desc,
                        "fqn": table.get("fullyQualifiedName"),
                    }
                )

            for column in table.get("columns") or []:
                column_desc = (column.get("description") or "").strip()
                if not column_desc:
                    continue
                records.append(
                    {
                        "level": "column",
                        "database": database,
                        "schema": schema,
                        "table": table_name,
                        "column": column.get("name"),
                        "description": column_desc,
                        "fqn": column.get("fullyQualifiedName"),
                    }
                )

        return pd.DataFrame.from_records(records, columns=DESCRIPTION_COLUMNS)

    @staticmethod
    def _db_schema(table: dict) -> tuple[Optional[str], Optional[str]]:
        """Resolve (database, schema) names from a table entity.

        Prefers the structured entity references; falls back to parsing the
        fully-qualified name (``service.database.schema.table``).
        """
        database = (table.get("database") or {}).get("name")
        schema = (table.get("databaseSchema") or {}).get("name")
        if database and schema:
            return database, schema

        parts = _split_fqn(table.get("fullyQualifiedName") or "")
        if len(parts) >= 4:
            return parts[-3], parts[-2]
        return database, schema
