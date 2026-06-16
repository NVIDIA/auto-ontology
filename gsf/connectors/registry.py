# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and cache SQLDatabase connectors from CONNECTION_STRINGS."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from nemo_retriever.tabular_data.sql_database import SQLDatabase

from gsf.connectors.duckdb import DuckDBDatabase
from gsf.connectors.heavydb import HeavyDBDatabase
from gsf.connectors.postgres import PostgresDatabase
from gsf.connectors.snowflake import SnowflakeDatabase

logger = logging.getLogger(__name__)

CONNECTOR_REGISTRY: dict[str, type[SQLDatabase]] = {
    "postgresql": PostgresDatabase,
    "postgres": PostgresDatabase,
    "duckdb": DuckDBDatabase,
    "snowflake": SnowflakeDatabase,
    "heavydb": HeavyDBDatabase,
}

_connectors: dict[str, SQLDatabase] | None = None


def create_connector(connection_string: str) -> SQLDatabase:
    """Parse *connection_string*, select a connector class, and return an instance."""
    try:
        parsed = urlparse(connection_string)
        scheme = parsed.scheme

        if not scheme:
            raise ValueError("Invalid connection string: No protocol scheme found.")

        db_type = scheme.split("+", 1)[0].lower()
        connector_class = CONNECTOR_REGISTRY.get(db_type)

        if connector_class is None:
            raise ValueError(
                f"Unsupported database type: {db_type!r}. "
                f"Connection string: {connection_string}"
            )

        return connector_class(connection_string)

    except Exception:
        logger.exception(
            "Failed to initialize connector for connection string: %s",
            connection_string,
        )
        raise


def get_connectors() -> list[SQLDatabase]:
    """Return cached connectors from ``CONNECTION_STRINGS``.

    One instance per database (keyed internally by ``database_name``). The
    return value is a list because NeMo text-to-SQL agents expect
    ``list[SQLDatabase]``, not a mapping.

    Reads ``CONNECTION_STRINGS`` from the environment (set in ``.env``).
    """
    global _connectors
    if _connectors is None:
        raw = os.environ.get("CONNECTION_STRINGS", "")
        connection_strings = [cs.strip() for cs in raw.split(",") if cs.strip()]
        if not connection_strings:
            logger.warning(
                "CONNECTION_STRINGS is not set. Add it to your .env, e.g.:\n\n"
                "    CONNECTION_STRINGS=postgresql://user:password@host:5432/dbname"
            )
            return []

        loaded: dict[str, SQLDatabase] = {}
        for cs in connection_strings:
            connector = create_connector(cs)
            loaded[connector.database_name] = connector
        _connectors = loaded
    return list(_connectors.values())
