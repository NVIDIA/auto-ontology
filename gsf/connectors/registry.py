# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and cache SQLDatabase connectors from CONNECTION_STRINGS."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from nemo_retriever.tabular_data.sql_database import SQLDatabase

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.connectors.duckdb import DuckDBDatabase
from gsf.connectors.heavydb import HeavyDBDatabase
from gsf.connectors.postgres import PostgresDatabase
from gsf.connectors.snowflake import SnowflakeDatabase
from gsf.connectors.sqlite import SQLiteDatabase

logger = logging.getLogger(__name__)

CONNECTOR_REGISTRY: dict[str, type[SQLDatabase]] = {
    "postgres": PostgresDatabase,
    "postgresql": PostgresDatabase,
    "duckdb": DuckDBDatabase,
    "snowflake": SnowflakeDatabase,
    "heavydb": HeavyDBDatabase,
    "sqlite": SQLiteDatabase,
}

_connectors: list[SQLDatabase] | None = None


def invalidate_connectors_cache() -> None:
    """Drop cached connectors so the next :func:`get_connectors` reloads."""
    global _connectors
    if _connectors is not None:
        for connector in _connectors:
            try:
                connector.close()
            except Exception:
                logger.exception("Failed to close connector during cache invalidation")
    _connectors = None


def create_connector(connection_string: str) -> SQLDatabase:
    """Parse *connection_string*, select a connector class, and return an instance."""
    try:
        parsed = urlparse(connection_string)
        scheme = parsed.scheme

        if not scheme:
            raise ValueError("Invalid connection string: No protocol scheme found.")

        connector_type = scheme.split("+", 1)[0].lower()
        connector_class = CONNECTOR_REGISTRY.get(connector_type)

        if connector_class is None:
            raise ValueError(
                f"Unsupported database type: {connector_type!r}. "
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
    """Return cached connectors for all configured connections.

    NeMo text-to-SQL resolves the execution connector from
    ``relevant_tables[*].database_name`` (see
    ``nemo_retriever.tabular_data.retrieval.text_to_sql.connector_routing``).
    Each connector's ``database_name`` must therefore be unique across the
    returned list — use ``metadata_database`` on Snowflake URLs (or distinct
    physical databases) when wiring multiple connections.
    """
    global _connectors
    if _connectors is None:
        from gsf.dal.connections import list_connections

        try:
            connection_strings = [
                build_connection_string(conn) for conn in list_connections()
            ]
        except Exception:
            logger.exception("Failed to load connection strings from Neo4j DB nodes")
            connection_strings = []
        if not connection_strings:
            raw = os.environ.get("CONNECTION_STRINGS", "")
            connection_strings = [cs.strip() for cs in raw.split(",") if cs.strip()]
        if not connection_strings:
            logger.warning(
                "No connections configured. Add CONNECTION_STRINGS to your .env "
                "or create a connection in Settings → Connections."
            )
            return []

        loaded: list[SQLDatabase] = []
        seen_database_names: set[str] = set()
        for cs in connection_strings:
            connector = create_connector(cs)
            database_name = connector.database_name
            if database_name in seen_database_names:
                logger.warning(
                    "Duplicate connector database_name %r — NeMo routes SQL by "
                    "database_name, so only one connector per name can be used. "
                    "Set metadata_database on the connection string to disambiguate.",
                    database_name,
                )
            seen_database_names.add(database_name)
            loaded.append(connector)
        _connectors = loaded
    return list(_connectors)
