# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection lifecycle: test, persist credentials, and link catalog databases."""

from __future__ import annotations

import json
import logging
from typing import Any

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.connectors.registry import create_connector, invalidate_connectors_cache
from gsf.connectors.vault import delete_secrets, is_vault_configured, write_secret
from gsf.server.ingestion.proxy import trigger_ingest, trigger_ingest_delete
from gsf.server.chat.worker import refresh_chat_workers
from gsf.dal.connections import insert_connection, list_connections

logger = logging.getLogger(__name__)


def _database_already_connected(database_name: str) -> bool:
    """True if a UI-managed connection already exists for this database."""
    return any(
        str(conn.get("database") or "").strip().lower() == database_name.lower()
        for conn in list_connections()
    )


def test_connection(connection: dict[str, Any]) -> list[str]:
    """Validate credentials for the settings UI test action.

    Returns the connection's schemas when the connector supports enumerating
    them (currently Databricks and Snowflake); enumerating doubles as the
    connectivity check. Connectors without schema enumeration just ``ping()``
    and return ``[]``.
    """
    database_name = str(connection.get("database") or "").strip()
    if not database_name:
        raise ValueError("Database name is required")

    if _database_already_connected(database_name):
        raise ValueError(f"A connection for database {database_name!r} already exists")

    connection_string = build_connection_string(connection)
    connector = create_connector(connection_string)
    try:
        # ``get_schemas`` runs a real query, so it validates connectivity and
        # credentials just like ``ping`` while also returning the schema list.
        get_schemas = getattr(connector, "get_schemas", None)
        if get_schemas is not None:
            return list(get_schemas())
        connector.ping()
        return []
    finally:
        connector.close()


def create_connection(
    *,
    connection: dict[str, Any],
) -> dict[str, Any]:
    """Create a UI-managed connection stored in Neo4j."""

    database_name = str(connection.get("database"))
    if not database_name:
        raise ValueError("Database name is required")

    if _database_already_connected(database_name):
        raise ValueError(f"A connection for database {database_name!r} already exists")

    # With Vault configured, store credentials in Vault and keep them off the
    # Neo4j node; otherwise fall back to persisting the JSON on the node.
    vault_configured = is_vault_configured()
    if vault_configured:
        write_secret(database_name, connection)
    insert_connection(
        connection=json.dumps(connection) if not vault_configured else "",
        database_name=database_name,
    )

    invalidate_connectors_cache()
    refresh_chat_workers()

    trigger_ingest(connection)

    return connection


def delete_connection(database_name: str) -> dict[str, str]:
    """Delete a UI-managed connection and tear down its ingested database graph."""
    if is_vault_configured():
        delete_secrets(database_name)

    invalidate_connectors_cache()
    refresh_chat_workers()

    trigger_ingest_delete(database_name)

    return {"database_name": database_name}
