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
from gsf.server.ingestion.proxy import trigger_ingest, trigger_reset
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

    When the form names a specific ``schema``, the test also confirms that schema
    exists — the UI then skips schema selection, so a typo would otherwise only
    surface much later as an ingest that silently finds nothing.
    """
    database_name = str(connection.get("database") or "").strip()
    if not database_name:
        raise ValueError("Database name is required")

    if _database_already_connected(database_name):
        raise ValueError(f"A connection for database {database_name!r} already exists")

    requested_schema = str(connection.get("schema") or "").strip()

    connection_string = build_connection_string(connection)
    connector = create_connector(connection_string)
    try:
        # ``get_schemas`` runs a real query, so it validates connectivity and
        # credentials just like ``ping`` while also returning the schema list.
        get_schemas = getattr(connector, "get_schemas", None)
        if get_schemas is None:
            connector.ping()
            return []

        schemas = list(get_schemas())
        if requested_schema and not any(
            str(name).strip().casefold() == requested_schema.casefold()
            for name in schemas
        ):
            raise ValueError(
                f"Schema {requested_schema!r} was not found in {database_name!r}."
            )
        return schemas
    finally:
        connector.close()


def create_connection(
    *,
    connection: dict[str, Any],
) -> dict[str, Any]:
    """Create a UI-managed connection stored in Neo4j."""

    # ``schema`` is the form's shorthand for a single-schema allowlist; the caller
    # turns it into ``schemas``. Storing the raw field too would leave two sources of
    # truth for what gets ingested, so drop it.
    connection = {key: value for key, value in connection.items() if key != "schema"}

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


def set_sso_federation(*, database_name: str, enabled: bool) -> dict[str, Any]:
    """Toggle "authenticate as signed-in user" on an existing connection.

    Deliberately narrow: it rewrites only the ``sso_federation`` flag on the
    stored connection, so callers never have to re-send credentials to change
    it. Ingestion is unaffected — it always uses the stored access token — so
    no re-ingest is triggered.
    """
    connection = next(
        (
            conn
            for conn in list_connections()
            if str(conn.get("database") or "").strip() == database_name
        ),
        None,
    )
    if connection is None:
        raise ValueError(f"No connection found for database {database_name!r}")

    updated = {**connection, "sso_federation": enabled}

    if is_vault_configured():
        write_secret(database_name, updated)
    else:
        insert_connection(
            connection=json.dumps(updated),
            database_name=database_name,
        )

    # Connectors cache the credential, and chat workers hold connectors, so both
    # must be rebuilt for the change to take effect on the next question.
    invalidate_connectors_cache()
    refresh_chat_workers()

    # Bust the SSO-federation flag cache so the chat endpoint picks up the
    # change on the very next request rather than waiting for the TTL to expire.
    from gsf.connectors.databricks_oauth import invalidate_sso_federation_cache

    invalidate_sso_federation_cache()

    return {"database_name": database_name, "sso_federation": enabled}


def delete_connection(database_name: str) -> dict[str, str]:
    """Delete a UI-managed connection and tear down its ingested database graph."""
    if is_vault_configured():
        delete_secrets(database_name)

    invalidate_connectors_cache()
    refresh_chat_workers()

    trigger_reset(database_name)

    return {"database_name": database_name}
