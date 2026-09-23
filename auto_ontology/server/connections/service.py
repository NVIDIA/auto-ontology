# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection lifecycle: test, persist credentials, and link catalog databases."""

from __future__ import annotations

import json
import logging
from typing import Any

from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.registry import (
    create_connector,
    invalidate_connectors_cache,
)
from auto_ontology.connectors.vault import (
    delete_secrets,
    is_vault_configured,
    write_secret,
)
from auto_ontology.server.ingestion.proxy import trigger_ingest, trigger_reset
from auto_ontology.server.chat.worker import refresh_chat_workers
from auto_ontology.dal.connections import (
    clear_connection,
    insert_connection,
    list_connections,
)

logger = logging.getLogger(__name__)

# Credentials, stripped from every payload the UI receives. An edit form
# therefore opens with these blank, which is why the update path reads blank as
# "unchanged" rather than "clear it".
SECRET_FIELDS = frozenset(
    {
        "password",
        "password_env",
        "private_key",
        "private_key_passphrase",
        "truststore_password",
    }
)


def _database_already_connected(database_name: str) -> bool:
    """True if a UI-managed connection already exists for this database."""
    return any(
        str(conn.get("database") or "").strip().lower() == database_name.lower()
        for conn in list_connections()
    )


def _stored_connection(database_name: str) -> dict[str, Any] | None:
    """The stored connection for a catalog database, or ``None`` if there is none."""
    return next(
        (
            conn
            for conn in list_connections()
            if str(conn.get("database") or "").strip() == database_name
        ),
        None,
    )


def _with_stored_secrets(
    connection: dict[str, Any], stored: dict[str, Any]
) -> dict[str, Any]:
    """Fill blank credentials from the stored connection.

    The UI never gets credentials back, so it cannot send them back either.
    Taking blank literally would mean re-typing an access token just to correct
    a hostname, and would silently wipe the credential of anyone who didn't.
    """
    merged = dict(connection)
    for field in SECRET_FIELDS:
        if not str(merged.get(field) or "").strip() and stored.get(field):
            merged[field] = stored[field]
    return merged


def _ingest_scope(connection: dict[str, Any]) -> list[str]:
    """The schema allowlist that decides what ingestion pulls in."""
    return sorted(str(schema) for schema in (connection.get("schemas") or []))


def _refresh_connection_caches() -> None:
    """Rebuild everything holding a connector or a cached connection flag.

    Each refresh is best-effort, because callers run this *after* they have
    already persisted. Letting one raise would abandon the rest of the write:
    an update re-ingests only when the schema allowlist moved, so failing here
    would skip that ingest with the new allowlist already stored, and the retry
    would compare the new list against itself and skip it again — permanently.
    A stale cache, by contrast, costs one refresh.
    """
    # Imported here rather than at module scope: ``databricks_oauth`` reaches
    # back into the server package, so a top-level import is circular.
    from auto_ontology.connectors.databricks_oauth import (
        invalidate_sso_federation_cache,
    )

    caches = (
        ("connectors", invalidate_connectors_cache),
        ("chat workers", refresh_chat_workers),
        ("SSO federation", invalidate_sso_federation_cache),
    )
    for name, refresh in caches:
        try:
            refresh()
        except Exception:
            logger.exception("Failed to refresh the %s cache", name)


def test_connection(
    connection: dict[str, Any], *, replacing: str | None = None
) -> list[str]:
    """Validate credentials for the settings UI test action.

    Returns the connection's schemas when the connector supports enumerating
    them (currently Databricks and Snowflake); enumerating doubles as the
    connectivity check. Connectors without schema enumeration just ``ping()``
    and return ``[]``.

    When the form names a specific ``schema``, the test also confirms that schema
    exists — the UI then skips schema selection, so a typo would otherwise only
    surface much later as an ingest that silently finds nothing.

    ``replacing`` names the connection an edit is about to overwrite: the
    "already connected" guard would otherwise reject a connection for being
    itself, and blank credentials are filled from the stored ones so the test
    exercises exactly what the update would save.
    """
    database_name = str(connection.get("database") or "").strip()
    if not database_name:
        raise ValueError("Database name is required")

    if replacing is None:
        if _database_already_connected(database_name):
            raise ValueError(
                f"A connection for database {database_name!r} already exists"
            )
    else:
        stored = _stored_connection(replacing)
        if stored is None:
            raise LookupError(f"No connection found for database {replacing!r}")
        if database_name != replacing:
            raise ValueError("A connection's database name cannot be changed")
        connection = _with_stored_secrets(connection, stored)

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
    """Create a UI-managed connection stored on the catalog database row."""

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
    # catalog row; otherwise fall back to persisting the JSON on the row.
    vault_configured = is_vault_configured()
    if vault_configured:
        write_secret(database_name, connection)
    insert_connection(
        connection=json.dumps(connection) if not vault_configured else "",
        database_name=database_name,
    )

    _refresh_connection_caches()

    trigger_ingest(connection)

    return connection


def update_connection(
    *,
    database_name: str,
    connection: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite an existing connection's settings, keeping its identity.

    ``database`` and ``type`` are the connection's identity — the catalog row is
    keyed by the name, and its ingested graph was built by that connector — so
    pointing the record at a different database or driver is a new connection,
    not an edit. Everything else, credentials included, can change.

    Re-ingest only when the schema allowlist moved. Rotating a token or fixing a
    hostname leaves the ingested graph correct, and re-ingesting it would be a
    long, surprising side effect of saving a form.
    """
    # Same shorthand the create path drops: `schema` is the single-schema form
    # of `schemas`, and storing both leaves two sources of truth.
    connection = {key: value for key, value in connection.items() if key != "schema"}

    stored = _stored_connection(database_name)
    if stored is None:
        raise LookupError(f"No connection found for database {database_name!r}")

    incoming_database = str(connection.get("database") or "").strip()
    if incoming_database and incoming_database != database_name:
        raise ValueError("A connection's database name cannot be changed")

    stored_type = str(stored.get("type") or "").strip()
    incoming_type = str(connection.get("type") or "").strip()
    if incoming_type and stored_type and incoming_type != stored_type:
        raise ValueError("A connection's type cannot be changed")

    updated = _with_stored_secrets(
        {
            **connection,
            "type": stored_type or incoming_type,
            "database": database_name,
        },
        stored,
    )

    if is_vault_configured():
        write_secret(database_name, updated)
        # Blank out any credentials the row still carries from before Vault was
        # configured, so there is only ever one copy of them.
        insert_connection(connection="", database_name=database_name)
    else:
        insert_connection(
            connection=json.dumps(updated),
            database_name=database_name,
        )

    _refresh_connection_caches()

    if _ingest_scope(updated) != _ingest_scope(stored):
        trigger_ingest(updated)

    return updated


def set_sso_federation(*, database_name: str, enabled: bool) -> dict[str, Any]:
    """Toggle "authenticate as signed-in user" on an existing connection.

    Deliberately narrow: it rewrites only the ``sso_federation`` flag on the
    stored connection, so callers never have to re-send credentials to change
    it. Ingestion is unaffected — it always uses the stored access token — so
    no re-ingest is triggered.
    """
    connection = _stored_connection(database_name)
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

    # Connectors cache the credential, chat workers hold connectors, and the
    # chat endpoint caches this very flag, so all three have to be rebuilt for
    # the change to take effect on the next question rather than after a TTL.
    _refresh_connection_caches()

    return {"database_name": database_name, "sso_federation": enabled}


def delete_connection(database_name: str) -> dict[str, str] | None:
    """Delete a UI-managed connection and tear down its ingested database graph.

    ``None`` when no connection carries that name, which the route turns into a
    404.

    Removing the record is done here rather than delegated to the ingestion
    service: a connection is a catalog row carrying connection metadata, and
    clearing that is a write this process can do. Delegating it meant a delete
    silently did nothing whenever the service was unreachable — and still
    answered 200, because the call to it is best-effort.

    Deleting the ingested graph stays the service's job and stays best-effort.
    It is the slow half, and failing it leaves stale data rather than a
    connection that comes back from the dead.
    """
    if _stored_connection(database_name) is None:
        return None

    # Vault first: `list_connections` prefers a secret over the row, so failing
    # after the row is cleared would resurrect the connection, while failing
    # after the secret is gone cannot.
    if is_vault_configured():
        delete_secrets(database_name)

    clear_connection(database_name=database_name)

    _refresh_connection_caches()

    trigger_reset(database_name)

    return {"database_name": database_name}
