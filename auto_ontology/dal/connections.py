# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UI-managed database connections.

Connection metadata lives on the catalog database row, so the UI connection and
the catalog database are the same record and cannot drift apart. A database is a
UI-managed connection when it has a ``connection`` value; the catalog name
doubles as the connection's identity.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.connectors.vault import read_secret
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store

logger = logging.getLogger(__name__)


def list_connections() -> list[dict[str, Any]]:
    """Every catalog database's resolved connection.

    Vault wins when it holds a secret for the database name; otherwise the
    stored value is used. Databases with neither are skipped, which is what
    makes this "connections" rather than "databases" — an ingested catalog with
    no UI connection has nothing to return.
    """
    connections: list[dict[str, Any]] = []
    for row in store().query_read(
        select(s.catalog_database.c.name, s.catalog_database.c.connection).order_by(
            s.catalog_database.c.name
        )
    ):
        secret = read_secret(str(row["name"]))
        if secret:
            connections.append(secret)
            continue
        stored = row["connection"]
        if stored:
            # jsonb here, a JSON *string* on the graph node. Callers get the
            # object either way, so decoding a string keeps a database written
            # before this column existed readable.
            connections.append(
                json.loads(stored) if isinstance(stored, str) else stored
            )
    return connections


def insert_connection(
    *,
    connection: str,
    database_name: str,
) -> dict[str, Any]:
    """Attach connection metadata to the catalog database, creating it if needed.

    Creating it matters: a connection is normally configured *before* anything
    is ingested, so there is usually no database row to attach to yet.
    """
    # An empty string means "no connection JSON on the row" -- the Vault path
    # writes the credentials to Vault and deliberately keeps them off the
    # catalog. `json.loads("")` raises, and because the Vault write happens
    # first, the secret was already stored when the insert 500s: no
    # catalog_database row, so the connection is invisible to list_connections
    # and cannot be ingested.
    payload = json.loads(connection) if connection and connection.strip() else None
    statement = insert(s.catalog_database).values(
        name=database_name, connection=payload
    )
    rows = store().query_write(
        statement.on_conflict_do_update(
            index_elements=[s.catalog_database.c.name],
            set_={"connection": statement.excluded.connection},
        ).returning(s.catalog_database)
    )
    return dict(rows[0])


def clear_connection(*, database_name: str) -> None:
    """Detach connection metadata from the catalog database.

    The counterpart to :func:`insert_connection`, and the write that actually
    makes a database stop being a connection — "connections" are exactly the
    catalog rows carrying a connection value.

    The row itself stays. ``catalog_schema`` cascades off it, so deleting it
    here would drop the ingested catalog while leaving behind the semantic rows
    and pgvector embeddings that are identified *through* it — the half-state
    :func:`auto_ontology.dal.reset.delete_all_data` orders its own deletes to avoid.
    Tearing the graph down is that function's job.
    """
    store().query_write(
        update(s.catalog_database)
        .where(s.catalog_database.c.name == database_name)
        .values(connection=None)
    )


def verify_connectivity() -> None:
    """Probe the store. Raises if it is unreachable.

    Backs the health endpoint, so it must fail rather than report healthy — the
    ``SELECT 1`` is deliberately the cheapest statement that still proves a
    connection can be checked out of the pool and used.
    """
    store().query_read("SELECT 1")
