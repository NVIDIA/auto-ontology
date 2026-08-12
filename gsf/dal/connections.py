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

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from gsf.connectors.vault import read_secret
from gsf.dal import schema as s
from gsf.dal.session import store

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
    statement = insert(s.catalog_database).values(
        name=database_name, connection=json.loads(connection)
    )
    rows = store().query_write(
        statement.on_conflict_do_update(
            index_elements=[s.catalog_database.c.name],
            set_={"connection": statement.excluded.connection},
        ).returning(s.catalog_database)
    )
    return dict(rows[0])


def verify_connectivity() -> None:
    """Probe the store. Raises if it is unreachable.

    Backs the health endpoint, so it must fail rather than report healthy — the
    ``SELECT 1`` is deliberately the cheapest statement that still proves a
    connection can be checked out of the pool and used.
    """
    store().query_read("SELECT 1")
