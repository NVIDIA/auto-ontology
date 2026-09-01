# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j data access for UI-managed database connections.

Connection metadata is stored directly on ``Labels.DB`` nodes so that the UI
connection and the catalog database share a single node.  A DB node is treated
as a UI-managed connection when it has a ``connection`` set (a JSON string of
the structured form fields); the catalog database name (``db.name``) doubles as
the connection's identity and label.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.connectors.vault import read_secret

logger = logging.getLogger(__name__)


def list_connections() -> list[dict[str, Any]]:
    """Return the resolved connection object for every catalog database.

    Each connection is resolved from Vault when a secret exists for that
    database name; otherwise it falls back to the ``connection`` stored on the
    Neo4j DB node. Databases with neither are skipped.
    """
    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB})
        RETURN properties(db) AS props
        ORDER BY db.name
        """
    )
    connections: list[dict[str, Any]] = []
    for row in rows:
        props = dict(row["props"])
        database_name = str(props.get("name"))
        secret = read_secret(database_name)
        if secret:
            connections.append(secret)
            continue
        connection_raw = props.get("connection")
        if connection_raw:
            connections.append(json.loads(connection_raw))
    return connections


def insert_connection(
    *,
    connection: str,
    database_name: str,
) -> dict[str, Any]:
    """Attach the JSON-encoded connection metadata to the catalog DB node.

    Returns the public payload (the node's properties).
    """
    rows = get_neo4j_conn().query_write(
        f"""
        MERGE (db:{Labels.DB} {{name: $database_name}})
        ON CREATE SET db.id = randomUUID()
        SET db.connection = $connection
        RETURN properties(db) AS props
        """,
        {
            "database_name": database_name,
            "connection": connection,
        },
    )
    assert rows
    return dict(rows[0]["props"])


def verify_connectivity() -> None:
    """Probe the Neo4j connection. Raises if the database is unreachable."""
    get_neo4j_conn().verify_connectivity()
