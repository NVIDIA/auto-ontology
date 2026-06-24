# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a connector connection string from a structured connection object.

UI-managed connections are stored as a JSON ``connection`` object (the form
fields) on the catalog DB node. The connectors still consume a connection
string, so this module is the single place that converts the structured form
into the ``libpq``/Snowflake URL the connectors expect.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote

DEFAULT_POSTGRES_PORT = "5432"
DEFAULT_HEAVYDB_PORT = "6274"
DEFAULT_HEAVYDB_PROTOCOL = "binary"


def _require(connection: Mapping[str, Any], key: str) -> str:
    value = str(connection.get(key) or "").strip()
    if not value:
        raise ValueError(f"Connection is missing required field: {key!r}")
    return value


def _enc(value: str) -> str:
    """Percent-encode a URL component, escaping reserved chars like ``/`` and ``@``."""
    return quote(value, safe="")


def build_connection_string(connection: Mapping[str, Any]) -> str:
    """Return a connector connection string for a structured *connection*."""
    conn_type = str(connection.get("type") or "").strip().lower()

    if conn_type in ("postgresql", "postgres"):
        host = _require(connection, "host").rstrip("/")
        user = _require(connection, "user")
        password = _require(connection, "password")
        database = _require(connection, "database")
        port = str(connection.get("port") or "").strip() or DEFAULT_POSTGRES_PORT
        return (
            f"postgresql://{_enc(user)}:{_enc(password)}@{host}:{port}/{_enc(database)}"
        )

    if conn_type == "snowflake":
        account = _require(connection, "account")
        warehouse = _require(connection, "warehouse")
        user = _require(connection, "user")
        password = _require(connection, "password")
        database = _require(connection, "database")
        return (
            f"snowflake://{_enc(user)}:{_enc(password)}@{account}"
            f"?warehouse={_enc(warehouse)}&database={_enc(database)}"
        )

    if conn_type == "heavydb":
        host = _require(connection, "host").rstrip("/")
        user = _require(connection, "user")
        password = _require(connection, "password")
        database = _require(connection, "database")
        port = str(connection.get("port") or "").strip() or DEFAULT_HEAVYDB_PORT
        protocol = (
            str(connection.get("protocol") or "").strip() or DEFAULT_HEAVYDB_PROTOCOL
        )
        return (
            f"heavydb://{_enc(user)}:{_enc(password)}@{host}:{port}/{_enc(database)}"
            f"?protocol={_enc(protocol)}"
        )

    raise ValueError(f"Unsupported connection type: {conn_type!r}")
