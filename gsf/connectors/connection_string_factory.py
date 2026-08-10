# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a connector connection string from a structured connection object.

UI-managed connections are stored as a JSON ``connection`` object on the
catalog DB node or in Vault. The connectors still consume a connection string,
so this module converts the structured form into the URL the connectors expect.
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

    if conn_type == "databricks":
        host = _require(connection, "host").rstrip("/")
        if host.startswith(("https://", "http://")):
            host = host.split("://", 1)[1]
        http_path = _require(connection, "http_path")
        # ``access_token_override`` carries a Databricks token exchanged from the
        # caller's SSO identity, so the query runs with that user's privileges
        # instead of the connection's stored PAT.
        access_token = str(connection.get("access_token_override") or "").strip()
        federated = bool(access_token)
        if not access_token:
            access_token = _require(connection, "password")
        catalog = _require(connection, "database")
        url = (
            f"databricks://token:{_enc(access_token)}@{host}/{_enc(catalog)}"
            f"?http_path={_enc(http_path)}"
        )
        # The token itself is opaque, so record which credential it is. The
        # connector logs this alongside every statement it runs.
        if federated:
            url += "&auth=sso"
        return url

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
