# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a connector connection string from a structured connection object.

UI-managed connections are stored as a JSON ``connection`` object on the
catalog DB node or in Vault. The connectors still consume a connection string,
so this module converts the structured form into the URL the connectors expect.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping
from urllib.parse import quote

DEFAULT_POSTGRES_PORT = "5432"
DEFAULT_MYSQL_PORT = "3306"
DEFAULT_HEAVYDB_PORT = "6274"
DEFAULT_KYUUBI_PORT = "10000"
DEFAULT_HEAVYDB_PROTOCOL = "binary"


def _require(connection: Mapping[str, Any], key: str) -> str:
    value = str(connection.get(key) or "").strip()
    if not value:
        raise ValueError(f"Connection is missing required field: {key!r}")
    return value


def _enc(value: str) -> str:
    """Percent-encode a URL component, escaping reserved chars like ``/`` and ``@``."""
    return quote(value, safe="")


def _encode_private_key(pem: str) -> str:
    """Return URL-safe base64 for a PEM private key.

    A PEM cannot travel through a query string as-is: its newlines and its ``+``
    and ``/`` characters get mangled, and ``parse_qs`` decodes ``+`` as a space,
    which corrupts the key silently rather than failing. Base64 sidesteps all of
    that. An already-encoded value is passed through so re-building a connection
    string is idempotent.
    """
    if "BEGIN" not in pem:
        return pem
    return base64.urlsafe_b64encode(pem.encode()).decode()


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

    if conn_type == "mysql":
        host = _require(connection, "host").rstrip("/")
        user = _require(connection, "user")
        password = _require(connection, "password")
        database = _require(connection, "database")
        port = str(connection.get("port") or "").strip() or DEFAULT_MYSQL_PORT
        return f"mysql://{_enc(user)}:{_enc(password)}@{host}:{port}/{_enc(database)}"

    if conn_type == "snowflake":
        account = _require(connection, "account")
        warehouse = _require(connection, "warehouse")
        user = _require(connection, "user")
        database = _require(connection, "database")
        params = f"warehouse={_enc(warehouse)}&database={_enc(database)}"

        # Snowflake accounts that enforce MFA reject passwords for PERSON users and
        # forbid them on SERVICE users, so a key pair is the only credential an
        # unattended service can use. Password auth stays supported for accounts
        # that still permit it.
        private_key = str(connection.get("private_key") or "").strip()
        if private_key:
            params += f"&private_key={_enc(_encode_private_key(private_key))}"
            passphrase = str(connection.get("private_key_passphrase") or "").strip()
            if passphrase:
                params += f"&private_key_passphrase={_enc(passphrase)}"
            return f"snowflake://{_enc(user)}@{account}?{params}"

        password = _require(connection, "password")
        return f"snowflake://{_enc(user)}:{_enc(password)}@{account}?{params}"

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

    if conn_type == "kyuubi":
        host = _require(connection, "host").rstrip("/")
        if host.startswith(("https://", "http://")):
            host = host.split("://", 1)[1]
        user = _require(connection, "user")
        password = _require(connection, "password")
        catalog = _require(connection, "database")
        port = str(connection.get("port") or "").strip() or DEFAULT_KYUUBI_PORT

        # An SSA client mints a fresh token per hour, so the connector needs the
        # token endpoint alongside the credentials. Without ``ssa_url`` the
        # password is treated as a JWT that nothing can refresh.
        ssa_url = str(connection.get("ssa_url") or "").strip().rstrip("/")
        params = [f"auth={'ssa' if ssa_url else 'token'}"]
        if ssa_url:
            params.append(f"ssa_url={_enc(ssa_url)}")

        # NVIDIA ships its internal CAs as a Java truststore; the connector
        # converts it to PEM since Python's ssl module cannot read a JKS. The
        # keystore can arrive either as base64 bytes uploaded through the UI --
        # which keeps it with the connection instead of requiring a file to
        # exist on every pod -- or as a path, for env-var connection strings.
        truststore_file = str(connection.get("truststore_file") or "").strip()
        truststore = str(connection.get("truststore") or "").strip()
        if truststore_file:
            params.append(f"truststore_data={_enc(truststore_file)}")
        elif truststore:
            params.append(f"truststore={_enc(truststore)}")
        if truststore_file or truststore:
            truststore_password = str(
                connection.get("truststore_password") or ""
            ).strip()
            if truststore_password:
                params.append(f"truststore_password={_enc(truststore_password)}")

        return (
            f"kyuubi://{_enc(user)}:{_enc(password)}@{host}:{port}/{_enc(catalog)}"
            f"?{'&'.join(params)}"
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
