# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake connector implementing the NeMo-Retriever SQLDatabase ABC."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)


def _quoted_identifier(name: str) -> str:
    """Return a Snowflake-quoted identifier (preserves case and special chars)."""
    return '"' + name.replace('"', '""') + '"'


def _load_private_key(encoded: str, passphrase: str | None) -> bytes:
    """Return the DER bytes for a PEM private key supplied in a connection string.

    Accepts either a bare PEM or URL-safe base64 of one. Base64 is what
    :func:`build_connection_string` emits, since a PEM's newlines and ``+``/``/``
    characters do not survive a query string intact -- ``parse_qs`` decodes ``+``
    as a space, which silently corrupts the key.

    The driver documents DER bytes as the accepted form, so the key is normalised
    here rather than passing a key object through.
    """
    text = encoded.strip()
    if "BEGIN" in text:
        pem = text.encode()
    else:
        try:
            # Tolerate stripped padding; base64 requires a multiple of 4.
            pem = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "Snowflake private_key must be a PEM key or base64-encoded PEM"
            ) from exc

    # A PEM pasted through JSON often arrives with literal backslash-n instead of
    # real newlines, which the PEM parser rejects with an unhelpful error.
    if b"\\n" in pem:
        pem = pem.replace(b"\\n", b"\n")

    try:
        key = serialization.load_pem_private_key(
            pem, password=passphrase.encode() if passphrase else None
        )
    except TypeError as exc:
        raise ValueError(
            "Snowflake private_key is encrypted; supply private_key_passphrase"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Snowflake private_key could not be parsed: {exc}") from exc

    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _parse_connection_string(
    connection_string: str,
) -> tuple[dict[str, Any], str, str]:
    """Parse a Snowflake URL into connector kwargs, warehouse, and database name.

    Required URL parts: ``user``, ``account`` (host), ``warehouse``, ``database``
    query param, and one credential -- either a password or a ``private_key``.

    Expected formats::

        snowflake://USER:PASSWORD@ACCOUNT?warehouse=WH&database=SF_DB
        snowflake://USER@ACCOUNT?warehouse=WH&database=SF_DB&private_key=BASE64_PEM

    Key-pair auth is not merely an alternative: Snowflake accounts that enforce
    MFA reject password sign-in for ``PERSON`` users and forbid passwords on
    ``SERVICE`` users entirely, leaving a key pair as the only credential an
    unattended service can present.
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "snowflake":
        raise ValueError(f"Not a Snowflake URL: {connection_string}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid Snowflake connection string (missing account): {connection_string}"
        )

    user = unquote(parsed.username or "")
    if not user:
        raise ValueError(
            "Snowflake connection string requires user in the URL, e.g. "
            "snowflake://user:pass@account?warehouse=COMPUTE_WH&database=MY_DB"
        )

    query = parse_qs(parsed.query)

    password = unquote(parsed.password) if parsed.password is not None else ""
    private_key = query.get("private_key", [None])[0]
    if not password and not private_key:
        raise ValueError(
            "Snowflake connection string requires either a password or a private_key, "
            "e.g. snowflake://user:pass@account?warehouse=COMPUTE_WH&database=MY_DB "
            "or snowflake://user@account?warehouse=COMPUTE_WH&database=MY_DB"
            "&private_key=BASE64_PEM"
        )

    warehouse = query.get("warehouse", [None])[0]
    if not warehouse:
        raise ValueError("Snowflake connection string requires ?warehouse=COMPUTE_WH")

    database = query.get("database", [None])[0]
    if database:
        database = unquote(database)
    else:
        database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError(
            "Snowflake connection string requires ?database=SNOWFLAKE_DB, e.g. "
            "snowflake://user:pass@account?warehouse=COMPUTE_WH&database=MY_DB"
        )

    connect_kwargs: dict[str, Any] = {
        "user": user,
        "account": parsed.hostname,
        "database": database,
        "warehouse": warehouse,
        "login_timeout": 10,
    }

    # A key pair takes precedence: if one is supplied it is the credential the
    # account will actually accept, so a stale password alongside it is ignored.
    if private_key:
        connect_kwargs["private_key"] = _load_private_key(
            private_key, query.get("private_key_passphrase", [None])[0]
        )
    else:
        connect_kwargs["password"] = password

    role = query.get("role", [None])[0]
    if role:
        connect_kwargs["role"] = role

    schema = query.get("schema", [None])[0]
    if schema:
        connect_kwargs["schema"] = schema

    return connect_kwargs, warehouse, database


class SnowflakeDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``snowflake-connector-python``.

    Parameters
    ----------
    connection_string:
        ``snowflake://user:password@account?warehouse=COMPUTE_WH&database=MY_DB``
    """

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        (
            self._connect_kwargs,
            self._warehouse,
            self._database_name,
        ) = _parse_connection_string(connection_string)
        # Optional ingestion allowlist: an explicit structured-connection
        # selection takes precedence; otherwise a URL ``schema=`` parameter
        # scopes env-var connections such as CONNECTION_STRINGS. Without this
        # fallback, ``schema=`` only sets Snowflake's current schema while
        # introspection still returns every visible schema in the database.
        url_schema = self._connect_kwargs.get("schema")
        filter_schemas = schemas or ([str(url_schema)] if url_schema else None)

        # Empty/None means "all schemas".
        # Compared case-insensitively (Snowflake upper-cases unquoted names).
        self._schema_filter: set[str] | None = (
            {s.upper() for s in filter_schemas if s and s.strip()}
            if filter_schemas
            else None
        )
        if self._schema_filter:
            logger.info(
                "Snowflake ingestion restricted to schemas: %s",
                sorted(self._schema_filter),
            )

    def _filter_by_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict a schema-introspection frame to the configured allowlist.

        No-op when no filter is set or the frame lacks a ``table_schema`` column.
        """
        if self._schema_filter is None or df.empty or "table_schema" not in df.columns:
            return df
        return df[df["table_schema"].str.upper().isin(self._schema_filter)]

    @property
    def dialect(self) -> str:
        return "snowflake"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        with snowflake.connector.connect(**self._connect_kwargs) as conn:
            with conn.cursor() as cur:
                cur.execute(f"USE WAREHOUSE {_quoted_identifier(self._warehouse)}")
                if parameters:
                    cur.execute(sql, parameters)
                else:
                    cur.execute(sql)
                if cur.description is None:
                    return pd.DataFrame()
                # Return the real column names as Snowflake reports them (unquoted
                # identifiers come back UPPERCASE). The introspection queries below
                # quote their aliases to keep their lowercase result keys stable.
                columns = [desc[0] for desc in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=columns)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """List the database's schemas (excluding ``INFORMATION_SCHEMA``).

        Used by the connection UI to let the user pick which schemas to ingest.
        """
        df = self.execute("""
            SELECT SCHEMA_NAME AS "schema_name"
            FROM INFORMATION_SCHEMA.SCHEMATA
            WHERE SCHEMA_NAME != 'INFORMATION_SCHEMA'
            ORDER BY SCHEMA_NAME
        """)
        if df.empty:
            return []
        return [str(name) for name in df["schema_name"].tolist()]

    def get_tables(self) -> pd.DataFrame:
        return self._filter_by_schema(
            self.execute("""
            SELECT
                TABLE_SCHEMA AS "table_schema",
                TABLE_NAME   AS "table_name",
                TABLE_TYPE   AS "table_type"
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        )

    def get_columns(self) -> pd.DataFrame:
        df = self.execute("""
            SELECT
                TABLE_SCHEMA     AS "table_schema",
                TABLE_NAME       AS "table_name",
                COLUMN_NAME      AS "column_name",
                DATA_TYPE        AS "data_type",
                IS_NULLABLE      AS "is_nullable",
                ORDINAL_POSITION AS "ordinal_position"
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA'
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """)
        if df.empty:
            return df

        # https://stackoverflow.com/questions/64667418/snowflake-sys-facade-in-information-schema-columns
        df["column_name"] = df["column_name"].str.removesuffix("$SYS_FACADE$0")
        df["column_name"] = df["column_name"].str.removesuffix("$SYS_FACADE$1")
        df = df.drop_duplicates(subset=["table_schema", "table_name", "column_name"])
        return self._filter_by_schema(df)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Return recent queries from ``INFORMATION_SCHEMA.QUERY_HISTORY``."""
        try:
            df = self.execute(f"""
                SELECT
                    END_TIME   AS "end_time",
                    QUERY_TEXT AS "query_text"
                FROM TABLE(
                    INFORMATION_SCHEMA.QUERY_HISTORY(
                        DATEADD(hour, -{hours}, CURRENT_TIMESTAMP()),
                        CURRENT_TIMESTAMP(),
                        RESULT_LIMIT => 10000
                    )
                )
                WHERE QUERY_TYPE NOT IN (
                    'USE', 'SHOW', 'GRANT', 'CREATE_USER', 'CREATE_ROLE',
                    'DROP', 'COMMIT', 'ALTER_SESSION', 'CALL'
                )
                  AND EXECUTION_STATUS = 'SUCCESS'
                  AND NULLIF(TRIM(QUERY_TEXT), '') IS NOT NULL
                  AND LOWER(QUERY_TEXT) NOT LIKE '%information_schema%'
                ORDER BY END_TIME DESC
            """)
            return df[["end_time", "query_text"]]
        except snowflake.connector.errors.Error:
            logger.exception("Failed to fetch Snowflake query history")
            return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        db = _quoted_identifier(self._database_name)
        df = self.execute(f"SHOW VIEWS IN DATABASE {db}")
        if df.empty:
            return pd.DataFrame(
                columns=["table_schema", "table_name", "view_definition"]
            )

        # SHOW VIEWS returns fixed metadata columns; normalize their case here
        # since execute() no longer lower-cases result headers.
        df.columns = [str(c).lower() for c in df.columns]
        df = df.loc[df["schema_name"] != "INFORMATION_SCHEMA"]
        df = df.rename(
            columns={
                "schema_name": "table_schema",
                "name": "table_name",
                "text": "view_definition",
            }
        )[["table_schema", "table_name", "view_definition"]]
        return self._filter_by_schema(df)

    def get_pks(self) -> pd.DataFrame:
        db = _quoted_identifier(self._database_name)
        df = self.execute(f"SHOW PRIMARY KEYS IN DATABASE {db}")
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "table_schema",
                    "table_name",
                    "column_name",
                    "ordinal_position",
                ]
            )

        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(
            columns={
                "schema_name": "table_schema",
                "key_sequence": "ordinal_position",
            }
        )[["table_schema", "table_name", "column_name", "ordinal_position"]]
        return self._filter_by_schema(df)

    def get_fks(self) -> pd.DataFrame:
        db = _quoted_identifier(self._database_name)
        df = self.execute(f"SHOW IMPORTED KEYS IN DATABASE {db}")
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "table_schema",
                    "table_name",
                    "column_name",
                    "referenced_schema",
                    "referenced_table",
                    "referenced_column",
                ]
            )

        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(
            columns={
                "fk_schema_name": "table_schema",
                "fk_table_name": "table_name",
                "fk_column_name": "column_name",
                "pk_schema_name": "referenced_schema",
                "pk_table_name": "referenced_table",
                "pk_column_name": "referenced_column",
            }
        )[
            [
                "table_schema",
                "table_name",
                "column_name",
                "referenced_schema",
                "referenced_table",
                "referenced_column",
            ]
        ]
        return self._filter_by_schema(df)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify credentials, the warehouse, and that schemas are visible."""
        with snowflake.connector.connect(**self._connect_kwargs) as conn:
            conn.execute_string(
                f"USE WAREHOUSE {_quoted_identifier(self._warehouse)}; SHOW SCHEMAS"
            )

    def close(self) -> None:
        """No persistent connection to close (connections are per-query)."""
