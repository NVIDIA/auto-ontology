# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Postgres connection string URL encoding.

Special characters in database credentials (such as '@' or ':' in passwords)
must be percent-encoded so that URL parsers in SQLAlchemy and psycopg do not
misinterpret them as delimiter tokens (e.g. host or port delimiters).
"""

from __future__ import annotations

from psycopg.conninfo import conninfo_to_dict
import pytest
from sqlalchemy.engine import make_url

from auto_ontology.infra.postgres import get_postgres_connection_string


def test_postgres_connection_string_escapes_special_characters_in_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "abc@1234:pass/word%")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DATABASE", "test_db")

    conn_str = get_postgres_connection_string()

    # The raw '@' in the password must be encoded to avoid ambiguous host delimiters
    assert "abc@1234" not in conn_str
    assert "abc%401234" in conn_str

    # Verify SQLAlchemy make_url parses credentials back accurately
    sa_url = make_url(conn_str)
    assert sa_url.username == "test_user"
    assert sa_url.password == "abc@1234:pass/word%"
    assert sa_url.host == "127.0.0.1"
    assert sa_url.port == 5432
    assert sa_url.database == "test_db"

    # Verify psycopg connection info parser parses credentials back accurately
    conninfo = conninfo_to_dict(conn_str)
    assert conninfo["user"] == "test_user"
    assert conninfo["password"] == "abc@1234:pass/word%"
    assert conninfo["host"] == "127.0.0.1"
    assert conninfo["port"] == "5432"
    assert conninfo["dbname"] == "test_db"


def test_postgres_connection_string_escapes_user_with_at_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_USER", "user@domain")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DATABASE", "app_db")

    conn_str = get_postgres_connection_string()

    assert "user%40domain" in conn_str

    sa_url = make_url(conn_str)
    assert sa_url.username == "user@domain"
    assert sa_url.password == "secret"
    assert sa_url.database == "app_db"


def test_postgres_connection_string_default_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_DATABASE", raising=False)

    conn_str = get_postgres_connection_string()
    assert conn_str == "postgresql://postgres:postgres@localhost:5432/auto_ontology"
