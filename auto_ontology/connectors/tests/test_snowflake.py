# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pytest import MonkeyPatch

from auto_ontology.connectors.snowflake import (
    SnowflakeDatabase,
    _parse_connection_string,
)


def _connection_string(query: str = "") -> str:
    suffix = f"&{query}" if query else ""
    return f"snowflake://user:password@account?warehouse=warehouse&database=db{suffix}"


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    """One key for the whole module; generation is the slow part of these tests."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key: rsa.RSAPrivateKey, passphrase: bytes | None = None) -> str:
    encryption = (
        serialization.BestAvailableEncryption(passphrase)
        if passphrase
        else serialization.NoEncryption()
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    ).decode()


def _keypair_connection_string(pem: str, query: str = "") -> str:
    encoded = base64.urlsafe_b64encode(pem.encode()).decode()
    suffix = f"&{query}" if query else ""
    return (
        f"snowflake://user@account?warehouse=warehouse&database=db"
        f"&private_key={encoded}{suffix}"
    )


def _tables() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "table_schema": ["GPU_FLEET", "GPU_FLEET", "CRM"],
            "table_name": ["GPUS", "JOBS", "CUSTOMERS"],
        }
    )


def test_url_schema_filters_introspection() -> None:
    database = SnowflakeDatabase(_connection_string("schema=gpu_fleet"))

    filtered = database._filter_by_schema(_tables())

    assert filtered["table_name"].tolist() == ["GPUS", "JOBS"]


def test_explicit_schema_selection_overrides_url_schema() -> None:
    database = SnowflakeDatabase(
        _connection_string("schema=gpu_fleet"),
        schemas=["crm"],
    )

    filtered = database._filter_by_schema(_tables())

    assert filtered["table_name"].tolist() == ["CUSTOMERS"]


def test_missing_schema_keeps_all_visible_schemas() -> None:
    database = SnowflakeDatabase(_connection_string())

    filtered = database._filter_by_schema(_tables())

    assert filtered.equals(_tables())


def test_password_auth_passes_password_and_no_key() -> None:
    kwargs, warehouse, database = _parse_connection_string(_connection_string())

    assert kwargs["password"] == "password"
    assert "private_key" not in kwargs
    assert (warehouse, database) == ("warehouse", "db")


def test_private_key_replaces_password(rsa_key: rsa.RSAPrivateKey) -> None:
    kwargs, _, _ = _parse_connection_string(_keypair_connection_string(_pem(rsa_key)))

    # The driver takes DER bytes, and no password may be sent alongside a key.
    assert isinstance(kwargs["private_key"], bytes)
    assert "password" not in kwargs
    assert kwargs["user"] == "user"


def test_private_key_der_matches_the_supplied_key(rsa_key: rsa.RSAPrivateKey) -> None:
    kwargs, _, _ = _parse_connection_string(_keypair_connection_string(_pem(rsa_key)))

    expected = rsa_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    assert kwargs["private_key"] == expected


def test_bare_pem_is_accepted_without_base64(rsa_key: rsa.RSAPrivateKey) -> None:
    """Hand-written connection strings should not have to pre-encode the key."""
    from urllib.parse import quote

    conn = (
        f"snowflake://user@account?warehouse=warehouse&database=db"
        f"&private_key={quote(_pem(rsa_key), safe='')}"
    )

    kwargs, _, _ = _parse_connection_string(conn)

    assert isinstance(kwargs["private_key"], bytes)


def test_pem_with_literal_backslash_n_is_repaired(rsa_key: rsa.RSAPrivateKey) -> None:
    """A PEM round-tripped through JSON often loses its real newlines."""
    mangled = _pem(rsa_key).replace("\n", "\\n")

    kwargs, _, _ = _parse_connection_string(_keypair_connection_string(mangled))

    assert isinstance(kwargs["private_key"], bytes)


def test_encrypted_key_accepts_passphrase(rsa_key: rsa.RSAPrivateKey) -> None:
    conn = _keypair_connection_string(
        _pem(rsa_key, passphrase=b"s3cret"), query="private_key_passphrase=s3cret"
    )

    kwargs, _, _ = _parse_connection_string(conn)

    assert isinstance(kwargs["private_key"], bytes)


def test_encrypted_key_without_passphrase_says_so(rsa_key: rsa.RSAPrivateKey) -> None:
    conn = _keypair_connection_string(_pem(rsa_key, passphrase=b"s3cret"))

    with pytest.raises(ValueError, match="private_key_passphrase"):
        _parse_connection_string(conn)


def test_missing_both_credentials_is_rejected() -> None:
    with pytest.raises(ValueError, match="password or a private_key"):
        _parse_connection_string("snowflake://user@account?warehouse=wh&database=db")


def test_unparseable_private_key_is_rejected() -> None:
    conn = _keypair_connection_string("not a key at all")

    with pytest.raises(ValueError, match="could not be parsed"):
        _parse_connection_string(conn)


def test_query_history_excludes_blank_query_text(monkeypatch: MonkeyPatch) -> None:
    database = SnowflakeDatabase(_connection_string())
    captured_sql = ""

    def execute(sql: str) -> pd.DataFrame:
        nonlocal captured_sql
        captured_sql = sql
        return pd.DataFrame(columns=["end_time", "query_text"])

    monkeypatch.setattr(database, "execute", execute)

    database.get_queries()

    assert "NULLIF(TRIM(QUERY_TEXT), '') IS NOT NULL" in captured_sql
