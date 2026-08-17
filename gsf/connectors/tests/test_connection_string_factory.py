# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from gsf.connectors.connection_string_factory import build_connection_string
from gsf.connectors.snowflake import _parse_connection_string
from gsf.connectors.sqlite import _sqlite_path_from_connection_string

PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzk4OcbXcT\n"
    "-----END PRIVATE KEY-----\n"
)


def _snowflake(**overrides: object) -> dict[str, object]:
    connection = {
        "type": "snowflake",
        "account": "acct",
        "warehouse": "WH",
        "user": "svc_user",
        "database": "DB",
    }
    connection.update(overrides)
    return connection


def test_password_form_is_unchanged() -> None:
    """The old Snowflake account still authenticates with a password."""
    result = build_connection_string(_snowflake(password="p@ss/word"))

    assert result == "snowflake://svc_user:p%40ss%2Fword@acct?warehouse=WH&database=DB"


def test_private_key_omits_the_password_segment() -> None:
    result = build_connection_string(_snowflake(private_key=PEM))

    parsed = urlparse(result)
    assert parsed.password is None
    assert parsed.username == "svc_user"
    assert "private_key" in parse_qs(parsed.query)


def test_private_key_is_base64_encoded_in_the_url() -> None:
    """A raw PEM cannot survive a query string: parse_qs turns ``+`` into a space."""
    result = build_connection_string(_snowflake(private_key=PEM))

    encoded = parse_qs(urlparse(result).query)["private_key"][0]
    assert "BEGIN" not in encoded
    # Surrounding whitespace is trimmed; a PEM parses without a trailing newline.
    assert base64.urlsafe_b64decode(encoded).decode() == PEM.strip()


def test_already_encoded_key_is_not_double_encoded() -> None:
    """Rebuilding a connection string from a stored value must be idempotent."""
    encoded = base64.urlsafe_b64encode(PEM.encode()).decode()

    result = build_connection_string(_snowflake(private_key=encoded))

    assert parse_qs(urlparse(result).query)["private_key"][0] == encoded


def test_passphrase_is_carried_when_present() -> None:
    result = build_connection_string(
        _snowflake(private_key=PEM, private_key_passphrase="s3cret")
    )

    assert parse_qs(urlparse(result).query)["private_key_passphrase"] == ["s3cret"]


def test_passphrase_is_omitted_when_blank() -> None:
    result = build_connection_string(
        _snowflake(private_key=PEM, private_key_passphrase="   ")
    )

    assert "private_key_passphrase" not in parse_qs(urlparse(result).query)


def test_private_key_takes_precedence_over_a_stale_password() -> None:
    """A leftover password must not be sent to an account that rejects passwords."""
    result = build_connection_string(_snowflake(private_key=PEM, password="old"))

    assert urlparse(result).password is None


def test_missing_both_credentials_is_rejected() -> None:
    with pytest.raises(ValueError, match="password"):
        build_connection_string(_snowflake())


@pytest.mark.parametrize("field", ["account", "warehouse", "user", "database"])
def test_required_fields_are_still_enforced_for_key_pair(field: str) -> None:
    connection = _snowflake(private_key=PEM)
    connection[field] = ""

    with pytest.raises(ValueError, match=field):
        build_connection_string(connection)


def test_round_trip_reaches_the_driver_kwargs() -> None:
    """The builder's output must be consumable by the connector that receives it."""
    connection = _snowflake(private_key=PEM, private_key_passphrase="")
    encoded = parse_qs(urlparse(build_connection_string(connection)).query)

    assert encoded["warehouse"] == ["WH"]
    assert encoded["database"] == ["DB"]
    # The PEM above is a stub rather than a real key, so parsing must fail at the
    # key itself -- proving the value arrived intact rather than being mangled.
    with pytest.raises(ValueError, match="could not be parsed"):
        _parse_connection_string(build_connection_string(connection))


def test_sqlite_path_becomes_a_uri_the_connector_can_parse() -> None:
    result = build_connection_string(
        {"type": "sqlite", "database": "regional_sales", "path": "/data/rs.sqlite"}
    )

    assert result == "sqlite:///data/rs.sqlite"
    assert _sqlite_path_from_connection_string(result) == Path("/data/rs.sqlite")


def test_sqlite_requires_a_path() -> None:
    with pytest.raises(ValueError, match="path"):
        build_connection_string({"type": "sqlite", "database": "regional_sales"})


def test_sqlite_rejects_a_relative_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        build_connection_string({"type": "sqlite", "path": "rs.sqlite"})
