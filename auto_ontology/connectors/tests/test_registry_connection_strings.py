# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.connectors.registry import _redact, parse_connection_strings


def test_redact_masks_the_password_slot_not_the_first_substring() -> None:
    """``str.replace`` replaces the first occurrence *anywhere*.

    A username containing the password as a substring (the repo's own test
    config runs user == password) got masked instead of the password, and
    the real password stayed in clear text in the log line. Issue #693.
    """
    redacted = _redact("postgresql://s3cr3t:s3cr3t@db.example.com:5432/mydb")
    assert redacted == "postgresql://s3cr3t:***@db.example.com:5432/mydb"

    redacted = _redact(
        "postgresql://auto_ontology:auto_ontology@localhost:5433/auto_ontology"
    )
    assert redacted == "postgresql://auto_ontology:***@localhost:5433/auto_ontology"

    # Redaction is idempotent: an already-masked string round-trips.
    assert _redact("postgresql://user:***@db.example.com/mydb") == (
        "postgresql://user:***@db.example.com/mydb"
    )


def test_redact_masks_the_password_when_the_username_is_empty() -> None:
    """A pattern anchored on a nonempty username would skip ":pass@host".

    ``urlparse`` reads a password for an empty username too, and the
    connector-failure log line must not carry it.
    """
    assert _redact("postgresql://:s3cr3t@db.example.com:5432/mydb") == (
        "postgresql://:***@db.example.com:5432/mydb"
    )


def test_redact_masks_a_password_containing_at() -> None:
    """``urlparse`` ends the password at the LAST '@' of the netloc.

    A pattern stopping at the first '@' masked only the part of the
    password before it and left the rest in clear text.
    """
    assert _redact("postgresql://user:p@ss@db.example.com/mydb") == (
        "postgresql://user:***@db.example.com/mydb"
    )


def test_redact_masks_the_password_when_the_url_contains_a_tab() -> None:
    """Python 3.12+ ``urlparse`` strips tabs before parsing.

    A tab inside the userinfo made the parsed netloc differ from the
    raw string, so a netloc-based replacement on the raw string would
    miss and leave the password in the log line. Found by the
    CodeRabbit review on #742.
    """
    raw = "postgresql://us" + chr(9) + "ter:" + "s3c" + "r3t" + "@db.example.com/mydb"
    assert _redact(raw) == "postgresql://uster:***@db.example.com/mydb"


def test_parse_connection_strings_accepts_commas_and_newlines() -> None:
    raw = """
        postgresql://localhost/first
        clickhouse://localhost/second,

        snowflake://account?database=third
    """

    assert parse_connection_strings(raw) == [
        "postgresql://localhost/first",
        "clickhouse://localhost/second",
        "snowflake://account?database=third",
    ]
