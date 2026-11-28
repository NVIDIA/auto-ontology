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
    assert redacted == ("postgresql://auto_ontology:***@localhost:5433/auto_ontology")


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
