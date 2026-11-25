# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch

import pytest

from auto_ontology.connectors.registry import (
    PostgresDatabase,
    _redact,
    create_connector,
    parse_connection_strings,
)


@pytest.mark.parametrize(
    "connection_string, expected",
    [
        (
            "postgresql://test-secret:test-secret@db.example/test",
            "postgresql://test-secret:***@db.example/test",
        ),
        (
            "postgresql://my-test-secret:test-secret@db.example/test",
            "postgresql://my-test-secret:***@db.example/test",
        ),
        (
            "postgresql://user:p%40ss%3Aword@[::1]:5432/test",
            "postgresql://user:***@[::1]:5432/test",
        ),
        ("postgresql://user@db.example/test", "postgresql://user@db.example/test"),
    ],
)
def test_redact_masks_the_password_segment(
    connection_string: str, expected: str
) -> None:
    assert _redact(connection_string) == expected


def test_connector_failure_logs_the_redacted_password(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection_string = "postgresql://test-secret:test-secret@db.example/test"
    with patch.object(
        PostgresDatabase,
        "__init__",
        side_effect=RuntimeError("test connection failure"),
    ):
        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(RuntimeError, match="test connection failure"),
        ):
            create_connector(connection_string)

    assert "postgresql://test-secret:***@db.example/test" in caplog.text
    assert connection_string not in caplog.text


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
