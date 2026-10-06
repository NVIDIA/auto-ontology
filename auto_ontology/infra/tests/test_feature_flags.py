# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading instance-wide feature flags."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from auto_ontology.infra import feature_flags
from auto_ontology.infra.feature_flags import (
    get_sql_query_timeout_seconds,
    is_distinct_value_probing_enabled,
    read_configuration_flag,
    read_configuration_int,
)


class _Cursor:
    def __init__(self, row: tuple[Any, ...] | None) -> None:
        self._row = row
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor

    def __enter__(self) -> "_Connection":
        return self

    def __exit__(self, *args: object) -> None:
        return None


@contextmanager
def _patched(
    monkeypatch: pytest.MonkeyPatch, row: tuple[Any, ...] | None
) -> Iterator[_Cursor]:
    cursor = _Cursor(row)
    monkeypatch.setattr(
        feature_flags, "get_postgres_connection_string", lambda: "postgres://stub"
    )
    monkeypatch.setattr(
        feature_flags.psycopg, "connect", lambda *a, **k: _Connection(cursor)
    )
    yield cursor


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        (" true ", True),
        ("false", False),
        ("FALSE", False),
    ],
)
def test_stored_value_wins_over_the_default(
    monkeypatch: pytest.MonkeyPatch, stored: str, expected: bool
) -> None:
    # Casing and stray whitespace must not flip a flag; the frontend writes
    # lowercase, but nothing stops an operator editing the row by hand.
    with _patched(monkeypatch, (stored,)):
        assert read_configuration_flag("k", default=not expected) is expected


@pytest.mark.parametrize("stored", ["", "1", "yes", "null", "  "])
def test_unrecognised_value_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, stored: str
) -> None:
    """Must agree with the frontend, which reads opt-out flags as != 'false'.

    Treating junk as False would render the toggle ON while the backend
    silently stopped probing -- the divergence the route comment rules out.
    """
    with _patched(monkeypatch, (stored,)):
        assert read_configuration_flag("k", default=True) is True
        assert read_configuration_flag("k", default=False) is False


def test_distinct_probing_survives_a_junk_row(monkeypatch: pytest.MonkeyPatch) -> None:
    with _patched(monkeypatch, ("",)):
        assert is_distinct_value_probing_enabled() is True


def test_missing_row_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _patched(monkeypatch, None):
        assert read_configuration_flag("k", default=True) is True
        assert read_configuration_flag("k", default=False) is False


def test_db_error_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Before the frontend's first migration the table does not exist. That must
    # not crash a service at startup, and must not silently flip an opt-out
    # flag to off.
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr(
        feature_flags, "get_postgres_connection_string", lambda: "postgres://stub"
    )
    monkeypatch.setattr(feature_flags.psycopg, "connect", _boom)

    assert read_configuration_flag("k", default=True) is True
    assert read_configuration_flag("k", default=False) is False


def test_distinct_probing_defaults_to_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-out: upgrading an existing instance must not silently stop probing."""
    with _patched(monkeypatch, None) as cursor:
        assert is_distinct_value_probing_enabled() is True
    assert cursor.executed[0][1] == ("distinct_value_probing_enabled",)


def test_distinct_probing_honours_an_explicit_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _patched(monkeypatch, ("false",)):
        assert is_distinct_value_probing_enabled() is False


@pytest.mark.parametrize(
    ("stored", "expected"),
    [("45", 45), (" 45 ", 45), ("1", 1), ("3600", 3600)],
)
def test_sql_query_timeout_honours_a_stored_value(
    monkeypatch: pytest.MonkeyPatch, stored: str, expected: int
) -> None:
    with _patched(monkeypatch, (stored,)) as cursor:
        assert get_sql_query_timeout_seconds() == expected
    assert cursor.executed[0][1] == ("sql_query_timeout_seconds",)


@pytest.mark.parametrize("stored", ["", "abc", "1.5", "0", "-5", "3601"])
def test_unusable_int_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, stored: str
) -> None:
    with _patched(monkeypatch, (stored,)):
        assert read_configuration_int("k", default=30, minimum=1, maximum=3600) == 30


def test_sql_query_timeout_defaults_to_30_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An instance running without the frontend never writes the row."""
    with _patched(monkeypatch, None):
        assert get_sql_query_timeout_seconds() == 30
