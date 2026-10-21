# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choosing whose grants a ClickHouse connection's chat SQL runs under."""

from __future__ import annotations

from typing import Any

import pytest

from auto_ontology.connectors import execute_as
from auto_ontology.connectors.clickhouse import ClickHouseDatabase
from auto_ontology.connectors.execute_as import (
    ExecuteAsError,
    apply_execute_as,
    uses_execute_as_user,
    username_from_email,
)

_LIST_CONNECTIONS = "auto_ontology.dal.connections.list_connections"


class _OtherConnector:
    """Stands in for any non-ClickHouse connector."""

    def __init__(self, database_name: str) -> None:
        self.database_name = database_name


def _clickhouse(database: str) -> ClickHouseDatabase:
    return ClickHouseDatabase(f"clickhouse://svc:pw@ch.example.com:8123/{database}")


@pytest.fixture(autouse=True)
def _fresh_flag_cache() -> Any:
    """Keep a cached answer from leaking into other test modules."""
    execute_as.invalidate_execute_as_cache()
    yield
    execute_as.invalidate_execute_as_cache()


def _stored(monkeypatch: pytest.MonkeyPatch, *connections: dict[str, Any]) -> None:
    monkeypatch.setattr(_LIST_CONNECTIONS, lambda: list(connections))


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("alice.smith@nvidia.com", "alice.smith"),
        # As written: ClickHouse user names are case-sensitive.
        ("Alice.Smith@nvidia.com", "Alice.Smith"),
        ("  bob@nvidia.com  ", "bob"),
        ("first+tag@nvidia.com", "first+tag"),
        ("o-neil_2@nvidia.com", "o-neil_2"),
    ],
)
def test_username_is_the_local_part_as_written(email: str, expected: str) -> None:
    assert username_from_email(email) == expected


@pytest.mark.parametrize(
    "email",
    ["no-at-sign", "@nvidia.com", "a`b@nvidia.com", "a b@nvidia.com", "a'b@x.com"],
)
def test_username_refuses_an_unusable_local_part(email: str) -> None:
    with pytest.raises(ValueError):
        username_from_email(email)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("1", True), (None, False)],
)
def test_flag_accepts_bool_and_string_forms(value: object, expected: bool) -> None:
    assert uses_execute_as_user({"execute_as_user": value}) is expected


def test_flagged_clickhouse_connector_runs_as_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stored(monkeypatch, {"database": "shop", "execute_as_user": True})
    connector = _clickhouse("shop")

    (applied,) = apply_execute_as([connector], "alice.smith")

    assert applied is not connector
    assert applied.auth_mode == "execute_as:alice.smith"


def test_unflagged_connectors_are_returned_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stored(
        monkeypatch,
        {"database": "shop", "execute_as_user": True},
        {"database": "other"},
    )
    shop, other, pg = _clickhouse("shop"), _clickhouse("other"), _OtherConnector("pg")

    applied = apply_execute_as([shop, other, pg], "alice.smith")  # type: ignore[list-item]

    assert applied[1] is other
    assert applied[2] is pg


def test_nothing_flagged_needs_no_user(monkeypatch: pytest.MonkeyPatch) -> None:
    _stored(monkeypatch, {"database": "shop"})
    connectors = [_clickhouse("shop")]

    assert apply_execute_as(connectors, None) is connectors


def test_a_flagged_connection_without_a_user_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stored(monkeypatch, {"database": "shop", "execute_as_user": True})

    with pytest.raises(ExecuteAsError):
        apply_execute_as([_clickhouse("shop")], None)


def test_a_flagged_non_clickhouse_connection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stored(monkeypatch, {"database": "pg", "execute_as_user": True})

    with pytest.raises(ExecuteAsError):
        apply_execute_as([_OtherConnector("pg")], "alice.smith")  # type: ignore[list-item]


def _store_down() -> list[dict[str, Any]]:
    raise RuntimeError("store unreachable")


def test_an_unreadable_store_fails_closed_with_clickhouse_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_LIST_CONNECTIONS, _store_down)

    with pytest.raises(ExecuteAsError):
        apply_execute_as([_clickhouse("shop")], "alice.smith")


def test_an_unreadable_store_does_not_block_other_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_LIST_CONNECTIONS, _store_down)
    connectors = [_OtherConnector("pg")]

    assert apply_execute_as(connectors, None) is connectors  # type: ignore[arg-type]


def test_flag_check_is_cached_and_invalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    stored = [{"database": "shop", "execute_as_user": False}]

    def list_connections() -> list[dict[str, Any]]:
        calls.append(1)
        return stored

    monkeypatch.setattr(_LIST_CONNECTIONS, list_connections)
    execute_as.invalidate_execute_as_cache()

    assert execute_as.any_connection_uses_execute_as_user() is False
    stored[0]["execute_as_user"] = True
    assert execute_as.any_connection_uses_execute_as_user() is False
    assert len(calls) == 1

    execute_as.invalidate_execute_as_cache()
    assert execute_as.any_connection_uses_execute_as_user() is True


def test_flag_check_fails_closed_when_the_store_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_LIST_CONNECTIONS, _store_down)
    execute_as.invalidate_execute_as_cache()

    assert execute_as.any_connection_uses_execute_as_user() is True
