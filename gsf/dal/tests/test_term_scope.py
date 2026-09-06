# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic Term merge scope policy tests."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from gsf.dal import terms


def test_term_scope_policy_defaults_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TERM_AUTO_MERGE_SCOPE", raising=False)
    assert terms._term_scope_policy() == "global"
    monkeypatch.setenv("TERM_AUTO_MERGE_SCOPE", "schema")
    assert terms._term_scope_policy() == "schema"
    monkeypatch.setenv("TERM_AUTO_MERGE_SCOPE", "guess")
    with pytest.raises(ValueError, match="TERM_AUTO_MERGE_SCOPE"):
        terms._term_scope_policy()


@pytest.mark.parametrize(
    ("policy", "target", "represented", "conflicts"),
    [
        ("global", ("db-a", "sales"), {("db-b", "sales")}, False),
        ("database", ("db-a", "sales"), {("db-a", "other")}, False),
        ("database", ("db-a", "sales"), {("db-b", "sales")}, True),
        ("schema", ("db-a", "sales"), {("db-a", "sales")}, False),
        ("schema", ("db-a", "sales"), {("db-a", "other")}, True),
        ("schema", ("db-a", "sales"), set(), False),
    ],
)
def test_scope_conflict_policy(
    policy: str,
    target: tuple[str, str],
    represented: set[tuple[str, str]],
    conflicts: bool,
) -> None:
    assert terms._scope_conflicts(policy, target, represented) is conflicts


def test_schema_policy_rejects_existing_exact_name_in_another_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM_AUTO_MERGE_SCOPE", "schema")
    store = MagicMock()
    store.query_read.side_effect = [
        [{"database_name": "catalog", "schema_name": "manufacturing"}],
        [{"database_name": "catalog", "schema_name": "commerce"}],
    ]

    with patch("gsf.dal.terms.store", return_value=store):
        with pytest.raises(terms._TermScopeConflictError, match="outside.*schema"):
            terms._guard_term_scope("Product", "table-id")


def test_merge_serializes_scope_check_with_term_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM_AUTO_MERGE_SCOPE", "schema")
    store = MagicMock()
    store.query_read.return_value = [{"id": "table-id"}]
    locked = False

    def query_write(statement, parameters=None):
        nonlocal locked
        if isinstance(statement, str):
            assert "pg_advisory_xact_lock" in statement
            assert parameters == {"key": "semantic:Product"}
            locked = True
            return []
        if not locked:
            raise AssertionError("Term write occurred before advisory scope lock")
        if "RETURNING term.id" in str(statement):
            return [{"id": "term-id"}]
        return []

    store.query_write.side_effect = query_write

    def guard(*_args, **_kwargs) -> None:
        assert locked

    with (
        patch("gsf.dal.terms.store", return_value=store),
        patch("gsf.dal.terms.write_transaction", return_value=nullcontext()),
        patch("gsf.dal.terms._guard_term_scope", side_effect=guard) as scope_guard,
    ):
        assert terms.merge_term("Product", "A product", "table-id") == "term-id"

    scope_guard.assert_called_once_with("Product", "table-id", policy="schema")


def test_schema_policy_allows_existing_exact_name_in_same_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM_AUTO_MERGE_SCOPE", "schema")
    store = MagicMock()
    store.query_read.side_effect = [
        [{"database_name": "catalog", "schema_name": "commerce"}],
        [{"database_name": "catalog", "schema_name": "commerce"}],
    ]

    with patch("gsf.dal.terms.store", return_value=store):
        terms._guard_term_scope("Customer", "table-id")
