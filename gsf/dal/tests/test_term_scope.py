# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic Term merge scope policy tests."""

from __future__ import annotations

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
