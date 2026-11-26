# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for find_unresolved_columns."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from auto_ontology.catalog.parsers.unresolved_columns import find_unresolved_columns


def _schemas() -> dict:
    rows = [
        ("orders", "id"),
        ("orders", "customer_id"),
        ("orders", "amount"),
        ("customers", "id"),
        ("customers", "name"),
    ]
    df = pd.DataFrame(rows, columns=["table_name", "column_name"])
    return {"sales": SimpleNamespace(columns_df=df)}


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT SUM(CASE WHEN missing >= 20 THEN amount END) AS bulk "
            "FROM sales.orders",
            ["missing"],
        ),
        ("SELECT o.missing FROM sales.orders AS o", ["o.missing"]),
        (
            "SELECT name, missing FROM orders o JOIN customers c "
            "ON o.customer_id = c.id",
            ["missing"],
        ),
        ("WITH t AS (SELECT customer_id FROM orders) SELECT total FROM t", ["total"]),
        ("SELECT d.nope FROM (SELECT SUM(amount) AS tot FROM orders) AS d", ["d.nope"]),
        ("SELECT id FROM orders UNION ALL SELECT nope FROM customers", ["nope"]),
        ("SELECT id FROM customers WHERE id IN (SELECT nope FROM orders)", ["nope"]),
        (
            "SELECT id FROM customers c WHERE EXISTS "
            "(SELECT 1 FROM orders o WHERE o.customer_id = zzz)",
            ["zzz"],
        ),
        ("SELECT missing AS missing FROM orders", ["missing"]),
        ("SELECT amount AS total FROM orders WHERE total > 1", ["total"]),
        ("SELECT amount AS missing, missing FROM orders", ["missing"]),
        (
            "SELECT orders.foo FROM orders JOIN other ON orders.id = other.id",
            ["orders.foo"],
        ),
    ],
    ids=[
        "bare",
        "qualified",
        "join",
        "cte-output",
        "derived-output",
        "union-branch",
        "in-subquery",
        "correlated-subquery",
        "self-alias",
        "alias-in-where",
        "alias-name-collision",
        "qualified-beside-unknown-table",
    ],
)
def test_reports_columns_the_catalog_rules_out(sql: str, expected: list[str]) -> None:
    assert find_unresolved_columns(sql, "postgres", _schemas()) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(CASE WHEN customer_id >= 20 THEN amount END) AS bulk "
        "FROM sales.orders",
        "SELECT SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY total",
        "SELECT customer_id, SUM(amount) AS total FROM orders "
        "GROUP BY customer_id HAVING total > 1",
        "WITH t AS (SELECT customer_id, SUM(amount) AS tot FROM orders GROUP BY 1) "
        "SELECT tot FROM t",
        "WITH t(a) AS (SELECT amount FROM orders) SELECT a FROM t",
        "SELECT d.a FROM (SELECT amount FROM orders) AS d(a)",
        "WITH t AS (SELECT * FROM orders) SELECT anything FROM t",
        "SELECT foo FROM orders JOIN other ON orders.id = other.id",
        "SELECT amount FROM elsewhere.orders",
        "SELECT id FROM customers c WHERE EXISTS "
        "(SELECT 1 FROM orders o WHERE o.customer_id = c.id AND name <> '')",
        "SELECT c.name, x.m FROM customers c CROSS JOIN LATERAL "
        "(SELECT MAX(amount) AS m FROM orders o WHERE o.customer_id = c.id) AS x",
        "SELECT amount, g FROM orders, generate_series(1, 3) AS g",
        "SELECT amount, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount) "
        "AS rn FROM orders",
        "SELECT amount FROM orders JOIN customers USING (id)",
        'SELECT "Amount" FROM orders',
        "SELECT o.*, COUNT(*) FILTER (WHERE amount > 1) FROM orders AS o GROUP BY 1",
    ],
    ids=[
        "valid",
        "order-by-alias",
        "having-alias",
        "cte",
        "cte-column-aliases",
        "derived-column-aliases",
        "star-source",
        "unknown-table",
        "unknown-schema",
        "correlated-outer-column",
        "lateral",
        "table-function",
        "window",
        "using",
        "case-insensitive",
        "star-and-filter",
    ],
)
def test_stays_silent_when_nothing_is_proven_missing(sql: str) -> None:
    assert find_unresolved_columns(sql, "postgres", _schemas()) == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT amount AS total FROM orders WHERE total > 1",
        "SELECT o.amount AS total FROM orders o JOIN customers c ON total = c.id",
    ],
    ids=["where-alias", "join-on-alias"],
)
def test_sqlite_allows_output_aliases_in_where_and_join_on(sql: str) -> None:
    assert find_unresolved_columns(sql, "sqlite", _schemas()) == []


def test_an_empty_catalog_verifies_nothing() -> None:
    assert find_unresolved_columns("SELECT missing FROM orders", "postgres", {}) == []
