# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.react import run_sql as tool
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    SHOP_CATALOG,
    SHOP_SCRIPT,
    FakeCatalog,
    SQLiteConnector,
    show_shop_tables,
)

CATALOG = FakeCatalog(SHOP_CATALOG)


def _problems(sql: str) -> list[str]:
    return tool.check_sql(sql, "sqlite", CATALOG).problems


def test_known_names_pass() -> None:
    assert (
        _problems(
            "SELECT c.status, SUM(o.amount - o.discount) AS revenue "
            "FROM orders o JOIN customers c ON c.customer_id = o.cust_ref "
            "GROUP BY c.status ORDER BY revenue DESC"
        )
        == []
    )


def test_unknown_table_is_reported() -> None:
    assert _problems("SELECT id FROM clients") == ["Unknown table: clients"]


def test_unknown_qualified_column_names_the_closest_real_one() -> None:
    problems = _problems("SELECT o.ammount FROM orders o")
    assert problems == ["Unknown column: o.ammount (did you mean: amount?)"]


def test_unknown_unqualified_column_is_reported() -> None:
    problems = _problems("SELECT customer_name FROM customers")
    assert len(problems) == 1
    assert problems[0].startswith("Unknown column: customer_name")


def test_output_alias_and_cte_columns_are_not_reported() -> None:
    assert _problems("SELECT amount AS total FROM orders ORDER BY total") == []
    assert (
        _problems(
            "WITH t AS (SELECT cust_ref, SUM(amount) AS spent FROM orders "
            "GROUP BY cust_ref) SELECT spent FROM t"
        )
        == []
    )


@pytest.mark.parametrize(
    "sql, message",
    [
        ("DELETE FROM orders", "Only read queries"),
        ("SELECT 1; SELECT 2", "exactly one SQL statement"),
        ("SELECT FROM WHERE (", "does not parse"),
    ],
)
def test_non_queries_are_refused(sql: str, message: str) -> None:
    assert message in _problems(sql)[0]


def test_render_rows_counts_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_RESULT_ROWS", "2")
    text = tool.render_rows(pd.DataFrame({"x": [1, 2, 3], "y": [None, "b", "c"]}))
    assert text.splitlines() == ["3 rows, first 2 shown", "x | y", "1 | NULL", "2 | b"]


def test_render_rows_explains_an_empty_result() -> None:
    assert tool.render_rows(pd.DataFrame({"x": []})).startswith("0 rows.")


def _ctx(monkeypatch: pytest.MonkeyPatch) -> tuple[ToolContext, SQLiteConnector]:
    connector = SQLiteConnector(SHOP_SCRIPT)
    monkeypatch.setattr(tool, "catalog_for", lambda *_: CATALOG)
    return show_shop_tables(
        ToolContext(retriever=None, connectors=[connector])
    ), connector


def test_run_sql_checks_before_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, connector = _ctx(monkeypatch)
    text = tool.run_sql(ctx, "SELECT bogus FROM orders")
    assert text.startswith("Not run.")
    assert connector.executed == []
    assert ctx.sql_runs == 0


def test_run_sql_refuses_a_table_retrieve_never_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = SQLiteConnector(SHOP_SCRIPT)
    monkeypatch.setattr(tool, "catalog_for", lambda *_: CATALOG)
    ctx = ToolContext(retriever=None, connectors=[connector])
    ctx.show_table(None, "orders")

    text = tool.run_sql(
        ctx,
        "WITH t AS (SELECT cust_ref FROM orders) "
        "SELECT c.status FROM t JOIN customers c ON c.customer_id = t.cust_ref",
    )

    assert text.splitlines() == [
        "Not run. Fix these first:",
        "- Not in view: customers. Retrieve it (its exact name works) before using it.",
    ]
    assert connector.executed == []
    assert ctx.sql_runs == 0


@pytest.mark.parametrize(
    "schema, expected",
    [(None, True), ("sales", True), ("SALES", True), ("archive", False)],
)
def test_was_shown_matches_the_schema_only_when_both_name_one(
    schema: str | None, expected: bool
) -> None:
    ctx = ToolContext(retriever=None, connectors=[])
    ctx.show_table("sales", "Orders")

    assert ctx.was_shown(schema, "orders") is expected


def test_run_sql_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _ctx(monkeypatch)
    text = tool.run_sql(ctx, "SELECT DISTINCT status FROM customers ORDER BY status;")
    assert text.splitlines() == ["2 rows", "status", "ACTIVE", "CHURNED"]
    assert ctx.sql_runs == 1


def test_run_sql_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_SQL_RUNS", "1")
    ctx, _ = _ctx(monkeypatch)
    tool.run_sql(ctx, "SELECT status FROM customers")
    assert "no run_sql calls left" in tool.run_sql(ctx, "SELECT amount FROM orders")


def test_run_sql_reports_database_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _ctx(monkeypatch)
    text = tool.run_sql(ctx, "SELECT amount / FROM orders")
    assert "does not parse" in text or text.startswith("Error from the database")


def test_connector_for_prefers_the_target_database() -> None:
    first, second = SQLiteConnector(SHOP_SCRIPT), SQLiteConnector(SHOP_SCRIPT)
    second.database_name = "other"
    ctx = ToolContext(None, [first, second], database_name="other")
    assert tool.connector_for(ctx) is second


def test_catalog_for_loads_only_the_connectors_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids_by_db = {"db_a": ["schema-a"], "db_b": ["schema-b"]}
    loaded: list[list[str]] = []
    monkeypatch.setattr(tool, "fetch_schema_ids_for_database", ids_by_db.__getitem__)
    monkeypatch.setattr(tool, "fetch_all_schema_ids", lambda: ["schema-a", "schema-b"])
    monkeypatch.setattr(
        tool, "get_schemas_by_ids", lambda ids: loaded.append(ids) or {"public": ids}
    )
    db_a, db_b = SQLiteConnector(SHOP_SCRIPT), SQLiteConnector(SHOP_SCRIPT)
    db_a.database_name, db_b.database_name = "db_a", "db_b"
    ctx = ToolContext(None, [db_a, db_b])

    tool.catalog_for(ctx, db_a)
    tool.catalog_for(ctx, db_a)
    tool.catalog_for(ctx, db_b)

    assert loaded == [["schema-a"], ["schema-b"]]
    assert ctx.schemas == {
        "db_a": {"public": ["schema-a"]},
        "db_b": {"public": ["schema-b"]},
    }


def test_catalog_for_falls_back_to_every_schema_without_a_database_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool, "fetch_all_schema_ids", lambda: ["schema-a", "schema-b"])
    monkeypatch.setattr(tool, "get_schemas_by_ids", lambda ids: {"public": ids})
    connector = SQLiteConnector(SHOP_SCRIPT)
    connector.database_name = None
    ctx = ToolContext(None, [connector])

    tool.catalog_for(ctx, connector)

    assert ctx.schemas == {None: {"public": ["schema-a", "schema-b"]}}
