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


@pytest.mark.parametrize(
    "sql, problem",
    [
        (
            "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
            "DELETE",
        ),
        (
            "WITH i AS (INSERT INTO orders (amount) VALUES (1) RETURNING *) "
            "SELECT * FROM i",
            "INSERT",
        ),
        ("SELECT * INTO orders_copy FROM orders", "INTO"),
        ("SELECT amount FROM orders FOR UPDATE", "LOCK"),
        ("SELECT setval('orders_id_seq', 1)", "function setval()"),
        ("SELECT nextval('orders_id_seq')", "function nextval()"),
        ("SELECT pg_advisory_lock(1)", "function pg_advisory_lock()"),
        (
            "SELECT pg_catalog.pg_terminate_backend(42)",
            "function pg_terminate_backend()",
        ),
        ("SELECT PG_CANCEL_BACKEND(42)", "function pg_cancel_backend()"),
        ("SELECT set_config('search_path', 'x', false)", "function set_config()"),
        ("SELECT * FROM dblink('db', 'SELECT 1') AS t(a INT)", "function dblink()"),
        ("SELECT lo_export(1, '/tmp/x')", "function lo_export()"),
        (
            "SELECT query_to_xml('DELETE FROM orders', true, true, '')",
            "function query_to_xml()",
        ),
    ],
)
def test_side_effects_are_refused(sql: str, problem: str) -> None:
    problems = tool.check_sql(sql, "postgres", CATALOG).problems
    assert problems == [f"Not allowed in a read query: {problem}"]


def test_harmless_functions_still_pass() -> None:
    sql = (
        "SELECT pg_typeof(amount), COALESCE(discount, 0), UPPER(status), COUNT(*) "
        "FROM orders o JOIN customers c ON c.customer_id = o.cust_ref "
        "GROUP BY 1, 2, 3"
    )
    assert tool.check_sql(sql, "postgres", CATALOG).problems == []


def test_render_rows_counts_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_RESULT_ROWS", "2")
    text = tool.render_rows(pd.DataFrame({"x": [1, 2, 3], "y": [None, "b", "c"]}))
    assert text.splitlines() == ["3 rows, first 2 shown", "x | y", "1 | NULL", "2 | b"]


def test_render_rows_explains_an_empty_result() -> None:
    assert tool.render_rows(pd.DataFrame({"x": []})).startswith("0 rows.")


def _ctx(monkeypatch: pytest.MonkeyPatch) -> tuple[ToolContext, SQLiteConnector]:
    connector = SQLiteConnector(SHOP_SCRIPT)
    monkeypatch.setattr(tool, "catalog_for", lambda *_: CATALOG)
    return show_shop_tables(ToolContext(connectors=[connector])), connector


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
    ctx = ToolContext(connectors=[connector])
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


class _StubSchema:
    def __init__(self, tables: dict[str, list[str]]):
        self.columns_map = {t: dict.fromkeys(cols) for t, cols in tables.items()}

    def table_exists(self, table: str) -> bool:
        return table.lower() in self.columns_map


def test_schema_catalog_guesses_no_copy_for_an_unqualified_name() -> None:
    catalog = tool.SchemaCatalog(
        {
            "sales": _StubSchema({"orders": ["order_id", "amount"]}),
            "archive": _StubSchema({"orders": ["order_id", "archived_at"]}),
            "crm": _StubSchema({"customers": ["customer_id"]}),
        }
    )

    assert catalog.columns("orders", "archive") == ["order_id", "archived_at"]
    assert catalog.columns("orders", None) == ["order_id", "amount", "archived_at"]
    assert catalog.columns("customers", None) == ["customer_id"]
    assert catalog.columns("customers", "sales") is None
    assert sorted(catalog.schemas_with("orders")) == ["archive", "sales"]


class _ThreeLevelConnector(SQLiteConnector):
    """A Trino-like engine: names are ``catalog.schema.table``, bound to ``lake``."""

    dialect = "trino"

    def qualify(self, schema: str | None, table: str) -> str:
        return f'"lake"."{schema}"."{table}"'


def test_the_bound_catalog_comes_from_the_connectors_own_qualify() -> None:
    assert tool.bound_catalog(_ThreeLevelConnector(SHOP_SCRIPT)) == "lake"
    assert tool.bound_catalog(SQLiteConnector(SHOP_SCRIPT)) is None


def test_run_sql_refuses_a_table_in_another_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ThreeLevelConnector(SHOP_SCRIPT)
    monkeypatch.setattr(tool, "catalog_for", lambda *_: CATALOG)
    ctx = ToolContext(connectors=[connector])
    ctx.show_table("sales", "orders")

    text = tool.run_sql(ctx, "SELECT order_id FROM other.sales.orders")

    assert text.splitlines() == [
        "Not run. Fix these first:",
        "- Not allowed: catalog 'other' in other.sales.orders. This database's "
        "catalog is 'lake'; leave the catalog out or use it.",
    ]
    assert connector.executed == []
    assert ctx.sql_runs == 0


@pytest.mark.parametrize(
    "table", ["sales.orders", "lake.sales.orders", "LAKE.sales.orders"]
)
def test_run_sql_allows_the_bound_catalog_or_none(
    monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    connector = _ThreeLevelConnector("CREATE TABLE orders (order_id INTEGER);")
    monkeypatch.setattr(tool, "catalog_for", lambda *_: CATALOG)
    ctx = ToolContext(connectors=[connector])
    ctx.show_table("sales", "orders")

    check = tool.check_query(ctx, connector, f"SELECT order_id FROM {table}")

    assert check.problems == []


@pytest.mark.parametrize(
    "shown, schema, owners, expected",
    [
        ("sales", "sales", ["archive", "sales"], True),
        ("sales", "SALES", ["archive", "sales"], True),
        ("sales", "archive", ["archive", "sales"], False),
        ("sales", None, ["sales"], True),
        ("sales", None, ["archive", "sales"], False),
        ("sales", None, [], False),
        (None, "sales", ["sales"], True),
        (None, "archive", ["archive", "sales"], False),
        (None, "archive", [], False),
        (None, None, ["archive", "sales"], True),
        (None, None, [], True),
    ],
)
def test_was_shown_matches_the_table_each_name_resolves_to(
    shown: str | None, schema: str | None, owners: list[str], expected: bool
) -> None:
    ctx = ToolContext(connectors=[])
    ctx.show_table(shown, "Orders")

    assert ctx.was_shown(schema, "orders", owners) is expected


@pytest.mark.parametrize(
    "shown, sql, problem",
    [
        (
            (None, "orders"),
            "SELECT amount FROM archive.orders",
            "Not in view: archive.orders. "
            "Retrieve it (its exact name works) before using it.",
        ),
        (
            ("archive", "orders"),
            "SELECT amount FROM orders",
            "Ambiguous: orders is in archive, sales. Name its schema.",
        ),
    ],
)
def test_run_sql_refuses_a_same_named_table_in_another_schema(
    monkeypatch: pytest.MonkeyPatch,
    shown: tuple[str | None, str],
    sql: str,
    problem: str,
) -> None:
    connector = SQLiteConnector(SHOP_SCRIPT)
    catalog = FakeCatalog(SHOP_CATALOG, {"orders": ["sales", "archive"]})
    monkeypatch.setattr(tool, "catalog_for", lambda *_: catalog)
    ctx = ToolContext(connectors=[connector])
    ctx.show_table(*shown)

    text = tool.run_sql(ctx, sql)

    assert text.splitlines() == ["Not run. Fix these first:", f"- {problem}"]
    assert connector.executed == []


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
    ctx = ToolContext([first, second], database_name="other")
    assert tool.connector_for(ctx) is second


def _shown(database_name: str, schema: str, name: str) -> dict:
    return {
        "id": f"{database_name}.{schema}.{name}",
        "database_name": database_name,
        "schema_name": schema,
        "name": name,
    }


def _connectors(*names: str) -> list[SQLiteConnector]:
    connectors = []
    for name in names:
        connector = SQLiteConnector(SHOP_SCRIPT)
        connector.database_name = name
        connectors.append(connector)
    return connectors


def test_connector_for_routes_by_the_tables_the_sql_reads() -> None:
    census, works = _connectors("census", "adventureworks")
    ctx = ToolContext(connectors=[census, works])
    for table in (
        _shown("census", "acs", "state_2014"),
        _shown("adventureworks", "main", "salesorderheader"),
        _shown("census", "acs", "county_2014"),
    ):
        ctx.seen_tables[table["id"]] = table

    assert tool.connector_for(ctx) is None
    sql = "SELECT SUM(totaldue) FROM main.salesorderheader"
    assert tool.connector_for(ctx, sql) is works
    assert tool.connector_for(ctx, "SELECT 1 FROM main.missing") is None


def test_a_name_in_two_databases_has_nowhere_to_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shop, works = _connectors("shop", "adventureworks")
    monkeypatch.setattr(tool, "catalog_for", lambda *_: CATALOG)
    ctx = show_shop_tables(ToolContext(connectors=[shop, works]))
    for table in (
        _shown("adventureworks", "main", "orders"),
        _shown("shop", "main", "orders"),
    ):
        ctx.seen_tables[table["id"]] = table
    sql = "SELECT COUNT(*) FROM orders"

    assert tool.connector_for(ctx, sql) is None
    assert tool.run_sql(ctx, sql).startswith(
        "Error: the tables this query reads are in more than one database "
        "(adventureworks, shop)"
    )
    ctx.database_name = "adventureworks"
    assert tool.connector_for(ctx, sql) is works


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
    ctx = ToolContext([db_a, db_b])

    tool.catalog_for(ctx, db_a)
    tool.catalog_for(ctx, db_a)
    tool.catalog_for(ctx, db_b)

    assert loaded == [["schema-a"], ["schema-b"]]
    assert ctx.schemas == {
        "db_a": {"public": ["schema-a"]},
        "db_b": {"public": ["schema-b"]},
    }


def test_a_database_with_no_schemas_gets_an_empty_catalog_not_every_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object) -> None:
        raise AssertionError("must not load schemas for a database that has none")

    monkeypatch.setattr(tool, "fetch_schema_ids_for_database", lambda _db: [])
    monkeypatch.setattr(tool, "fetch_all_schema_ids", lambda: ["schema-other"])
    monkeypatch.setattr(tool, "get_schemas_by_ids", fail)
    connector = SQLiteConnector(SHOP_SCRIPT)
    connector.database_name = "empty_db"
    ctx = ToolContext([connector])

    catalog = tool.catalog_for(ctx, connector)

    assert ctx.schemas == {"empty_db": {}}
    assert catalog.columns("orders", None) is None


def test_catalog_for_falls_back_to_every_schema_without_a_database_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool, "fetch_all_schema_ids", lambda: ["schema-a", "schema-b"])
    monkeypatch.setattr(tool, "get_schemas_by_ids", lambda ids: {"public": ids})
    connector = SQLiteConnector(SHOP_SCRIPT)
    connector.database_name = None
    ctx = ToolContext([connector])

    tool.catalog_for(ctx, connector)

    assert ctx.schemas == {None: {"public": ["schema-a", "schema-b"]}}


GA_CATALOG = FakeCatalog(
    {"ga_sessions_*": ["visitId", "totals", "_TABLE_SUFFIX"], "visitors": ["id"]},
    owners={"ga_sessions_*": ["ga"], "visitors": ["ga"]},
)


@pytest.mark.parametrize(
    "table",
    [
        "ga_sessions_*",
        "ga_sessions_20170801",
        "ga_sessions_2017*",
        "ga_sessions_201708*",
    ],
)
def test_shards_and_partial_wildcards_check_as_the_wildcard_table(table: str) -> None:
    sql = f"SELECT visitId FROM `ga.{table}`"
    check = tool.check_sql(sql, "bigquery", GA_CATALOG)
    ctx = ToolContext(connectors=[])
    ctx.show_table("ga", "ga_sessions_*")

    assert check.problems == []
    assert tool.tables_not_shown(check.statement, ctx, GA_CATALOG) == []


def test_a_shard_name_without_a_wildcard_table_stays_unknown() -> None:
    check = tool.check_sql(
        "SELECT id FROM `ga.visitors_20170801`", "bigquery", GA_CATALOG
    )

    assert check.problems == ['Unknown table: "ga"."visitors_20170801"']


def test_an_unshown_wildcard_is_not_in_view_by_its_shard_name() -> None:
    check = tool.check_sql(
        "SELECT visitId FROM `ga.ga_sessions_20170801`", "bigquery", GA_CATALOG
    )

    problems = tool.tables_not_shown(
        check.statement, ToolContext(connectors=[]), GA_CATALOG
    )

    assert problems == [
        "Not in view: ga.ga_sessions_20170801. Retrieve it (its exact name works) "
        "before using it."
    ]
