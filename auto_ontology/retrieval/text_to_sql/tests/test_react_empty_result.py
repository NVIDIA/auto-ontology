# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from langchain_core.messages import AIMessage

from auto_ontology.retrieval.text_to_sql.db_probe import executor as executor_module
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.react import empty_result, loop
from auto_ontology.retrieval.text_to_sql.react import run_sql as run_sql_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    FakeCatalog,
    ScriptedModel,
    SQLiteConnector,
    call,
)

# site_id is declared without a type, as on the SQLite source in #257: it holds
# integers, and SQLite does not convert '1' to match one.
SCRIPT = """
CREATE TABLE sites (site_id, code TEXT);
CREATE TABLE sales_orders (order_id INTEGER, site_id, status TEXT, amount REAL);
INSERT INTO sites VALUES (1, 'S0001'), (2, 'S0002');
INSERT INTO sales_orders VALUES
  (10, 1, 'OPEN', 20), (11, 1, 'SHIPPED', 50), (12, 2, 'OPEN', 100);
"""
CATALOG = {
    "sites": ["site_id", "code"],
    "sales_orders": ["order_id", "site_id", "status", "amount"],
}
TYPE_MISMATCH = "SELECT order_id FROM sales_orders WHERE site_id = '1'"


@pytest.fixture(autouse=True)
def enable_empty_result_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_EMPTY_RESULT_CHECK", "true")


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    catalog = FakeCatalog(CATALOG)
    monkeypatch.setattr(run_sql_module, "catalog_for", lambda *_: catalog)
    monkeypatch.setattr(empty_result, "catalog_for", lambda *_: catalog)
    context = ToolContext(connectors=[SQLiteConnector(SCRIPT)])
    context.connectors[0].supports_statement_timeout = True
    for name in CATALOG:
        context.show_table(None, name)
    return context


def _executed(ctx: ToolContext) -> list[str]:
    return ctx.connectors[0].executed


def test_an_empty_filtered_result_is_held_back_once_then_accepted(
    ctx: ToolContext,
) -> None:
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[call("submit", "1", sql=TYPE_MISMATCH)]),
            AIMessage(content="", tool_calls=[call("submit", "2", sql=TYPE_MISMATCH)]),
        ]
    )

    result = loop.run_episode(model, ctx, "Which orders came from site S0001?")

    assert result.reason == "submitted"
    assert result.submission.sql == TYPE_MISMATCH
    assert result.submission.db_response.result == ["[]"]
    first, second = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert first.startswith("Not accepted yet. The query returned 0 rows.")
    assert "submit this version again" in first
    assert second == "Accepted."


def test_disabling_empty_result_check_accepts_and_does_not_probe(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_EMPTY_RESULT_CHECK", "false")

    observation, submission = loop.submit(ctx, TYPE_MISMATCH)

    assert observation == "Accepted."
    assert submission is not None
    assert len(_executed(ctx)) == 1
    assert not ctx.empty_result_checked


def test_a_literal_of_the_wrong_type_shows_the_stored_values(ctx: ToolContext) -> None:
    observation, submission = loop.submit(ctx, TYPE_MISMATCH)

    assert submission is None
    assert (
        "- site_id = '1': matches no row of sales_orders on its own. Stored "
        "values equal to it as text, ignoring case: 1. They are stored as "
        "numbers: compare without quotes."
    ) in observation


def test_a_value_in_the_wrong_case_shows_the_stored_one(ctx: ToolContext) -> None:
    observation, _ = loop.submit(
        ctx, "SELECT order_id FROM sales_orders o WHERE o.status IN ('open', 'lost')"
    )

    assert "Stored values equal to it as text, ignoring case: 'OPEN'." in observation


def test_a_missing_value_shows_a_sample_of_the_stored_ones(ctx: ToolContext) -> None:
    observation, _ = loop.submit(
        ctx, "SELECT order_id FROM sales_orders WHERE status = 'CANCELLED'"
    )

    assert "matches no row of sales_orders on its own." in observation
    assert "Stored values: 'OPEN', 'SHIPPED'." in observation


def test_a_range_filter_shows_the_stored_range(ctx: ToolContext) -> None:
    observation, _ = loop.submit(
        ctx, "SELECT order_id FROM sales_orders WHERE amount > 1000"
    )

    assert "- amount > 1000: matches no row of sales_orders on its own." in observation
    assert "Stored range: 20.0 to 100.0." in observation


def test_range_keys_are_read_whatever_their_case(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = ctx.connectors[0]
    execute = connector.execute

    def upper_case_columns(sql: str, timeout_s: float | None = None):
        frame = execute(sql, timeout_s=timeout_s)
        return frame.rename(columns=str.upper)

    monkeypatch.setattr(connector, "execute", upper_case_columns)

    observation, _ = loop.submit(
        ctx, "SELECT order_id FROM sales_orders WHERE amount > 1000"
    )

    assert "Stored range: 20.0 to 100.0." in observation


def test_a_comparison_under_a_not_is_not_probed(ctx: ToolContext) -> None:
    observation, _ = loop.submit(
        ctx,
        "SELECT order_id FROM sales_orders "
        "WHERE NOT (status = 'OPEN' OR status = 'SHIPPED')",
    )

    assert "could not be checked" in observation
    assert "no filter compares a column with a value" in observation
    assert len(_executed(ctx)) == 1


def test_filters_that_each_match_point_at_how_they_combine(ctx: ToolContext) -> None:
    observation, _ = loop.submit(
        ctx,
        "SELECT o.order_id FROM sales_orders o JOIN sites s ON s.site_id = o.site_id "
        "WHERE s.code = 'S0002' AND o.status = 'SHIPPED'",
    )

    assert "- s.code = 'S0002': matches rows on its own" in observation
    assert "- o.status = 'SHIPPED': matches rows on its own" in observation


def test_an_aggregate_filter_is_named_as_not_checked(ctx: ToolContext) -> None:
    observation, _ = loop.submit(
        ctx,
        "SELECT site_id FROM sales_orders WHERE status = 'OPEN' "
        "GROUP BY site_id HAVING COUNT(*) > 5",
    )

    assert "- status = 'OPEN': matches rows on its own" in observation
    assert "- Not checked: COUNT(*) > 5" in observation


def test_each_new_empty_query_is_checked_and_a_resubmission_accepted(
    ctx: ToolContext,
) -> None:
    first = "SELECT order_id FROM sales_orders WHERE status = 'OPNE'"
    second = "SELECT order_id FROM sales_orders WHERE status = 'CLOESD'"

    assert loop.submit(ctx, first)[1] is None
    observation, submission = loop.submit(ctx, second)

    assert submission is None
    assert "Stored values: 'OPEN', 'SHIPPED'." in observation
    assert loop.submit(ctx, second)[0] == "Accepted."


def test_a_having_alias_is_not_probed_as_a_table_column(ctx: ToolContext) -> None:
    observation, submission = loop.submit(
        ctx,
        "SELECT site_id, SUM(amount) AS amount FROM sales_orders "
        "GROUP BY site_id HAVING amount > 500",
    )

    assert submission is None
    assert "amount > 500" in observation
    assert "Stored range" not in observation
    assert len(_executed(ctx)) == 1


def test_an_empty_result_without_a_filter_is_accepted(ctx: ToolContext) -> None:
    observation, submission = loop.submit(
        ctx, "SELECT order_id FROM sales_orders LIMIT 0"
    )

    assert observation == "Accepted."
    assert submission.db_response.result == ["[]"]
    assert not ctx.empty_result_checked


def test_a_filtered_result_with_rows_is_accepted_without_probes(
    ctx: ToolContext,
) -> None:
    sql = "SELECT order_id FROM sales_orders WHERE site_id = 1"

    observation, submission = loop.submit(ctx, sql)

    assert observation == "Accepted."
    assert submission.sql == sql
    assert len(_executed(ctx)) == 1
    assert not ctx.empty_result_checked


def test_without_evidence_the_rejection_says_the_filters_were_not_checked(
    ctx: ToolContext,
) -> None:
    ctx.cost_status = lambda: "BUDGET: 5 coins left."
    ctx.can_run_queries = lambda: True

    observation, submission = loop.submit(ctx, TYPE_MISMATCH)

    assert submission is None
    assert "could not be checked against the stored values" in observation
    assert "queries here are charged" in observation
    assert len(_executed(ctx)) == 1
    observation, submission = loop.submit(ctx, TYPE_MISMATCH)
    assert observation == "Accepted."


def test_failing_probes_are_reported_as_missing_evidence(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = ctx.connectors[0]
    execute = connector.execute

    def first_query_only(sql: str, timeout_s: float | None = None):
        if connector.executed:
            raise RuntimeError("permission denied")
        return execute(sql, timeout_s=timeout_s)

    monkeypatch.setattr(connector, "execute", first_query_only)

    observation, submission = loop.submit(ctx, TYPE_MISMATCH)

    assert submission is None
    assert "could not be checked against the stored values" in observation
    assert "permission denied" in observation


def test_all_probes_share_one_time_budget(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = ctx.connectors[0]
    timeouts: list[float | None] = []
    execute = connector.execute

    def record_timeout(sql: str, timeout_s: float | None = None):
        timeouts.append(timeout_s)
        return execute(sql, timeout_s=timeout_s)

    times = iter([100.0, 103.0, 111.0])
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: next(times, 111.0))
    monkeypatch.setattr(connector, "execute", record_timeout)
    executor = ProbeExecutor(connector, total_timeout_s=10)

    assert executor.run("SELECT 1")["ok"] is True
    assert executor.run("SELECT 2")["error"] == "probe time budget exhausted"
    assert timeouts == [pytest.approx(7.0)]


def test_probes_are_skipped_without_statement_timeout(ctx: ToolContext) -> None:
    ctx.connectors[0].supports_statement_timeout = False

    observation, submission = loop.submit(ctx, TYPE_MISMATCH)

    assert submission is None
    assert "connector cannot cap probe statements in time" in observation
    assert len(_executed(ctx)) == 1


def test_the_last_answer_attempt_is_accepted_without_probes(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "1")
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[call("retrieve", "1", query="a")]),
            AIMessage(content="", tool_calls=[call("submit", "2", sql=TYPE_MISMATCH)]),
        ]
    )
    monkeypatch.setattr(loop, "retrieve", lambda *_a, **_k: "TABLES")

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert result.submission is not None
    assert result.submission.db_response is not None
    assert result.submission.db_response.result == ["[]"]
    assert len(_executed(ctx)) == 1


def test_a_final_run_sql_is_told_to_run_again(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_SUBMIT_TOOL", "false")
    observation, _, _ = loop._dispatch(
        ctx, "run_sql", {"sql": TYPE_MISMATCH, "final": True}, ["run_sql"]
    )

    assert "run this version with final=true again" in observation
