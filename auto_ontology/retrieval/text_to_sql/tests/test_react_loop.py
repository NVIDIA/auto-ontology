# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from auto_ontology.retrieval.text_to_sql.prompts import format_dialect_rules
from auto_ontology.retrieval.text_to_sql.react import loop, prompts
from auto_ontology.retrieval.text_to_sql.react import run_sql as run_sql_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    SHOP_CATALOG,
    SHOP_SCRIPT,
    FakeCatalog,
    ScriptedModel,
    SQLiteConnector,
    call,
)

FINAL_SQL = (
    "SELECT SUM(o.amount - o.discount) FROM orders o "
    "JOIN customers c ON c.customer_id = o.cust_ref WHERE c.status = 'ACTIVE'"
)


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    catalog = FakeCatalog(SHOP_CATALOG)
    monkeypatch.setattr(run_sql_module, "catalog_for", lambda _ctx: catalog)
    monkeypatch.setattr(loop, "catalog_for", lambda _ctx: catalog)
    monkeypatch.setattr(
        loop,
        "retrieve",
        lambda _ctx, query, kinds=None: f"TABLE orders (for {query})",
    )
    return ToolContext(retriever=None, connectors=[SQLiteConnector(SHOP_SCRIPT)])


def _reply(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def test_search_check_and_submit(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="revenue per active customer")),
            _reply(call("run_sql", "2", sql="SELECT DISTINCT status FROM customers")),
            _reply(
                call(
                    "submit",
                    "3",
                    sql=FINAL_SQL,
                    response="Revenue of active customers.",
                )
            ),
        ]
    )

    result = loop.run_episode(model, ctx, "What is revenue from active customers?")

    assert result.reason == "submitted"
    assert result.tool_calls == 3
    assert result.submission.sql == FINAL_SQL
    assert json.loads(result.submission.db_response.result[0]) == [
        {"SUM(o.amount - o.discount)": 160.0}
    ]
    tool_results = [m.content for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert tool_results[0] == "TABLE orders (for revenue per active customer)"
    assert "ACTIVE" in tool_results[1]


def test_cost_status_follows_the_question_and_each_tool_result(ctx: ToolContext) -> None:
    left = iter([9, 8])
    ctx.cost_status = lambda: f"BUDGET: {next(left)} coins left."
    model = ScriptedModel(
        [
            _reply(call("run_sql", "1", sql="SELECT DISTINCT status FROM customers")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    loop.run_episode(model, ctx, "q")

    assert model.seen[0][1].content.endswith("BUDGET: 9 coins left.")
    tool_result = [m.content for m in model.seen[-1] if isinstance(m, ToolMessage)][0]
    assert "ACTIVE" in tool_result
    assert tool_result.endswith("BUDGET: 8 coins left.")


def test_no_cost_status_leaves_messages_unchanged(ctx: ToolContext) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q")

    assert model.seen[0][1].content == prompts.user_message("q", None)


def test_rejected_submission_lets_the_model_fix_it(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("submit", "1", sql="SELECT revenue FROM orders", response="x")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    rejection = [t for t in result.trajectory if t["role"] == "tool"][0]["content"]
    assert rejection.startswith("Not accepted.")
    assert "Unknown column: revenue" in rejection


def test_repeated_call_is_not_run_again(ctx: ToolContext) -> None:
    same = dict(query="orders")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", **same)),
            _reply(call("retrieve", "2", **same)),
            _reply(call("submit", "3", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.tool_calls == 2
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[1].startswith("You already made this exact call")


def test_budget_ends_with_a_submit_only_turn(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "2")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="a")),
            _reply(call("retrieve", "2", query="b")),
            _reply(call("submit", "3", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert model.bound_tool_names[-1] == ["submit"]


def test_budget_exhausted_without_submit(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "1")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="a")),
            _reply(call("submit", "2", sql="SELECT nope FROM orders", response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.submission is None
    assert result.reason == "budget_exhausted"


def test_model_that_never_calls_a_tool_gives_up(ctx: ToolContext) -> None:
    model = ScriptedModel([AIMessage(content="hmm")] * 3)

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "no_submission"
    assert result.tool_calls == 0


def test_unknown_tool_and_bad_arguments_are_observations(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("delete_everything", "1")),
            _reply(call("run_sql", "2", query="SELECT 1")),
            _reply(call("submit", "3", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0].startswith("Unknown tool")
    assert observations[1].startswith("Wrong arguments for run_sql")
    assert result.reason == "submitted"


def test_evidence_reaches_the_model(ctx: ToolContext) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", evidence="active means status = 'ACTIVE'")

    assert "EVIDENCE\nactive means status = 'ACTIVE'" in model.seen[0][1].content


@pytest.mark.parametrize("dialect", ["mysql", "sqlite", "oracle"])
def test_dialect_rules_reach_the_model(ctx: ToolContext, dialect: str) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", dialect=dialect)

    system = model.seen[0][0].content
    rules = format_dialect_rules(dialect).strip()
    base = prompts.SYSTEM_PROMPT.format(dialect=dialect)
    # A dialect with rules gets them after the base prompt; one without gets
    # the base prompt alone.
    assert system == (f"{base}\n\n{rules}" if rules else base)
    assert ("Do NOT CAST a computed value" in system) == (dialect == "mysql")


@pytest.mark.parametrize("dialect", [None, "postgresql", "mysql", "sqlite"])
def test_column_order_rule_reaches_the_model(ctx: ToolContext, dialect: str | None) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", dialect=dialect)

    assert "same order the question" in model.seen[0][0].content


class InternalServerError(Exception):
    """Named like the client's error for a throttled gateway, which the retry matches by type."""


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr("auto_ontology.utils.llm_invoke.time.sleep", waited.append)
    return waited


def test_a_throttled_model_call_is_retried_and_the_episode_goes_on(
    ctx: ToolContext, sleeps: list[float]
) -> None:
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="revenue")),
            InternalServerError("Error code: 500 - throttling_error 429"),
            InternalServerError("Error code: 500 - throttling_error 429"),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert len(sleeps) == 2
    # The retried turn saw the earlier search result: nothing was restarted.
    assert any(isinstance(m, ToolMessage) for m in model.seen[-1])


def test_a_model_error_that_is_not_transient_is_not_retried(
    ctx: ToolContext, sleeps: list[float]
) -> None:
    model = ScriptedModel([ValueError("bad request")])

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "model_error"
    assert sleeps == []


def test_retries_end_and_the_episode_gives_up(
    ctx: ToolContext, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MODEL_RETRIES", "3")
    model = ScriptedModel([InternalServerError("throttled")] * 5)

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "model_error"
    assert len(model.seen) == 3
    assert len(sleeps) == 2
