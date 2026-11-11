# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from auto_ontology.retrieval.text_to_sql.react import loop, prompts
from auto_ontology.retrieval.text_to_sql.react import run_sql as run_sql_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.grep import grep
from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    SHOP_CATALOG,
    SHOP_SCRIPT,
    FakeCatalog,
    ScriptedModel,
    SQLiteConnector,
    call,
)

ALL_ORDERS = "SELECT order_id, amount FROM orders ORDER BY order_id"


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_GREP", "true")
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_RESULT_ROWS", "2")
    catalog = FakeCatalog(SHOP_CATALOG)
    monkeypatch.setattr(run_sql_module, "catalog_for", lambda *_: catalog)
    monkeypatch.setattr(loop, "catalog_for", lambda *_: catalog)
    return ToolContext(retriever=None, connectors=[SQLiteConnector(SHOP_SCRIPT)])


def test_a_result_too_long_to_show_is_saved_as_a_page(ctx: ToolContext) -> None:
    text = run_sql_module.run_sql(ctx, ALL_ORDERS)

    assert text.splitlines()[0].startswith("4 rows, first 2 shown. Saved as page 1:")
    assert len(ctx.pages) == 1


def test_a_short_result_is_not_saved(ctx: ToolContext) -> None:
    text = run_sql_module.run_sql(ctx, "SELECT DISTINCT status FROM customers")

    assert text.splitlines()[0] == "2 rows"
    assert ctx.pages == []


def test_without_the_flag_nothing_is_saved(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEXT_TO_SQL_REACT_GREP")

    text = run_sql_module.run_sql(ctx, ALL_ORDERS)

    assert text.splitlines()[0] == "4 rows, first 2 shown"
    assert ctx.pages == []


def test_grep_finds_rows_past_the_ones_shown(ctx: ToolContext) -> None:
    run_sql_module.run_sql(ctx, ALL_ORDERS)

    assert grep(ctx, 1, "^13 ").splitlines() == [
        "1 of 4 rows match",
        "row | order_id | amount",
        "4 | 13 | 20.0",
    ]


def test_grep_without_a_pattern_pages_through_the_result(ctx: ToolContext) -> None:
    run_sql_module.run_sql(ctx, ALL_ORDERS)

    lines = grep(ctx, 1, offset=2).splitlines()

    assert lines[0] == "4 rows, 3 to 4 shown"
    assert [line.split(" | ")[0] for line in lines[2:]] == ["3", "4"]


def test_grep_reads_a_bad_pattern_as_plain_text() -> None:
    ctx = ToolContext(retriever=None, connectors=[])
    ctx.pages.append(pd.DataFrame({"name": ["a(b", "c"]}))

    assert grep(ctx, 1, "a(b").splitlines()[0] == "1 of 2 rows match"


def test_grep_explains_a_missing_page() -> None:
    ctx = ToolContext(retriever=None, connectors=[])
    assert grep(ctx, 1, "x").startswith("Error: no saved results.")
    ctx.pages.append(pd.DataFrame({"x": [1]}))
    assert grep(ctx, 2, "x") == "Error: no page 2; pages 1 to 1 exist."


def test_grep_is_offered_with_sql_tools_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_GREP", "true")
    names = lambda mode: [t["function"]["name"] for t in prompts.tools_for(mode)]  # noqa: E731

    assert names(EpisodeMode.SQL) == ["retrieve", "run_sql", "grep", "submit"]
    assert names(EpisodeMode.AUTO) == [
        "retrieve",
        "run_sql",
        "grep",
        "submit",
        "predict",
    ]
    assert names(EpisodeMode.PREDICT) == ["retrieve", "predict"]
    assert "- grep:" not in prompts.system_prompt_for(EpisodeMode.PREDICT, None)


def test_the_episode_offers_grep_only_with_the_flag(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = "SELECT SUM(amount) FROM orders"
    script = [
        AIMessage(content="", tool_calls=[call("run_sql", "1", sql=ALL_ORDERS)]),
        AIMessage(content="", tool_calls=[call("grep", "2", page=1, pattern="^13 ")]),
        AIMessage(
            content="", tool_calls=[call("submit", "3", sql=final, response="x")]
        ),
    ]
    model = ScriptedModel(list(script))

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert model.bound_tool_names[0] == ["retrieve", "run_sql", "grep", "submit"]
    assert "- grep:" in model.seen[0][0].content
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[1].startswith("1 of 4 rows match")

    monkeypatch.delenv("TEXT_TO_SQL_REACT_GREP")
    model = ScriptedModel(list(script))
    result = loop.run_episode(model, ToolContext(None, ctx.connectors), "q")

    assert model.bound_tool_names[0] == ["retrieve", "run_sql", "submit"]
    assert model.seen[0][0].content == prompts.SYSTEM_PROMPT.format(dialect="SQL")
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[1].startswith("Unknown tool 'grep'")
