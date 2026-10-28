# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for choosing between the semantic and schema-only SQL candidates."""

from __future__ import annotations

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_candidate_selection
from auto_ontology.retrieval.text_to_sql.agents.sql_candidate_selection import (
    SelectionContext,
    SQLCandidate,
    format_result_preview,
    result_key,
    select_candidate,
)
from auto_ontology.retrieval.text_to_sql.db_probe import executor as probe_executor
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel, SQLJudgeModel


def _candidate(source: str, sql: str) -> SQLCandidate:
    return SQLCandidate(
        source=source,
        response=SQLGenerationModel(
            thought="No assumptions.",
            sql_code=sql,
            description=f"What does the {source} query return?",
            response=f"The {source} answer.",
        ),
        messages=[],
    )


def _context(connector: object | None = None) -> SelectionContext:
    return SelectionContext(
        llm=object(),
        question="How many active accounts?",
        evidence="Active means status = 'active'.",
        tables_text="table\naccounts",
        connector=connector,
        dialect="sqlite",
        target_db=None,
        relevant_tables=[],
    )


@pytest.fixture
def results(monkeypatch) -> dict:
    """Map SQL text to the DataFrame (or exception) the fake database returns."""
    by_sql: dict = {}

    def fake_execute(_connector, sql, **_kwargs):
        outcome = next(v for k, v in by_sql.items() if sql.startswith(k))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(probe_executor, "execute_chat_sql", fake_execute)
    return by_sql


@pytest.fixture
def no_judge(monkeypatch) -> None:
    def fail(*_args):
        raise AssertionError("the judge must not run")

    monkeypatch.setattr(sql_candidate_selection, "_judge_once", fail)


def test_static_issue_disqualifies_a_candidate(no_judge) -> None:
    semantic = _candidate("semantic", "SELECT NULL FROM accounts")
    schema = _candidate("schema-only", "SELECT COUNT(*) FROM accounts")

    selection = select_candidate([semantic, schema], _context())

    assert selection.selected is schema
    assert selection.stage == "static"
    assert "constant/NULL" in semantic.static_issue


def test_unparseable_candidate_is_disqualified(no_judge) -> None:
    semantic = _candidate("semantic", "SELECT COUNT(* FROM accounts")
    schema = _candidate("schema-only", "SELECT COUNT(*) FROM accounts")

    selection = select_candidate([semantic, schema], _context())

    assert selection.selected is schema
    assert semantic.static_issue.startswith("does not parse")


def test_execution_error_or_empty_result_disqualifies(results, no_judge) -> None:
    results["SELECT bad"] = RuntimeError("no such column: bad")
    results["SELECT good"] = pd.DataFrame({"n": [3]})
    semantic = _candidate("semantic", "SELECT bad FROM accounts")
    schema = _candidate("schema-only", "SELECT good FROM accounts")

    selection = select_candidate([semantic, schema], _context(connector=object()))

    assert selection.selected is schema
    assert selection.stage == "execution"
    assert "no such column" in selection.reason

    results.clear()
    results["SELECT empty"] = pd.DataFrame({"n": []})
    results["SELECT nulls"] = pd.DataFrame({"n": [None, None]})
    results["SELECT good"] = pd.DataFrame({"n": [3]})
    for loser_sql in ("SELECT empty FROM accounts", "SELECT nulls FROM accounts"):
        loser = _candidate("semantic", loser_sql)
        winner = _candidate("schema-only", "SELECT good FROM accounts")
        assert select_candidate([loser, winner], _context(object())).selected is winner


def test_identical_results_keep_the_semantic_candidate(results, no_judge) -> None:
    results["SELECT a"] = pd.DataFrame({"x": [1, 2], "y": ["p", "q"]})
    results["SELECT b"] = pd.DataFrame({"other": [2.0, 1.0], "z": ["q", "p"]})
    semantic = _candidate("semantic", "SELECT a FROM t")
    schema = _candidate("schema-only", "SELECT b FROM t")

    selection = select_candidate([schema, semantic], _context(object()))

    assert selection.selected is semantic
    assert selection.stage == "agreement"


def test_judge_decides_only_when_both_orders_agree(results, monkeypatch) -> None:
    results["SELECT a"] = pd.DataFrame({"n": [3]})
    results["SELECT b"] = pd.DataFrame({"n": [4]})
    prompts: list[str] = []

    def judge_prefers_b_query(_llm, messages, schema):
        assert schema is SQLJudgeModel
        prompt = messages[-1].content
        prompts.append(prompt)
        b_is_first = prompt.index("SELECT b") < prompt.index("SELECT a")
        return SQLJudgeModel(
            thought="B filters correctly.", winner="A" if b_is_first else "B"
        )

    monkeypatch.setattr(
        sql_candidate_selection,
        "safe_invoke_with_structured_output",
        judge_prefers_b_query,
    )
    semantic = _candidate("semantic", "SELECT a FROM t")
    schema = _candidate("schema-only", "SELECT b FROM t")

    selection = select_candidate([semantic, schema], _context(object()))

    assert selection.selected is schema
    assert selection.stage == "judge"
    assert schema.judge_votes == ["schema-only", "schema-only"]
    assert len(prompts) == 2
    assert all("1 row(s), showing 1" in p for p in prompts)
    assert all("Active means status = 'active'." in p for p in prompts)


def test_split_judge_falls_back_to_rerank(results, monkeypatch) -> None:
    results["SELECT a"] = pd.DataFrame({"n": [3]})
    results["SELECT b"] = pd.DataFrame({"n": [4]})
    monkeypatch.setattr(
        sql_candidate_selection,
        "safe_invoke_with_structured_output",
        lambda *_args: SQLJudgeModel(thought="Position bias.", winner="A"),
    )
    monkeypatch.setattr(
        sql_candidate_selection,
        "rerank_hits",
        lambda _query, hits, **_kwargs: [
            {"_rerank_score": 0.8, **hits[1]},
            {"_rerank_score": 0.1, **hits[0]},
        ],
    )
    semantic = _candidate("semantic", "SELECT a FROM t")
    schema = _candidate("schema-only", "SELECT b FROM t")

    selection = select_candidate([semantic, schema], _context(object()))

    assert selection.selected is schema
    assert selection.stage == "rerank"
    assert schema.rerank_score == 0.8


def test_result_key_ignores_order_column_names_and_numeric_type() -> None:
    assert result_key([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]) == result_key(
        [{"n": 2.0, "m": "y"}, {"n": 1.0, "m": "x"}]
    )
    assert result_key([{"a": 1}]) != result_key([{"a": 2}])


def test_result_preview_is_bounded() -> None:
    rows = [{"id": i, "note": "word " * 40} for i in range(30)]
    preview = format_result_preview(
        {"ok": True, "rows": rows, "row_count": 30, "truncated": False}
    )

    assert preview.startswith("30 row(s), showing 10:")
    assert len(preview.splitlines()) == 12
    assert all(
        len(cell) <= 80 for line in preview.splitlines() for cell in line.split(" | ")
    )
    assert format_result_preview({"ok": False, "error": "boom"}) == "Error: boom"
    assert format_result_preview(None) == "(not executed)"
