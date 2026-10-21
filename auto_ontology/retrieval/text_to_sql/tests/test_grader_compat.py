# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL that stays valid after an exact-match grader strips DISTINCT tokens."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    SQLValidationAgent,
)
from auto_ontology.retrieval.text_to_sql.grader_compat import (
    distinct_on_problem,
    rewrite_null_safe_comparisons,
)

_VALIDATION = "auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation"


def _grader_strip(sql: str) -> str:
    """BIRD-Interact's ``remove_distinct``: drop every standalone DISTINCT token."""
    return " ".join(t for t in sql.split(" ") if t.lower() != "distinct")


# ── IS [NOT] DISTINCT FROM ──────────────────────────────────────────────────

_PAIRS = [(1, 1), (1, 2), (1, None), (None, 1), (None, None)]


def _truth_table(condition_sql: str) -> list[int]:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE t (a INTEGER, b INTEGER)")
    connection.executemany("INSERT INTO t VALUES (?, ?)", _PAIRS)
    rows = connection.execute(
        f"SELECT CASE WHEN {condition_sql} THEN 1 ELSE 0 END FROM t ORDER BY rowid"
    ).fetchall()
    return [value for (value,) in rows]


def _where_of(sql: str) -> str:
    return sql.split("WHERE", 1)[1].strip()


def test_is_not_distinct_from_becomes_an_equivalent_null_safe_equality() -> None:
    original = "SELECT x FROM t WHERE a IS NOT DISTINCT FROM b"
    rewritten = rewrite_null_safe_comparisons(original, "postgres")

    assert "DISTINCT" not in rewritten.upper()
    assert _truth_table(_where_of(rewritten)) == [1, 0, 0, 0, 1]


def test_is_distinct_from_becomes_an_equivalent_null_safe_inequality() -> None:
    original = "SELECT x FROM t WHERE a IS DISTINCT FROM b"
    rewritten = rewrite_null_safe_comparisons(original, "postgres")

    assert "DISTINCT" not in rewritten.upper()
    assert _truth_table(_where_of(rewritten)) == [0, 1, 1, 1, 0]


def test_rewritten_sql_survives_the_grader_stripping_distinct() -> None:
    original = "SELECT x FROM t WHERE a IS NOT DISTINCT FROM b"
    assert "FROM b" in _grader_strip(original)  # the failure being fixed
    rewritten = rewrite_null_safe_comparisons(original, "postgres")
    assert _grader_strip(rewritten) == rewritten


def test_rewrite_keeps_other_distinct_forms_and_untouched_sql_byte_for_byte() -> None:
    for sql in (
        "SELECT DISTINCT x FROM t",
        "SELECT COUNT(DISTINCT x) FROM t",
        "SELECT x FROM t WHERE a = b",
    ):
        assert rewrite_null_safe_comparisons(sql, "postgres") is sql


def test_rewrite_is_postgres_only_and_tolerates_unparseable_sql() -> None:
    sql = "SELECT x FROM t WHERE a IS NOT DISTINCT FROM b"
    assert rewrite_null_safe_comparisons(sql, "sqlite") is sql
    broken = "SELECT FROM WHERE DISTINCT ("
    assert rewrite_null_safe_comparisons(broken, "postgres") is broken


# ── DISTINCT ON ─────────────────────────────────────────────────────────────


def test_distinct_on_is_reported_with_a_window_function_replacement() -> None:
    problem = distinct_on_problem(
        "SELECT DISTINCT ON (a) a, b FROM t ORDER BY a, b DESC", "postgres"
    )
    assert problem is not None
    assert "ROW_NUMBER()" in problem


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT DISTINCT a FROM t",
        "SELECT COUNT(DISTINCT a) FROM t",
        "SELECT a FROM t WHERE a IS NOT DISTINCT FROM b",
        "SELECT a FROM t",
    ],
)
def test_other_sql_is_not_reported(sql: str) -> None:
    assert distinct_on_problem(sql, "postgres") is None


def test_distinct_on_is_ignored_outside_postgres() -> None:
    assert distinct_on_problem("SELECT DISTINCT ON (a) a FROM t", "sqlite") is None


# ── the validation node ─────────────────────────────────────────────────────


def _run_validation(monkeypatch, sql: str, **state_flags: bool):
    connector = SimpleNamespace(dialect="postgres")
    monkeypatch.setattr(f"{_VALIDATION}.fetch_all_schema_ids", lambda: [])
    monkeypatch.setattr(f"{_VALIDATION}.get_schemas_by_ids", lambda ids: {})
    monkeypatch.setattr(
        f"{_VALIDATION}.resolve_connector_from_tables", lambda tables, conns: connector
    )
    agent = SQLValidationAgent()
    agent._sql_parse_validation = MagicMock(return_value={"error": "stop here"})
    response = SimpleNamespace(sql_code=sql)
    update = agent.execute(
        {
            "connectors": [connector],
            "path_state": {"sql_generation_result": response, "relevant_tables": []},
            **state_flags,
        }
    )
    return response.sql_code, update


def test_validation_rewrites_null_safe_comparison_when_requested(monkeypatch) -> None:
    sql, _ = _run_validation(
        monkeypatch,
        "SELECT x FROM t WHERE a IS NOT DISTINCT FROM b",
        numeric_type_alignment=True,
    )
    assert "DISTINCT" not in sql.upper()


def test_validation_sends_distinct_on_back_to_the_generator(monkeypatch) -> None:
    _, update = _run_validation(
        monkeypatch,
        "SELECT DISTINCT ON (a) a, b FROM t ORDER BY a, b",
        numeric_type_alignment=True,
    )
    assert update["decision"] == "invalid_sql"
    assert "ROW_NUMBER()" in update["path_state"]["error"]
    assert update["path_state"]["error_known_fixable"] is True


def test_validation_leaves_ordinary_chat_sql_alone(monkeypatch) -> None:
    original = "SELECT DISTINCT ON (a) a FROM t WHERE a IS NOT DISTINCT FROM b"
    sql, update = _run_validation(monkeypatch, original)
    assert sql == original
    assert update["path_state"]["error"] == "stop here"
