# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sqlglot

from auto_ontology.retrieval.text_to_sql.db_probe.executor import (
    ProbeExecutor,
    _ensure_limit,
    is_read_only_select,
)


@pytest.mark.parametrize(
    "literal",
    [
        "Journey into Nyx",
        "Hero''s Path",
        "drop",
        "delete",
        "call",
        "update",
        "limit",
    ],
)
def test_keywords_and_apostrophes_inside_literals_are_accepted(literal: str) -> None:
    assert is_read_only_select(f"SELECT name FROM sets WHERE name = '{literal}'")


def test_quoted_identifiers_and_comments_with_forbidden_words_are_accepted() -> None:
    sql = (
        'SELECT "delete", "limit" FROM "drop" WHERE "update" = 1 /* call insert into */'
    )

    assert is_read_only_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO target VALUES (1)",
        "UPDATE target SET value = 1",
        "DELETE FROM target",
        "CREATE TABLE target (id INT)",
        "DROP TABLE target",
        "ALTER TABLE target ADD COLUMN value INT",
        "TRUNCATE TABLE target",
        "CALL refresh_target()",
    ],
)
def test_real_mutating_and_command_statements_are_rejected(sql: str) -> None:
    assert not is_read_only_select(sql)


def test_select_into_is_rejected() -> None:
    assert not is_read_only_select("SELECT * INTO copied_rows FROM source_rows")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT nextval('sequence_name')",
        "SELECT setval('sequence_name', 1)",
        "SELECT pg_advisory_lock(1)",
    ],
)
def test_side_effecting_select_functions_are_rejected(sql: str) -> None:
    assert not is_read_only_select(sql, "postgres")


@pytest.mark.parametrize(
    "sql",
    [
        "WITH changed AS (DELETE FROM target RETURNING *) SELECT * FROM changed",
        (
            "WITH changed AS (UPDATE target SET value = 1 RETURNING *) "
            "SELECT * FROM changed"
        ),
        (
            "WITH changed AS (INSERT INTO target VALUES (1) RETURNING *) "
            "SELECT * FROM changed"
        ),
    ],
)
def test_mutating_statements_inside_ctes_are_rejected(sql: str) -> None:
    assert not is_read_only_select(sql, "postgres")


def test_multiple_statements_are_rejected() -> None:
    assert not is_read_only_select("SELECT 1; SELECT 2")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT FROM",
        "SELECT * FRM target",
        "SELECT (",
        "SELEC 1",
    ],
)
def test_malformed_sql_is_rejected(sql: str) -> None:
    assert not is_read_only_select(sql)


def test_literal_containing_limit_still_receives_top_level_limit() -> None:
    bounded = _ensure_limit("SELECT 'limit' AS value", 17)
    parsed = sqlglot.parse_one(bounded)

    assert parsed.args.get("limit") is not None
    assert parsed.args["limit"].expression.this == "17"
    assert "'limit'" in bounded


def test_comment_containing_limit_still_receives_top_level_limit() -> None:
    bounded = _ensure_limit("SELECT 1 /* limit is discussed here */", 17)
    parsed = sqlglot.parse_one(bounded)

    assert parsed.args.get("limit") is not None
    assert parsed.args["limit"].expression.this == "17"


def test_existing_top_level_limit_is_preserved() -> None:
    sql = 'SELECT "limit" FROM "drop" LIMIT 3;'

    assert _ensure_limit(sql, 17) == 'SELECT "limit" FROM "drop" LIMIT 3'


def test_inner_limit_does_not_satisfy_top_level_limit() -> None:
    bounded = _ensure_limit(
        "SELECT * FROM (SELECT id FROM target LIMIT 1) AS nested",
        17,
    )
    parsed = sqlglot.parse_one(bounded)

    assert parsed.args.get("limit") is not None
    assert parsed.args["limit"].expression.this == "17"


def test_read_only_cte_and_union_are_accepted() -> None:
    assert is_read_only_select(
        "WITH values_cte AS (SELECT 1 AS value) "
        "SELECT value FROM values_cte UNION SELECT 2"
    )


def test_probe_rejection_reports_specific_reason(caplog) -> None:
    executor = ProbeExecutor(connector=SimpleNamespace(dialect="postgres"))

    with caplog.at_level("WARNING"):
        result = executor.run("SELECT * INTO copied_rows FROM source_rows")

    assert result["error"] == "rejected: SELECT INTO is not read-only"
    assert "SELECT INTO is not read-only" in caplog.text
    assert executor.calls == 0
