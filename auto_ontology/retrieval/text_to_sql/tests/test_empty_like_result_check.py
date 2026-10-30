# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A result whose every value is NULL is sent back once, and never costs the answer."""

from __future__ import annotations

import json
from typing import Any

import pytest

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    ALL_NULL_RESULT_ERROR,
    EmptyLikeResultCheckAgent,
    _build_null_jsonb_error,
    _is_all_null_result,
    _is_empty_db_result,
    restore_all_null_original_node,
)
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    route_sql_execution,
    route_sql_validation_or_original,
)

_SQL = 'SELECT "rate" FROM t ORDER BY "rate" ASC LIMIT 3'


def _records(*rows: dict) -> list[str]:
    """The shape ``sql_execution`` stores: one JSON records string."""
    return [json.dumps(list(rows))]


def _state(db_result: Any, sql: str = _SQL, **path_overrides: Any) -> dict:
    path_state = {
        "sql_code": sql,
        "sql_response_from_db": db_result,
        "sql_columns": ["t.rate"],
        **path_overrides,
    }
    return {"path_state": path_state, "shorten_answer": True, "messages": []}


@pytest.mark.parametrize(
    "db_result",
    [
        [(None,), (None,), (None,)],
        [{"rate": None}, {"rate": None}],
        [(None,)],
        _records({"rate": None}, {"rate": None}, {"rate": None}),
        _records({"total": None}),
    ],
)
def test_every_value_null_is_all_null(db_result: Any) -> None:
    assert _is_all_null_result(db_result)


@pytest.mark.parametrize(
    "db_result",
    [
        [(None, "Los Angeles")],
        [(None,), (0.04,)],
        [],
        _records({"rate": None, "city": "Los Angeles"}),
        _records(),
        None,
    ],
)
def test_any_value_or_no_rows_is_not_all_null(db_result: Any) -> None:
    assert not _is_all_null_result(db_result)


def test_empty_result_detection_is_unchanged() -> None:
    assert _is_empty_db_result([])
    assert _is_empty_db_result(_records())
    assert not _is_empty_db_result([(None,)])


@pytest.mark.parametrize(
    "db_result", [[(None,), (None,), (None,)], [{"rate": None}, {"rate": None}]]
)
def test_all_null_result_goes_back_for_reconstruction(db_result: Any) -> None:
    result = EmptyLikeResultCheckAgent().execute(_state(db_result))

    assert result["decision"] == "invalid_sql"
    path_state = result["path_state"]
    assert path_state["error"] == ALL_NULL_RESULT_ERROR
    assert path_state["all_null_retry_attempted"] is True
    assert path_state["all_null_original"]["sql_code"] == _SQL


def test_regular_runs_get_the_check_too() -> None:
    state = _state([(None,)])
    state["shorten_answer"] = False

    result = EmptyLikeResultCheckAgent().execute(state)

    assert result["decision"] == "invalid_sql"
    assert result["path_state"]["error"] == ALL_NULL_RESULT_ERROR


def test_a_non_null_value_passes() -> None:
    result = EmptyLikeResultCheckAgent().execute(_state([(None,), (0.04,)]))

    assert result["decision"] == "valid_sql"
    assert "error" not in result["path_state"]


def test_the_second_pass_does_not_loop() -> None:
    state = _state([(None,)], all_null_retry_attempted=True)

    result = EmptyLikeResultCheckAgent().execute(state)

    assert result["decision"] == "valid_sql"


def test_a_retry_that_is_still_all_null_restores_the_original() -> None:
    first = EmptyLikeResultCheckAgent().execute(_state([(None,), (None,)]))
    retried = {
        **first["path_state"],
        "sql_code": "SELECT NULL AS rate",
        "sql_response_from_db": [(None,)],
    }

    result = EmptyLikeResultCheckAgent().execute(
        {"path_state": retried, "shorten_answer": True}
    )

    assert result["decision"] == "valid_sql"
    path_state = result["path_state"]
    assert path_state["sql_code"] == _SQL
    assert path_state["sql_response_from_db"] == [(None,), (None,)]
    assert "all_null_original" not in path_state
    assert "error" not in path_state


def test_no_sql_from_reconstruction_submits_the_original() -> None:
    first = EmptyLikeResultCheckAgent().execute(_state([(None,), (None,)]))
    gave_up = {
        "decision": "unconstructable",
        "path_state": {
            **first["path_state"],
            "sql_code": "",
            "unconstructable_explanation": "SQL can't be constructed.",
        },
    }

    assert route_sql_validation_or_original(gave_up) == "restore_original"
    assert route_sql_execution(gave_up) == "restore_original"
    restored = restore_all_null_original_node(gave_up)

    assert restored["decision"] == "valid_sql"
    path_state = restored["path_state"]
    assert path_state["sql_code"] == _SQL
    assert path_state["sql_response_from_db"] == [(None,), (None,)]
    assert "unconstructable_explanation" not in path_state


def test_giving_up_without_a_retry_still_reports_unconstructable() -> None:
    state = {"decision": "unconstructable", "path_state": {}}

    assert route_sql_validation_or_original(state) == "unconstructable"


def test_jsonb_all_null_keeps_its_own_message() -> None:
    sql = "SELECT (data->>'score')::numeric AS score FROM t"
    state = _state(_records({"score": None}, {"score": None}), sql=sql)
    state["shorten_answer"] = False

    first = EmptyLikeResultCheckAgent().execute(state)

    assert first["decision"] == "invalid_sql"
    assert first["path_state"]["error"] == _build_null_jsonb_error(["score"])

    second = EmptyLikeResultCheckAgent().execute(
        {"path_state": first["path_state"], "shorten_answer": False}
    )
    assert second["decision"] == "valid_sql"
