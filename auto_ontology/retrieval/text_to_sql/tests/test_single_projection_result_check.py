# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_ontology.retrieval.text_to_sql.agents.single_projection_result_check import (
    SingleProjectionResultCheckAgent,
)
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    create_graph,
    route_single_projection_result_check,
)


def _generation(sql: str) -> SQLGenerationModel:
    return SQLGenerationModel(
        plan="Return the requested list.",
        thought="No assumptions.",
        sql_code=sql,
        response="Returns the requested list.",
    )


def _state(
    sql: str,
    rows: str,
    *,
    path_updates: dict | None = None,
) -> dict:
    path_state = {
        "sql_code": sql,
        "sql_generation_result": _generation(sql),
        "sql_response_from_db": [rows],
        "relevant_tables": [{"database_name": "db"}],
    }
    path_state.update(path_updates or {})
    return {
        "connectors": [SimpleNamespace(dialect="sqlite", database_name="db")],
        "path_state": path_state,
    }


def test_repeated_single_projection_adds_distinct_and_reexecutes() -> None:
    result = SingleProjectionResultCheckAgent().execute(
        _state(
            "SELECT city FROM customers ORDER BY city",
            '[{"city":"Paris"},{"city":"Paris"},{"city":"Rome"}]',
        )
    )

    assert result["decision"] == "retry_sql"
    path_state = result["path_state"]
    assert "SELECT DISTINCT city" in path_state["sql_code"]
    assert "ORDER BY city" in path_state["sql_code"]
    assert path_state["sql_generation_result"].sql_code == path_state["sql_code"]
    assert path_state["sql_response_from_db"] is None
    assert route_single_projection_result_check(result) == "retry_sql"


def test_unique_single_projection_is_unchanged() -> None:
    sql = "SELECT city FROM customers"
    result = SingleProjectionResultCheckAgent().execute(
        _state(sql, '[{"city":"Paris"},{"city":"Rome"}]')
    )

    assert result["decision"] == "valid_sql"
    assert result["path_state"]["sql_code"] == sql


def test_multiple_projected_fields_are_unchanged() -> None:
    sql = "SELECT city, country FROM customers"
    result = SingleProjectionResultCheckAgent().execute(
        _state(
            sql,
            '[{"city":"Paris","country":"France"},{"city":"Paris","country":"France"}]',
        )
    )

    assert result["decision"] == "valid_sql"
    assert result["path_state"]["sql_code"] == sql


def test_existing_distinct_is_not_rewritten() -> None:
    sql = "SELECT DISTINCT city FROM customers"
    result = SingleProjectionResultCheckAgent().execute(
        _state(sql, '[{"city":"Paris"},{"city":"Paris"}]')
    )

    assert result["decision"] == "valid_sql"
    assert result["path_state"]["sql_code"] == sql


def test_unfiltered_single_column_null_routes_to_reconstruction() -> None:
    result = SingleProjectionResultCheckAgent().execute(
        _state(
            "SELECT city AS place FROM customers ORDER BY city",
            '[{"place":null},{"place":"Paris"}]',
        )
    )

    assert result["decision"] == "invalid_sql"
    path_state = result["path_state"]
    assert path_state["single_projection_null_retry_attempted"] is True
    assert path_state["error_known_fixable"] is True
    assert "WHERE city IS NOT NULL" in path_state["error"]
    assert route_single_projection_result_check(result) == "invalid_sql"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT city FROM customers WHERE country = 'France'",
        "SELECT c.city FROM customers AS c JOIN countries AS x ON x.id = c.country_id",
        "SELECT city FROM customers WHERE city IN ('Paris', 'Rome')",
        "SELECT city FROM customers GROUP BY city HAVING COUNT(*) > 1",
    ],
)
def test_null_with_filter_or_join_is_unchanged(sql: str) -> None:
    result = SingleProjectionResultCheckAgent().execute(
        _state(sql, '[{"city":null},{"city":"Paris"}]')
    )

    assert result["decision"] == "valid_sql"
    assert result["path_state"]["sql_code"] == sql


def test_null_repair_is_attempted_only_once() -> None:
    result = SingleProjectionResultCheckAgent().execute(
        _state(
            "SELECT city FROM customers",
            '[{"city":null},{"city":"Paris"}]',
            path_updates={"single_projection_null_retry_attempted": True},
        )
    )

    assert result["decision"] == "valid_sql"


def test_graph_runs_projection_check_after_empty_result_repair() -> None:
    edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in create_graph().compile().get_graph().edges
    }

    assert ("check_value_repair", "check_single_projection_result", True) in edges
    assert ("check_single_projection_result", "execute_sql_query", True) in edges
    assert ("check_single_projection_result", "reconstruct_sql", True) in edges
    assert ("check_single_projection_result", "format_and_respond", True) in edges
