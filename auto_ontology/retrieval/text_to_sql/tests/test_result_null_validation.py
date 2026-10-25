# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import sqlglot

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    EmptyLikeResultCheckAgent,
    add_not_null_filters_for_result,
)
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    route_post_execution_result_check,
)


def _generation(sql: str) -> SQLGenerationModel:
    return SQLGenerationModel(
        plan="Read the requested fields.",
        thought="No assumptions.",
        sql_code=sql,
        response="Returns the requested fields.",
    )


def test_adds_not_null_filters_for_nullable_projected_fields() -> None:
    rewritten, columns = add_not_null_filters_for_result(
        "SELECT customer_id, city AS customer_city FROM customers",
        [
            '[{"customer_id":1,"customer_city":null},{"customer_id":2,"customer_city":"LA"}]'
        ],
        "sqlite",
    )

    assert columns == ["customer_city"]
    assert rewritten is not None
    normalized = sqlglot.parse_one(rewritten, read="sqlite")
    assert normalized.args.get("where") is not None
    assert "customer_city" not in normalized.args["where"].sql()
    assert "city" in normalized.args["where"].sql()
    assert "IS NOT NULL" in rewritten


def test_does_not_move_aggregate_projection_into_where() -> None:
    rewritten, columns = add_not_null_filters_for_result(
        "SELECT SUM(amount) AS total FROM orders",
        ['[{"total":null}]'],
        "sqlite",
    )

    assert rewritten is None
    assert columns == ["total"]


def test_non_null_results_are_unchanged() -> None:
    rewritten, columns = add_not_null_filters_for_result(
        "SELECT name FROM customers",
        ['[{"name":"Ada"}]'],
        "sqlite",
    )

    assert rewritten is None
    assert columns == []


def test_agent_reexecutes_deterministically_filtered_sql() -> None:
    sql = "SELECT customer_id, city FROM customers"
    result = EmptyLikeResultCheckAgent().execute(
        {
            "connectors": [
                SimpleNamespace(dialect="sqlite", database_name="customers")
            ],
            "path_state": {
                "sql_code": sql,
                "sql_generation_result": _generation(sql),
                "sql_response_from_db": ['[{"customer_id":1,"city":null}]'],
                "relevant_tables": [{"database_name": "customers"}],
            },
        }
    )

    assert result["decision"] == "retry_sql"
    path_state = result["path_state"]
    assert path_state["sql_response_from_db"] is None
    assert "NULL" in path_state["sql_code"]
    assert path_state["sql_generation_result"].sql_code == path_state["sql_code"]
    assert route_post_execution_result_check(result) == "retry_sql"


def test_unfilterable_null_result_routes_to_reconstruction() -> None:
    sql = "SELECT SUM(amount) AS total FROM orders"
    result = EmptyLikeResultCheckAgent().execute(
        {
            "connectors": [SimpleNamespace(dialect="sqlite", database_name="orders")],
            "path_state": {
                "sql_code": sql,
                "sql_generation_result": _generation(sql),
                "sql_response_from_db": ['[{"total":null}]'],
                "relevant_tables": [{"database_name": "orders"}],
            },
        }
    )

    assert result["decision"] == "invalid_sql"
    assert "IS NOT NULL" in result["path_state"]["error"]
    assert route_post_execution_result_check(result) == "invalid_sql"


def test_same_filtered_sql_is_not_reexecuted_indefinitely() -> None:
    sql = "SELECT city FROM customers WHERE NOT city IS NULL"
    result = EmptyLikeResultCheckAgent().execute(
        {
            "path_state": {
                "sql_code": sql,
                "sql_response_from_db": ['[{"city":null}]'],
                "null_result_filter_retry_sql": sql,
            },
        }
    )

    assert result["decision"] == "invalid_sql"
    assert "returned NULL" in result["path_state"]["error"]
