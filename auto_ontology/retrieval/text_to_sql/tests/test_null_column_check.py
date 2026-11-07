# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from auto_ontology.retrieval.text_to_sql.agents.null_column_check import (
    NullColumnCheckAgent,
    all_null_columns,
)


def _state(rows, question="Total weight per depot", **path_state):
    return {
        "initial_question": question,
        "path_state": {"sql_response_from_db": [json.dumps(rows)], **path_state},
    }


def test_all_null_columns_finds_only_columns_null_in_every_row() -> None:
    rows = [
        {"depot": "North", "carrier": None, "weight": 10},
        {"depot": "South", "carrier": None, "weight": None},
    ]

    assert all_null_columns([json.dumps(rows)]) == ["carrier"]
    assert all_null_columns(["[]"]) == []


def test_an_all_null_column_routes_once_to_reconstruction() -> None:
    agent = NullColumnCheckAgent()
    rows = [{"depot": "North", "carrier": None}, {"depot": "South", "carrier": None}]

    first = agent.execute(_state(rows))
    assert first["decision"] == "invalid_sql"
    assert "`carrier`" in first["path_state"]["error"]

    second = agent.execute({**_state(rows), "path_state": first["path_state"]})
    assert second["decision"] == "valid_sql"


def test_a_result_with_values_in_every_column_passes() -> None:
    rows = [
        {"depot": "North", "carrier": "Fleet A"},
        {"depot": "South", "carrier": None},
    ]

    assert NullColumnCheckAgent().execute(_state(rows))["decision"] == "valid_sql"


def test_a_question_asking_for_nulls_is_left_alone() -> None:
    rows = [{"depot": None, "total": 5}]
    question = "Add a grand total row in the format (null, total)"

    assert (
        NullColumnCheckAgent().execute(_state(rows, question=question))["decision"]
        == "valid_sql"
    )
