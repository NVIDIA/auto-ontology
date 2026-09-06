# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest

from gsf.retrieval.text_to_sql.agents import sql_reconstruction
from gsf.retrieval.text_to_sql.agents.sql_reconstruction import (
    SQLReconstructionAgent,
)
from gsf.retrieval.text_to_sql.state import AgentState
from gsf.retrieval.text_to_sql.text_to_sql_graph import route_sql_validation


def test_reconstruction_visits_activate_the_existing_graph_budget() -> None:
    path_state: dict = {}

    for expected in range(1, 7):
        assert sql_reconstruction._increment_reconstruction_count(path_state) == expected

    assert path_state["reconstruction_count"] == 6
    assert (
        route_sql_validation({"decision": "valid_sql", "path_state": path_state})
        == "skip_intent_validation"
    )


def test_reconstruction_counter_preserves_existing_attempts() -> None:
    path_state = {"reconstruction_count": 3}

    assert sql_reconstruction._increment_reconstruction_count(path_state) == 4
    assert path_state["reconstruction_count"] == 4


@pytest.mark.parametrize(
    ("path_state", "expected_database"),
    [
        ({"retrieval_database": "retrieved-db"}, "retrieved-db"),
        (
            {
                "target_db": "target-db",
                "retrieval_database": "retrieved-db",
            },
            "target-db",
        ),
    ],
)
def test_table_discovery_uses_retrieval_database(
    monkeypatch: pytest.MonkeyPatch,
    path_state: dict,
    expected_database: str,
) -> None:
    database_names: list[str | None] = []

    def fake_get_relevant_tables(
        _retriever: object,
        _query_text: str,
        *,
        k: int,
        database_name: str | None,
    ) -> list[dict]:
        assert k == 3
        database_names.append(database_name)
        return []

    monkeypatch.setattr(
        sql_reconstruction,
        "get_relevant_tables",
        fake_get_relevant_tables,
    )
    state = cast(
        AgentState,
        {
            "data_retriever": object(),
            "path_state": path_state,
        },
    )

    result = SQLReconstructionAgent()._discover_tables(
        state,
        ["orders"],
        [],
    )

    assert result == []
    assert database_names == [expected_database]
