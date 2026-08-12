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
