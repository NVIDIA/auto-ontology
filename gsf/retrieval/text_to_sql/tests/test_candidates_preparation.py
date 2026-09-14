# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import cast

from gsf.retrieval.text_to_sql.agents import candidates_preparation
from gsf.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from gsf.retrieval.text_to_sql.prompts import SQL_GEN_MAX_ENTITIES
from gsf.retrieval.text_to_sql.state import AgentState


def test_retrieve_additional_tables_searches_question_and_entities(
    monkeypatch,
) -> None:
    seen: list[tuple[str, int, str | None]] = []

    def fake_get_relevant_tables(retriever, query, k=1, database_name=None, **kwargs):
        seen.append((query, k, database_name))
        return [{"id": query, "name": query, "database_name": database_name}]

    monkeypatch.setattr(
        candidates_preparation, "get_relevant_tables", fake_get_relevant_tables
    )
    monkeypatch.setattr(
        candidates_preparation,
        "dedupe_merge_relevant_tables",
        lambda tables: tables,
    )

    out = CandidatePreparationAgent()._retrieve_additional_tables(
        object(), "q", ["e1", "e2"], "db"
    )

    assert [q for q, _, _ in seen] == ["q", "e1", "e2"]
    expected_k = max(1, SQL_GEN_MAX_ENTITIES // len(seen))
    assert all(k == expected_k for _, k, _ in seen)
    assert all(db == "db" for _, _, db in seen)
    assert len(out) == 3


def test_additional_table_retrieve_starts_before_anchor_returns(monkeypatch) -> None:
    retrieve_started = threading.Event()
    release_retrieve = threading.Event()
    retrieve_was_running = []

    def fake_retrieve(self, retriever, question, entities, target_db):
        retrieve_started.set()
        assert release_retrieve.wait(timeout=2)
        return [{"id": "extra", "name": "extra", "database_name": "db"}]

    def fake_anchor(self, state, question, contexts):
        retrieve_was_running.append(retrieve_started.wait(timeout=2))
        release_retrieve.set()
        return "a1", "anchor reason"

    monkeypatch.setattr(
        CandidatePreparationAgent, "_retrieve_additional_tables", fake_retrieve
    )
    monkeypatch.setattr(CandidatePreparationAgent, "_identify_anchor", fake_anchor)
    monkeypatch.setattr(
        CandidatePreparationAgent,
        "_filter_tables_by_relevance",
        lambda self, state, question, tables, custom_analyses=None, attribute_join_paths=None: (
            tables,
            "",
        ),
    )
    monkeypatch.setattr(
        candidates_preparation, "fetch_custom_analyses_with_sql", lambda ids: []
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_attr_column_contexts",
        lambda ids, database_name=None: {
            "a1": {
                "attr_name": "Revenue",
                "col_name": "rev",
                "col_id": "c1",
                "table_id": "t1",
                "table_name": "sales",
                "schema_name": "main",
                "database_name": "db",
            }
        },
    )
    monkeypatch.setattr(candidates_preparation, "fetch_term_synonyms", lambda ids: {})
    monkeypatch.setattr(
        candidates_preparation,
        "get_relevant_tables_from_candidates",
        lambda candidates: [],
    )
    monkeypatch.setattr(candidates_preparation, "fetch_tables_by_ids", lambda ids: [])
    monkeypatch.setattr(candidates_preparation, "find_join_path", lambda *a, **k: [])
    monkeypatch.setattr(
        candidates_preparation,
        "find_connected_junction_tables",
        lambda ids: ([], []),
    )

    state = cast(
        AgentState,
        {
            "initial_question": "question",
            "llm": object(),
            "data_retriever": object(),
            "path_state": {
                "entities": ["revenue"],
                "target_db": "db",
                "retrieved_column_attributes": [{"id": "a1"}],
            },
        },
    )

    result = CandidatePreparationAgent().execute(state)

    assert retrieve_was_running == [True]
    names = [t.get("name") for t in result["path_state"]["relevant_tables"]]
    assert "extra" in names


def test_connected_junctions_are_forced_in_with_their_join_hops(monkeypatch) -> None:
    base = {
        "id": "base",
        "name": "orders",
        "schema_name": "public",
        "database_name": "db",
        "columns": [{"name": "id", "sample_values": [1]}],
    }
    junction = {
        "id": "junction",
        "name": "order_tag",
        "schema_name": "public",
        "database_name": "db",
        "columns": [],
    }
    hop = {
        "source_schema": "public",
        "source_table": "order_tag",
        "source_column": "order_id",
        "target_schema": "public",
        "target_table": "orders",
        "target_column": "id",
    }

    monkeypatch.setattr(
        CandidatePreparationAgent,
        "_retrieve_additional_tables",
        lambda self, retriever, question, entities, target_db: [],
    )
    monkeypatch.setattr(
        CandidatePreparationAgent,
        "_filter_tables_by_relevance",
        lambda self, state, question, tables, custom_analyses=None, attribute_join_paths=None: (
            tables,
            "",
        ),
    )
    monkeypatch.setattr(
        candidates_preparation, "fetch_custom_analyses_with_sql", lambda ids: []
    )
    monkeypatch.setattr(
        candidates_preparation, "fetch_tables_from_custom_analyses", lambda ids: []
    )
    monkeypatch.setattr(
        candidates_preparation,
        "get_relevant_tables_from_candidates",
        lambda candidates: [base],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_tables_by_ids",
        lambda ids: [junction] if ids == ["junction"] else [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "find_connected_junction_tables",
        lambda ids: ([{"id": "junction", "name": "order_tag"}], [[hop]]),
    )

    state = cast(
        AgentState,
        {
            "initial_question": "orders and their tags",
            "data_retriever": object(),
            "path_state": {
                "target_db": "db",
                "retrieved_custom_analyses": [{"id": "analysis"}],
            },
        },
    )

    result = CandidatePreparationAgent().execute(state)["path_state"]

    assert [table["id"] for table in result["relevant_tables"]] == [
        "base",
        "junction",
    ]
    assert {"path": [hop]} in result["attribute_join_paths"]
