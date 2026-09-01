# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import cast

from gsf.retrieval.text_to_sql.agents import candidates_preparation
from gsf.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
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
    # k_per_query = max(1, 10 // len(search_queries)) — see
    # candidates_preparation.py's _retrieve_additional_tables.
    assert all(k == 3 for _, k, _ in seen)
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
