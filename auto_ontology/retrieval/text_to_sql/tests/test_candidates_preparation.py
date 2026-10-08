# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import cast

import pytest

from auto_ontology.retrieval.text_to_sql.agents import candidates_preparation
from auto_ontology.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from auto_ontology.retrieval.text_to_sql.models import (
    AnchorColumnModel,
    TableRelevanceModel,
    TableRemovalModel,
)
from auto_ontology.retrieval.text_to_sql.prompts import SQL_GEN_MAX_ENTITIES
from auto_ontology.retrieval.text_to_sql.state import AgentState


def test_anchor_maps_evidence_to_schema_table_column(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeLlm:
        def bind(self, **_kwargs):
            return self

    def fake_invoke(_llm, messages, _model):
        captured["prompt"] = messages[-1].content
        return AnchorColumnModel(anchor_id="frpm", reasoning="Evidence matches")

    monkeypatch.setattr(
        candidates_preparation, "invoke_with_structured_output", fake_invoke
    )
    state = cast(
        AgentState,
        {
            "llm": FakeLlm(),
            "evidence": "eligible rate = FRPM / Enrollment",
        },
    )
    contexts = {
        "frpm": {
            "attr_name": "FRPM Count (K-12)",
            "schema_name": "main",
            "table_name": "frpm",
            "col_name": "frpm_count_k12",
            "attr_description": "Eligible meal-program students.",
        },
        "enrollment": {
            "attr_name": "Enrollment (K-12)",
            "schema_name": "main",
            "table_name": "frpm",
            "col_name": "enrollment_k12",
            "attr_description": "K-12 student enrollment.",
        },
    }

    anchor_id, _ = CandidatePreparationAgent()._identify_anchor(
        state, "Which school has the highest eligible rate?", contexts
    )

    assert anchor_id == "frpm"
    assert "Authoritative evidence:" in captured["prompt"]
    assert "main.frpm.frpm_count_k12" in captured["prompt"]
    assert "main.frpm.enrollment_k12" in captured["prompt"]
    assert "FRPM Count (K-12)" not in captured["prompt"]
    assert "schema.table.column physical references" in captured["prompt"]


def test_column_metadata_backfill_requires_samples_and_nullability() -> None:
    assert candidates_preparation._needs_column_metadata_backfill(
        {
            "columns": [
                {
                    "name": "status",
                    "sample_values": ["open"],
                }
            ]
        }
    )
    assert candidates_preparation._needs_column_metadata_backfill(
        {
            "columns": [
                {
                    "name": "status",
                    "sample_values": [],
                    "is_nullable": True,
                }
            ]
        }
    )
    assert not candidates_preparation._needs_column_metadata_backfill(
        {
            "columns": [
                {
                    "name": "status",
                    "sample_values": ["open"],
                    "is_nullable": None,
                }
            ]
        }
    )


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

    # The searches run on a thread pool, so the order they are called in is the
    # scheduler's; the order their results are collected in is not. (The deduper
    # is stubbed out here, so `out` shows the collection order directly.)
    assert sorted(q for q, _, _ in seen) == ["e1", "e2", "q"]
    assert [t["name"] for t in out] == ["q", "e1", "e2"]
    expected_k = max(1, SQL_GEN_MAX_ENTITIES // len(seen))
    assert all(k == expected_k for _, k, _ in seen)
    assert all(db == "db" for _, _, db in seen)
    assert len(out) == 3


def test_a_table_two_searches_return_merges_the_same_way_whichever_finishes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real deduper keeps the first non-empty description it sees.

    The question's search is made to finish last. Collected in completion order,
    the entity's description would win; collected in submission order, the
    question's does, every time.
    """
    entity_done = threading.Event()

    def fake_get_relevant_tables(
        retriever: object,
        query: str,
        k: int = 1,
        database_name: str | None = None,
        **kwargs: object,
    ) -> list[dict]:
        if query == "q":
            assert entity_done.wait(5)
        else:
            entity_done.set()
        return [{"id": "t1", "name": "t1", "description": f"found by {query}"}]

    monkeypatch.setattr(
        candidates_preparation, "get_relevant_tables", fake_get_relevant_tables
    )

    out = CandidatePreparationAgent()._retrieve_additional_tables(
        object(), "q", ["e1"], "db"
    )

    assert [t["id"] for t in out] == ["t1"]
    assert out[0]["description"] == "found by q"


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


def test_anchors_naming_a_column_in_scope_are_dropped(monkeypatch) -> None:
    enrollment = {
        "id": "enrollment",
        "name": "enrollment",
        "schema_name": "public",
        "database_name": "db",
        "columns": [{"name": "student_id"}, {"name": "Headcount (Full-Time)"}],
    }
    attendance_type = {
        "phrase": "full-time",
        "kind": "value",
        "tbl": "students",
        "col": "Attendance Type",
        "stored_value": "Full-Time",
    }
    region = {
        "phrase": "northside",
        "kind": "value",
        "tbl": "students",
        "col": "Region",
        "stored_value": "Northside",
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
        lambda candidates: [enrollment],
    )
    monkeypatch.setattr(candidates_preparation, "fetch_tables_by_ids", lambda ids: [])
    monkeypatch.setattr(
        candidates_preparation, "find_table_id_by_name", lambda name, db=None: None
    )
    monkeypatch.setattr(
        candidates_preparation, "find_connected_junction_tables", lambda ids: ([], [])
    )

    state = cast(
        AgentState,
        {
            "initial_question": (
                "Which region has the highest full-time headcount in Northside?"
            ),
            "data_retriever": object(),
            "value_anchors": [attendance_type, region],
            "path_state": {
                "target_db": "db",
                "retrieved_custom_analyses": [{"id": "analysis"}],
            },
        },
    )

    assert CandidatePreparationAgent().execute(state)["value_anchors"] == [region]


def test_the_table_an_anchor_points_at_reaches_the_relevance_filter(
    monkeypatch,
) -> None:
    """q758's shape: the anchor found 'Human' in a table retrieval never saw."""
    superhero = {
        "id": "superhero",
        "name": "superhero",
        "schema_name": "main",
        "database_name": "db",
        "columns": [{"name": "race_id"}, {"name": "height_cm"}],
    }
    race = {
        "id": "race",
        "name": "race",
        "schema_name": "main",
        "database_name": "db",
        "columns": [{"name": "id"}, {"name": "race"}],
    }
    human = {
        "phrase": "human",
        "kind": "value",
        "tbl": "race",
        "col": "race",
        "stored_value": "Human",
    }
    judged: list[list[str]] = []

    monkeypatch.setattr(
        CandidatePreparationAgent,
        "_retrieve_additional_tables",
        lambda self, retriever, question, entities, target_db: [],
    )

    def record_and_keep(
        self,
        state,
        question,
        tables,
        custom_analyses=None,
        attribute_join_paths=None,
    ):
        judged.append([table["name"] for table in tables])
        return tables, ""

    monkeypatch.setattr(
        CandidatePreparationAgent, "_filter_tables_by_relevance", record_and_keep
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
        lambda candidates: [superhero],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "find_table_id_by_name",
        lambda name, db=None: "race" if name == "race" else None,
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_tables_by_ids",
        lambda ids: [race] if list(ids) == ["race"] else [],
    )
    monkeypatch.setattr(
        candidates_preparation, "find_connected_junction_tables", lambda ids: ([], [])
    )
    bridge_calls: list[list[str]] = []

    def fake_bridges(ids, allowed=None):
        bridge_calls.append(list(ids))
        return [], [], 0

    monkeypatch.setattr(candidates_preparation, "find_kept_table_bridges", fake_bridges)

    state = cast(
        AgentState,
        {
            "initial_question": "hair colour of the human superhero 185 cm tall",
            "data_retriever": object(),
            "value_anchors": [human],
            "path_state": {
                "target_db": "db",
                "retrieved_custom_analyses": [{"id": "analysis"}],
            },
        },
    )

    result = CandidatePreparationAgent().execute(state)

    assert judged == [["superhero", "race"]]
    assert bridge_calls == [["superhero", "race"]]
    assert "race" in [t["name"] for t in result["path_state"]["relevant_tables"]]
    # The anchor still names data, so §6 keeps it: the table arriving in scope
    # is not what the schema-naming check looks at.
    assert result["value_anchors"] == [human]


# --------------------------------------------------------------------------
# Relevance filter: a removal is applied only when both checks hold
# --------------------------------------------------------------------------


def _table(name: str) -> dict:
    return {
        "id": name,
        "name": name,
        "schema_name": "main",
        "database_name": "db",
        "description": f"the {name} table",
    }


def _removal(
    table: str,
    *,
    no_column: bool = True,
    cannot_change_rows: bool = True,
) -> TableRemovalModel:
    return TableRemovalModel(
        table=table,
        supplies_no_needed_column=no_column,
        cannot_change_qualifying_rows=cannot_change_rows,
        justification="checked both",
    )


def _stub_filter_llm(monkeypatch, result: TableRelevanceModel) -> None:
    monkeypatch.setattr(
        candidates_preparation,
        "invoke_with_structured_output",
        lambda llm, messages, model: result,
    )


def _run_filter(tables: list[dict]) -> list[dict]:
    kept, _ = CandidatePreparationAgent()._filter_tables_by_relevance(
        cast(AgentState, {"llm": object(), "domain_rules": []}), "q", tables
    )
    return kept


def test_removal_is_applied_when_both_checks_hold(monkeypatch) -> None:
    _stub_filter_llm(
        monkeypatch,
        TableRelevanceModel(
            reasoning="drop b", tables_to_remove=[_removal("db.main.b")]
        ),
    )

    kept = _run_filter([_table(n) for n in ("a", "b", "c")])

    assert [t["name"] for t in kept] == ["a", "c"]


def test_removal_is_declined_when_the_table_may_restrict_rows(monkeypatch) -> None:
    """The 70% case: no column of it is projected, but it scopes the row set."""
    _stub_filter_llm(
        monkeypatch,
        TableRelevanceModel(
            reasoning="frpm supplies no output column",
            tables_to_remove=[
                _removal("db.main.frpm", no_column=True, cannot_change_rows=False)
            ],
        ),
    )

    kept = _run_filter([_table(n) for n in ("schools", "frpm", "satscores")])

    assert "frpm" in [t["name"] for t in kept]


def test_removal_is_declined_when_the_table_supplies_a_needed_column(
    monkeypatch,
) -> None:
    _stub_filter_llm(
        monkeypatch,
        TableRelevanceModel(
            reasoning="only used for scoping",
            tables_to_remove=[
                _removal("db.main.frpm", no_column=False, cannot_change_rows=True)
            ],
        ),
    )

    kept = _run_filter([_table(n) for n in ("schools", "frpm", "satscores")])

    assert "frpm" in [t["name"] for t in kept]
