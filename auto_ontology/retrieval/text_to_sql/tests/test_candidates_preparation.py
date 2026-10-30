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
    TableRelevanceModel,
    TableRemovalModel,
)
from auto_ontology.retrieval.text_to_sql.prompts import SQL_GEN_MAX_ENTITIES
from auto_ontology.retrieval.text_to_sql.state import AgentState


def _hop(source: str, source_col: str, target: str, target_col: str) -> dict:
    return {
        "source_schema": "main",
        "source_table": source,
        "source_column": source_col,
        "target_schema": "main",
        "target_table": target,
        "target_column": target_col,
    }


def _catalog_table(table_id: str, name: str) -> dict:
    return {
        "id": table_id,
        "name": name,
        "schema_name": "main",
        "database_name": "db",
        "columns": [],
    }


def _stub_preparation(monkeypatch, contexts: dict, tables: list[dict]) -> None:
    by_id = {t["id"]: t for t in tables}
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
        candidates_preparation,
        "fetch_attr_column_contexts",
        lambda ids, database_name=None: contexts,
    )
    monkeypatch.setattr(candidates_preparation, "fetch_term_synonyms", lambda ids: {})
    monkeypatch.setattr(
        candidates_preparation, "get_relevant_tables_from_candidates", lambda c: []
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_tables_by_ids",
        lambda ids: [by_id[i] for i in ids if i in by_id],
    )
    monkeypatch.setattr(
        candidates_preparation, "find_connected_junction_tables", lambda ids: ([], [])
    )
    monkeypatch.setattr(
        candidates_preparation,
        "find_kept_table_bridges",
        lambda ids, allowed: ([], [], 0),
    )


def _attr_context(col: str, table_id: str, table: str) -> dict:
    return {
        "attr_name": col.title(),
        "col_name": col,
        "col_id": f"col-{col}",
        "table_id": table_id,
        "table_name": table,
        "schema_name": "main",
        "database_name": "db",
    }


def _execute(attr_ids: list[str]) -> dict:
    state = cast(
        AgentState,
        {
            "initial_question": "What is the average loan amount by male borrowers?",
            "llm": object(),
            "data_retriever": object(),
            "path_state": {
                "entities": ["loan amount", "male borrower"],
                "target_db": "db",
                "retrieved_column_attributes": [{"id": a} for a in attr_ids],
            },
        },
    )
    return CandidatePreparationAgent().execute(state)["path_state"]


def test_join_paths_connect_every_pair_of_matched_tables(monkeypatch) -> None:
    contexts = {
        "amount": _attr_context("amount", "t-loan", "loan"),
        "gender": _attr_context("gender", "t-client", "client"),
    }
    _stub_preparation(
        monkeypatch,
        contexts,
        [_catalog_table("t-loan", "loan"), _catalog_table("t-client", "client")],
    )
    seen: list[list[str]] = []
    path = [
        _hop("loan", "account_id", "account", "account_id"),
        _hop("account", "account_id", "disp", "account_id"),
        _hop("disp", "client_id", "client", "client_id"),
    ]

    def fake_paths(table_ids: list[str]) -> list[dict]:
        seen.append(table_ids)
        return [{"table_ids": ("t-loan", "t-client"), "path": path}]

    monkeypatch.setattr(candidates_preparation, "find_table_join_paths", fake_paths)

    out = _execute(["amount", "gender"])

    assert seen == [["t-loan", "t-client"]]
    assert [a["col_name"] for a in out["matched_attributes"]] == ["amount", "gender"]
    assert "primary_attribute" not in out
    # account and disp never reached the final tables, so the path is unusable.
    assert out["attribute_join_paths"] == []


def test_a_pair_path_whose_tables_all_reach_sql_generation_is_kept(
    monkeypatch,
) -> None:
    contexts = {
        "amount": _attr_context("amount", "t-loan", "loan"),
        "frequency": _attr_context("frequency", "t-account", "account"),
    }
    _stub_preparation(
        monkeypatch,
        contexts,
        [_catalog_table("t-loan", "loan"), _catalog_table("t-account", "account")],
    )
    path = [_hop("loan", "account_id", "account", "account_id")]
    monkeypatch.setattr(
        candidates_preparation,
        "find_table_join_paths",
        lambda ids: [{"table_ids": ("t-loan", "t-account"), "path": path}],
    )

    out = _execute(["amount", "frequency"])

    assert out["attribute_join_paths"] == [{"pair": True, "path": path}]


def test_hub_siblings_are_found_from_every_matched_table(monkeypatch) -> None:
    contexts = {
        "amount": _attr_context("amount", "t-loan", "loan"),
        "gender": _attr_context("gender", "t-client", "client"),
    }
    _stub_preparation(
        monkeypatch,
        contexts,
        [
            _catalog_table("t-loan", "loan"),
            _catalog_table("t-client", "client"),
            _catalog_table("t-account", "account"),
            _catalog_table("t-disp", "disp"),
        ],
    )
    monkeypatch.setattr(candidates_preparation, "find_table_join_paths", lambda ids: [])
    monkeypatch.setattr(candidates_preparation, "_HUB_SIBLING_EXPANSION_ENABLED", True)
    siblings = {
        "t-loan": [
            {"id": "t-account", "target_table": "account", "is_hub": True},
            {"id": "t-disp", "target_table": "disp", "is_hub": False},
        ],
        "t-client": [
            {"id": "t-disp", "target_table": "disp", "is_hub": False},
            {"id": "t-loan", "target_table": "loan", "is_hub": False},
        ],
    }
    monkeypatch.setattr(
        candidates_preparation,
        "find_anchor_hub_siblings",
        lambda table_id, max_siblings=None: (siblings[table_id], 0),
    )

    out = _execute(["amount", "gender"])

    names = sorted(t["name"] for t in out["relevant_tables"])
    assert names == ["account", "client", "disp", "loan"]
    [structural] = out["attribute_join_paths"]
    assert sorted(h["id"] for h in structural["path"]) == ["t-account", "t-disp"]


def test_column_metadata_backfill_requires_samples_nullability_and_semantic_fk() -> (
    None
):
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
                    "semantic_fk": None,
                }
            ]
        }
    )
    assert candidates_preparation._needs_column_metadata_backfill(
        {
            "columns": [
                {
                    "name": "status",
                    "sample_values": ["open"],
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
                    "semantic_fk": None,
                }
            ]
        }
    )


@pytest.mark.parametrize("source", ["custom_analysis", "sql_attribute"])
def test_linked_tables_are_backfilled_before_relevance_filter(
    monkeypatch, source: str
) -> None:
    captured_tables: list[dict] = []
    linked_table = {
        "id": "linked",
        "name": "orders",
        "database_name": "db",
        "schema_name": "public",
        "columns": [
            {
                "name": "customer_id",
                "sample_values": ["1"],
                "is_nullable": False,
            }
        ],
    }
    enriched_table = {
        **linked_table,
        "columns": [
            {
                **linked_table["columns"][0],
                "semantic_fk": "public.customers.id",
            }
        ],
    }

    monkeypatch.setattr(
        CandidatePreparationAgent,
        "_retrieve_additional_tables",
        lambda self, retriever, question, entities, target_db: [],
    )

    def capture_filter(
        self,
        state,
        question,
        tables,
        custom_analyses=None,
        attribute_join_paths=None,
    ):
        captured_tables.extend(tables)
        return tables, ""

    monkeypatch.setattr(
        CandidatePreparationAgent, "_filter_tables_by_relevance", capture_filter
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_custom_analyses_with_sql",
        lambda ids: [{"id": "analysis"}] if source == "custom_analysis" else [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_tables_from_custom_analyses",
        lambda ids: [linked_table] if source == "custom_analysis" else [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_sql_attributes_with_sql",
        lambda ids: [{"id": "attribute"}] if source == "sql_attribute" else [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_tables_from_sql_attributes",
        lambda ids: [linked_table] if source == "sql_attribute" else [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "get_relevant_tables_from_candidates",
        lambda candidates: [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_tables_by_ids",
        lambda ids: [enriched_table] if ids == ["linked"] else [],
    )
    monkeypatch.setattr(
        candidates_preparation,
        "find_connected_junction_tables",
        lambda ids: ([], []),
    )
    monkeypatch.setattr(
        candidates_preparation,
        "find_kept_table_bridges",
        lambda kept_ids, candidate_ids: ([], [], 0),
    )

    state = cast(
        AgentState,
        {
            "initial_question": "Count orders",
            "data_retriever": object(),
            "path_state": {
                "target_db": "db",
                (
                    "retrieved_custom_analyses"
                    if source == "custom_analysis"
                    else "retrieved_sql_attributes"
                ): [
                    {"id": ("analysis" if source == "custom_analysis" else "attribute")}
                ],
            },
        },
    )

    CandidatePreparationAgent().execute(state)

    assert captured_tables[0]["columns"][0]["semantic_fk"] == "public.customers.id"


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


def test_additional_table_retrieve_starts_before_join_paths_return(
    monkeypatch,
) -> None:
    retrieve_started = threading.Event()
    release_retrieve = threading.Event()
    retrieve_was_running = []

    def fake_retrieve(self, retriever, question, entities, target_db):
        retrieve_started.set()
        assert release_retrieve.wait(timeout=2)
        return [{"id": "extra", "name": "extra", "database_name": "db"}]

    def fake_join_paths(table_ids):
        retrieve_was_running.append(retrieve_started.wait(timeout=2))
        release_retrieve.set()
        return []

    monkeypatch.setattr(
        CandidatePreparationAgent, "_retrieve_additional_tables", fake_retrieve
    )
    monkeypatch.setattr(
        candidates_preparation, "find_table_join_paths", fake_join_paths
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


def test_retrieved_columns_take_the_attribute_description() -> None:
    tables = [
        {
            "id": "t1",
            "name": "orders",
            "columns": [
                {"name": "status", "description": "Order status."},
                {"name": "total", "description": "Order total."},
                {"name": "region", "description": "Sales region."},
            ],
        },
        {"id": "t2", "name": "stores", "columns": [{"name": "status"}]},
    ]
    contexts = {
        "a1": {
            "table_id": "t1",
            "col_name": "status",
            "attr_description": "Fulfilment state of the order.",
            "value_description": "1 = open, 2 = shipped",
        },
        "a2": {
            "table_id": "t1",
            "col_name": "total",
            "attr_description": "",
            "value_description": "",
        },
    }

    described = candidates_preparation._describe_matched_columns(tables, contexts)

    columns = {c["name"]: c["description"] for c in described[0]["columns"]}
    assert columns == {
        "status": "Fulfilment state of the order. Values: 1 = open, 2 = shipped",
        "total": "Order total.",
        "region": "Sales region.",
    }
    assert described[1]["columns"] == [{"name": "status"}]
    assert tables[0]["columns"][0]["description"] == "Order status."


def test_resolvers_reach_every_column_and_the_relevance_summary(monkeypatch) -> None:
    tables = [
        {
            "id": "t1",
            "name": "results",
            "columns": [{"name": "points"}, {"name": "grid"}],
        },
        {"id": "t2", "name": "standings", "columns": [{"name": "points"}]},
    ]
    monkeypatch.setattr(
        candidates_preparation,
        "fetch_ambiguity_resolvers",
        lambda ids: {
            ("t1", "points"): "Points from one race.",
            ("t2", "points"): "Season total after a race.",
        },
    )

    attached = candidates_preparation._attach_ambiguity_resolvers(tables)

    assert attached[0]["columns"] == [
        {"name": "points", "ambiguity_resolver": "Points from one race."},
        {"name": "grid"},
    ]
    assert attached[1]["columns"][0]["ambiguity_resolver"] == (
        "Season total after a race."
    )
    assert "ambiguity_resolver" not in tables[0]["columns"][0]
    summary = candidates_preparation._build_relevance_tables_summary(attached)
    assert "points (unknown) | when to use: Points from one race." in summary
    assert "grid (unknown)\n" in summary
