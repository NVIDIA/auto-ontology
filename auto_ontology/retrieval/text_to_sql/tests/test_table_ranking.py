# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql import table_ranking
from auto_ontology.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_entity_tables_for_prompt,
    format_tables_for_prompt,
)
from auto_ontology.retrieval.text_to_sql.table_ranking import (
    rank_tables_by_entity,
    score_tables_by_entity,
    unique_entities,
)


def _table(tid: str) -> dict:
    return {"id": tid, "name": tid, "schema_name": "public", "columns": []}


def _hit(score: float, column: str = "col") -> dict:
    return {"score": score, "column": column}


def test_last_entity_best_table_is_not_left_at_the_end() -> None:
    tables = [_table(t) for t in ["a1", "a2", "a3", "b1", "e1"]]
    scores = {
        "order": {"a1": _hit(0.1), "a2": _hit(0.2), "a3": _hit(0.3)},
        "customer": {"b1": _hit(0.4)},
        "product": {"e1": _hit(0.5)},
    }

    ranked = rank_tables_by_entity(tables, scores, ["order", "customer", "product"])

    assert [t["id"] for t in ranked] == ["a1", "b1", "e1", "a2", "a3"]


def test_table_matching_more_entities_outranks_a_closer_single_match() -> None:
    tables = [_table(t) for t in ["closest", "shared", "other"]]
    scores = {
        "region": {"closest": _hit(0.1), "shared": _hit(0.4)},
        "revenue": {"shared": _hit(0.5), "other": _hit(0.2)},
    }

    ranked = rank_tables_by_entity(tables, scores, ["region", "revenue"])

    assert [t["id"] for t in ranked] == ["shared", "closest", "other"]
    assert ranked[0]["entity_matches"] == [
        {"entity": "region", "column": "col", "rank": 1},
        {"entity": "revenue", "column": "col", "rank": 1},
    ]


def test_lead_table_goes_first_and_unmatched_tables_keep_their_order() -> None:
    tables = [_table(t) for t in ["x", "a1", "y", "anchor", "b1", "z"]]
    scores = {"e1": {"a1": _hit(0.1)}, "e2": {"b1": _hit(0.2)}}

    ranked = rank_tables_by_entity(tables, scores, ["e1", "e2"], "anchor")

    assert [t["id"] for t in ranked] == ["anchor", "a1", "b1", "x", "y", "z"]
    assert "entity_matches" not in ranked[0]


def test_each_match_records_its_rank_within_the_entity() -> None:
    tables = [_table("order_history"), _table("orders")]
    scores = {
        "order": {
            "order_history": _hit(0.4, "ORDER_DATE"),
            "orders": _hit(0.2, "ORDER_DATE"),
        }
    }

    ranked = rank_tables_by_entity(tables, scores, ["order"])

    assert [t["id"] for t in ranked] == ["orders", "order_history"]
    assert ranked[0]["entity_matches"] == [
        {"entity": "order", "column": "ORDER_DATE", "rank": 1}
    ]
    assert ranked[1]["entity_matches"] == [
        {"entity": "order", "column": "ORDER_DATE", "rank": 2}
    ]
    assert "entity_matches" not in tables[0]


def test_scores_out_of_scope_or_empty_leave_tables_unchanged() -> None:
    tables = [_table("t1"), _table("t2")]

    assert rank_tables_by_entity(tables, {}, ["e"]) == tables
    assert rank_tables_by_entity(tables, {"e": {"other": _hit(0.1)}}, ["e"]) == tables


def test_unique_entities_strips_and_dedupes_in_order() -> None:
    assert unique_entities([" a ", "b", "a", "", None, "b "]) == ["a", "b"]


def test_score_tables_keeps_best_hit_per_table_and_isolates_failures(
    monkeypatch,
) -> None:
    def fake_search(_retriever, entity, **_kwargs):
        if entity == "broken":
            raise RuntimeError("vdb down")
        return [
            {"id": "attr1", "score": 0.3},
            {"id": "attr2", "score": 0.1},
            {"id": "attr3", "score": 0.5},
        ]

    contexts = {
        "attr1": {"table_id": "t1", "col_name": "A"},
        "attr2": {"table_id": "t1", "col_name": "B"},
        "attr3": {"table_id": "t2", "col_name": "C"},
    }
    monkeypatch.setattr(table_ranking, "search_semantic_index", fake_search)
    monkeypatch.setattr(
        table_ranking,
        "fetch_attr_column_contexts",
        lambda ids, database_name=None: {i: contexts[i] for i in ids},
    )

    scores = score_tables_by_entity(object(), ["ok", "broken"], "db")

    assert scores == {
        "ok": {
            "t1": {"score": 0.1, "column": "B"},
            "t2": {"score": 0.5, "column": "C"},
        }
    }


def test_prompt_shows_entity_matches_without_steering_to_one_table() -> None:
    tables = rank_tables_by_entity(
        [_table("order_history"), _table("orders")],
        {
            "order": {
                "order_history": _hit(0.4, "ORDER_DATE"),
                "orders": _hit(0.2, "ORDER_DATE"),
            }
        },
        ["order"],
    )

    rendered = format_tables_for_prompt(tables)

    assert rendered.startswith("Tables are ordered by how many")
    assert rendered.count('matches: "order" via ORDER_DATE') == 2
    assert "best match" not in rendered
    assert "prefer" not in rendered.lower()


def test_prompt_has_no_entity_note_without_matches() -> None:
    rendered = format_tables_for_prompt([_table("t1")])

    assert "Tables are ordered" not in rendered
    assert "matches:" not in rendered


def test_entities_section_lists_each_entitys_tables_in_rank_order() -> None:
    tables = rank_tables_by_entity(
        [_table("order_history"), _table("customers"), _table("orders")],
        {
            "customer": {"customers": _hit(0.3, "NAME")},
            "order": {
                "order_history": _hit(0.4, "ORDER_DATE"),
                "orders": _hit(0.2, "ORDER_DATE"),
                "customers": _hit(0.6, "LAST_ORDER"),
            },
        },
        ["order", "customer"],
    )

    rendered = format_entity_tables_for_prompt(tables, ["order", "customer"])

    assert rendered.splitlines()[1:] == [
        '- "order": public.customers (via LAST_ORDER), '
        "public.orders (via ORDER_DATE), public.order_history (via ORDER_DATE)",
        '- "customer": public.customers (via NAME)',
    ]


def test_entities_section_is_empty_without_matches() -> None:
    assert format_entity_tables_for_prompt([_table("t1")], ["order"]) == ""


def test_hub_siblings_are_capped_by_shared_entity_scores() -> None:
    hops = [
        {"id": "hub", "is_hub": True},
        {"id": "s1"},
        {"id": "s2"},
        {"id": "s3"},
    ]
    scores = {"e1": {"s3": _hit(0.1)}, "e2": {"s2": _hit(0.2), "s1": _hit(0.9)}}

    kept, truncated = CandidatePreparationAgent()._rank_and_cap_hub_siblings(
        hops, cap=2, scores=scores
    )

    assert [h["id"] for h in kept] == ["hub", "s3", "s2"]
    assert truncated == 1


def test_hub_siblings_fall_back_to_unranked_cap_without_scores() -> None:
    hops = [{"id": "hub", "is_hub": True}, {"id": "s1"}, {"id": "s2"}]

    kept, truncated = CandidatePreparationAgent()._rank_and_cap_hub_siblings(
        hops, cap=1, scores={}
    )

    assert [h["id"] for h in kept] == ["hub", "s1"]
    assert truncated == 1


def test_entity_scoring_searches_the_semantic_index(monkeypatch) -> None:
    from auto_ontology.retrieval.text_to_sql.agents import candidates_preparation

    used: list[object] = []
    semantic, data = object(), object()

    def fake_score(retriever, entities, target_db):
        used.append(retriever)
        return {}

    monkeypatch.setattr(candidates_preparation, "score_tables_by_entity", fake_score)
    monkeypatch.setattr(
        CandidatePreparationAgent,
        "_retrieve_additional_tables",
        lambda self, retriever, question, entities, target_db: [],
    )
    for name, value in {
        "fetch_custom_analyses_with_sql": lambda ids: [],
        "fetch_tables_from_custom_analyses": lambda ids: [],
        "get_relevant_tables_from_candidates": lambda candidates: [_table("t1")],
        "fetch_tables_by_ids": lambda ids: [],
        "find_connected_junction_tables": lambda ids: ([], []),
        "find_kept_table_bridges": lambda ids, candidate_ids: ([], [], 0),
        "find_direct_table_joins": lambda ids: [],
    }.items():
        monkeypatch.setattr(candidates_preparation, name, value)

    CandidatePreparationAgent().execute(
        {
            "initial_question": "orders per customer",
            "data_retriever": data,
            "semantic_retriever": semantic,
            "path_state": {
                "target_db": "db",
                "entities": ["order", "customer"],
                "retrieved_custom_analyses": [{"id": "analysis"}],
            },
        }
    )

    assert used == [semantic]
