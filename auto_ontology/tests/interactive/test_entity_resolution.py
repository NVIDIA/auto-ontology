# SPDX-FileCopyrightText: Copyright (c) 2026, Ali Tharwat
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from auto_ontology.retrieval.interactive.entity_resolution import (
    _find_unresolvable_entities,
)


def _mock_ec_state(
    entities: list[str],
    col_hits: list[dict],
    candidates: list[dict] | None = None,
    uncovered_entities: list[str] | None = None,
) -> dict:
    return {
        "entities": entities,
        "retrieved_column_attributes": col_hits,
        "final_response": {
            "candidates": candidates or [],
            "uncovered_entities": uncovered_entities or [],
        },
    }


def test_ambiguous_entity_with_filler_word_is_demoted_to_unresolvable() -> None:
    raw_entity = "Price-to-Book (P/B) ratio"
    norm_entity = "price-to-book (p/b)"
    ec_state = _mock_ec_state(
        entities=[raw_entity],
        col_hits=[
            {
                "id": "col_pb_reported",
                "text": "ColumnAttribute: Reported P/B.",
                "score": 0.30,
                "query_entities": [raw_entity],
            },
            {
                "id": "col_pb_adjusted",
                "text": "ColumnAttribute: Adjusted P/B.",
                "score": 0.32,
                "query_entities": [raw_entity],
            },
        ],
        candidates=[
            {"id": "col_pb_reported", "attribute": "reported_pb", "term": "Valuation"},
            {"id": "col_pb_adjusted", "attribute": "adjusted_pb", "term": "Valuation"},
        ],
    )
    ranked_hits = [
        {
            "id": "col_pb_reported",
            "text": "ColumnAttribute: Reported P/B.",
            "score": 0.30,
        },
        {
            "id": "col_pb_adjusted",
            "text": "ColumnAttribute: Adjusted P/B.",
            "score": 0.32,
        },
    ]

    with (
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution._run_entity_coverage_pipeline",
            return_value=ec_state,
        ),
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution.search_semantic_index",
            return_value=ranked_hits,
        ),
    ):
        ambiguous_hits: dict[str, list[dict]] = {}
        unresolvable, resolved_hits, *_ = _find_unresolvable_entities(
            "What is the Price-to-Book (P/B) ratio?",
            MagicMock(),
            "test_db",
            ambiguous_hits_out=ambiguous_hits,
        )
        assert unresolvable == [(norm_entity, None)]
        assert resolved_hits == []
        assert norm_entity in ambiguous_hits
        assert len(ambiguous_hits[norm_entity]) == 2

        unresolvable_no_out, resolved_hits_no_out, *_ = _find_unresolvable_entities(
            "What is the Price-to-Book (P/B) ratio?",
            MagicMock(),
            "test_db",
            ambiguous_hits_out=None,
        )
        assert unresolvable_no_out == [(norm_entity, None)]
        assert resolved_hits_no_out == []


def test_ambiguous_entity_rescued_by_kb_is_not_unresolvable_or_resolved_hit() -> None:
    raw_entity = "Price-to-Book (P/B) ratio"
    norm_entity = "price-to-book (p/b)"
    ec_state = _mock_ec_state(
        entities=[raw_entity],
        col_hits=[
            {
                "id": "col_pb_reported",
                "text": "ColumnAttribute: Reported P/B.",
                "score": 0.30,
                "query_entities": [raw_entity],
            },
            {
                "id": "col_pb_adjusted",
                "text": "ColumnAttribute: Adjusted P/B.",
                "score": 0.32,
                "query_entities": [raw_entity],
            },
        ],
        candidates=[
            {"id": "col_pb_reported", "attribute": "reported_pb", "term": "Valuation"},
            {"id": "col_pb_adjusted", "attribute": "adjusted_pb", "term": "Valuation"},
        ],
    )
    formatted_kb = (
        "- Price-to-Book (P/B) Ratio\n"
        "  Description: Equity valuation multiple\n"
        "  Definition: Market value divided by book value\n"
    )

    with (
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution._run_entity_coverage_pipeline",
            return_value=ec_state,
        ),
        patch(
            "auto_ontology.retrieval.interactive.kb_coverage.safe_invoke_text_nr",
            return_value=(f"{raw_entity}: YES ;; Price-to-Book (P/B) Ratio"),
        ),
    ):
        unresolvable, resolved_hits, relevant_kb, *_ = _find_unresolvable_entities(
            "What is the Price-to-Book (P/B) ratio?",
            MagicMock(),
            "test_db",
            formatted_kb=formatted_kb,
        )

    assert unresolvable == []
    assert resolved_hits == []
    assert "Price-to-Book (P/B) Ratio" in relevant_kb
    assert norm_entity not in {norm for norm, *_ in resolved_hits}


def test_colliding_entities_reconcile_without_dropping_non_colliding_ambiguity() -> (
    None
):
    ec_state = _mock_ec_state(
        entities=["Alpha", "Beta", "Signal Score"],
        col_hits=[
            {
                "id": "col_1",
                "text": "ColumnAttribute: Primary Metric.",
                "score": 0.20,
                "query_entities": ["Alpha"],
            },
            {
                "id": "col_1",
                "text": "ColumnAttribute: Primary Metric.",
                "score": 0.28,
                "query_entities": ["Beta"],
            },
            {
                "id": "col_3",
                "text": "ColumnAttribute: Raw Signal.",
                "score": 0.30,
                "query_entities": ["Signal Score"],
            },
            {
                "id": "col_4",
                "text": "ColumnAttribute: Filtered Signal.",
                "score": 0.32,
                "query_entities": ["Signal Score"],
            },
        ],
        candidates=[
            {"id": "col_1", "attribute": "primary_metric", "term": "Perf"},
            {"id": "col_2", "attribute": "secondary_metric", "term": "Perf"},
            {"id": "col_3", "attribute": "raw_signal", "term": "Signal"},
            {"id": "col_4", "attribute": "filtered_signal", "term": "Signal"},
        ],
        uncovered_entities=["Beta"],
    )

    def _fake_search(
        _retriever: object, query: str, **_kwargs: object
    ) -> list[dict[str, object]]:
        if query == "Alpha":
            return [
                {
                    "id": "col_1",
                    "text": "ColumnAttribute: Primary Metric.",
                    "score": 0.20,
                }
            ]
        if query == "Beta":
            return [
                {
                    "id": "col_1",
                    "text": "ColumnAttribute: Primary Metric.",
                    "score": 0.28,
                },
                {
                    "id": "col_2",
                    "text": "ColumnAttribute: Secondary Metric.",
                    "score": 0.32,
                },
            ]
        return []

    with (
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution._run_entity_coverage_pipeline",
            return_value=ec_state,
        ),
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution.search_semantic_index",
            side_effect=_fake_search,
        ),
    ):
        unresolvable, resolved_hits, *_ = _find_unresolvable_entities(
            "Compare Alpha, Beta, and Signal Score",
            MagicMock(),
            "test_db",
        )

    assert unresolvable == [("signal", None)]
    resolved_by_norm = {norm: hit_id for norm, _text, _score, hit_id in resolved_hits}
    assert resolved_by_norm == {"alpha": "col_1", "beta": "col_2"}


def test_dropped_collision_entity_becomes_unresolvable() -> None:
    ec_state = _mock_ec_state(
        entities=["Alpha", "Beta"],
        col_hits=[
            {
                "id": "col_1",
                "text": "ColumnAttribute: Shared Col.",
                "score": 0.25,
                "query_entities": ["Alpha"],
            },
            {
                "id": "col_1",
                "text": "ColumnAttribute: Shared Col.",
                "score": 0.26,
                "query_entities": ["Beta"],
            },
        ],
    )

    with (
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution._run_entity_coverage_pipeline",
            return_value=ec_state,
        ),
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution.search_semantic_index",
            return_value=[
                {"id": "col_1", "text": "ColumnAttribute: Shared Col.", "score": 0.25}
            ],
        ),
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution._llm_disambiguate_collision",
            return_value=None,
        ),
    ):
        unresolvable, resolved_hits, *_ = _find_unresolvable_entities(
            "Compare Alpha and Beta",
            MagicMock(),
            "test_db",
        )

    assert unresolvable == [("beta", None)]
    assert [norm for norm, *_ in resolved_hits] == ["alpha"]


def test_colliding_alias_does_not_clear_ambiguous_entity_sharing_norm() -> None:
    ec_state = _mock_ec_state(
        entities=["Alpha", "Average Alpha", "Beta"],
        col_hits=[
            {
                "id": "col_alpha_a",
                "text": "ColumnAttribute: Alpha Primary.",
                "score": 0.30,
                "query_entities": ["Alpha"],
            },
            {
                "id": "col_alpha_b",
                "text": "ColumnAttribute: Alpha Secondary.",
                "score": 0.32,
                "query_entities": ["Alpha"],
            },
            {
                "id": "col_shared",
                "text": "ColumnAttribute: Shared.",
                "score": 0.20,
                "query_entities": ["Average Alpha"],
            },
            {
                "id": "col_shared",
                "text": "ColumnAttribute: Shared.",
                "score": 0.28,
                "query_entities": ["Beta"],
            },
        ],
    )

    def _fake_search(
        _retriever: object, query: str, **_kwargs: object
    ) -> list[dict[str, object]]:
        if query == "Average Alpha":
            return [
                {
                    "id": "col_shared",
                    "text": "ColumnAttribute: Shared.",
                    "score": 0.20,
                }
            ]
        if query == "Beta":
            return [
                {
                    "id": "col_shared",
                    "text": "ColumnAttribute: Shared.",
                    "score": 0.28,
                },
                {
                    "id": "col_beta",
                    "text": "ColumnAttribute: Beta Alt.",
                    "score": 0.32,
                },
            ]
        return []

    with (
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution._run_entity_coverage_pipeline",
            return_value=ec_state,
        ),
        patch(
            "auto_ontology.retrieval.interactive.entity_resolution.search_semantic_index",
            side_effect=_fake_search,
        ),
    ):
        unresolvable, resolved_hits, *_ = _find_unresolvable_entities(
            "Compare Alpha, Average Alpha, and Beta",
            MagicMock(),
            "test_db",
        )

    assert unresolvable == [("alpha", None)]
    assert {norm for norm, *_ in resolved_hits} == {"beta"}
