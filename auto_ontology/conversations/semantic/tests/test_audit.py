# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the edge audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from auto_ontology.conversations.semantic.audit import (
    MergeAnswer,
    MergeAnswers,
    MutualAnswer,
    MutualAnswers,
    PairAnswer,
    PairAnswers,
    RevisionReport,
    SubjectAnswer,
    SubjectAnswers,
    audit_revisions,
    audit_structure,
    check_derived,
    check_subjects,
    resolve_mutual,
    review_merges,
    unsound_pairs,
)


@pytest.fixture(autouse=True)
def _no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_ontology.utils import llm_invoke

    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_: object())


def node(
    id_: str,
    name: str,
    *,
    value: str = "",
    key: str = "",
    turns: list[int] | None = None,
    occurred_on: str | None = None,
) -> dict[str, Any]:
    return {
        "id": id_,
        "name": name,
        "value": value,
        "attribute_key": key,
        "turn_indexes": turns if turns is not None else [0],
        "occurred_on": occurred_on,
    }


def edge(source: str, target: str, role: str) -> dict[str, Any]:
    return {"source_id": source, "target_id": target, "role": role}


class TestStructure:
    def test_a_containment_holding_both_ways_is_reported(self) -> None:
        nodes = [node("a", "Autocomplete"), node("b", "Loading spinner")]
        edges = [edge("a", "b", "has_component"), edge("b", "a", "has_component")]

        report = audit_structure(nodes, edges)

        assert len(report.mutual) == 1
        assert report.total == 1

    def test_the_same_role_in_one_direction_only_is_fine(self) -> None:
        nodes = [node("a", "Autocomplete"), node("b", "Loading spinner")]

        report = audit_structure(nodes, [edge("a", "b", "has_component")])

        assert report.total == 0

    def test_a_symmetric_role_may_hold_both_ways(self) -> None:
        nodes = [node("a", "One"), node("b", "Two")]
        edges = [edge("a", "b", "refines"), edge("b", "a", "refines")]

        assert audit_structure(nodes, edges).total == 0

    def test_an_edge_to_itself_is_reported(self) -> None:
        report = audit_structure([node("a", "One")], [edge("a", "a", "refines")])

        assert len(report.loops) == 1

    def test_precedes_against_its_own_dates_is_reported(self) -> None:
        nodes = [
            node("a", "Later", occurred_on="2024-04-05"),
            node("b", "Earlier", occurred_on="2024-03-11"),
        ]

        report = audit_structure(nodes, [edge("a", "b", "precedes")])

        assert len(report.misdated) == 1

    def test_precedes_agreeing_with_its_dates_is_not(self) -> None:
        nodes = [
            node("a", "Earlier", occurred_on="2024-03-11"),
            node("b", "Later", occurred_on="2024-04-05"),
        ]

        assert audit_structure(nodes, [edge("a", "b", "precedes")]).total == 0

    def test_precedes_without_dates_is_left_alone(self) -> None:
        nodes = [node("a", "One"), node("b", "Two")]

        assert audit_structure(nodes, [edge("a", "b", "precedes")]).total == 0


_NODES = [
    node("a", "Error rate 1%", value="1%", key="error_rate_pct", turns=[44]),
    node("b", "Error rate 12%", value="12%", key="error_rate_pct", turns=[44]),
    node("c", "Coverage 78%", value="78%", key="coverage_pct", turns=[60]),
    node("d", "Coverage 65%", value="65%", key="coverage_pct", turns=[20]),
]
_TEXTS = {
    20: "coverage is at sixty-five percent",
    44: "the error rate went from twelve percent to one percent",
    60: "coverage reached seventy-eight percent",
}


def answers(*items: tuple[int, str]) -> PairAnswers:
    return PairAnswers(
        answers=[PairAnswer(number=n, verdict=v, reason="") for n, v in items]
    )


class TestRevisions:
    def test_only_the_revision_roles_are_put_to_the_model(self) -> None:
        edges = [
            edge("a", "b", "contradicts"),
            edge("c", "d", "refines"),
            edge("c", "d", "about"),
        ]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "changed")),
        ) as called:
            report = audit_revisions(_NODES, edges, _TEXTS)

        assert report.checked == 1
        assert called.call_count == 1

    def test_a_graph_with_no_revisions_asks_nothing(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            report = audit_revisions(_NODES, [edge("c", "d", "about")], _TEXTS)

        assert report.checked == 0
        assert called.call_count == 0

    def test_a_pair_whose_turns_are_missing_is_left_unchecked(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            report = audit_revisions(_NODES, [edge("a", "b", "supersedes")], {})

        assert len(report.unchecked) == 1
        assert called.call_count == 0

    def test_a_pair_with_an_endpoint_outside_the_graph_is_left_unchecked(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            report = audit_revisions(_NODES, [edge("a", "gone", "supersedes")], _TEXTS)

        assert len(report.unchecked) == 1
        assert called.call_count == 0

    def test_the_turns_of_both_endpoints_reach_the_prompt(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "changed")),
        ) as called:
            audit_revisions(_NODES, [edge("c", "d", "supersedes")], _TEXTS)

        prompt = called.call_args.args[1][1].content
        assert "turn 20" in prompt
        assert "turn 60" in prompt

    def test_verdicts_land_in_their_own_buckets(self) -> None:
        edges = [edge("a", "b", "contradicts"), edge("c", "d", "supersedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "separate"), (2, "changed")),
        ):
            report = audit_revisions(_NODES, edges, _TEXTS)

        assert len(report.separate) == 1
        assert len(report.changed) == 1
        assert report.separate[0].edge["source_id"] == "a"

    def test_a_pair_the_model_skipped_is_unchecked_not_assumed(self) -> None:
        edges = [edge("a", "b", "contradicts"), edge("c", "d", "supersedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "changed")),
        ):
            report = audit_revisions(_NODES, edges, _TEXTS)

        assert report.checked == 1
        assert len(report.unchecked) == 1

    def test_a_number_the_model_invented_is_ignored(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((9, "separate"), (1, "changed")),
        ):
            report = audit_revisions(_NODES, [edge("a", "b", "supersedes")], _TEXTS)

        assert len(report.changed) == 1
        assert not report.separate

    def test_a_model_failure_leaves_the_batch_unchecked(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            side_effect=RuntimeError("no"),
        ):
            report = audit_revisions(_NODES, [edge("a", "b", "supersedes")], _TEXTS)

        assert len(report.unchecked) == 1
        assert report.checked == 0

    def test_no_answer_at_all_leaves_the_batch_unchecked(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=None,
        ):
            report = audit_revisions(_NODES, [edge("a", "b", "supersedes")], _TEXTS)

        assert len(report.unchecked) == 1

    @pytest.mark.parametrize("size", [1, 3])
    def test_the_pairs_are_batched(self, size: int) -> None:
        edges = [edge("a", "b", "supersedes"), edge("c", "d", "supersedes")]
        every = answers((1, "changed"), (2, "changed"))
        with (
            patch("auto_ontology.conversations.semantic.audit.BATCH_PAIRS", size),
            patch(
                "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
                return_value=every,
            ) as called,
        ):
            audit_revisions(_NODES, edges, _TEXTS)

        assert called.call_count == (2 if size == 1 else 1)


def subject_answers(*items: tuple[int, str, str]) -> SubjectAnswers:
    return SubjectAnswers(
        answers=[SubjectAnswer(number=n, verdict=v, belongs_to=b) for n, v, b in items]
    )


@dataclass
class _Draft:
    """Enough of a ``NodeDraft`` for the subject check."""

    name: str
    kinds: list[str]
    key: str = ""
    value: str = ""
    type: str = "Feature"
    description: str = ""
    evidence: str = ""
    turn_indexes: set[int] = field(default_factory=lambda: {1})


@dataclass
class _DraftEdge:
    source: str
    target: str
    role: str = "about"


@dataclass
class _DraftGraph:
    nodes: list[_Draft]
    edges: list[_DraftEdge]


_TURNS = [
    {"turn_index": 1, "question": "how is it going", "answer": "fine"},
    {"turn_index": 2, "question": "and now", "answer": "also fine"},
]


def reading(name: str, key: str = "coverage_pct") -> _Draft:
    return _Draft(name=name, kinds=["fact"], key=key, value="78%")


def thing(name: str) -> _Draft:
    return _Draft(name=name, kinds=["object"])


class TestCheckingSubjects:
    def test_a_keyed_reading_is_checked(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "right", "")),
        ):
            report = check_subjects(graph, _TURNS)

        assert len(report.right) == 1

    def test_a_statement_with_no_key_is_skipped(self) -> None:
        graph = _DraftGraph(
            nodes=[_Draft("A remark", ["fact"]), thing("A whole")],
            edges=[_DraftEdge("a remark", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            report = check_subjects(graph, _TURNS)

        assert report.checked == 0
        assert called.call_count == 0

    def test_other_roles_are_not_subjects(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole", "refines")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            check_subjects(graph, _TURNS)

        assert called.call_count == 0

    def test_a_reading_whose_turn_is_missing_is_left_unchecked(self) -> None:
        orphan = reading("A reading")
        orphan.turn_indexes = {99}
        graph = _DraftGraph(
            nodes=[orphan, thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            report = check_subjects(graph, _TURNS)

        assert report.unchecked == ["A reading"]
        assert called.call_count == 0

    def test_a_wrong_subject_naming_a_thing_moves_the_edge(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole"), thing("The other thing")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "wrong", "The other thing")),
        ):
            report = check_subjects(graph, _TURNS)

        assert graph.edges[0].target == "the other thing"
        assert report.moved == 1

    def test_a_wrong_subject_naming_nothing_is_left_alone(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "wrong", "Never mentioned")),
        ):
            report = check_subjects(graph, _TURNS)

        assert graph.edges[0].target == "a whole"
        assert len(report.wrong) == 1
        assert report.moved == 0

    def test_a_wrong_subject_naming_a_statement_is_left_alone(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole"), reading("Another reading")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "wrong", "Another reading")),
        ):
            report = check_subjects(graph, _TURNS)

        assert graph.edges[0].target == "a whole"
        assert report.moved == 0

    def test_a_wrong_subject_naming_where_it_already_is_moves_nothing(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "wrong", "A whole")),
        ):
            report = check_subjects(graph, _TURNS)

        assert graph.edges[0].target == "a whole"
        assert report.moved == 0

    def test_an_unsure_verdict_changes_nothing(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole"), thing("The other thing")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "unsure", "The other thing")),
        ):
            report = check_subjects(graph, _TURNS)

        assert graph.edges[0].target == "a whole"
        assert len(report.unsure) == 1

    def test_a_model_failure_leaves_every_subject_as_assigned(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            side_effect=RuntimeError("no"),
        ):
            report = check_subjects(graph, _TURNS)

        assert graph.edges[0].target == "a whole"
        assert report.unchecked == ["A reading"]

    def test_a_reading_the_model_skipped_is_unchecked(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((2, "right", "")),
        ):
            report = check_subjects(graph, _TURNS)

        assert report.checked == 0
        assert report.unchecked == ["A reading"]

    def test_the_cited_turn_reaches_the_prompt(self) -> None:
        graph = _DraftGraph(
            nodes=[reading("A reading"), thing("A whole")],
            edges=[_DraftEdge("a reading", "a whole")],
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=subject_answers((1, "right", "")),
        ) as called:
            check_subjects(graph, _TURNS)

        prompt = called.call_args.args[1][1].content
        assert "turn 1" in prompt
        assert "turn 2" not in prompt


@dataclass
class _Edge:
    source: str
    target: str
    role: str
    evidence: str = ""


@dataclass
class _Graph:
    edges: list[_Edge]


def mutual_answers(*items: tuple[int, str]) -> MutualAnswers:
    return MutualAnswers(answers=[MutualAnswer(number=n, keep=k) for n, k in items])


class TestResolvingMutual:
    def test_the_losing_direction_is_dropped(self) -> None:
        keep = _Edge("Whole", "Part", "has_component", "the part of the whole")
        drop = _Edge("Part", "Whole", "has_component", "the part of the whole")
        graph = _Graph([keep, drop])
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=mutual_answers((1, "a")),
        ):
            assert resolve_mutual(graph) == 1

        assert graph.edges == [keep]

    def test_choosing_b_drops_the_first(self) -> None:
        first = _Edge("Part", "Whole", "has_component")
        second = _Edge("Whole", "Part", "has_component")
        graph = _Graph([first, second])
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=mutual_answers((1, "b")),
        ):
            resolve_mutual(graph)

        assert graph.edges == [second]

    def test_undecided_keeps_both(self) -> None:
        graph = _Graph(
            [
                _Edge("Whole", "Part", "has_component"),
                _Edge("Part", "Whole", "has_component"),
            ]
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=mutual_answers((1, "neither")),
        ):
            assert resolve_mutual(graph) == 0

        assert len(graph.edges) == 2

    def test_a_pair_the_model_skipped_keeps_both(self) -> None:
        graph = _Graph(
            [
                _Edge("Whole", "Part", "has_component"),
                _Edge("Part", "Whole", "has_component"),
            ]
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=mutual_answers((2, "a")),
        ):
            assert resolve_mutual(graph) == 0

        assert len(graph.edges) == 2

    def test_a_graph_with_no_mutual_pair_asks_nothing(self) -> None:
        graph = _Graph([_Edge("Whole", "Part", "has_component")])
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            assert resolve_mutual(graph) == 0

        assert called.call_count == 0

    def test_a_symmetric_role_is_not_a_mutual_pair(self) -> None:
        graph = _Graph([_Edge("One", "Two", "refines"), _Edge("Two", "One", "refines")])
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            assert resolve_mutual(graph) == 0

        assert len(graph.edges) == 2
        assert called.call_count == 0

    def test_names_are_matched_case_insensitively(self) -> None:
        graph = _Graph(
            [
                _Edge("Whole", "Part", "has_component"),
                _Edge("part", "whole", "has_component"),
            ]
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=mutual_answers((1, "a")),
        ):
            assert resolve_mutual(graph) == 1

    def test_both_sentences_reach_the_prompt(self) -> None:
        graph = _Graph(
            [
                _Edge("Whole", "Part", "has_component", "first sentence"),
                _Edge("Part", "Whole", "has_component", "second sentence"),
            ]
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=mutual_answers((1, "a")),
        ) as called:
            resolve_mutual(graph)

        prompt = called.call_args.args[1][1].content
        assert "first sentence" in prompt
        assert "second sentence" in prompt

    def test_a_model_failure_keeps_both(self) -> None:
        graph = _Graph(
            [
                _Edge("Whole", "Part", "has_component"),
                _Edge("Part", "Whole", "has_component"),
            ]
        )
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            side_effect=RuntimeError("no"),
        ):
            assert resolve_mutual(graph) == 0

        assert len(graph.edges) == 2


@dataclass
class _Merge:
    canonical: str
    duplicates: tuple[str, ...]
    similarity: float = 0.95


def merge_answers(*items: tuple[int, str, list[str]]) -> MergeAnswers:
    return MergeAnswers(
        answers=[
            MergeAnswer(number=n, verdict=v, not_the_same=names)
            for n, v, names in items
        ]
    )


_MERGE_NODES = [
    _Draft("API module", ["object"]),
    _Draft("API integration module", ["object"]),
    _Draft("Autocomplete input latency", ["object"]),
]


class TestReviewingMerges:
    def test_a_group_the_model_accepts_still_merges(self) -> None:
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((1, "same", [])),
        ):
            kept = review_merges(merges, _MERGE_NODES)

        assert kept == merges

    def test_a_name_the_model_keeps_apart_is_removed_from_the_group(self) -> None:
        merges = [
            _Merge(
                "api integration module",
                ("api module", "autocomplete input latency"),
            )
        ]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers(
                (1, "same", ["Autocomplete input latency"]),
            ),
        ):
            kept = review_merges(merges, _MERGE_NODES)

        assert kept[0].duplicates == ("api module",)

    def test_a_group_wholly_rejected_disappears(self) -> None:
        merges = [_Merge("api integration module", ("autocomplete input latency",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((1, "different", [])),
        ):
            assert review_merges(merges, _MERGE_NODES) == []

    def test_a_name_that_was_never_offered_is_ignored(self) -> None:
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((1, "same", ["Something else entirely"])),
        ):
            kept = review_merges(merges, _MERGE_NODES)

        assert kept[0].duplicates == ("api module",)

    def test_a_group_the_model_skipped_still_merges(self) -> None:
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((2, "different", [])),
        ):
            assert review_merges(merges, _MERGE_NODES) == merges

    def test_a_model_failure_applies_the_plan_as_planned(self) -> None:
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            side_effect=RuntimeError("no"),
        ):
            assert review_merges(merges, _MERGE_NODES) == merges

    def test_nothing_planned_asks_nothing(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            assert review_merges([], _MERGE_NODES) == []

        assert called.call_count == 0

    def test_the_types_reach_the_prompt(self) -> None:
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((1, "same", [])),
        ) as called:
            review_merges(merges, _MERGE_NODES)

        prompt = called.call_args.args[1][1].content
        assert "API integration module" in prompt
        assert "API module" in prompt

    def test_the_sentence_each_name_came_from_reaches_the_prompt(self) -> None:
        nodes = [
            _Draft("API module", ["object"], evidence="split out the API module"),
            _Draft("API integration module", ["object"], evidence="the integration"),
        ]
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((1, "same", [])),
        ) as called:
            review_merges(merges, nodes)

        prompt = called.call_args.args[1][1].content
        assert "split out the API module" in prompt
        assert "the integration" in prompt

    def test_a_name_with_no_sentence_is_still_shown(self) -> None:
        merges = [_Merge("api integration module", ("api module",))]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=merge_answers((1, "same", [])),
        ) as called:
            review_merges(merges, _MERGE_NODES)

        assert "said as" not in called.call_args.args[1][1].content


@dataclass
class _Derived:
    """Enough of a ``SemanticEdge`` for the derived check."""

    source_id: str
    target_id: str
    role: str


class TestCheckingDerived:
    def test_a_pair_that_is_not_a_revision_never_reaches_the_write(self) -> None:
        derived = [_Derived("a", "b", "contradicts")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "separate")),
        ):
            turned, dropped = check_derived(derived, _NODES, _TEXTS)

        assert (turned, dropped) == (0, 1)
        assert derived == []

    def test_a_pair_facing_the_wrong_way_is_turned_round(self) -> None:
        derived = [_Derived("a", "b", "supersedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "reversed")),
        ):
            turned, dropped = check_derived(derived, _NODES, _TEXTS)

        assert (turned, dropped) == (1, 0)
        assert (derived[0].source_id, derived[0].target_id) == ("b", "a")

    def test_the_real_frozen_edge_can_be_turned_round(self) -> None:
        # The stand-in above is mutable and the edge the pipeline hands over is
        # not, which is a difference a verdict of 'reversed' used to discover
        # in production and nowhere else.
        from auto_ontology.conversations.semantic.relate import SemanticEdge

        derived = [
            SemanticEdge(
                source_id="a",
                target_id="b",
                role="supersedes",
                reason=None,
                distance=None,
                derived=True,
            )
        ]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "reversed")),
        ):
            turned, dropped = check_derived(derived, _NODES, _TEXTS)

        assert (turned, dropped) == (1, 0)
        assert (derived[0].source_id, derived[0].target_id) == ("b", "a")
        assert derived[0].derived is True

    def test_a_real_revision_is_left_exactly_as_it_was(self) -> None:
        derived = [_Derived("a", "b", "supersedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "changed")),
        ):
            assert check_derived(derived, _NODES, _TEXTS) == (0, 0)

        assert (derived[0].source_id, derived[0].target_id) == ("a", "b")

    def test_the_right_edge_is_dropped_when_several_are_checked(self) -> None:
        keep = _Derived("c", "d", "supersedes")
        derived = [_Derived("a", "b", "contradicts"), keep]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "separate"), (2, "changed")),
        ):
            check_derived(derived, _NODES, _TEXTS)

        assert derived == [keep]

    def test_an_edge_to_itself_goes_without_asking_anything(self) -> None:
        derived = [_Derived("a", "a", "precedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            turned, dropped = check_derived(derived, _NODES, _TEXTS)

        assert (turned, dropped) == (0, 1)
        assert derived == []
        assert called.call_count == 0

    def test_a_precedes_against_its_own_dates_is_dropped(self) -> None:
        nodes = [
            node("x", "Later", occurred_on="2024-04-05"),
            node("y", "Earlier", occurred_on="2024-03-11"),
        ]
        derived = [_Derived("x", "y", "precedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            # Reported by the structural pass, but misdating is not grounds for
            # removal on its own: the dates may be the wrong field, not the edge.
            turned, dropped = check_derived(derived, nodes, _TEXTS)

        assert (turned, dropped) == (0, 0)
        assert called.call_count == 0

    def test_nothing_derived_asks_nothing(self) -> None:
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output"
        ) as called:
            assert check_derived([], _NODES, _TEXTS) == (0, 0)

        assert called.call_count == 0

    def test_a_model_failure_writes_every_edge_unchanged(self) -> None:
        derived = [_Derived("a", "b", "supersedes")]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            side_effect=RuntimeError("no"),
        ):
            assert check_derived(derived, _NODES, _TEXTS) == (0, 0)

        assert len(derived) == 1


class TestWhatIsDropped:
    def test_only_the_separate_pairs_are_unsound(self) -> None:
        edges = [
            edge("a", "b", "contradicts"),
            edge("c", "d", "supersedes"),
            edge("d", "c", "supersedes"),
        ]
        with patch(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            return_value=answers((1, "separate"), (2, "changed"), (3, "reversed")),
        ):
            report = audit_revisions(_NODES, edges, _TEXTS)

        assert [pair["source_id"] for pair in unsound_pairs(report)] == ["a"]

    def test_an_empty_report_drops_nothing(self) -> None:
        assert unsound_pairs(RevisionReport()) == []
