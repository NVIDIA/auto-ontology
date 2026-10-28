# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for checking edges against the turns they came from."""

from __future__ import annotations

import pytest

from auto_ontology.conversations.semantic.extract import EdgeDraft, NodeDraft
from auto_ontology.conversations.semantic.verify import (
    BATCH_EDGES,
    CitationReport,
    NodeCitation,
    VerificationReport,
    _batch,
    build_denials,
    _render_claims,
    _render_edges,
    _render_turns,
    detach_bad_citations,
    drop_unsupported,
    verify_edges,
    verify_node_citations,
)


def edge(source: str, role: str, target: str, turns: set[int]) -> EdgeDraft:
    return EdgeDraft(source=source, target=target, role=role, turn_indexes=set(turns))


def statement(name: str, turns: set[int], **kwargs) -> NodeDraft:
    return NodeDraft(
        name=name,
        type=kwargs.pop("type", "Metric"),
        kinds=kwargs.pop("kinds", ["fact"]),
        turn_indexes=set(turns),
        **kwargs,
    )


def thing(name: str, turns: set[int]) -> NodeDraft:
    return NodeDraft(
        name=name, type="Feature", kinds=["object"], turn_indexes=set(turns)
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_edges_citing_no_known_turn_are_not_checked(self):
        known = edge("a", "uses", "b", {1})
        stray = edge("c", "uses", "d", {99})
        batches, unchecked = _batch([known, stray], {1: "text"})

        assert unchecked == [stray]
        assert batches == [[known]]

    def test_an_edge_with_no_turns_at_all_is_unchecked(self):
        orphan = edge("a", "uses", "b", set())
        batches, unchecked = _batch([orphan], {1: "text"})

        assert unchecked == [orphan]
        assert batches == []

    def test_batches_are_ordered_by_first_cited_turn(self):
        late = edge("late", "uses", "x", {9})
        early = edge("early", "uses", "x", {1})
        batches, _ = _batch([late, early], {1: "one", 9: "nine"})

        assert [item.source for batch in batches for item in batch] == [
            "early",
            "late",
        ]

    def test_edge_count_caps_a_batch(self):
        edges = [edge(f"n{i}", "uses", "x", {1}) for i in range(BATCH_EDGES + 5)]
        batches, _ = _batch(edges, {1: "short"})

        assert len(batches) == 2
        assert len(batches[0]) == BATCH_EDGES
        assert len(batches[1]) == 5

    def test_text_budget_splits_long_turns(self):
        turns = {index: "x" * 5000 for index in range(1, 5)}
        edges = [edge(f"n{i}", "uses", "x", {i}) for i in range(1, 5)]
        batches, _ = _batch(edges, turns)

        assert len(batches) > 1

    def test_edges_sharing_a_turn_do_not_pay_for_it_twice(self):
        turns = {1: "x" * 7000}
        edges = [edge(f"n{i}", "uses", "x", {1}) for i in range(5)]
        batches, _ = _batch(edges, turns)

        assert len(batches) == 1

    def test_every_checkable_edge_lands_in_exactly_one_batch(self):
        turns = {index: f"turn {index}" for index in range(1, 30)}
        edges = [edge(f"n{i}", "uses", "x", {i}) for i in range(1, 30)]
        batches, unchecked = _batch(edges, turns)

        assert not unchecked
        assert sorted(item.source for b in batches for item in b) == sorted(
            item.source for item in edges
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_turns_are_labelled_and_ordered(self):
        rendered = _render_turns({3, 1}, {1: "first", 3: "third"})

        assert rendered == "[turn 1] first\n\n[turn 3] third"

    def test_edges_are_numbered_from_one(self):
        rendered = _render_edges(
            [edge("a", "uses", "b", {1}), edge("c", "builds", "d", {2})]
        )

        assert rendered.startswith("1. a --uses--> b")
        assert "2. c --builds--> d" in rendered

    def test_edges_show_the_turns_they_cite(self):
        rendered = _render_edges([edge("a", "uses", "b", {5, 2})])

        assert "(from turn 2, 5)" in rendered


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, verdicts):
        self.verdicts = verdicts


class _Item:
    def __init__(self, number, verdict, reason=""):
        self.number, self.verdict, self.reason = number, verdict, reason


@pytest.fixture
def answer(monkeypatch):
    """Make the model return a fixed verdict list, and record what it was asked."""
    asked: list[str] = []

    def install(items):
        def fake_invoke(_client, messages, _schema):
            asked.append(messages[-1].content)
            return _Result(items)

        monkeypatch.setattr(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output", fake_invoke
        )
        monkeypatch.setattr(
            "auto_ontology.utils.llm_invoke.get_llm_client", lambda **_: object()
        )
        return asked

    return install


class TestVerify:
    def test_verdicts_are_bucketed(self, answer):
        answer(
            [
                _Item(1, "supported"),
                _Item(2, "unsupported", "turns say it was removed"),
                _Item(3, "absent"),
            ]
        )
        edges = [
            edge("a", "uses", "b", {1}),
            edge("c", "uses", "d", {1}),
            edge("e", "uses", "f", {1}),
        ]

        report = verify_edges(edges, {1: "text"})

        assert [item.edge.source for item in report.supported] == ["a"]
        assert [item.edge.source for item in report.unsupported] == ["c"]
        assert [item.edge.source for item in report.absent] == ["e"]
        assert report.checked == 3

    def test_the_reason_is_kept(self, answer):
        answer([_Item(1, "unsupported", "  never deployed  ")])

        report = verify_edges([edge("a", "uses", "b", {1})], {1: "text"})

        assert report.unsupported[0].reason == "never deployed"

    def test_a_verdict_for_an_unshown_relation_is_ignored(self, answer):
        answer([_Item(7, "unsupported")])

        report = verify_edges([edge("a", "uses", "b", {1})], {1: "text"})

        assert report.checked == 0

    def test_a_relation_judged_twice_takes_the_first_verdict(self, answer):
        answer([_Item(1, "supported"), _Item(1, "unsupported")])

        report = verify_edges([edge("a", "uses", "b", {1})], {1: "text"})

        assert len(report.supported) == 1
        assert not report.unsupported

    def test_an_unjudged_relation_is_in_no_bucket(self, answer):
        answer([_Item(1, "supported")])
        edges = [edge("a", "uses", "b", {1}), edge("c", "uses", "d", {1})]

        report = verify_edges(edges, {1: "text"})

        assert report.checked == 1
        assert not report.unchecked  # it was asked about, just not answered

    def test_the_turns_are_shown_to_the_model(self, answer):
        asked = answer([_Item(1, "supported")])

        verify_edges([edge("a", "uses", "b", {4})], {4: "the real turn text"})

        assert "the real turn text" in asked[0]
        assert "a --uses--> b" in asked[0]

    def test_only_the_cited_turns_are_shown(self, answer):
        asked = answer([_Item(1, "supported")])

        verify_edges([edge("a", "uses", "b", {1})], {1: "cited", 2: "uncited"})

        assert "cited" in asked[0]
        assert "uncited" not in asked[0]

    def test_nothing_is_asked_when_no_edge_can_be_checked(self, answer):
        asked = answer([])

        report = verify_edges([edge("a", "uses", "b", {99})], {1: "text"})

        assert not asked
        assert len(report.unchecked) == 1

    def test_no_verdicts_returned_leaves_every_edge_alone(self, monkeypatch):
        monkeypatch.setattr(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            lambda *_: None,
        )
        monkeypatch.setattr(
            "auto_ontology.utils.llm_invoke.get_llm_client", lambda **_: object()
        )

        report = verify_edges([edge("a", "uses", "b", {1})], {1: "text"})

        assert report.checked == 0
        assert not report.wrong()


# ---------------------------------------------------------------------------
# Dropping
# ---------------------------------------------------------------------------


class TestDrop:
    def test_only_unsupported_edges_are_dropped(self, answer):
        answer([_Item(1, "supported"), _Item(2, "unsupported"), _Item(3, "absent")])
        edges = [
            edge("a", "uses", "b", {1}),
            edge("c", "uses", "d", {1}),
            edge("e", "uses", "f", {1}),
        ]

        kept = drop_unsupported(edges, verify_edges(edges, {1: "text"}))

        assert [item.source for item in kept] == ["a", "e"]

    def test_an_empty_report_drops_nothing(self):
        edges = [edge("a", "uses", "b", {1})]

        assert drop_unsupported(edges, VerificationReport()) == edges

    def test_identical_looking_edges_are_distinguished_drop(self, answer):
        """Two edges can compare equal and still be separate graph entries."""
        answer([_Item(1, "supported"), _Item(2, "unsupported")])
        first = edge("a", "uses", "b", {1})
        second = edge("a", "uses", "b", {1})

        report = verify_edges([first, second], {1: "text"})
        kept = drop_unsupported([first, second], report)

        assert len(kept) == 1
        assert kept[0] is first


# ---------------------------------------------------------------------------
# Node citations
# ---------------------------------------------------------------------------


class _CItem:
    def __init__(self, claim, turn, verdict, reason=""):
        self.claim, self.turn = claim, turn
        self.verdict, self.reason = verdict, reason


class TestNodeCitations:
    def test_single_turn_statements_are_not_checked(self, answer):
        asked = answer([])

        report = verify_node_citations([statement("a", {1})], {1: "text"})

        assert not asked
        assert report.checked == 0

    def test_things_are_not_checked_however_many_turns(self, answer):
        asked = answer([])

        report = verify_node_citations(
            [thing("Autocomplete", {1, 2, 3})], {1: "a", 2: "b", 3: "c"}
        )

        assert not asked
        assert report.checked == 0

    def test_a_denial_is_reported(self, answer):
        answer([_CItem(1, 16, "supports"), _CItem(1, 35, "denies", "says never")])
        node = statement("API key obtained", {16, 35})

        report = verify_node_citations([node], {16: "got the key", 35: "never got one"})

        assert [item.turn_index for item in report.denies] == [35]
        assert report.denies[0].reason == "says never"
        assert len(report.supports) == 1

    def test_unrelated_and_denies_are_both_wrong(self, answer):
        answer([_CItem(1, 1, "denies"), _CItem(1, 2, "unrelated")])
        node = statement("a", {1, 2})

        report = verify_node_citations([node], {1: "x", 2: "y"})

        assert len(report.wrong()) == 2

    def test_a_verdict_on_an_uncited_turn_is_ignored(self, answer):
        answer([_CItem(1, 99, "denies")])
        node = statement("a", {1, 2})

        report = verify_node_citations([node], {1: "x", 2: "y"})

        assert report.checked == 0

    def test_a_verdict_for_an_unshown_claim_is_ignored(self, answer):
        answer([_CItem(7, 1, "denies")])

        report = verify_node_citations([statement("a", {1, 2})], {1: "x", 2: "y"})

        assert report.checked == 0

    def test_the_claim_is_rendered_with_its_key_and_turns(self):
        rendered = _render_claims(
            [
                statement(
                    "Coverage",
                    {5, 9},
                    key="test_coverage_pct",
                    value="85%",
                    description="The suite covers 85%.",
                )
            ]
        )

        assert "[test_coverage_pct = 85%]" in rendered
        assert "The suite covers 85%." in rendered
        assert "cited from turns: 5, 9" in rendered

    def test_turns_are_shown_to_the_model(self, answer):
        asked = answer([_CItem(1, 1, "supports")])

        verify_node_citations([statement("a", {1, 2})], {1: "first body", 2: "second"})

        assert "first body" in asked[0]
        assert "second" in asked[0]


class TestDetach:
    def test_a_denied_turn_is_removed(self, answer):
        answer([_CItem(1, 16, "supports"), _CItem(1, 35, "denies")])
        node = statement("API key obtained", {16, 35})

        report = verify_node_citations([node], {16: "got it", 35: "never"})
        assert detach_bad_citations(report) == 1
        assert node.turn_indexes == {16}

    def test_the_last_citation_is_never_removed(self, answer):
        answer([_CItem(1, 1, "denies"), _CItem(1, 2, "denies")])
        node = statement("a", {1, 2})

        report = verify_node_citations([node], {1: "x", 2: "y"})
        detach_bad_citations(report)

        assert len(node.turn_indexes) == 1

    def test_a_detached_turn_loses_its_verified_mark(self, answer):
        answer([_CItem(1, 35, "denies"), _CItem(1, 16, "supports")])
        node = statement("a", {16, 35})
        node.verified_turn_indexes = {16, 35}

        detach_bad_citations(verify_node_citations([node], {16: "x", 35: "y"}))

        assert node.verified_turn_indexes == {16}

    def test_an_empty_report_detaches_nothing(self):
        assert detach_bad_citations(CitationReport()) == 0

    def test_supported_citations_are_left_alone(self, answer):
        answer([_CItem(1, 1, "supports"), _CItem(1, 2, "supports")])
        node = statement("a", {1, 2})

        detach_bad_citations(verify_node_citations([node], {1: "x", 2: "y"}))

        assert node.turn_indexes == {1, 2}


# ---------------------------------------------------------------------------
# Building the missing denial
# ---------------------------------------------------------------------------


class _Denial:
    def __init__(self, name, description="said so", evidence="", value=""):
        self.name, self.description = name, description
        self.evidence, self.value = evidence, value


TURN_35 = "I've never actually obtained an API key for this project."


def _denies(node, turn=35):
    """A report with one ``denies`` verdict against *node*."""
    report = CitationReport()
    report.denies.append(
        NodeCitation(node=node, turn_index=turn, verdict="denies", reason="")
    )
    return report


class TestBuildDenials:
    def test_a_quoted_denial_becomes_a_statement(self, answer):
        answer([])  # installs the patched client
        claim = statement("API key obtained", {16, 35}, description="Key obtained.")
        self._returns(
            _Denial("Never obtained an API key", evidence="never actually obtained")
        )

        built = build_denials(_denies(claim), {35: TURN_35})

        assert len(built) == 1
        assert built[0].name == "Never obtained an API key"
        assert built[0].turn_indexes == {35}
        assert built[0].verified_turn_indexes == {35}
        assert built[0].kinds == ["fact"]

    def test_a_quote_not_in_the_turn_is_discarded(self, answer):
        answer([])
        self._returns(_Denial("Never obtained a key", evidence="I threw the key away"))

        built = build_denials(
            _denies(statement("API key obtained", {16, 35})), {35: TURN_35}
        )

        assert built == []

    def test_an_empty_quote_is_discarded(self, answer):
        answer([])
        self._returns(_Denial("Never obtained a key", evidence="  "))

        built = build_denials(
            _denies(statement("API key obtained", {16, 35})), {35: TURN_35}
        )

        assert built == []

    def test_reusing_the_claims_name_is_discarded(self, answer):
        answer([])
        self._returns(_Denial("API key obtained", evidence="never actually obtained"))

        built = build_denials(
            _denies(statement("API key obtained", {16, 35})), {35: TURN_35}
        )

        assert built == []

    def test_the_claims_key_is_reused_with_the_opposite_value(self, answer):
        answer([])
        claim = statement(
            "API key obtained", {16, 35}, key="api_key_obtained", value="yes"
        )
        self._returns(
            _Denial("Never obtained one", evidence="never actually", value="no")
        )

        built = build_denials(_denies(claim), {35: TURN_35})

        assert built[0].key == "api_key_obtained"
        assert built[0].value == "no"

    def test_a_value_is_dropped_when_the_claim_has_no_key(self, answer):
        answer([])
        self._returns(
            _Denial("Never obtained one", evidence="never actually", value="no")
        )

        built = build_denials(
            _denies(statement("API key obtained", {16, 35})), {35: TURN_35}
        )

        assert built[0].key == ""
        assert built[0].value == ""

    def test_a_missing_turn_is_skipped(self, answer):
        answer([])
        self._returns(_Denial("Never obtained one", evidence="never actually"))

        built = build_denials(_denies(statement("a", {16, 35})), {16: "other"})

        assert built == []

    def test_no_answer_builds_nothing(self, monkeypatch):
        monkeypatch.setattr(
            "auto_ontology.utils.llm_invoke.invoke_with_structured_output",
            lambda *_: None,
        )
        monkeypatch.setattr(
            "auto_ontology.utils.llm_invoke.get_llm_client", lambda **_: object()
        )

        built = build_denials(_denies(statement("a", {16, 35})), {35: TURN_35})

        assert built == []

    def test_an_empty_report_builds_nothing(self, answer):
        answer([])

        assert build_denials(CitationReport(), {35: TURN_35}) == []

    def _returns(self, item):
        import auto_ontology.utils.llm_invoke as llm

        llm.invoke_with_structured_output = lambda *_: item
