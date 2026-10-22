# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The subject pass: what it offers, what it accepts, what it drops."""

from __future__ import annotations

from auto_ontology.conversations.semantic.extract import (
    EdgeDraft,
    GraphDraft,
    NodeDraft,
    normalise_name,
)
from auto_ontology.conversations.semantic.subjects import (
    SubjectChoice,
    SubjectChoices,
    _accept,
    _batch,
    _drop_drafted_subjects,
    assign_subjects,
)


def thing(name: str, *, turn: int = 0) -> NodeDraft:
    return NodeDraft(
        name=name, type="Application", kinds=["object"], turn_indexes={turn}
    )


def statement(name: str, *, turn: int = 0, kinds: list[str] | None = None) -> NodeDraft:
    return NodeDraft(
        name=name, type="Measurement", kinds=kinds or ["state"], turn_indexes={turn}
    )


def edge(source: str, target: str, role: str) -> EdgeDraft:
    """A draft edge as extraction builds one: both endpoints normalised.

    Spelled out here because constructing one with raw names is the bug this
    file exists to pin down -- it cost a build 334 subject edges that were
    assigned, reported, and never written.
    """
    return EdgeDraft(
        source=normalise_name(source), target=normalise_name(target), role=role
    )


class TestDroppingDraftedSubjects:
    def test_a_statements_subject_edge_goes(self) -> None:
        graph = GraphDraft(
            nodes=[statement("Latency 220ms"), thing("Weather app")],
            edges=[edge("Latency 220ms", "Weather app", "about")],
        )
        assert _drop_drafted_subjects(graph) == 1
        assert graph.edges == []

    def test_a_statement_about_a_statement_goes(self) -> None:
        graph = GraphDraft(
            nodes=[statement("Latency 220ms"), statement("Optimize fetch latency")],
            edges=[
                edge("Latency 220ms", "Optimize fetch latency", "measured_by"),
            ],
        )
        assert _drop_drafted_subjects(graph) == 1
        assert graph.edges == []

    def test_a_relation_between_two_things_stays(self) -> None:
        """``about`` is only replaced for statements -- two things relating is not
        a question this pass answers."""
        graph = GraphDraft(
            nodes=[thing("Weather app"), thing("OpenWeather API")],
            edges=[edge("Weather app", "OpenWeather API", "about")],
        )
        assert _drop_drafted_subjects(graph) == 0
        assert len(graph.edges) == 1

    def test_a_non_subject_role_stays(self) -> None:
        graph = GraphDraft(
            nodes=[statement("Switched to fetch"), thing("Weather app")],
            edges=[
                EdgeDraft(
                    source="Switched to fetch", target="Weather app", role="covers"
                )
            ],
        )
        assert _drop_drafted_subjects(graph) == 0


class TestEdgeEndpoints:
    """The edges the pass appends have to be written, which means normalised ends.

    ``replace_graph`` resolves an edge's ends through a map keyed by normalised
    name and skips an unresolvable one without inserting it. Measured once: 334
    subjects assigned, 334 reported written, 0 in the table. The names were all
    correct and all in the wrong form.
    """

    def _assign(self, monkeypatch, choices: list[SubjectChoice]) -> GraphDraft:
        from auto_ontology.utils import llm_invoke

        graph = GraphDraft(
            nodes=[statement("Daily Quota 1,200/day"), thing("Weather App")],
            edges=[],
        )
        monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_: object())
        monkeypatch.setattr(
            llm_invoke,
            "invoke_with_structured_output",
            lambda *_, **__: SubjectChoices(choices=choices),
        )
        assign_subjects(
            graph,
            [{"turn_index": 0, "question": "how fast?", "answer": "220ms"}],
        )
        return graph

    def test_the_appended_edge_resolves_the_way_the_write_looks_it_up(
        self, monkeypatch
    ) -> None:
        graph = self._assign(
            monkeypatch, [SubjectChoice(statement="s1", thing="Weather App")]
        )

        assert len(graph.edges) == 1
        appended = graph.edges[0]
        assert appended.role == "about"
        # The form the write's id map is keyed by, which is what was wrong.
        assert appended.source == normalise_name("Daily Quota 1,200/day")
        assert appended.target == normalise_name("Weather App")

    def test_the_endpoints_are_not_the_raw_names(self, monkeypatch) -> None:
        """Stated separately so the assertion above cannot pass by the two forms
        happening to be equal for the name chosen."""
        graph = self._assign(
            monkeypatch, [SubjectChoice(statement="s1", thing="Weather App")]
        )
        appended = graph.edges[0]

        assert appended.source != "Daily Quota 1,200/day"
        assert appended.target != "Weather App"

    def test_the_turns_come_from_the_statement(self, monkeypatch) -> None:
        graph = self._assign(
            monkeypatch, [SubjectChoice(statement="s1", thing="Weather App")]
        )
        assert graph.edges[0].turn_indexes == {0}

    def test_a_rejected_answer_appends_nothing(self, monkeypatch) -> None:
        graph = self._assign(
            monkeypatch, [SubjectChoice(statement="s1", thing="Nonexistent thing")]
        )
        assert graph.edges == []

    def test_dropping_matches_normalised_sources(self) -> None:
        """``_drop_drafted_subjects`` compares against edge endpoints, which are
        normalised, so comparing raw names would drop nothing at all."""
        graph = GraphDraft(
            nodes=[statement("Latency 220ms"), thing("Weather App")],
            edges=[edge("Latency 220ms", "Weather App", "about")],
        )
        assert _drop_drafted_subjects(graph) == 1


class TestBatching:
    def test_statements_are_grouped_by_their_earliest_turn(self) -> None:
        turn_text = {i: f"turn {i}" for i in range(25)}
        statements = [statement(f"s{i}", turn=i) for i in range(0, 25, 5)]
        batches = _batch(statements, turn_text)

        assert [indexes[0] for indexes, _ in batches] == [0, 10, 20]
        assert [[node.name for node in members] for _, members in batches] == [
            ["s0", "s5"],
            ["s10", "s15"],
            ["s20"],
        ]

    def test_a_statement_spanning_turns_is_asked_about_once(self) -> None:
        """Keyed on the earliest turn, so a node accumulated across two batches'
        worth of turns does not get two answers to reconcile."""
        turn_text = {i: f"turn {i}" for i in range(20)}
        node = statement("Latency", turn=2)
        node.turn_indexes = {2, 15}
        batches = _batch([node], turn_text)

        assert sum(len(members) for _, members in batches) == 1
        assert batches[0][0][0] == 0

    def test_turns_with_no_statements_make_no_call(self) -> None:
        turn_text = {i: f"turn {i}" for i in range(30)}
        batches = _batch([statement("only one", turn=25)], turn_text)
        assert len(batches) == 1


class TestAcceptance:
    def setup_method(self) -> None:
        self.things = [thing("Weather app"), thing("OpenWeather API")]
        self.statements = [statement("Latency 220ms"), statement("Quota 1200/day")]

    def test_a_valid_pair_resolves_to_names(self) -> None:
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="n1")], self.things, self.statements
        )
        assert resolved == {"Latency 220ms": "Weather app"}

    def test_a_thing_named_instead_of_labelled_resolves(self) -> None:
        """The first live run answered 412 of 412 this way -- `s1 -> Weather app`
        rather than `s1 -> n1` -- and a label-only lookup discarded all of them."""
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="Weather app")],
            self.things,
            self.statements,
        )
        assert resolved == {"Latency 220ms": "Weather app"}

    def test_both_sides_named_instead_of_labelled_resolves(self) -> None:
        resolved = _accept(
            [SubjectChoice(statement="Latency 220ms", thing="Weather app")],
            self.things,
            self.statements,
        )
        assert resolved == {"Latency 220ms": "Weather app"}

    def test_a_thing_name_not_offered_is_still_dropped(self) -> None:
        """Accepting names costs nothing as long as the list stays closed: that,
        not the labelling, is what makes an invented subject detectable."""
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="Weather application")],
            self.things,
            self.statements,
        )
        assert resolved == {}

    def test_a_statement_named_as_the_subject_is_dropped(self) -> None:
        """Names resolve on both sides, so the roles have to be enforced apart."""
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="Quota 1200/day")],
            self.things,
            self.statements,
        )
        assert resolved == {}

    def test_a_thing_label_not_offered_is_dropped(self) -> None:
        """The whole point of answering in labels: an invented subject is
        detectable, where an invented name would not be."""
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="n9")], self.things, self.statements
        )
        assert resolved == {}

    def test_a_statement_label_not_in_this_batch_is_dropped(self) -> None:
        resolved = _accept(
            [SubjectChoice(statement="s7", thing="n1")], self.things, self.statements
        )
        assert resolved == {}

    def test_a_statement_label_used_as_a_subject_is_dropped(self) -> None:
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="s2")], self.things, self.statements
        )
        assert resolved == {}

    def test_a_statement_answered_twice_keeps_neither(self) -> None:
        resolved = _accept(
            [
                SubjectChoice(statement="s1", thing="n1"),
                SubjectChoice(statement="s1", thing="n2"),
            ],
            self.things,
            self.statements,
        )
        assert resolved == {}

    def test_one_bad_choice_does_not_cost_the_good_ones(self) -> None:
        resolved = _accept(
            [
                SubjectChoice(statement="s1", thing="n1"),
                SubjectChoice(statement="s2", thing="n42"),
            ],
            self.things,
            self.statements,
        )
        assert resolved == {"Latency 220ms": "Weather app"}


class TestAnswerWording:
    """However the model words a choice, it resolves to the node it names."""

    things = [
        NodeDraft(
            name="Minimum character threshold for autocomplete (3 chars)",
            type="Feature",
            kinds=("object",),
        )
    ]
    statements = [
        NodeDraft(
            name="User retrying after hitting rate limit concern",
            type="Fact",
            kinds=("fact",),
        )
    ]

    def test_whole_rendered_line_resolves(self) -> None:
        """The wording that dropped 42 subjects of 42 in one build."""
        resolved = _accept(
            [
                SubjectChoice(
                    statement="s1 | User retrying after hitting rate limit concern",
                    thing=(
                        "n1 | Minimum character threshold for autocomplete (3 chars)"
                    ),
                )
            ],
            self.things,
            self.statements,
        )
        assert resolved == {
            "User retrying after hitting rate limit concern": (
                "Minimum character threshold for autocomplete (3 chars)"
            )
        }

    def test_name_with_the_type_appended_resolves(self) -> None:
        resolved = _accept(
            [
                SubjectChoice(
                    statement="s1",
                    thing=(
                        "Minimum character threshold for autocomplete "
                        "(3 chars) (Feature)"
                    ),
                )
            ],
            self.things,
            self.statements,
        )
        assert len(resolved) == 1

    def test_bare_label_and_bare_name_still_resolve(self) -> None:
        for thing in ("n1", "Minimum character threshold for autocomplete (3 chars)"):
            resolved = _accept(
                [SubjectChoice(statement="s1", thing=thing)],
                self.things,
                self.statements,
            )
            assert len(resolved) == 1, thing

    def test_an_invented_thing_is_still_refused(self) -> None:
        """The wording is widened; the vocabulary is not."""
        resolved = _accept(
            [SubjectChoice(statement="s1", thing="n9 | Redis cache")],
            self.things,
            self.statements,
        )
        assert resolved == {}
