# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the conversation graph: windows, kinds, derivation, rendering.

The LLM calls are not exercised here. What is under test is everything around
them -- which turn a quote resolves to, which nodes collapse into one, which
edges follow from the rows without asking anything, and how a retrieved
subgraph reads.
"""

from __future__ import annotations

import threading

from datetime import date

import pytest

from auto_ontology.conversations.semantic.derive import derive_edges
from auto_ontology.conversations.semantic.extract import (
    NODE_KINDS,
    STATEMENT_KINDS,
    NodeItem,
    _clean_kinds,
    _node_key,
    _pair_key_and_value,
    node_text,
    normalise_name,
    normalise_role,
    parse_date,
)
from auto_ontology.conversations.semantic.keys import (
    KeyGroup,
    _accept,
    _candidates,
    _valued,
)
from auto_ontology.conversations.semantic.note import note_text
from auto_ontology.conversations.semantic.relate import (
    ROLES,
    PairVerdict,
    _not_a_revision,
    _orient,
    candidate_pairs,
)
from auto_ontology.conversations.semantic.windows import build_windows

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _turn(
    index: int, question: str = "q", answer: str = "a", anchor: str | None = None
):
    return {
        "turn_index": index,
        "question": question,
        "answer": answer,
        "time_anchor": anchor,
    }


def test_a_turn_is_never_split_across_windows() -> None:
    """Half a turn is not a usable citation."""
    windows = build_windows([_turn(0, "x" * 500), _turn(1, "y" * 500)], max_chars=600)

    assert [w.turn_indexes for w in windows] == [[0], [1]]


def test_an_oversized_turn_becomes_its_own_window() -> None:
    windows = build_windows([_turn(0, "x" * 5000)], max_chars=100)

    assert [w.turn_indexes for w in windows] == [[0]]


def test_the_date_is_on_the_label_not_left_in_the_turn() -> None:
    """Most turns carry no date of their own, so the label has to."""
    window = build_windows([_turn(7, anchor="March-15-2024")])[0]

    assert "[turn 7, March-15-2024]" in window.labelled_text


def test_a_quote_resolves_to_its_turn() -> None:
    window = build_windows(
        [_turn(3, "the quota is 1000 calls per day"), _turn(4, "unrelated text here")]
    )[0]

    assert window.find_quote("quota is 1000 calls per day") == 3


def test_an_ambiguous_quote_resolves_to_nothing() -> None:
    """A false citation is worse than a missing one."""
    window = build_windows(
        [_turn(3, "the same sentence"), _turn(4, "the same sentence")]
    )[0]

    assert window.find_quote("the same sentence") is None


def test_a_turn_outside_the_window_does_not_resolve() -> None:
    """A model citing a turn it was not shown invented the number."""
    window = build_windows([_turn(3), _turn(4)])[0]

    assert window.resolve(4) == 4
    assert window.resolve(99) is None


# ---------------------------------------------------------------------------
# Kinds and identity
# ---------------------------------------------------------------------------


def test_object_is_exclusive_of_the_statement_kinds() -> None:
    """A node the model marked both is one it could not decide about."""
    assert _clean_kinds(["object", "state"]) == ["state"]
    assert _clean_kinds(["object"]) == ["object"]


def test_a_node_can_be_several_statement_kinds_at_once() -> None:
    """A deadline has a date and moved; forcing one label picks which question
    it is allowed to answer."""
    assert _clean_kinds(["event", "state"]) == ["event", "state"]


def test_unknown_kinds_are_dropped() -> None:
    assert _clean_kinds(["nonsense"]) == []
    assert _clean_kinds(["nonsense", "fact"]) == ["fact"]


def test_every_statement_kind_is_a_node_kind() -> None:
    assert set(STATEMENT_KINDS) < set(NODE_KINDS)


def _item(**kwargs) -> NodeItem:
    base = {"name": "n", "type": "Thing", "kinds": ["fact"]}
    return NodeItem(**{**base, **kwargs})


def test_two_readings_of_one_property_stay_two_nodes() -> None:
    """Keying a statement by property alone makes a revision unrecoverable."""
    first = _item(kinds=["state"], key="daily_quota", value="1000 calls/day")
    second = _item(kinds=["state"], key="daily_quota", value="1200 calls/day")

    assert _node_key(first, ["state"]) != _node_key(second, ["state"])


def test_one_property_stated_twice_collapses() -> None:
    first = _item(name="Quota", kinds=["state"], key="daily_quota", value="1200/day")
    second = _item(
        name="Daily quota", kinds=["state"], key="daily_quota", value="1200/day"
    )

    assert _node_key(first, ["state"]) == _node_key(second, ["state"])


def test_a_key_without_a_value_is_dropped_and_the_node_keys_by_name() -> None:
    """An unreusable key is worse than none: it also splits one node in two.

    ``feature_city_not_found`` names this node rather than a property, so a
    second mention under a drifted label would be a second row.
    """
    first = _item(name="City not found handling", key="feature_city_not_found")
    second = _item(name="City not found handling", key="error_city_not_found")
    assert _node_key(first, ["fact"]) != _node_key(second, ["fact"])

    for item in (first, second):
        _pair_key_and_value(item)

    assert (first.key, first.value) == ("", "")
    assert _node_key(first, ["fact"]) == _node_key(second, ["fact"])


def test_a_value_without_a_key_is_dropped() -> None:
    """A reading of an unnamed property is comparable with nothing either."""
    item = _item(kinds=["event"], value="2024-03-10")
    _pair_key_and_value(item)

    assert (item.key, item.value) == ("", "")


def test_a_paired_key_and_value_survives() -> None:
    item = _item(kinds=["state"], key="test_coverage_current", value="85%")
    _pair_key_and_value(item)

    assert (item.key, item.value) == ("test_coverage_current", "85%")


class TestKeyCandidates:
    """Which readings are offered for key reconciliation, and which never are."""

    @staticmethod
    def _row(**kwargs) -> dict:
        base = {
            "attribute_key": "latency",
            "value": "220ms",
            "kinds": ["state"],
            "about": ["Weather app"],
        }
        return {**base, **kwargs}

    def test_a_reading_is_offered(self) -> None:
        assert _valued([self._row()])

    def test_a_bound_is_not_a_reading(self) -> None:
        """ "below 250ms" is where the figure was asked to get to.

        Merged into the key of the figure itself, the distance between a goal
        and a measurement reads as the measurement moving.
        """
        for value in ("below 250ms", "under 2.5MB", "<300ms", "at least 99.9%"):
            assert not _valued([self._row(value=value)]), value

    def test_a_goal_is_not_offered_whatever_its_value(self) -> None:
        assert not _valued([self._row(kinds=["target"])])

    def test_an_unvalued_statement_is_not_offered(self) -> None:
        assert not _valued([self._row(value="")])

    def test_one_subject_with_one_key_is_not_worth_asking_about(self) -> None:
        rows = [self._row(), self._row(value="250ms")]

        assert _candidates(rows) == {}

    def test_two_keys_on_one_subject_are_asked_about(self) -> None:
        rows = [self._row(), self._row(attribute_key="fetch_latency", value="250ms")]

        assert sorted(_candidates(rows)) == ["Weather app"]

    def test_keys_split_across_subjects_are_not_asked_about(self) -> None:
        """Two subjects with one key each is not a question about either."""
        rows = [
            self._row(),
            self._row(attribute_key="fetch_latency", about=["Autocomplete feature"]),
        ]

        assert _candidates(rows) == {}


class TestKeyAcceptance:
    """What the model may come back with, and what is refused."""

    CANDIDATES = {
        "Weather app": [
            {"attribute_key": "fetch_latency", "turn_indexes": [19]},
            {"attribute_key": "api_response_time", "turn_indexes": [75]},
        ],
        "OpenWeather API": [{"attribute_key": "daily_quota", "turn_indexes": [33]}],
    }

    def test_the_earliest_stated_key_becomes_canonical(self) -> None:
        """So the chain reads from the name the transcript used first."""
        groups = [KeyGroup(keys=["api_response_time", "fetch_latency"])]

        assert _accept(groups, self.CANDIDATES) == {
            "api_response_time": "fetch_latency"
        }

    def test_a_group_spanning_two_subjects_is_refused(self) -> None:
        groups = [KeyGroup(keys=["fetch_latency", "daily_quota"])]

        assert _accept(groups, self.CANDIDATES) == {}

    def test_a_key_nobody_offered_is_refused(self) -> None:
        groups = [KeyGroup(keys=["fetch_latency", "invented_key"])]

        assert _accept(groups, self.CANDIDATES) == {}

    def test_a_key_claimed_twice_is_refused_the_second_time(self) -> None:
        groups = [
            KeyGroup(keys=["fetch_latency", "api_response_time"]),
            KeyGroup(keys=["fetch_latency", "daily_quota"]),
        ]

        assert _accept(groups, self.CANDIDATES) == {
            "api_response_time": "fetch_latency"
        }

    def test_a_group_of_one_is_not_a_group(self) -> None:
        assert _accept([KeyGroup(keys=["fetch_latency"])], self.CANDIDATES) == {}


def test_a_thing_is_identified_by_its_name_alone() -> None:
    assert _node_key(_item(name="Flask", kinds=["object"]), ["object"]) == (
        "flask",
        "",
        "",
    )


def test_names_normalise_across_spelling() -> None:
    assert normalise_name("OpenWeather API!") == normalise_name("openweather  api")


def test_roles_normalise_to_snake_case() -> None:
    assert normalise_role("built With") == "built_with"


def test_node_text_is_one_definition_for_merge_and_for_search() -> None:
    assert node_text("Quota", "1,200/day") == "Quota: 1,200/day"
    assert node_text("Quota", None) == "Quota: "


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-03-10", date(2024, 3, 10)),
        ("March-10-2024", date(2024, 3, 10)),
        ("obtained on April 9, 2024", date(2024, 4, 9)),
        ("Sept 1 2024", date(2024, 9, 1)),
        ("", None),
        (None, None),
        ("sometime last spring", None),
        ("2024-13-45", None),
    ],
)
def test_dates_parse_from_both_forms(value, expected) -> None:
    """The extractor is asked for ISO; BEAM's own anchors are March-15-2024."""
    assert parse_date(value) == expected


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _statement(row_id: str, **kwargs) -> dict:
    base = {
        "id": row_id,
        "name": row_id,
        "kinds": ["state"],
        "description": row_id,
        "attribute_key": None,
        "value": None,
        "turn_indexes": [],
        "occurred_on": None,
    }
    return {**base, **kwargs}


def test_a_revised_value_is_derived_without_a_model() -> None:
    """Two readings of one key: later supersedes earlier, by turn number."""
    rows = [
        _statement(
            "a", attribute_key="daily_quota", value="1000/day", turn_indexes=[12]
        ),
        _statement(
            "b", attribute_key="daily_quota", value="1200/day", turn_indexes=[48]
        ),
    ]

    edges, decided = derive_edges(rows)

    assert [(e.source_id, e.role, e.target_id) for e in edges] == [
        ("b", "supersedes", "a")
    ]
    assert decided == {frozenset(("a", "b"))}
    assert edges[0].derived is True


def test_one_value_stated_twice_is_not_a_revision() -> None:
    rows = [
        _statement(
            "a", attribute_key="daily_quota", value="1200/day", turn_indexes=[12]
        ),
        _statement(
            "b", attribute_key="daily_quota", value="1200/day", turn_indexes=[48]
        ),
    ]

    assert derive_edges(rows)[0] == []


def test_two_values_in_one_turn_contradict_rather_than_supersede() -> None:
    """Turn order is the only thing that makes one of them later."""
    rows = [
        _statement("a", attribute_key="coverage", value="78%", turn_indexes=[20]),
        _statement("b", attribute_key="coverage", value="85%", turn_indexes=[20]),
    ]

    edges, _ = derive_edges(rows)

    assert [e.role for e in edges] == ["contradicts"]


def test_three_readings_chain_rather_than_pair_up() -> None:
    """ "Third supersedes first" would read as the first being replaced twice."""
    rows = [
        _statement("a", attribute_key="q", value="1", turn_indexes=[1]),
        _statement("b", attribute_key="q", value="2", turn_indexes=[2]),
        _statement("c", attribute_key="q", value="3", turn_indexes=[3]),
    ]

    edges, _ = derive_edges(rows)

    assert {(e.source_id, e.target_id) for e in edges} == {("b", "a"), ("c", "b")}


def test_dated_events_order_by_date_not_by_turn() -> None:
    """A transcript often mentions two events in the opposite order to the one
    they occurred in."""
    rows = [
        _statement(
            "wireframe",
            kinds=["event"],
            turn_indexes=[21],
            occurred_on=date(2024, 3, 12),
        ),
        _statement(
            "key", kinds=["event"], turn_indexes=[16], occurred_on=date(2024, 3, 10)
        ),
    ]

    edges, _ = derive_edges(rows)

    assert [(e.source_id, e.role, e.target_id) for e in edges] == [
        ("key", "precedes", "wireframe")
    ]


def test_the_interval_is_computed_onto_the_edge() -> None:
    """ "How many days between" is the question the edge exists for."""
    rows = [
        _statement("a", kinds=["event"], occurred_on=date(2024, 3, 10)),
        _statement("b", kinds=["event"], occurred_on=date(2024, 3, 12)),
    ]

    edges, _ = derive_edges(rows)

    assert "2 day(s)" in (edges[0].reason or "")


def test_events_on_one_day_get_no_ordering() -> None:
    rows = [
        _statement("a", kinds=["event"], occurred_on=date(2024, 3, 10)),
        _statement("b", kinds=["event"], occurred_on=date(2024, 3, 10)),
    ]

    assert derive_edges(rows)[0] == []


def test_undated_events_are_left_for_adjudication() -> None:
    rows = [
        _statement("a", kinds=["event"], turn_indexes=[1]),
        _statement("b", kinds=["event"], turn_indexes=[2]),
    ]

    assert derive_edges(rows)[0] == []


def test_every_event_pair_gets_an_edge_not_only_consecutive_ones() -> None:
    """An interval question names two events that need not be adjacent, and the
    prompt only renders edges whose both ends were retrieved."""
    rows = [
        _statement("a", kinds=["event"], occurred_on=date(2024, 3, 1)),
        _statement("b", kinds=["event"], occurred_on=date(2024, 3, 2)),
        _statement("c", kinds=["event"], occurred_on=date(2024, 3, 3)),
    ]

    edges, _ = derive_edges(rows)

    assert {(e.source_id, e.target_id) for e in edges} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }


# ---------------------------------------------------------------------------
# Candidate proposal and orientation
# ---------------------------------------------------------------------------


def test_event_pairs_are_proposed_regardless_of_similarity() -> None:
    """ "Obtained an API key" and "completed the wireframes" share no vocabulary
    yet precedes between them is what an interval question needs."""
    rows = [
        {"id": "a", "kinds": ["event"], "vector": [1.0, 0.0]},
        {"id": "b", "kinds": ["event"], "vector": [0.0, 1.0]},
    ]

    assert len(candidate_pairs(rows)) == 1


def test_distant_non_events_are_not_proposed() -> None:
    rows = [
        {"id": "a", "kinds": ["fact"], "vector": [1.0, 0.0]},
        {"id": "b", "kinds": ["fact"], "vector": [0.0, 1.0]},
    ]

    assert candidate_pairs(rows) == []


def test_candidates_come_back_closest_first() -> None:
    """A transcript hitting the cap should spend it on the closest pairs."""
    rows = [
        {"id": "a", "kinds": ["fact"], "vector": [1.0, 0.0]},
        {"id": "b", "kinds": ["fact"], "vector": [0.99, 0.14]},
        {"id": "c", "kinds": ["fact"], "vector": [0.8, 0.6]},
    ]

    scores = [score for _, _, score in candidate_pairs(rows)]

    assert scores == sorted(scores, reverse=True)


def test_supersedes_falls_back_to_the_later_turn_when_no_side_was_given() -> None:
    """Presentation order would make the direction depend on extraction order."""
    left = {"id": "a", "turn_indexes": [12]}
    right = {"id": "b", "turn_indexes": [48]}

    source, target = _orient(PairVerdict(role="supersedes", source=None), left, right)

    assert (source["id"], target["id"]) == ("b", "a")


def test_precedes_falls_back_to_the_earlier_turn() -> None:
    left = {"id": "a", "turn_indexes": [48]}
    right = {"id": "b", "turn_indexes": [12]}

    source, target = _orient(PairVerdict(role="precedes", source=None), left, right)

    assert (source["id"], target["id"]) == ("b", "a")


def test_an_explicit_side_wins_over_the_turn_numbers() -> None:
    left = {"id": "a", "turn_indexes": [12]}
    right = {"id": "b", "turn_indexes": [48]}

    source, _ = _orient(PairVerdict(role="supersedes", source="A"), left, right)

    assert source["id"] == "a"


def test_a_revision_needs_both_sides_to_be_about_the_same_thing() -> None:
    """The adjudicator ruled the app's 220ms a revision of the autocomplete's
    280ms, and the note then told the reader 280ms was no longer true."""
    app = {"about": ["Weather app"], "value": "220ms"}
    autocomplete = {"about": ["Autocomplete feature"], "value": "280ms"}

    assert "different subjects" in (_not_a_revision(app, autocomplete) or "")


def test_a_shared_subject_is_enough_when_a_statement_has_several() -> None:
    """A latency measured on a feature *and* in a browser keeps both."""
    left = {"about": ["Autocomplete feature", "Chrome v112"], "value": "520ms"}
    right = {"about": ["Autocomplete feature"], "value": "290ms"}

    assert _not_a_revision(left, right) is None


def test_an_unrecorded_subject_does_not_block_a_revision() -> None:
    """Subjects come off the knowledge edges, so a statement the extractor left
    unattached would otherwise be exempt from every revision it belongs to."""
    left: dict = {"value": "1%"}
    right = {"about": ["Weather app"], "value": "2.9%"}

    assert _not_a_revision(left, right) is None


def test_the_same_value_on_both_sides_is_a_duplicate_not_a_revision() -> None:
    """One 220ms reading keyed twice. Nothing was revised, so there is nothing
    for a reader to prefer."""
    left = {"about": ["Weather app"], "value": "220ms"}
    right = {"about": ["Weather app"], "value": "220ms "}

    assert "the same value" in (_not_a_revision(left, right) or "")


def test_a_statement_with_no_value_is_still_revisable() -> None:
    """Only properties carry a value; a plan can still be superseded."""
    left = {"about": ["Weather app"], "value": None}
    right = {"about": ["Weather app"], "value": None}

    assert _not_a_revision(left, right) is None


def test_unrelated_is_a_role_the_adjudicator_can_reach() -> None:
    """It is most of the answer, and storing it stops the next pass paying to
    reach it again."""
    assert "unrelated" in ROLES


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _node(name: str, kinds: list[str], description: str, turns: list[int]) -> dict:
    return {
        "id": name,
        "name": name,
        "type": "Thing",
        "kinds": kinds,
        "description": description,
        "turn_indexes": turns,
    }


class TestSittings:
    """Which sitting a line was said in, for a question phrased in sittings."""

    LAYOUT = {("2", 3): 0, ("2", 4): 0, ("2", 40): 1, ("2", 80): 2}

    def _note(self, turns: list[int], layout: dict | None = None) -> str:
        node = _node("Quota", ["fact"], "the quota is 1,200/day", turns)
        node["conversation_id"] = "2"
        return note_text(
            [node], [], session_index_by_turn=self.LAYOUT if layout is None else layout
        )

    def test_a_statement_names_its_sitting_one_based(self) -> None:
        """One-based, as "of 3" implies, and as the turn rows already render it."""
        assert "session 2 of 3" in self._note([40])

    def test_a_statement_spanning_two_sittings_names_both(self) -> None:
        assert "sessions 1 and 3 of 3" in self._note([3, 80])

    def test_a_statement_inside_one_sitting_is_not_listed_twice(self) -> None:
        assert "session 1 of 3" in self._note([3, 4])

    def test_a_node_spanning_every_sitting_says_nothing(self) -> None:
        """That the subject ran throughout rules nothing out, so it is not worth
        the context."""
        assert "session" not in self._note([3, 40, 80])

    def test_a_single_sitting_transcript_says_nothing(self) -> None:
        assert "session" not in self._note([3], {("2", 3): 0})

    def test_no_layout_renders_as_before(self) -> None:
        """A caller rendering rows it already holds should not need a database."""
        assert "session" not in note_text(
            [_node("Quota", ["fact"], "the quota is 1,200/day", [40])], []
        )

    def test_another_transcripts_turn_five_is_not_this_ones(self) -> None:
        """Turn numbering is per-transcript and twenty share a database."""
        node = _node("Quota", ["fact"], "a quota", [40])
        node["conversation_id"] = "7"
        assert "session" not in note_text([node], [], session_index_by_turn=self.LAYOUT)


def test_nothing_retrieved_renders_nothing() -> None:
    """A heading over an empty list tells a model the graph found nothing."""
    assert note_text([], []) == ""


def test_things_are_rendered_apart_from_what_was_said_about_them() -> None:
    note = note_text(
        [
            _node("Flask", ["object"], "a Python web framework", [3]),
            _node("Stack", ["fact"], "the project uses vanilla JS", [22]),
        ],
        [],
    )

    assert "Things this conversation is about:" in note
    assert "Flask (Thing) -- a Python web framework (turn 3)" in note
    assert "[fact] the project uses vanilla JS (turn 22)" in note
    # The thing is not folded into the list of things said.
    assert "[object]" not in note


def test_knowledge_edges_render_as_arrows() -> None:
    """The role is the content of the edge, so it is shown rather than glossed."""
    note = note_text(
        [_node("Project", ["object"], "", [1]), _node("Flask", ["object"], "", [2])],
        [
            {
                "type": "knowledge",
                "role": "built_with",
                "source_name": "Project",
                "target_name": "Flask",
                "turn_indexes": [2],
            }
        ],
    )

    assert "Project --built_with--> Flask (turn 2)" in note


def test_a_revision_renders_as_an_instruction_not_as_a_relation_name() -> None:
    """ "A supersedes B" needs the reader to know what the word means here."""
    note = note_text(
        [],
        [
            {
                "type": "semantic",
                "role": "supersedes",
                "source_description": "the quota is 1,200/day",
                "source_turns": [48],
                "target_description": "the quota is 1,000/day",
                "target_turns": [12],
            }
        ],
    )

    assert "no longer true" in note
    assert "the quota is 1,200/day (turn 48)" in note
    assert "rather than: the quota is 1,000/day (turn 12)" in note
    assert "supersedes" not in note


def test_an_ordering_does_not_read_as_a_correction() -> None:
    """ "A rather than B" under precedes tells a model to prefer the earlier
    event, when the point is that both happened."""
    note = note_text(
        [],
        [
            {
                "type": "semantic",
                "role": "precedes",
                "source_description": "got the key",
                "target_description": "finished the wireframes",
            }
        ],
    )

    assert "and then: finished the wireframes" in note
    # The connector form, so the closing line's prose does not collide.
    assert "rather than:" not in note


def test_an_unresolved_contradiction_says_not_to_pick_one() -> None:
    """Measured: demanding a verdict here cost contradiction resolution
    0.938 -> 0.438."""
    note = note_text(
        [],
        [
            {
                "type": "semantic",
                "role": "contradicts",
                "source_description": "feature-complete April 9",
                "target_description": "deadline April 5",
            }
        ],
    )

    assert "never resolved" in note
    assert "but also: deadline April 5" in note


def test_a_revision_is_rendered_before_an_ordering() -> None:
    """A superseded value produces a confidently wrong answer; an ordering only
    changes how the answer reads."""
    note = note_text(
        [],
        [
            {"type": "semantic", "role": "precedes", "source_description": "ordered"},
            {"type": "semantic", "role": "supersedes", "source_description": "revised"},
        ],
    )

    assert note.index("revised") < note.index("ordered")


def test_the_reason_rides_along_with_the_pair() -> None:
    note = note_text(
        [],
        [
            {
                "type": "semantic",
                "role": "precedes",
                "source_description": "a",
                "target_description": "b",
                "reason": "2024-03-10 is 2 day(s) before 2024-03-12",
            }
        ],
    )

    assert "2024-03-10 is 2 day(s) before 2024-03-12" in note


def _ordering(source: str, source_on: date, target: str, target_on: date) -> dict:
    return {
        "type": "semantic",
        "role": "precedes",
        "source_id": source,
        "source_description": source,
        "source_occurred_on": source_on,
        "source_turns": [1],
        "target_id": target,
        "target_description": target,
        "target_occurred_on": target_on,
        "target_turns": [2],
    }


def test_dated_orderings_collapse_into_one_timeline() -> None:
    """Stored all-pairs, rendered once per event: three pairs over three events
    is three lines, not three sentences naming six."""
    note = note_text(
        [],
        [
            _ordering("a", date(2024, 3, 1), "b", date(2024, 3, 2)),
            _ordering("a", date(2024, 3, 1), "c", date(2024, 3, 3)),
            _ordering("b", date(2024, 3, 2), "c", date(2024, 3, 3)),
        ],
    )

    assert note.count("2024-03-01") == 1
    assert note.count("  - ") == 3
    assert "and then" not in note


def test_the_timeline_is_sorted_by_date_not_by_edge_order() -> None:
    """The interval between any two events is the difference of two visible
    dates, which only works if they are in order."""
    note = note_text(
        [],
        [
            _ordering("late", date(2024, 4, 10), "latest", date(2024, 5, 1)),
            _ordering("early", date(2024, 3, 1), "late", date(2024, 4, 10)),
        ],
    )

    assert [
        line.split()[1] for line in note.splitlines() if line.startswith("  - ")
    ] == [
        "2024-03-01",
        "2024-04-10",
        "2024-05-01",
    ]


def test_an_undated_ordering_still_renders_as_a_pair() -> None:
    """ "The deployment precedes the uptime window that follows it" has no date
    to sort by, so a timeline would have to invent a position for it."""
    note = note_text(
        [],
        [
            {
                "type": "semantic",
                "role": "precedes",
                "source_id": "a",
                "source_description": "the deployment",
                "target_id": "b",
                "target_description": "the uptime window",
            }
        ],
    )

    assert "and then: the uptime window" in note


def test_about_edges_are_not_rendered_as_arrows() -> None:
    """The statement's own text already names the thing it is about."""
    note = note_text(
        [],
        [
            {
                "type": "knowledge",
                "role": "about",
                "source_name": "quota 1,200/day",
                "target_name": "OpenWeather API",
            }
        ],
    )

    assert note == ""


class TestAdjudicateConcurrently:
    """The parallel adjudication must not change what the sequential one built."""

    def _attributes(self, count: int) -> list[dict]:
        return [{"id": str(n), "name": f"n{n}"} for n in range(count)]

    def _pairs(self, count: int) -> list[tuple[int, int, float]]:
        return [(n, n + 1, 0.9) for n in range(count)]

    def test_verdicts_come_back_in_pair_order(self, monkeypatch):
        """Not in completion order -- the slowest call is deliberately first."""
        import time as _time

        from auto_ontology.conversations.semantic import relate

        def slow_first(left, right):
            if left["name"] == "n0":
                _time.sleep(0.05)
            return left["name"]

        monkeypatch.setattr(relate, "adjudicate", slow_first)

        verdicts = relate._adjudicate_all(self._pairs(6), self._attributes(7))

        assert verdicts == ["n0", "n1", "n2", "n3", "n4", "n5"]

    def test_every_pair_is_adjudicated_exactly_once(self, monkeypatch):
        from collections import Counter

        from auto_ontology.conversations.semantic import relate

        seen: Counter = Counter()
        lock = threading.Lock()

        def count(left, right):
            with lock:
                seen[left["name"]] += 1
            return None

        monkeypatch.setattr(relate, "adjudicate", count)
        relate._adjudicate_all(self._pairs(30), self._attributes(31))

        assert len(seen) == 30
        assert set(seen.values()) == {1}

    def test_a_failed_call_is_a_none_in_place(self, monkeypatch):
        from auto_ontology.conversations.semantic import relate

        monkeypatch.setattr(
            relate,
            "adjudicate",
            lambda left, right: None if left["name"] == "n1" else left["name"],
        )

        verdicts = relate._adjudicate_all(self._pairs(3), self._attributes(4))

        assert verdicts == ["n0", None, "n2"]

    def test_no_pairs_makes_no_calls(self, monkeypatch):
        from auto_ontology.conversations.semantic import relate

        def fail(left, right):
            raise AssertionError("should not be called")

        monkeypatch.setattr(relate, "adjudicate", fail)

        assert relate._adjudicate_all([], self._attributes(2)) == []


class TestSubjectIsTheAboutEdge:
    """Only ``about`` names a statement's subject; the other verbs do not."""

    def _edge(self, role, source, target):
        return {
            "type": "knowledge",
            "role": role,
            "source_id": source,
            "target_name": target,
        }

    def test_an_about_edge_is_the_subject(self):
        from auto_ontology.conversations.semantic.store import subjects_of

        got = subjects_of([self._edge("about", "A", "Autocomplete")])

        assert got == {"A": ["Autocomplete"]}

    def test_obtained_at_is_not_a_subject(self):
        """A date a statement points at is not what the statement is about."""
        from auto_ontology.conversations.semantic.store import subjects_of

        assert subjects_of([self._edge("obtained_at", "A", "2024-03-10")]) == {}

    def test_built_with_is_not_a_subject(self):
        from auto_ontology.conversations.semantic.store import subjects_of

        assert subjects_of([self._edge("built_with", "A", "dotenv")]) == {}

    def test_semantic_edges_are_still_excluded(self):
        from auto_ontology.conversations.semantic.store import subjects_of

        edge = self._edge("about", "A", "Autocomplete") | {"type": "semantic"}

        assert subjects_of([edge]) == {}

    def test_a_statement_with_only_a_date_has_no_subject(self):
        """The case that cost a real edge, end to end.

        ``OpenWeather API key obtained`` carried only ``obtained_at``, so its
        subject read as the date and the retraction of it was rejected as a
        revision for naming a different subject.
        """
        from auto_ontology.conversations.semantic.relate import _not_a_revision
        from auto_ontology.conversations.semantic.store import subjects_of

        subjects = subjects_of(
            [
                self._edge("obtained_at", "A", "2024-03-10"),
                self._edge("about", "B", "Christina"),
            ]
        )
        claim = {
            "name": "API key obtained",
            "about": subjects.get("A", []),
            "value": "",
        }
        denial = {"name": "Never obtained one", "about": subjects["B"], "value": ""}

        assert _not_a_revision(claim, denial) is None

    def test_genuinely_different_subjects_are_still_rejected(self):
        """The guard must keep doing its job: 220ms is not a revision of 280ms."""
        from auto_ontology.conversations.semantic.relate import _not_a_revision

        left = {"name": "220ms", "about": ["Weather app"], "value": "220ms"}
        right = {"name": "280ms", "about": ["Autocomplete"], "value": "280ms"}

        assert _not_a_revision(left, right) is not None
