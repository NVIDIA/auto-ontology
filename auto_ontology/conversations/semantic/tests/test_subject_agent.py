# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agentic subject pass: what it reads, when it stops, what it writes."""

from __future__ import annotations

from typing import Any

from auto_ontology.conversations.semantic.extract import (
    EdgeDraft,
    GraphDraft,
    NodeDraft,
    normalise_name,
)
from auto_ontology.conversations.semantic.subject_agent import (
    MAX_OBSERVATION_CHARS,
    MAX_SEARCH_HITS,
    MAX_TOOL_CALLS,
    Action,
    AgentStep,
    Regrouping,
    ToolName,
    _find_turns,
    _read_turns,
    _regroup,
    _regroup_candidates,
    _render_things,
    _subjects_of,
    assign_subjects_agentic,
)
from auto_ontology.conversations.semantic.subjects import SubjectChoice
from auto_ontology.conversations.semantic.tests.test_subjects import statement, thing

TURNS: list[dict[str, Any]] = [
    {"turn_index": 0, "question": "how fast is the prototype?", "answer": "220ms"},
    {"turn_index": 7, "question": "which app did I mean?", "answer": "the weather app"},
]


def _answer(**kwargs: Any) -> AgentStep:
    return AgentStep(action=Action.ANSWER, **kwargs)


class TestRenderingTheChoices:
    def test_each_thing_carries_the_turn_it_was_first_said_in(self) -> None:
        """The one addition over the non-agentic pass, and the whole point.

        Without it the model cannot tell which candidates it has the evidence
        for and which it is taking on description alone.
        """
        rendered = _render_things([thing("Weather app", turn=7)])
        assert "first said in turn 7" in rendered

    def test_a_thing_from_no_turn_says_nothing_about_turns(self) -> None:
        node = thing("Weather app")
        node.turn_indexes = set()
        assert "first said in" not in _render_things([node])


class TestReadingTurns:
    def test_a_named_turn_comes_back_whole(self) -> None:
        out = _read_turns([7], {7: "which app did I mean?\nthe weather app"})
        assert "the weather app" in out

    def test_a_turn_outside_the_transcript_is_not_invented(self) -> None:
        assert _read_turns([99], {7: "..."}) == "No such turns."

    def test_one_turn_asked_for_twice_is_read_once(self) -> None:
        out = _read_turns([7, 7], {7: "body"})
        assert out.count("### turn 7") == 1

    def test_a_long_read_is_clipped(self) -> None:
        out = _read_turns([7], {7: "x" * (MAX_OBSERVATION_CHARS + 500)})
        assert len(out) <= MAX_OBSERVATION_CHARS + len("\n[truncated]")
        assert out.endswith("[truncated]")


class TestFindingTurns:
    def test_it_answers_where_a_phrase_was_used(self) -> None:
        out = _find_turns("prototype", {0: "the prototype is fast", 7: "unrelated"})
        assert "### turn 0" in out
        assert "### turn 7" not in out

    def test_the_search_ignores_case(self) -> None:
        assert "### turn 0" in _find_turns("PROTOTYPE", {0: "the prototype"})

    def test_a_phrase_in_no_turn_says_so(self) -> None:
        assert "No turn uses" in _find_turns("absent", {0: "the prototype"})

    def test_an_empty_phrase_is_refused(self) -> None:
        assert _find_turns("   ", {0: "body"}) == "No phrase given."

    def test_a_phrase_in_many_turns_is_capped_and_counted(self) -> None:
        """A word used everywhere is not a lead, and should not read as one."""
        text = {index: "the app" for index in range(MAX_SEARCH_HITS + 4)}
        out = _find_turns("app", text)
        assert out.count("### turn ") == MAX_SEARCH_HITS
        assert f"{MAX_SEARCH_HITS + 4} turns use it" in out


class TestTheLoop:
    """What the pass does with the steps it gets back.

    The model is replaced by a scripted list of steps, so these pin the
    control flow -- how many calls are made, when reading stops, what is
    written -- and not the model's judgement.
    """

    def _run(
        self, monkeypatch, steps: list[AgentStep | None]
    ) -> tuple[GraphDraft, int]:
        from auto_ontology.utils import llm_invoke

        graph = GraphDraft(
            nodes=[statement("Prototype latency 220ms"), thing("Weather app", turn=7)],
            edges=[],
        )
        remaining = list(steps)
        calls = {"n": 0}

        def _next(*_: Any, **__: Any) -> AgentStep | None:
            calls["n"] += 1
            return remaining.pop(0) if remaining else None

        monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_: object())
        monkeypatch.setattr(llm_invoke, "invoke_with_structured_output", _next)
        assign_subjects_agentic(graph, TURNS)
        return graph, calls["n"]

    def test_an_answer_on_the_first_step_reads_nothing(self, monkeypatch) -> None:
        graph, calls = self._run(
            monkeypatch,
            [_answer(choices=[SubjectChoice(statement="s1", thing="Weather app")])],
        )
        assert calls == 1
        assert len(graph.edges) == 1

    def test_a_tool_call_is_answered_and_the_loop_continues(self, monkeypatch) -> None:
        graph, calls = self._run(
            monkeypatch,
            [
                AgentStep(action=Action.TOOL, tool=ToolName.READ_TURNS, turns=[7]),
                _answer(choices=[SubjectChoice(statement="s1", thing="Weather app")]),
            ],
        )
        assert calls == 2
        assert len(graph.edges) == 1

    def test_the_edge_resolves_the_way_the_write_looks_it_up(self, monkeypatch) -> None:
        """Raw names are dropped by the write without a word. Measured once:
        334 assigned, 334 reported, 0 in the table."""
        graph, _ = self._run(
            monkeypatch,
            [_answer(choices=[SubjectChoice(statement="s1", thing="Weather app")])],
        )
        assert graph.edges[0].source == normalise_name("Prototype latency 220ms")
        assert graph.edges[0].target == normalise_name("Weather app")
        assert graph.edges[0].role == "about"

    def test_reading_forever_still_ends(self, monkeypatch) -> None:
        """A model that only ever asks for more must not loop. The last step
        inside the budget is taken as the answer, empty or not."""
        graph, calls = self._run(
            monkeypatch,
            [AgentStep(action=Action.TOOL, tool=ToolName.READ_TURNS, turns=[7])] * 20,
        )
        assert calls == MAX_TOOL_CALLS + 1
        assert graph.edges == []

    def test_a_failed_call_leaves_the_statement_without_a_subject(
        self, monkeypatch
    ) -> None:
        """No subject rather than a guessed one: a missing edge is recoverable
        by rerunning, a wrong one is not distinguishable later from a true one.
        """
        graph, _ = self._run(monkeypatch, [None])
        assert graph.edges == []

    def test_an_unknown_tool_does_not_end_the_batch(self, monkeypatch) -> None:
        graph, _ = self._run(
            monkeypatch,
            [
                AgentStep(action=Action.TOOL, tool=None),
                _answer(choices=[SubjectChoice(statement="s1", thing="Weather app")]),
            ],
        )
        assert len(graph.edges) == 1

    def test_an_omitted_statement_is_left_alone(self, monkeypatch) -> None:
        """Abstention is a wanted answer, not a failure."""
        graph, _ = self._run(monkeypatch, [_answer(choices=[])])
        assert graph.edges == []

    def test_a_subject_outside_the_list_is_refused(self, monkeypatch) -> None:
        graph, _ = self._run(
            monkeypatch,
            [_answer(choices=[SubjectChoice(statement="s1", thing="Chrome v112")])],
        )
        assert graph.edges == []


def reading(name: str, *, key: str, value: str, turn: int) -> NodeDraft:
    node = statement(name, turn=turn)
    node.key = key
    node.value = value
    return node


def _about(source: str, target: str) -> EdgeDraft:
    return EdgeDraft(
        source=normalise_name(source), target=normalise_name(target), role="about"
    )


class TestFindingWhatWasSplit:
    """Which keys are worth asking about, decided without a model."""

    def _subjects(self, graph: GraphDraft, statements: list[NodeDraft]):
        return _subjects_of(graph, statements)

    def test_one_key_under_two_subjects_is_a_candidate(self) -> None:
        """The shape of a revision that does not exist: the daily quota
        recorded against the API at 1,200 and the rate limiter at 1,000."""
        rows = [
            reading("Quota 1,000/day", key="rate_limit_per_day", value="1000", turn=16),
            reading("Quota 1,200/day", key="rate_limit_per_day", value="1200", turn=33),
        ]
        graph = GraphDraft(
            nodes=[*rows, thing("OpenWeather API"), thing("Rate limiting")],
            edges=[
                _about("Quota 1,000/day", "Rate limiting"),
                _about("Quota 1,200/day", "OpenWeather API"),
            ],
        )
        found = _regroup_candidates(rows, self._subjects(graph, rows))
        assert list(found) == ["rate_limit_per_day"]

    def test_one_key_under_one_subject_is_left_alone(self) -> None:
        """Already comparable, so ``derive`` will chain it without help."""
        rows = [
            reading("Coverage 65%", key="test_coverage_pct", value="65%", turn=57),
            reading("Coverage 78%", key="test_coverage_pct", value="78%", turn=64),
        ]
        graph = GraphDraft(
            nodes=[*rows, thing("API integration module")],
            edges=[
                _about("Coverage 65%", "API integration module"),
                _about("Coverage 78%", "API integration module"),
            ],
        )
        assert _regroup_candidates(rows, self._subjects(graph, rows)) == {}

    def test_two_subjects_but_one_value_is_not_a_revision(self) -> None:
        """Nothing changed, so there is no missing edge to go looking for."""
        rows = [
            reading("Uptime 100%", key="uptime_pct", value="100%", turn=76),
            reading("Uptime 100% again", key="uptime_pct", value="100%", turn=95),
        ]
        graph = GraphDraft(
            nodes=[*rows, thing("Weather app"), thing("Monitoring")],
            edges=[
                _about("Uptime 100%", "Weather app"),
                _about("Uptime 100% again", "Monitoring"),
            ],
        )
        assert _regroup_candidates(rows, self._subjects(graph, rows)) == {}

    def test_a_proposed_reading_is_never_a_candidate(self) -> None:
        """A goal beside a measurement is not the figure having moved -- the
        same exclusion ``derive`` makes."""
        rows = [
            reading("Target 290ms", key="input_lag_ms", value="290ms", turn=46),
            reading("Measured 520ms", key="input_lag_ms", value="520ms", turn=60),
        ]
        rows[0].kinds = ["target"]
        graph = GraphDraft(
            nodes=[*rows, thing("Autocomplete"), thing("Weather app")],
            edges=[
                _about("Target 290ms", "Weather app"),
                _about("Measured 520ms", "Autocomplete"),
            ],
        )
        assert _regroup_candidates(rows, self._subjects(graph, rows)) == {}


class TestRegrouping:
    """What the answer does to the graph."""

    def _run(self, monkeypatch, step: AgentStep | None) -> GraphDraft:
        from auto_ontology.utils import llm_invoke

        rows = [
            reading("Quota 1,000/day", key="rate_limit_per_day", value="1000", turn=16),
            reading("Quota 1,200/day", key="rate_limit_per_day", value="1200", turn=33),
        ]
        graph = GraphDraft(
            nodes=[*rows, thing("OpenWeather API"), thing("Rate limiting")],
            edges=[
                _about("Quota 1,000/day", "Rate limiting"),
                _about("Quota 1,200/day", "OpenWeather API"),
            ],
        )
        monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_: object())
        monkeypatch.setattr(
            llm_invoke, "invoke_with_structured_output", lambda *_, **__: step
        )
        _regroup(
            graph,
            rows,
            [
                node
                for node in graph.nodes
                if node.name.startswith(("OpenWeather", "Rate"))
            ],
            {16: "quota is 1,000 a day", 33: "the quota is now 1,200 a day"},
        )
        return graph

    def test_both_readings_end_up_on_the_named_thing(self, monkeypatch) -> None:
        """Which is what makes ``derive`` able to see the revision at all."""
        graph = self._run(
            monkeypatch,
            _answer(
                regroupings=[
                    Regrouping(key="rate_limit_per_day", thing="OpenWeather API")
                ]
            ),
        )
        targets = {edge.target for edge in graph.edges if edge.role == "about"}
        assert targets == {normalise_name("OpenWeather API")}

    def test_no_revision_edge_is_written_here(self, monkeypatch) -> None:
        """The subject is the answer; ``derive`` emits the edge from it, and a
        second one written here would be a second source of truth."""
        graph = self._run(
            monkeypatch,
            _answer(
                regroupings=[
                    Regrouping(key="rate_limit_per_day", thing="OpenWeather API")
                ]
            ),
        )
        assert {edge.role for edge in graph.edges} == {"about"}

    def test_separate_quantities_are_left_where_they_are(self, monkeypatch) -> None:
        """Answering nothing is the wanted answer for a key that is two keys."""
        graph = self._run(monkeypatch, _answer(regroupings=[]))
        targets = {edge.target for edge in graph.edges if edge.role == "about"}
        assert targets == {
            normalise_name("Rate limiting"),
            normalise_name("OpenWeather API"),
        }

    def test_a_thing_outside_the_list_is_refused(self, monkeypatch) -> None:
        graph = self._run(
            monkeypatch,
            _answer(
                regroupings=[Regrouping(key="rate_limit_per_day", thing="Chrome v112")]
            ),
        )
        assert {edge.target for edge in graph.edges} == {
            normalise_name("Rate limiting"),
            normalise_name("OpenWeather API"),
        }

    def test_a_key_that_was_not_asked_about_is_refused(self, monkeypatch) -> None:
        graph = self._run(
            monkeypatch,
            _answer(
                regroupings=[Regrouping(key="made_up_key", thing="OpenWeather API")]
            ),
        )
        assert {edge.target for edge in graph.edges} == {
            normalise_name("Rate limiting"),
            normalise_name("OpenWeather API"),
        }

    def test_a_failed_call_changes_nothing(self, monkeypatch) -> None:
        graph = self._run(monkeypatch, None)
        assert {edge.target for edge in graph.edges} == {
            normalise_name("Rate limiting"),
            normalise_name("OpenWeather API"),
        }
