# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agentic SQL construction loop.

The loop's whole job is to stay bounded and to always come back with
something, so that is what these cover: each cap, the terminal rules, and the
guarantee that a model cannot reach the database with anything but a read-only
query.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from auto_ontology.retrieval.text_to_sql.sql_agent import config
from auto_ontology.retrieval.text_to_sql.sql_agent.loop import run_sql_agent
from auto_ontology.retrieval.text_to_sql.sql_agent.tools import SQLAgentToolbox


class FakeToolLLM:
    """A tool-calling model with a scripted reply per turn.

    The last reply repeats once the script runs out, which is what makes a
    "model that never stops" testable against the caps.
    """

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = replies
        self.invocations: list[list] = []
        self.bound_tools: list | None = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        index = min(len(self.invocations) - 1, len(self._replies) - 1)
        return self._replies[index]


class ExplodingConnector:
    """A connector that fails the test if any SQL reaches it."""

    dialect = "sqlite"
    database_name = "test"

    def execute(self, sql, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError(f"connector was reached with: {sql}")


class FakeConnector:
    """Returns a single row for anything, unless *empty_when* says otherwise."""

    dialect = "sqlite"
    database_name = "test"

    def __init__(self, empty_when=None) -> None:
        self._empty_when = empty_when or (lambda sql: False)
        self.queries: list[str] = []

    def execute(self, sql, *args, **kwargs):
        self.queries.append(sql)
        return pd.DataFrame([] if self._empty_when(sql) else [{"x": 1}])


def _tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _toolbox(connector=None) -> SQLAgentToolbox:
    return SQLAgentToolbox(
        connector=connector or FakeConnector(),
        dialect="sqlite",
        database_name="test",
        relevant_tables=[],
    )


@pytest.fixture
def prompt() -> list:
    return [HumanMessage(content="How many drivers finished the race?")]


@pytest.fixture(autouse=True)
def quiet_checks(monkeypatch):
    """Silence the submission gate's analysers by default.

    They reach for the live catalog, the probe suite and the relationship
    graph, which these tests have no business exercising — they are about the
    loop's control flow. A test that wants a finding patches the one it cares
    about back.
    """
    monkeypatch.setattr(
        SQLAgentToolbox, "static_findings", lambda self, sql: [], raising=True
    )
    monkeypatch.setattr(
        SQLAgentToolbox, "probe_findings", lambda self, sql: [], raising=True
    )
    monkeypatch.setattr(
        SQLAgentToolbox,
        "elaboration_findings",
        lambda self, sql, preview=None: [],
        raising=True,
    )


def test_submit_sql_ends_the_loop(prompt) -> None:
    llm = FakeToolLLM(
        [
            _tool_call(
                "submit_sql",
                {
                    "thought": "Counted the finishers.",
                    "sql_code": "SELECT COUNT(*) FROM results",
                    "response": "The number of finishers.",
                },
            )
        ]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "submitted"
    assert outcome.turns == 1
    assert outcome.result is not None
    assert outcome.result.sql_code == "SELECT COUNT(*) FROM results"
    assert len(llm.invocations) == 1


def test_turn_cap_returns_the_last_checked_draft(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "3")
    draft = "SELECT COUNT(*) FROM results"
    llm = FakeToolLLM([_tool_call("check_sql", {"sql": draft})])

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "turn_cap"
    assert outcome.turns == 3
    assert len(llm.invocations) == 3
    assert outcome.result is not None
    assert outcome.result.sql_code == draft


def test_deadline_returns_without_another_turn(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "8")
    monkeypatch.setenv("SQL_AGENT_DEADLINE_S", "0")
    draft = "SELECT COUNT(*) FROM results"
    llm = FakeToolLLM([_tool_call("check_sql", {"sql": draft})])

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "deadline"
    # One turn, not eight: the deadline is checked before the next call, and
    # an expired one must not buy another round of tools.
    assert outcome.turns == 1
    assert len(llm.invocations) == 1
    assert outcome.result is not None
    assert outcome.result.sql_code == draft


def test_unconstructable_is_reported_not_guessed(prompt) -> None:
    llm = FakeToolLLM(
        [_tool_call("report_unconstructable", {"reason": "No lap-time column."})]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "unconstructable"
    assert outcome.result is None
    assert outcome.unconstructable_reason == "No lap-time column."


def test_an_empty_submission_is_not_accepted_as_sql(prompt) -> None:
    llm = FakeToolLLM(
        [
            _tool_call(
                "submit_sql",
                {"thought": "t", "sql_code": "   ", "response": "r"},
            )
        ]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.result is None
    assert outcome.stop_reason == "unconstructable"


def test_a_reply_with_no_tool_call_is_prompted_again(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "4")
    llm = FakeToolLLM(
        [
            AIMessage(content="Here is my answer in prose."),
            _tool_call(
                "submit_sql",
                {"thought": "t", "sql_code": "SELECT 1 FROM t", "response": "r"},
            ),
        ]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "submitted"
    assert outcome.turns == 2


def test_a_collapsed_tool_call_is_recovered(prompt) -> None:
    """The gateway sometimes folds every argument into the first one.

    Payload copied from a real run, where this cost the question outright: the
    agent had already verified a working query and the submission was scored
    as "could not construct".
    """
    llm = FakeToolLLM(
        [
            _tool_call(
                "submit_sql",
                {
                    "thought": (
                        "I join trans to disp on account_id to reach the client."
                        "</thought>\n"
                        '<parameter name="sql_code">SELECT t.trans_id FROM trans AS t '
                        "JOIN disp AS d ON t.account_id = d.account_id "
                        "WHERE d.client_id = 3356</parameter>\n"
                        '<parameter name="response">The withdrawal transactions.'
                    ),
                },
            )
        ]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "submitted"
    assert outcome.result is not None
    assert outcome.result.sql_code.startswith("SELECT t.trans_id FROM trans AS t")
    assert "parameter name" not in outcome.result.sql_code
    # The leaked markup is stripped off the argument it was hiding in, too.
    assert outcome.result.thought.endswith("to reach the client.")
    assert outcome.result.response == "The withdrawal transactions."


def test_a_submission_with_no_query_falls_back_to_the_draft(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "3")
    draft = "SELECT COUNT(*) FROM results"
    llm = FakeToolLLM(
        [
            _tool_call("check_sql", {"sql": draft}, call_id="a"),
            _tool_call(
                "submit_sql",
                {"thought": "t", "sql_code": "", "response": "r"},
                call_id="b",
            ),
        ]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "submitted"
    assert outcome.result is not None
    assert outcome.result.sql_code == draft


def test_an_unknown_tool_is_answered_not_fatal(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "2")
    llm = FakeToolLLM(
        [
            _tool_call("drop_everything", {"sql": "DROP TABLE results"}),
            _tool_call(
                "submit_sql",
                {"thought": "t", "sql_code": "SELECT 1 FROM t", "response": "r"},
            ),
        ]
    )

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.stop_reason == "submitted"


def test_a_failed_turn_does_not_lose_the_question(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "4")

    class DyingLLM(FakeToolLLM):
        def invoke(self, messages):
            super().invoke(messages)
            raise RuntimeError("gateway exploded")

    draft = "SELECT COUNT(*) FROM results"
    llm = DyingLLM([_tool_call("check_sql", {"sql": draft})])
    toolbox = _toolbox()
    # Pretend an earlier turn had already checked a draft, which is what the
    # loop is expected to fall back on.
    toolbox.calls.append({"tool": "check_sql", "args": {"sql": draft}, "result": "ok"})

    outcome = run_sql_agent(llm, prompt, toolbox)

    assert outcome.stop_reason == "turn_cap"
    assert outcome.turns == 1


# ----------------------------------------------------------------------
# The submission gate
# ----------------------------------------------------------------------


def _submit(sql: str, call_id: str = "s") -> AIMessage:
    return _tool_call(
        "submit_sql",
        {"thought": "t", "sql_code": sql, "response": "r"},
        call_id=call_id,
    )


def test_an_empty_result_sends_the_submission_back(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "4")
    llm = FakeToolLLM(
        [
            _submit("SELECT x FROM t WHERE x = 'typo'", call_id="a"),
            _submit("SELECT x FROM t WHERE x = 'real'", call_id="b"),
        ]
    )
    # Only the misspelled literal comes back empty, which is the case the gate
    # exists to catch: a query that runs clean and answers nothing.
    toolbox = _toolbox(FakeConnector(empty_when=lambda sql: "typo" in sql))

    outcome = run_sql_agent(llm, prompt, toolbox)

    assert outcome.stop_reason == "submitted"
    assert outcome.turns == 2
    assert outcome.result is not None
    assert outcome.result.sql_code.endswith("'real'")


def test_the_gate_gives_up_rather_than_looping(prompt, monkeypatch) -> None:
    """A model that insists must eventually be allowed to submit.

    The gate's checks are heuristics, so an honest disagreement has to cost a
    revision and not the question.
    """
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "9")
    monkeypatch.setenv("SQL_AGENT_MAX_BOUNCES", "2")
    sql = "SELECT x FROM t"
    llm = FakeToolLLM([_submit(sql)])
    toolbox = _toolbox(FakeConnector(empty_when=lambda sql: True))

    outcome = run_sql_agent(llm, prompt, toolbox)

    assert outcome.stop_reason == "submitted"
    assert outcome.result is not None
    assert outcome.result.sql_code == sql
    # Two bounces, then accepted on the third submission.
    assert outcome.turns == 3


def test_static_problems_send_the_submission_back(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "4")
    seen: list[str] = []

    def fake_static(self, sql):
        seen.append(sql)
        return ["GROUP BY is unsafe: toy finding"] if len(seen) == 1 else []

    monkeypatch.setattr(SQLAgentToolbox, "static_findings", fake_static, raising=True)
    llm = FakeToolLLM([_submit("SELECT a FROM t", "a"), _submit("SELECT b FROM t", "b")])

    outcome = run_sql_agent(llm, prompt, _toolbox())

    assert outcome.turns == 2
    assert outcome.result is not None
    assert outcome.result.sql_code == "SELECT b FROM t"


def test_a_query_that_cannot_execute_is_sent_back(prompt, monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "4")
    llm = FakeToolLLM(
        [_submit("SELECT nope FROM t", "a"), _submit("SELECT x FROM t", "b")]
    )

    class BrokenThenFine(FakeConnector):
        def execute(self, sql, *args, **kwargs):
            if "nope" in sql:
                raise RuntimeError('column "nope" does not exist')
            return super().execute(sql, *args, **kwargs)

    outcome = run_sql_agent(llm, prompt, _toolbox(BrokenThenFine()))

    assert outcome.turns == 2
    assert outcome.result is not None
    assert outcome.result.sql_code == "SELECT x FROM t"


# ----------------------------------------------------------------------
# Tool guarantees
# ----------------------------------------------------------------------


def test_only_read_only_sql_reaches_the_database() -> None:
    toolbox = _toolbox(ExplodingConnector())

    for statement in (
        "DELETE FROM results",
        "DROP TABLE results",
        "SELECT 1; DROP TABLE results",
        "UPDATE results SET x = 1",
    ):
        assert "rejected" in toolbox.run_sql(statement)

    # ExplodingConnector asserts on contact, so reaching here is the test.
    assert toolbox.executor.calls == 0


def test_run_sql_stops_at_its_own_budget(monkeypatch) -> None:
    monkeypatch.setenv("SQL_AGENT_MAX_SQL", "2")
    toolbox = _toolbox()

    first = toolbox.run_sql("DELETE FROM results")
    second = toolbox.run_sql("DELETE FROM results")
    third = toolbox.run_sql("SELECT 1")

    assert "rejected" in first
    assert "rejected" in second
    # The third call never reaches the executor, which is why the connector
    # is never contacted even though this one is a valid SELECT.
    assert "budget exhausted" in third.lower()
    assert toolbox.sql_calls == 2


def test_inspect_table_names_what_is_in_scope() -> None:
    toolbox = SQLAgentToolbox(
        connector=ExplodingConnector(),
        dialect="sqlite",
        database_name="test",
        relevant_tables=[{"name": "results", "columns": []}],
    )

    message = toolbox.inspect_table("drivers")

    assert "results" in message
    assert "search_tables" in message


def test_search_tables_without_a_retriever_says_so() -> None:
    toolbox = _toolbox()

    assert "unavailable" in toolbox.search_tables("lap times")


def test_config_defaults_keep_the_feature_off(monkeypatch) -> None:
    monkeypatch.delenv("SQL_AGENT_ENABLED", raising=False)
    assert config.is_sql_agent_enabled() is False
    monkeypatch.setenv("SQL_AGENT_ENABLED", "true")
    assert config.is_sql_agent_enabled() is True


def test_every_turn_is_handed_a_conversation_ending_in_a_user_message(
    prompt, monkeypatch
) -> None:
    """Bedrock rejects a conversation that ends on the assistant's turn.

    It hoists system messages out of the message list, so mid-loop feedback
    sent as a ``SystemMessage`` leaves the last entry an assistant reply and
    the next call dies with "the conversation must end with a user message".
    That failure breaks the loop and throws away whatever the feedback was
    trying to fix, so it costs the question the very turn that was meant to
    repair it. Exercised through the two paths that append feedback: a reply
    with no tool call, and a submission the gate sends back.
    """
    monkeypatch.setenv("SQL_AGENT_MAX_TURNS", "5")
    monkeypatch.setattr(
        SQLAgentToolbox,
        "static_findings",
        lambda self, sql: ["Something is wrong with it."],
        raising=True,
    )
    llm = FakeToolLLM(
        [
            AIMessage(content="Here is my answer in prose."),
            _submit("SELECT 1 FROM t"),
            _submit("SELECT 2 FROM t"),
        ]
    )

    run_sql_agent(llm, prompt, _toolbox())

    assert len(llm.invocations) > 1
    for index, conversation in enumerate(llm.invocations[1:], start=2):
        last = conversation[-1]
        assert not isinstance(last, (SystemMessage, AIMessage)), (
            f"turn {index} was handed a conversation ending in "
            f"{type(last).__name__}"
        )
