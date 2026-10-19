# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.conversations.model`.

The pairing is where a transcript's shape is decided, and every downstream
count -- rows written, vectors indexed, turns retrieved -- is whatever it
returns. These cover the message sequences that are not a tidy alternation.
"""

from __future__ import annotations

from auto_ontology.conversations.model import (
    turns_from_messages,
    turns_from_sessions,
)


def _msg(role: str, content: str, **extra: object) -> dict[str, object]:
    return {"role": role, "content": content, **extra}


def test_pairs_a_simple_alternation() -> None:
    turns = turns_from_messages(
        [
            _msg("user", "what is the deadline?"),
            _msg("assistant", "March 15."),
            _msg("user", "and the budget?"),
            _msg("assistant", "$40k."),
        ]
    )

    assert [(t.question, t.answer) for t in turns] == [
        ("what is the deadline?", "March 15."),
        ("and the budget?", "$40k."),
    ]
    assert [t.index for t in turns] == [0, 1]


def test_joins_consecutive_messages_from_the_same_speaker() -> None:
    """Three lines then a wait is one question, not three."""
    turns = turns_from_messages(
        [
            _msg("user", "hi"),
            _msg("user", "quick question"),
            _msg("user", "when is the deadline?"),
            _msg("assistant", "March 15."),
            _msg("assistant", "Let me know if that slips."),
        ]
    )

    assert len(turns) == 1
    assert turns[0].question == "hi\n\nquick question\n\nwhen is the deadline?"
    assert turns[0].answer == "March 15.\n\nLet me know if that slips."


def test_keeps_a_question_nobody_answered() -> None:
    """The last thing asked is the turn most likely to be asked about."""
    turns = turns_from_messages(
        [
            _msg("user", "what is the deadline?"),
            _msg("assistant", "March 15."),
            _msg("user", "are you sure?"),
        ]
    )

    assert len(turns) == 2
    assert turns[1].question == "are you sure?"
    assert turns[1].answer == ""


def test_keeps_an_unprompted_opening_answer() -> None:
    turns = turns_from_messages([_msg("assistant", "Welcome back.")])

    assert len(turns) == 1
    assert turns[0].question == ""
    assert turns[0].answer == "Welcome back."


def test_skips_system_and_tool_messages() -> None:
    """Boilerplate in front of every vector would match everything equally."""
    turns = turns_from_messages(
        [
            _msg("system", "You are a helpful assistant."),
            _msg("user", "hello"),
            _msg("tool", '{"result": 1}'),
            _msg("assistant", "hi"),
        ]
    )

    assert len(turns) == 1
    assert turns[0].question == "hello"
    assert turns[0].answer == "hi"


def test_accepts_alternative_role_spellings() -> None:
    turns = turns_from_messages(
        [_msg("human", "ping"), _msg("ai", "pong")],
    )

    assert [(t.question, t.answer) for t in turns] == [("ping", "pong")]


def test_takes_the_time_anchor_from_the_first_question_in_a_run() -> None:
    turns = turns_from_messages(
        [
            _msg("user", "planning", time_anchor="March-15-2024"),
            _msg("user", "still planning", time_anchor="March-16-2024"),
            _msg("assistant", "ok"),
        ]
    )

    assert turns[0].time_anchor == "March-15-2024"


def test_the_stated_anchor_carries_forward_to_later_turns() -> None:
    """A transcript states the date once; every turn after it is still that day."""
    turns = turns_from_messages(
        [
            _msg("user", "a", time_anchor="March-15-2024"),
            _msg("assistant", "b"),
            _msg("user", "c"),
            _msg("assistant", "d"),
            _msg("user", "e"),
            _msg("assistant", "f"),
        ]
    )

    assert [t.time_anchor for t in turns] == ["March-15-2024"] * 3


def test_a_newly_stated_anchor_replaces_the_carried_one() -> None:
    turns = turns_from_messages(
        [
            _msg("user", "a", time_anchor="March-15-2024"),
            _msg("assistant", "b"),
            _msg("user", "c", time_anchor="April-02-2024"),
            _msg("assistant", "d"),
            _msg("user", "e"),
            _msg("assistant", "f"),
        ]
    )

    assert [t.time_anchor for t in turns] == [
        "March-15-2024",
        "April-02-2024",
        "April-02-2024",
    ]


def test_a_turn_before_any_anchor_stays_undated() -> None:
    """Carrying forward must not become carrying backward."""
    turns = turns_from_messages(
        [
            _msg("user", "a"),
            _msg("assistant", "b"),
            _msg("user", "c", time_anchor="March-15-2024"),
            _msg("assistant", "d"),
        ]
    )

    assert [t.time_anchor for t in turns] == [None, "March-15-2024"]


def test_the_carry_stops_at_a_session_boundary() -> None:
    """Sittings are days apart; last week's date is not an assumption to keep."""
    turns = turns_from_sessions(
        [
            [_msg("user", "a", time_anchor="March-15-2024"), _msg("assistant", "b")],
            [_msg("user", "c"), _msg("assistant", "d")],
        ]
    )

    assert [t.time_anchor for t in turns] == ["March-15-2024", None]


def test_missing_or_blank_time_anchor_is_none() -> None:
    turns = turns_from_messages(
        [
            _msg("user", "a", time_anchor="  "),
            _msg("assistant", "b"),
            _msg("user", "c"),
            _msg("assistant", "d"),
        ]
    )

    assert [t.time_anchor for t in turns] == [None, None]


def test_sessions_share_one_turn_sequence() -> None:
    """Numbering restarting per session would make 'the next turn' ambiguous."""
    turns = turns_from_sessions(
        [
            [_msg("user", "a"), _msg("assistant", "b")],
            [_msg("user", "c"), _msg("assistant", "d")],
            [_msg("user", "e"), _msg("assistant", "f")],
        ]
    )

    assert [t.index for t in turns] == [0, 1, 2]
    assert [t.session_index for t in turns] == [0, 1, 2]


def test_no_messages_is_no_turns() -> None:
    assert turns_from_messages([]) == []
    assert turns_from_sessions([]) == []
