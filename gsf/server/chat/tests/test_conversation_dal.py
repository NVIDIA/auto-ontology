# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shaping persisted messages into safe agent context."""

from gsf.server.chat.conversation_dal import _rows_to_history


def _row(
    role: str,
    content: str,
    *,
    sql_code: str | None = None,
    sql_response: str | None = None,
) -> dict:
    return {
        "role": role,
        "content": content,
        "sql_code": sql_code,
        "sql_response": sql_response,
    }


def test_history_excludes_results_charts_and_errors() -> None:
    rows = [
        _row("user", "Show revenue"),
        _row(
            "assistant",
            "Revenue query",
            sql_code="SELECT SUM(revenue) FROM sales",
        ),
        _row("assistant", "", sql_response='[{"revenue": 10}]'),
        _row("assistant", "```chart\n{}\n```"),
        _row("user", "Broken request"),
        _row("assistant", "Agent failed: unavailable"),
    ]

    history = _rows_to_history(rows)

    assert len(history) == 1
    assert history[0].question == "Show revenue"
    assert history[0].sql_code == "SELECT SUM(revenue) FROM sales"


def test_history_keeps_only_latest_five_completed_turns() -> None:
    rows = []
    for index in range(7):
        rows.extend(
            [
                _row("user", f"question {index}"),
                _row("assistant", f"answer {index}"),
            ]
        )

    history = _rows_to_history(rows)

    assert [turn.question for turn in history] == [
        "question 2",
        "question 3",
        "question 4",
        "question 5",
        "question 6",
    ]


def test_new_user_message_replaces_unanswered_pending_turn() -> None:
    rows = [
        _row("user", "cancelled question"),
        _row("user", "completed question"),
        _row("assistant", "completed answer"),
    ]

    history = _rows_to_history(rows)

    assert len(history) == 1
    assert history[0].question == "completed question"


def test_one_large_turn_is_bounded() -> None:
    history = _rows_to_history(
        [
            _row("user", "q" * 10_000),
            _row("assistant", "a" * 10_000, sql_code="s" * 10_000),
        ]
    )

    turn = history[0]
    assert len(turn.question) + len(turn.response) + len(turn.sql_code or "") <= 12_000
