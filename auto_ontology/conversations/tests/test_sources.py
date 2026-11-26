# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.conversations.sources`.

The weight of this file is on one property: reading a chat log through the
source interface must produce exactly what calling
:func:`~auto_ontology.conversations.model.turns_from_sessions` produced
before it existed. Every BEAM score was measured through the old call, and
the comparison is only honest while the text reaching the embedder is the
same byte for byte -- a stripped space would change every vector in the
corpus and quietly invalidate the number, without failing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_ontology.conversations.model import turns_from_sessions
from auto_ontology.conversations.sources import SOURCE_REGISTRY, create_source


def _write(root: Path, conversation_id: str, sessions: list) -> None:
    directory = root / "conversations"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{conversation_id}.json").write_text(
        json.dumps({"conversation_id": conversation_id, "chat": sessions}),
        encoding="utf-8",
    )


#: A transcript with the cases that are not a tidy alternation: a run of
#: consecutive user messages, a date stated once and relied on later, and a
#: second sitting. A format change severe enough to matter shows up in at
#: least one of them.
_SESSIONS = [
    [
        {"role": "user", "content": "deadline?", "time_anchor": "March-15-2024"},
        {"role": "assistant", "content": "March 15."},
        {"role": "user", "content": "and"},
        {"role": "user", "content": "the budget?"},
        {"role": "system", "content": "ignore me"},
        {"role": "assistant", "content": "$40k."},
    ],
    [
        {"role": "user", "content": "still on track?"},
        {"role": "assistant", "content": "yes."},
    ],
]


def test_chat_matches_turns_from_sessions_exactly(tmp_path: Path) -> None:
    _write(tmp_path, "1", _SESSIONS)

    through_source = create_source("chat", tmp_path).get_turns("1")

    assert through_source == turns_from_sessions(_SESSIONS)


def test_read_returns_a_conversation_with_every_turn(tmp_path: Path) -> None:
    _write(tmp_path, "1", _SESSIONS)

    conversation = create_source("chat", tmp_path).read("1")

    assert conversation.conversation_id == "1"
    assert len(conversation) == len(turns_from_sessions(_SESSIONS))
    # Numbering runs across sittings, not within them: it is what "the turn
    # after this one" resolves against, and every citation is an index.
    assert [turn.index for turn in conversation.turns] == list(
        range(len(conversation.turns))
    )


def test_lists_conversations_without_reading_them(tmp_path: Path) -> None:
    _write(tmp_path, "2", _SESSIONS)
    _write(tmp_path, "1", _SESSIONS)

    assert create_source("chat", tmp_path).get_conversations() == ["1", "2"]


def test_missing_root_lists_nothing(tmp_path: Path) -> None:
    # An absent export is an empty one. Enumerating is the cheap call callers
    # make to decide whether there is work, and raising would make them guard
    # it before they can ask.
    assert create_source("chat", tmp_path / "absent").get_conversations() == []


def test_unknown_source_names_the_known_ones(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown conversation source"):
        create_source("teams", tmp_path)


def test_every_registered_source_reports_its_own_key() -> None:
    # The key and the instance must agree, or a caller logging
    # ``source.source`` reports a format it is not reading.
    for key, source_class in SOURCE_REGISTRY.items():
        assert source_class(".").source == key


def _channel(root: Path, name: str, records: list) -> None:
    # Under ``export/``, which the source appends itself: callers pass the
    # revision, the same way a chat caller passes the scale and never names
    # ``conversations/``.
    directory = root / "export" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps({"collection": name, "records": records}), encoding="utf-8"
    )


def _record(ts: float, text: str, **generated: str) -> dict:
    return {
        "source": {"ts": ts, "thread_link": f"link/{ts}"},
        "text": text,
        "generated": generated,
    }


def test_slack_makes_one_turn_per_thread(tmp_path: Path) -> None:
    _channel(
        tmp_path,
        "qa",
        [
            _record(
                2.0,
                "2025-09-02 16:29:42 Dana Okonkwo: report it again?\n"
                "Reply 2025-09-02 16:31:18 Pat Lindqvist: no need\n"
                "Reply 2025-09-02 16:32:13 Dana Okonkwo: sounds good",
            ),
            _record(1.0, "2025-09-01 09:00:00 Robin Adeyemi: nobody replied to this"),
        ],
    )

    turns = create_source("slack", tmp_path).get_turns("qa")

    # Ordered by the export's timestamp, not by position in the file.
    assert [turn.time_anchor for turn in turns] == [
        "September-01-2025",
        "September-02-2025",
    ]
    # A thread nobody answered is still its own turn. Were it merged into the
    # next one, one person's question would be filed under another's thread.
    assert turns[0].answer == ""
    assert turns[1].question.endswith("Dana Okonkwo: report it again?")
    assert turns[1].answer.count("\n") == 1


def test_slack_dates_a_thread_from_its_earliest_message(tmp_path: Path) -> None:
    # The export leads with the message it chose as the root, which is not
    # always the first one sent -- true of 575 threads in one export. Dating
    # from the root files the thread under a day it did not happen on.
    _channel(
        tmp_path,
        "qa",
        [
            _record(
                1.0,
                "2025-09-03 08:00:00 Dana Okonkwo: root, stamped later\n"
                "Reply 2025-09-02 17:00:00 Pat Lindqvist: sent the day before",
            )
        ],
    )

    turns = create_source("slack", tmp_path).get_turns("qa")

    assert turns[0].time_anchor == "September-02-2025"


def test_slack_attaches_continuation_lines_to_the_message_above(tmp_path: Path) -> None:
    # Half the export's lines do not start a message. Split out on their own
    # they become speakerless fragments of somebody else's update.
    _channel(
        tmp_path,
        "standup",
        [
            _record(
                1.0,
                "2025-09-02 09:00:00 Robin Adeyemi: today\n- fixed a crash\n- reviewed",
            )
        ],
    )

    turns = create_source("slack", tmp_path).get_turns("standup")

    assert "- fixed a crash" in turns[0].question
    assert "- reviewed" in turns[0].question


def test_slack_recovers_a_workflow_author_and_drops_the_reminder(
    tmp_path: Path,
) -> None:
    _channel(
        tmp_path,
        "standup",
        [
            _record(
                1.0,
                "2025-09-02 09:00:00 Unknown User: :update: Jason Howard\nshipped it",
            ),
            _record(
                2.0,
                "2025-09-02 09:05:00 Unknown User: Please post your daily update",
            ),
        ],
    )

    turns = create_source("slack", tmp_path).get_turns("standup")

    # The app's own prompt is nobody's words and says the same thing daily.
    assert len(turns) == 1
    assert "Jason Howard" in turns[0].question
    # The form's first line is the name, so repeating it would open the turn
    # with its own author.
    assert ":update:" not in turns[0].question


def test_slack_redacts_a_disclosed_credential(tmp_path: Path) -> None:
    _channel(
        tmp_path,
        "qa",
        [
            _record(
                1.0,
                "2025-09-02 09:00:00 Dana Okonkwo: the branch password is hunter2swordfish",
            )
        ],
    )

    turns = create_source("slack", tmp_path).get_turns("qa")

    assert "hunter2swordfish" not in turns[0].question
    assert "[REDACTED]" in turns[0].question


def test_slack_leaves_an_emoji_shortcode_alone(tmp_path: Path) -> None:
    # ``still need the token :pensive:`` has the exact shape of a disclosure.
    # A value ending in a colon is a shortcode, not a secret.
    _channel(
        tmp_path,
        "qa",
        [
            _record(
                1.0, "2025-09-02 09:00:00 Dana Okonkwo: still need the token :pensive:"
            )
        ],
    )

    turns = create_source("slack", tmp_path).get_turns("qa")

    assert "[REDACTED]" not in turns[0].question


def test_slack_metadata_has_one_row_per_turn(tmp_path: Path) -> None:
    _channel(
        tmp_path,
        "qa",
        [
            _record(
                1.0,
                "2025-09-02 09:00:00 Dana Okonkwo: root\nReply 2025-09-02 09:01:00 Robin Adeyemi: yes",
                thread_title="A title",
            )
        ],
    )

    source = create_source("slack", tmp_path)
    metadata = source.thread_metadata("qa")

    assert len(metadata) == len(source.get_turns("qa"))
    assert metadata[0]["author"] == "Dana Okonkwo"
    assert metadata[0]["reply_count"] == 1
    # The export's own summaries stay out of the turn text: they are another
    # pipeline's model output, and a graph built on them recalls claims nobody
    # made.
    assert metadata[0]["thread_title"] == "A title"
    assert "A title" not in source.get_turns("qa")[0].question
