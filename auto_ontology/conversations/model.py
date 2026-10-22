# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shape a conversation arrives in, and how raw messages become turns.

Deliberately free of any benchmark's field names. A transcript reaches Auto
Ontology from a chat log, an export, or a benchmark archive, and each spells
its message dicts differently; the only thing they agree on is that somebody
asked and somebody answered. Callers translate into :class:`Turn` and the rest
of the package never learns where the text came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ASSISTANT_ROLES",
    "Conversation",
    "Turn",
    "USER_ROLES",
    "turns_from_messages",
    "turns_from_sessions",
]

#: Role strings treated as the asking side. Several spellings because the
#: exports disagree and normalising at the edge is cheaper than teaching every
#: caller which dialect it holds.
USER_ROLES = frozenset({"user", "human", "customer"})

#: Role strings treated as the answering side.
ASSISTANT_ROLES = frozenset({"assistant", "ai", "bot", "model", "agent"})

# How several messages from the same speaker in a row are joined before they
# become one side of a pair. A blank line, so the seam stays visible to a model
# reading the text back.
_RUN_SEPARATOR = "\n\n"

#: Benchmark bookkeeping that rides along in the message text, removed here so
#: that no later stage has to know about it.
#:
#: BEAM appends ``->-> 2,9`` -- session two, question nine -- to the turns its
#: questions are drawn from, 83 times in one transcript. Nobody said it, and
#: left in place it is read as content: the extractor turned
#: ``"brought it down from 12% to 1% ->-> 2,9"`` into ``error_rate = 2.9%``,
#: derivation built a revision chain on the invented figure, and the note
#: reported it as established fact. It also reaches the reader verbatim, since
#: retrieval hands back whole turns.
#:
#: Stripped at the parse rather than at each consumer because this is the one
#: place a turn's text is assembled, and the alternative is three stages each
#: carrying a regex for a dialect of someone else's dataset.
_METADATA_MARKERS = re.compile(r"\s*->->\s*\d+\s*,\s*\d+\s*")


@dataclass(frozen=True)
class Turn:
    """One question/answer exchange.

    ``index`` is the position in the whole transcript, not within a session:
    it is the ordering retrieval widens along, and restarting it per session
    would make "the turn after this one" ambiguous at every boundary.

    Either side may be empty. A transcript can open with an unprompted
    assistant greeting or end on a question nobody answered, and dropping
    those loses the two turns most likely to be asked about.
    """

    question: str
    answer: str
    index: int
    session_index: int | None = None
    time_anchor: str | None = None


@dataclass(frozen=True)
class Conversation:
    """A whole transcript, already split into turns."""

    conversation_id: str
    turns: tuple[Turn, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.turns)


def _role_of(
    message: Mapping[str, Any],
    user_roles: frozenset[str],
    assistant_roles: frozenset[str],
) -> str | None:
    """``"user"``, ``"assistant"``, or ``None`` for a role to skip."""
    role = str(message.get("role", "")).strip().lower()
    if role in user_roles:
        return "user"
    if role in assistant_roles:
        return "assistant"
    # System and tool messages fall here. Skipped rather than folded into the
    # neighbouring turn: a system prompt is not something either party said,
    # and embedding it with every pair would put the same boilerplate in front
    # of every vector in the conversation.
    return None


def turns_from_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    session_index: int | None = None,
    start_index: int = 0,
    user_roles: frozenset[str] = USER_ROLES,
    assistant_roles: frozenset[str] = ASSISTANT_ROLES,
) -> list[Turn]:
    """Pair a flat message list into turns, in order.

    One turn is the questions asked since the last answer, plus the answer that
    followed. Consecutive messages from the same speaker are joined rather than
    emitted separately -- a user who sends three lines then waits has asked one
    question, and splitting it would retrieve a fragment of their request
    without the rest.

    **A stated date carries forward.** Transcripts announce the date once, when
    a sitting opens, and say nothing on the turns that follow -- measured on
    BEAM, three of a conversation's 94 turns state one. Left as it arrives, the
    other 91 turns are undated, and a turn retrieved on its own gives a model
    no way to resolve "next Tuesday". Carrying the last stated anchor forward
    reproduces what a reader of the whole transcript would assume; a turn that
    states its own date overrides it from there on.

    *start_index* continues the numbering from an earlier session, so a
    multi-session transcript keeps one sequence across all of them. The carry
    does **not** continue with it: callers hand one sitting at a time (see
    :func:`turns_from_sessions`), and sittings are days apart, so the date that
    held in the last one is no longer a safe assumption in this one.
    """
    turns: list[Turn] = []
    pending: list[str] = []
    anchor: str | None = None
    carried: str | None = None
    index = start_index

    def flush(answer: str) -> None:
        nonlocal anchor, carried, index
        effective = anchor or carried
        carried = effective
        turns.append(
            Turn(
                question=_RUN_SEPARATOR.join(pending).strip(),
                answer=answer.strip(),
                index=index,
                session_index=session_index,
                time_anchor=effective,
            )
        )
        pending.clear()
        anchor = None
        index += 1

    answering: list[str] = []
    for message in messages:
        role = _role_of(message, user_roles, assistant_roles)
        if role is None:
            continue
        content = _METADATA_MARKERS.sub(" ", str(message.get("content") or ""))

        if role == "user":
            # A user message after an answer closes the previous turn.
            if answering:
                flush(_RUN_SEPARATOR.join(answering))
                answering = []
            pending.append(content)
            # The anchor of the turn is the one on its first question. A later
            # message in the same run restating it is the same "now".
            if anchor is None:
                anchor = _anchor_of(message)
            continue

        answering.append(content)

    if answering:
        flush(_RUN_SEPARATOR.join(answering))
    elif pending:
        # A question nobody answered: still a turn, with an empty answer.
        flush("")
    return turns


def _anchor_of(message: Mapping[str, Any]) -> str | None:
    value = message.get("time_anchor")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def turns_from_sessions(
    sessions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    user_roles: frozenset[str] = USER_ROLES,
    assistant_roles: frozenset[str] = ASSISTANT_ROLES,
) -> list[Turn]:
    """Pair a transcript that arrives as a list of sessions.

    Turn numbering runs across the whole transcript while ``session_index``
    records which sitting each turn belongs to. Both are needed and neither
    substitutes for the other: the sequence is what "later" means, and the
    session boundary is what makes a contradiction a change of mind rather
    than a correction.
    """
    turns: list[Turn] = []
    for session_index, messages in enumerate(sessions):
        turns.extend(
            turns_from_messages(
                messages,
                session_index=session_index,
                start_index=len(turns),
                user_roles=user_roles,
                assistant_roles=assistant_roles,
            )
        )
    return turns
