# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retrieve(purpose="conversation")``: what was said about this before.

A third source, beside the data and the catalog. The catalog says what the
database holds and SQL says what is in it; neither records that someone asked
for the rate limit to be raised, or that a number was revised, or which of two
contradictory statements came later. That is in the transcripts, and
:mod:`auto_ontology.retrieval.conversation` already searches them.

Same shape as :mod:`~auto_ontology.retrieval.text_to_sql.react.information`:
one search, rendered as itself rather than as tables to query.

What comes back is in two parts, and the order matters. The *note* is the
semantic layer's prose: which statements still hold, which were superseded,
what was asked for. It leads, because a reader that meets the raw turns first
has formed an answer before it learns that half of them were revised. The turns
follow as the evidence behind it.
"""

from __future__ import annotations

import logging

from auto_ontology.retrieval.conversation import search_semantic
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.information import _clip, _fit

logger = logging.getLogger(__name__)

#: Turns shown in full; the rest get one line.
#:
#: The same two tiers ``information`` uses for terms, and here it is the
#: difference between working and not. One conversation turn is a question and
#: an answer, often a pasted code block, and a handful of them exceed
#: :func:`config.max_observation_chars` on their own -- the BEAM reader is
#: given 29,000 characters of them. Spending the whole observation budget on
#: this leaves no room for the tables a SQL answer still needs, on a question
#: that may turn out to need both.
_FULL_TURNS = 3

#: One turn's question and answer, each clipped to this, when shown in full.
_TURN_CHARS = 700

#: A turn reduced to one line.
_LINE_CHARS = 160


def _turn_block(hit: object) -> str:
    """One turn in full: where it was said, what was asked, what was answered."""
    return "\n".join(
        [
            _turn_header(hit),
            f"  ASKED: {_clip(hit.question, _TURN_CHARS)}",  # type: ignore[attr-defined]
            f"  ANSWERED: {_clip(hit.answer, _TURN_CHARS)}",  # type: ignore[attr-defined]
        ]
    )


def _turn_line(hit: object) -> str:
    """One turn as a single line: enough to ask for it, not enough to answer."""
    return f"{_turn_header(hit)} {_clip(hit.question, _LINE_CHARS)}"


def _turn_header(hit: object) -> str:
    """``TURN`` with the session and date, when the transcript carried them.

    Session and turn numbers are kept because some questions are about the
    conversation rather than its subject -- when something was first raised,
    what came before what -- and those cannot be answered from the text alone.
    """
    parts = [f"TURN {hit.turn_index}"]  # type: ignore[attr-defined]
    if hit.session_index is not None:  # type: ignore[attr-defined]
        parts.append(f"session {hit.session_index}")  # type: ignore[attr-defined]
    if hit.time_anchor:  # type: ignore[attr-defined]
        parts.append(str(hit.time_anchor))  # type: ignore[attr-defined]
    return f"[{', '.join(parts)}]"


def conversation(ctx: ToolContext, query: str) -> str:
    """What past conversations on this database say about *query*.

    An empty *query* is an error rather than a listing. ``information`` lists
    every table for one because the set of tables is small, closed and useful
    on its own; the set of everything ever said is none of those.
    """
    query = (query or "").strip()
    if not query:
        return "Error: say what to look for. A conversation search needs a query."
    if not ctx.database_name:
        return "No database selected, so there are no conversations to search."

    hits = search_semantic(query, database_name=ctx.database_name)
    if not hits.note and not hits.turns:
        return (
            f"Nothing was said about {query!r} in this database's past "
            "conversations. They may hold nothing on it, or none may be "
            "recorded; either way the answer is not here."
        )

    blocks = []
    if hits.note:
        blocks.append(hits.note)
    for position, hit in enumerate(hits.turns):
        blocks.append(_turn_block(hit) if position < _FULL_TURNS else _turn_line(hit))
    return _fit(blocks, "turns", "narrow the query to see them", "\n\n")
