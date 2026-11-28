# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a conversation's rows into embedding-ready rows.

The counterpart of :mod:`auto_ontology.utils.embedding_rows` for chat
transcripts, and the same warning applies: :func:`build_turn_text` is the text
that gets embedded and vector-searched, so it is part of the retrieval
contract. Changing it changes every stored embedding, and a query embedded
against the new wording scores poorly against turns ingested under the old
one. Re-ingest the conversation after editing it.

**One embedding per question/answer pair.** That is the chunk. A pair is the
smallest span that still answers something on its own -- an assistant reply
retrieved without its prompt is a paragraph with no subject, and a question
retrieved without its reply is the part the asker already knew.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import pandas as pd

from auto_ontology.catalog.constants import Labels
from auto_ontology.utils.embedding_rows import build_embed_row

logger = logging.getLogger(__name__)

__all__ = ["ConversationEmbeddingRowsOp", "build_turn_text"]


class ConversationEmbeddingRowsOp:
    """Build an embedding-ready DataFrame from a conversation's turn rows.

    Expected input: a DataFrame with one row per turn, carrying the ids
    assigned by :func:`auto_ontology.conversations.ingest.ingest_conversation`
    -- ``id``, ``conversation_id``, ``turn_index``, ``question``, ``answer``,
    and optionally ``session_index`` and ``time_anchor``.

    Output columns: ``text, path, page_number, metadata`` -- the same shape
    :class:`~auto_ontology.utils.embedding_rows.CatalogEmbeddingRowsOp`
    produces, so both feed the same embed and VDB write path.
    """

    def __init__(self, *, database_name: str) -> None:
        self._database_name = database_name

    def __call__(self, turns_df: Any) -> pd.DataFrame:
        return self.process(turns_df)

    def process(self, turns_df: Any) -> pd.DataFrame:
        if not isinstance(turns_df, pd.DataFrame):
            raise TypeError(
                f"ConversationEmbeddingRowsOp expected a turns DataFrame for "
                f"database {self._database_name!r}, got {type(turns_df).__name__}."
            )

        rows = list(self._build_rows(turns_df))
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])

    def _build_rows(self, turns_df: pd.DataFrame) -> Iterable[dict[str, Any]]:
        turn_count = _extent(turns_df, "turn_index")
        session_count = _extent(turns_df, "session_index")
        opened_on = _opening_date(turns_df)

        for _, turn in turns_df.iterrows():
            conversation_id = str(turn.get("conversation_id", ""))
            turn_index = _as_int(turn.get("turn_index"))
            session_index = _as_int(turn.get("session_index"))
            question = _as_text(turn.get("question"))
            answer = _as_text(turn.get("answer"))
            time_anchor = _as_text(turn.get("time_anchor"))

            if not question and not answer:
                # Nothing was said. Embedding it would put a vector with no
                # content in the index, where it matches short queries on
                # length alone.
                continue

            text = build_turn_text(
                question=question,
                answer=answer,
                conversation_id=conversation_id,
                turn_index=turn_index,
                turn_count=turn_count,
                session_index=session_index,
                session_count=session_count,
                time_anchor=time_anchor,
                days_since_start=_days_between(opened_on, _parse_anchor(time_anchor)),
                database_name=self._database_name,
            )
            yield build_embed_row(
                text=text,
                node_id=str(turn.get("id", "")),
                label=Labels.CONVERSATION,
                # Turns have no name of their own, so the one shown wherever a
                # node is listed is its address in the transcript.
                name=f"{conversation_id}#{turn_index}",
                # No schema tier above a conversation; the field stays for the
                # shared row shape.
                schema_name="",
                database_name=self._database_name,
                extra={
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "session_index": session_index,
                },
            )


def _as_text(value: Any) -> str:
    """Empty string for anything absent, including a pandas NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extent(turns_df: pd.DataFrame, column: str) -> int | None:
    """How many positions the transcript has, from the highest index seen.

    ``max + 1``, not ``len``. The indexes are positions in the whole
    conversation, so a frame holding only part of one -- a resumed ingest, a
    re-embed of a handful of edited turns -- would otherwise render "turn 40
    of 3". Taking the extent makes the total mean the same thing whatever
    slice is in hand.
    """
    if column not in turns_df:
        return None
    indexes = [index for index in turns_df[column].map(_as_int) if index is not None]
    return max(indexes) + 1 if indexes else None


# Anchors are prose dates, not timestamps. BEAM writes "March-15-2024"; the
# others are here because no export agrees with another and a date that fails
# to parse costs only the day-offset clause.
_ANCHOR_FORMATS = ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_anchor(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in _ANCHOR_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _opening_date(turns_df: pd.DataFrame) -> date | None:
    """The date on the earliest turn that states one."""
    if "time_anchor" not in turns_df or "turn_index" not in turns_df:
        return None
    ordered = turns_df.sort_values("turn_index")
    for value in ordered["time_anchor"]:
        parsed = _parse_anchor(_as_text(value))
        if parsed is not None:
            return parsed
    return None


def _days_between(start: date | None, current: date | None) -> int | None:
    if start is None or current is None:
        return None
    return (current - start).days


# ── Text template ────────────────────────────────────────────────────────────


def build_turn_text(
    *,
    question: str,
    answer: str,
    conversation_id: str,
    turn_index: int | None,
    turn_count: int | None,
    session_index: int | None,
    session_count: int | None,
    time_anchor: str | None,
    days_since_start: int | None,
    database_name: str,
) -> str:
    """Build the embedding text for one question/answer pair.

    **Position is spelled out rather than left as a bare number.** Every turn
    in a sitting shares one date -- transcripts state the day when a session
    opens and never again -- so the anchor cannot say which of two turns came
    first, and ``turn: 4`` asks a model to infer both the direction and the
    scale of the number. ``turn 5 of 94`` answers both, and the day offset
    between sittings puts the session on a timeline. All of it is derived from
    the transcript; no clock time is invented, because none is recorded.

    Turn and session are rendered **one-based** here, as "of N" implies, while
    ``turn_index`` and ``session_index`` stay zero-based everywhere else. The
    text is read by a model and the columns are read by code, and each gets
    the convention its reader expects.

    Follows the catalog templates' ``key: value`` style so queries written for
    one tier do not have to be re-tuned for the other.

    The question and answer come last and are not truncated. Everything before
    them is short and bounded, so a transcript whose turns are long loses none
    of its identifying metadata to an embedding model's input limit -- the
    overflow comes off the end of the answer, which is the part a later
    retrieval can recover from the store.
    """
    parts = [
        f"database_name: {database_name}",
        f"conversation_id: {conversation_id}",
    ]
    if session_index is not None:
        parts.append(
            _ordinal("session", session_index, session_count, days_since_start)
        )
    if turn_index is not None:
        parts.append(_ordinal("turn", turn_index, turn_count, None))
    if time_anchor:
        parts.append(f"time_anchor: {time_anchor}")
    header = ", ".join(parts)
    # The two sides on their own lines rather than inline: a transcript turn
    # runs to paragraphs, and `question: ..., answer: ...` on one line leaves
    # nothing to tell a reader -- or a reranker -- where the reply starts.
    return f"{header}\nquestion: {question}\nanswer: {answer}"


def _ordinal(
    noun: str, index: int, total: int | None, days_since_start: int | None
) -> str:
    """``session 2 of 3 (21 days after the conversation began)``."""
    text = f"{noun} {index + 1}" + (f" of {total}" if total else "")
    if days_since_start:
        text += f" ({days_since_start} days after the conversation began)"
    elif days_since_start == 0:
        # The opening sitting. Said rather than omitted: "0 days after" reads
        # like a bug, and silence here would be read as "no date known".
        text += " (the day the conversation began)"
    return text
