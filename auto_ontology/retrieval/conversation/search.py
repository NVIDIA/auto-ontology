# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector search over a conversation's turns, returned in transcript order.

Three things this does that a bare similarity search does not, each because
a transcript is a sequence and a catalog is not:

**Returns hits in transcript order.** A vector store returns them by distance,
so an answer given in turn 12 can arrive above the question that prompted it
in turn 11. A model reading that order has to reconstruct the sequence before
it can reason about it, and on anything resembling "what did I decide first"
it will simply get it wrong. Sorting costs nothing and removes the whole class
of error.

**Widens a hit to its neighbours.** The pair is the chunk, but the context of
a pair is the pairs around it -- a decision is usually taken over two or three
exchanges, and the one that matched is rarely the one holding the conclusion.

**Reads the text back from the store.** The vector carries the *embedding*
text, which is a header plus the pair. The rows hold the transcript as it was
said, and that is what should reach a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auto_ontology.conversations.store import fetch_turns
from auto_ontology.utils import get_embed_params
from auto_ontology.utils.embedding import embed_texts
from auto_ontology.vdb import get_conversation_vdb

__all__ = ["TurnHit", "search_turns"]


@dataclass(frozen=True)
class TurnHit:
    """One turn on its way to a model.

    ``distance`` is ``None`` for a turn that no query matched and that is here
    only because it neighbours one that did. Keeping the distinction lets a
    caller weight, label or drop the context without a second lookup -- and
    stops a widened neighbour being reported as a retrieval result.
    """

    conversation_id: str
    turn_index: int
    session_index: int | None
    time_anchor: str | None
    question: str
    answer: str
    distance: float | None

    @property
    def matched(self) -> bool:
        return self.distance is not None


def search_turns(
    query: str,
    *,
    database_name: str,
    top_k: int = 10,
    widen: int = 0,
) -> list[TurnHit]:
    """Find the turns bearing on *query*, in the order they were said.

    *top_k* is how many turns the vector search returns. *widen* adds that
    many turns either side of each one, so ``widen=1`` turns a hit into the
    exchange before it, the hit, and the exchange after. Widening does not
    consume the *top_k* budget: the result can hold more than ``top_k`` turns,
    and the extras are flagged by a ``distance`` of ``None``.

    Returns an empty list when the query embeds to nothing or the memory holds
    no turns; neither is an error, and both are what an empty memory looks
    like before anything has been ingested into it.
    """
    stripped = query.strip()
    if not stripped or top_k <= 0:
        return []

    vectors = embed_texts([stripped], get_embed_params(), "query")
    vector = vectors[0] if vectors else None
    if vector is None:
        return []

    hits = get_conversation_vdb(database_name=database_name).retrieval(
        [vector],
        top_k=top_k,
        # Both predicates matter. `database_name` is the memory boundary and
        # leaving it off would search every transcript in the store; `label`
        # guards against anything else that is ever written to this
        # collection matching a conversational query.
        where={"database_name": database_name, "label": "Conversation"},
    )
    matched = _closest_by_turn(hits[0] if hits else [])
    if not matched:
        return []

    rows = fetch_turns(database_name, spans=_windows(matched, widen))
    return [
        TurnHit(
            conversation_id=str(row["conversation_id"]),
            turn_index=int(row["turn_index"]),
            session_index=row["session_index"],
            time_anchor=row["time_anchor"],
            question=row["question"] or "",
            answer=row["answer"] or "",
            distance=matched.get((str(row["conversation_id"]), int(row["turn_index"]))),
        )
        for row in rows
    ]


def _closest_by_turn(hits: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    """Distance per ``(conversation_id, turn_index)``, keeping the nearest.

    A turn should not appear twice, but the vector store is free to hold two
    rows for one turn -- an interrupted re-ingest, a write that raced a reset
    -- and a duplicate here would widen into overlapping spans and emit the
    same exchange several times.
    """
    best: dict[tuple[str, int], float] = {}
    for hit in hits:
        metadata = hit.get("metadata") or {}
        conversation_id = metadata.get("conversation_id")
        turn_index = metadata.get("turn_index")
        if conversation_id is None or turn_index is None:
            continue
        key = (str(conversation_id), int(turn_index))
        distance = float(hit.get("_distance", 0.0))
        if key not in best or distance < best[key]:
            best[key] = distance
    return best


def _windows(
    matched: dict[tuple[str, int], float], widen: int
) -> list[tuple[str, int, int]]:
    """Inclusive turn windows covering every hit, overlaps merged.

    Merging is per *run* of adjacent hits, not per conversation. Collapsing a
    conversation's hits into one window would read every turn between the
    first and the last -- two hits at turns 5 and 80 would return 76 turns,
    most of which nothing matched. Merging only what actually touches keeps
    the result the size the caller asked for while still emitting one
    predicate per contiguous stretch.
    """
    by_conversation: dict[str, list[int]] = {}
    for conversation_id, turn_index in matched:
        by_conversation.setdefault(conversation_id, []).append(turn_index)

    windows: list[tuple[str, int, int]] = []
    for conversation_id, turns in by_conversation.items():
        low = high = None
        for turn in sorted(turns):
            start, end = max(0, turn - widen), turn + widen
            if low is None:
                low, high = start, end
            elif start <= high + 1:
                # Touching or overlapping -- `+ 1` so two windows that meet
                # exactly (…9] and [10…) become one rather than two adjacent
                # predicates over the same unbroken stretch of turns.
                high = max(high, end)
            else:
                windows.append((conversation_id, low, high))
                low, high = start, end
        if low is not None:
            windows.append((conversation_id, low, high))
    return windows
