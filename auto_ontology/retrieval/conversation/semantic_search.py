# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval through a transcript's ontology rather than only its wording.

:func:`~auto_ontology.retrieval.conversation.search.search_turns` matches a
question against turn text. That works when the question reuses the
transcript's words and fails in two ways that the benchmark's hardest
abilities are made of:

**The answer is spread across turns that do not resemble the question.**
"How many days between obtaining the key and finishing the wireframes" shares
no vocabulary with either turn, and the two turns share none with each other.
A top-5 similarity search returns neither, or one of them.

**The matched turn is stale.** A quota stated in turn 12 and revised in turn 48
embed almost identically, so a search returns whichever is marginally closer
and the model answers confidently from it. Nothing in the turn says a revision
exists.

The route here goes the other way: the question matches *concepts* -- the
attributes extracted from the whole transcript -- and each attribute already
carries the turns that state it, found by a model that read the transcript
window by window rather than by a similarity that has to see the whole answer
in one chunk. The edges between the matched attributes then say how those
turns stand to one another, which is what :mod:`note` renders.

**Hybrid by default.** The attribute route is capped by construction: only
turns some attribute points at can ever be delivered, so an answer that rests
on a turn nothing was extracted from is unreachable through it. Vector search
has no such ceiling and no notion of staleness. Running both and taking the
union is strictly better than either -- the cost is context, not recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from auto_ontology.conversations.semantic import store as semantic_store
from auto_ontology.conversations.semantic.note import note_text
from auto_ontology.conversations.semantic.paths import graph_text
from auto_ontology.conversations.store import fetch_turns
from auto_ontology.dal.schema import CONVERSATION_NODE_SUBJECT
from auto_ontology.retrieval.conversation.search import TurnHit, search_turns
from auto_ontology.utils import get_embed_params
from auto_ontology.utils.embedding import embed_texts
from auto_ontology.vdb import get_conversation_semantic_vdb, get_one_hop_path_vdb

logger = logging.getLogger(__name__)

#: Extra rows to request so dropped subject nodes do not consume answer slots.
_SUBJECT_HEADROOM = 2

__all__ = ["SemanticHits", "search_semantic"]


@dataclass
class SemanticHits:
    """Turns to answer from, and what the ontology knows about them.

    *turns* is what to read, in transcript order. *note* is the prose block
    that goes in front of it. *nodes* and *edges* are the raw rows behind the
    note, kept so a caller can record what was available rather than only what
    was rendered -- the measurement needs that distinction and the prompt does
    not.

    *graph* is the same rows as *note*, rendered as paths rather than prose --
    the second of two ways to put the ontology in front of a model, and the
    cheaper one by an order of magnitude. Both are built because which one
    answers better is a question for measurement, not for argument.
    """

    turns: list[TurnHit] = field(default_factory=list)
    note: str = ""
    graph: str = ""
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    #: Ids of nodes that arrived only because a path they sit on matched.
    via_paths: list[str] = field(default_factory=list)

    @property
    def node_turns(self) -> int:
        """How many turns arrived through the node route alone."""
        return sum(1 for turn in self.turns if turn.distance is None)


def _matched_ids(query: str, *, database_name: str, top_k: int) -> list[str]:
    """Node ids matching *query*, nearest first, the subject excluded.

    Things and statements share one collection, so one search reaches both.
    They are resolved together and split by ``kinds`` downstream: the retriever
    does not need to know which it got, and a question that matched a thing
    wants that thing's statements anyway.

    The transcript's subject node is dropped. It matches broad questions
    strongly, because its description summarises the whole transcript and so
    overlaps anything asked about it, and it is the one node whose description a
    model can already infer from the turns it gets. Dropped here rather than in
    the query's ``where``: not every VDB backend filters on *extra* metadata,
    and over-fetching by the number we drop is cheap and backend-independent.

    ``top_k`` of zero turns the node route off, for measuring the path route on
    its own. Guarded here rather than left to the slice so the off-switch costs
    neither an embedding nor a query.
    """
    if top_k <= 0:
        return []
    vectors = embed_texts([query], get_embed_params(), "query")
    vector = vectors[0] if vectors else None
    if vector is None:
        return []

    hits = get_conversation_semantic_vdb(database_name=database_name).retrieval(
        [vector],
        top_k=top_k + _SUBJECT_HEADROOM,
        where={"database_name": database_name, "label": "Conversation"},
    )
    found: list[str] = []
    for hit in hits[0] if hits else []:
        metadata = hit.get("metadata") or {}
        row_id = metadata.get("id")
        if not row_id or metadata.get("type") == CONVERSATION_NODE_SUBJECT:
            continue
        found.append(str(row_id))
    return found[:top_k]


def _matched_path_endpoints(
    query: str, *, database_name: str, path_k: int
) -> list[str]:
    """Node ids sitting on the paths that match *query*, nearest path first.

    A path vector carries both ends and the verb between them, so it matches a
    question that names the *relation* and neither endpoint -- "what is the
    project built with" shares no wording with either node's description. The
    endpoints are returned rather than the path itself because the edge is
    already in Postgres: putting both ends into the node set is enough for
    :func:`~auto_ontology.conversations.semantic.store.fetch_edges` to return
    that edge under its both-ends rule, and the path then arrives rendered
    alongside everything else rather than as a second kind of thing.

    Source before target, because the source of a ``supersedes`` is the current
    value and of a ``precedes`` the earlier event: when the node budget runs out
    mid-path, the half worth keeping is the first one.
    """
    if path_k <= 0:
        return []
    vectors = embed_texts([query], get_embed_params(), "query")
    vector = vectors[0] if vectors else None
    if vector is None:
        return []

    hits = get_one_hop_path_vdb(database_name=database_name).retrieval(
        [vector],
        top_k=path_k,
        where={"database_name": database_name, "label": "Conversation"},
    )
    found: list[str] = []
    for hit in hits[0] if hits else []:
        metadata = hit.get("metadata") or {}
        for end in ("source_id", "target_id"):
            row_id = metadata.get(end)
            if row_id and str(row_id) not in found:
                found.append(str(row_id))
    return found


def search_semantic(
    query: str,
    *,
    database_name: str,
    top_k: int = 5,
    widen: int = 0,
    node_k: int = 8,
    path_k: int = 10,
    node_turns: int = 10,
    hybrid: bool = True,
) -> SemanticHits:
    """Find turns for *query* through the ontology, optionally with vector hits.

    *node_k* is how many nodes the concept search returns and *path_k* how many
    edges the path search returns; the paths' endpoints join the node set, so a
    relation can be reached even when neither of its ends matched on its own.
    *node_turns* caps how many turns they contribute between them, taken from
    the nearest nodes first.

    The cap is not optional tidying. A node's turns are all evidence a model
    judged sufficient for it, so there is no principled place to stop within
    one node -- but eight nodes averaging three turns each is
    twenty-four turns, five times the paper's depth, and a reader given that
    much spends longer reasoning than an endpoint will wait. Measured: an
    uncapped hybrid search returned 16 turns and every question timed out at
    120 seconds.

    Zero turns off entirely, for a caller that means to answer from
    :attr:`SemanticHits.graph` alone.

    *hybrid* unions in
    :func:`~auto_ontology.retrieval.conversation.search.search_turns`. The
    vector hits keep their distances and the node-only turns arrive with
    ``distance`` of ``None``, so a caller can still tell which route delivered
    what.

    Returns empty hits when the memory has no semantic layer, rather than
    raising: a transcript that was ingested but never had the semantic pass run
    is a normal state, and with *hybrid* on the vector route still answers.
    """
    stripped = query.strip()
    if not stripped:
        return SemanticHits()

    matched = _matched_ids(stripped, database_name=database_name, top_k=node_k)
    from_paths = _matched_path_endpoints(
        stripped, database_name=database_name, path_k=path_k
    )
    # Directly matched nodes first, then the ones a path dragged in. The order
    # is the order the prompt lists them in, and a node the question named
    # should be read before one that arrived because something it sits on was
    # named.
    ordered_ids = list(dict.fromkeys([*matched, *from_paths]))
    nodes = semantic_store.fetch_nodes(database_name, ids=ordered_ids)
    if not nodes:
        logger.debug(
            "No nodes matched %r in %s -- falling back to turn search.",
            stripped[:60],
            database_name,
        )

    rank = {row_id: position for position, row_id in enumerate(ordered_ids)}
    nodes.sort(key=lambda row: rank.get(str(row["id"]), len(rank)))
    via_paths = [row_id for row_id in from_paths if row_id not in set(matched)]
    if via_paths:
        logger.debug(
            "%d node(s) reached through a matching path rather than directly.",
            len(via_paths),
        )

    # Both layers at once. The knowledge edges say what the things are to each
    # other, the semantic edges say which of the statements still holds, and
    # answering a question usually needs both.
    edges = semantic_store.fetch_edges([str(row["id"]) for row in nodes])

    spans = _node_spans(nodes, widen=widen, limit=node_turns)
    rows = fetch_turns(database_name, spans=spans) if spans else []

    turns: dict[tuple[str, int], TurnHit] = {
        (str(row["conversation_id"]), int(row["turn_index"])): TurnHit(
            conversation_id=str(row["conversation_id"]),
            turn_index=int(row["turn_index"]),
            session_index=row["session_index"],
            time_anchor=row["time_anchor"],
            question=row["question"] or "",
            answer=row["answer"] or "",
            distance=None,
        )
        for row in rows
    }

    if hybrid:
        for hit in search_turns(
            stripped, database_name=database_name, top_k=top_k, widen=widen
        ):
            key = (hit.conversation_id, hit.turn_index)
            existing = turns.get(key)
            # A turn both routes found keeps its distance: it was matched, and
            # reporting it as a widened neighbour would understate the vector
            # route in the measurement.
            if existing is None or (
                existing.distance is None and hit.distance is not None
            ):
                turns[key] = hit

    ordered = sorted(turns.values(), key=lambda t: (t.conversation_id, t.turn_index))
    return SemanticHits(
        turns=ordered,
        note=note_text(
            nodes,
            edges,
            session_index_by_turn=_sessions(database_name) if nodes else None,
        ),
        graph=graph_text(nodes, edges),
        nodes=nodes,
        edges=edges,
        via_paths=via_paths,
    )


def _sessions(database_name: str) -> dict[tuple[str, int], int] | None:
    """The turn-to-sitting map the note renders from, or nothing on failure.

    One small query per retrieval, which is the price of not denormalising the
    sitting onto every node. Failure is swallowed: the sitting is context on a
    line that is already complete without it, and losing a whole answer over it
    would be the wrong trade.
    """
    try:
        return semantic_store.session_index_by_turn(database_name)
    except Exception:
        logger.exception(
            "Could not read the session layout of %s -- the note will not say "
            "which sitting a line came from.",
            database_name,
        )
        return None


def _node_spans(
    attributes: list[dict[str, Any]], *, widen: int, limit: int
) -> list[tuple[str, int, int]]:
    """Inclusive turn windows covering the attributes' cited turns, capped.

    One window per cited turn rather than one spanning an attribute's first to
    last. An attribute about a quota can cite turns 12 and 48 -- that is the
    point of it -- and a single span would read all 37 turns in between.

    *attributes* arrive ranked by question similarity, and the cap is applied
    across that order, so the turns kept are the nearest attributes' rather
    than the lowest-numbered. It is applied **breadth-first**: one turn from
    each attribute, then a second from each, and so on. Depth-first would spend
    the whole budget inside the first attribute, and the case this route exists
    for -- an interval between two events -- needs one turn from each of two
    attributes, not six from one.
    """
    queues = [
        [
            (str(attribute["conversation_id"]), int(index))
            for index in sorted(attribute.get("turn_indexes") or [])
        ]
        for attribute in attributes
    ]
    picked: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for depth in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if depth >= len(queue) or len(picked) >= limit:
                continue
            key = queue[depth]
            if key in seen:
                continue
            seen.add(key)
            picked.append(key)
        if len(picked) >= limit:
            break

    return [
        (conversation_id, max(0, index - widen), index + widen)
        for conversation_id, index in picked
    ]
