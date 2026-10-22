# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edges rendered as paths, for embedding whole relations rather than endpoints.

A node vector answers "what is this thing" and "what was said about it". It
cannot answer "what is the project built with", because that question names
neither endpoint's description and no statement at all -- it names the *verb*.
Searching nodes for it returns the project, the framework, or both, and the
reader is left to guess what joins them.

So every edge is also embedded, as the whole path it spells out:

    Budget tracker (Project) --built_with--> Flask (Technology, a Python web
    framework)

Both ends in full and the role between them, so the vector carries the fact the
edge asserts rather than a pointer to it.

**Both layers, one collection.** A ``knowledge`` path and a ``semantic`` path
read differently -- one is a relation in the world, the other is a relation
between claims -- but a question does not know which it needs, and splitting
them would mean guessing at query time. The ``type`` rides along in the metadata
for a caller that wants to filter after the fact.

**What goes in the text matters more than it looks.** The endpoint
*descriptions* are included, not just the names: "the quota is 1,200/day
supersedes the quota is 1,000/day" is matchable by a question about quotas,
while "Daily quota supersedes Daily quota" is matchable by nothing and is what
you get from names alone. The ``reason`` is included for the same cause -- on a
derived ``precedes`` it holds the day count, which is the literal answer to an
interval question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["node_text", "path_text", "path_rows", "graph_text"]


def node_text(node: dict[str, Any]) -> str:
    """One endpoint: name, type, kinds, and whatever it asserts.

    Kinds are included because they change what the endpoint *is* for -- the
    same name as a ``state`` and as a ``fact`` are a value that moves and one
    that does not, and a question about "the current" one should match the
    former more closely.
    """
    name = (node.get("name") or "").strip()
    parts = [f"{name} ({node.get('type', '')}"]
    kinds = "/".join(node.get("kinds") or ())
    parts.append(f": {kinds})" if kinds else ")")
    head = "".join(parts)

    detail = (node.get("description") or "").strip()
    key, value = node.get("attribute_key"), node.get("value")
    if key and value:
        detail = f"{detail} [{key}={value}]" if detail else f"{key}={value}"
    if (when := node.get("occurred_on")) is not None:
        detail = f"{detail} [occurred {when}]" if detail else f"occurred {when}"
    return f"{head} {detail}".strip() if detail else head


def path_text(
    source: dict[str, Any],
    edge: dict[str, Any],
    target: dict[str, Any],
) -> str:
    """One edge as the sentence it embeds as.

    Arrow-shaped rather than prose. The role is the content of the edge and has
    to survive verbatim: a question asking what something is *built with* should
    match on that phrase, and a prose rendering ("the project uses Flask")
    paraphrases away the only token the question shares with the row.
    """
    role = edge.get("role", "")
    kind = edge.get("type", "")
    head = f"{node_text(source)} --[{kind}: {role}]--> {node_text(target)}"
    reason = (edge.get("reason") or "").strip()
    return f"{head} ({reason})" if reason else head


def path_rows(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Embeddable rows for every edge whose both endpoints are present.

    An edge with a missing endpoint is dropped rather than rendered with a
    placeholder: the text is the whole value of the row, and half a path matches
    questions it cannot answer.

    ``unrelated`` is excluded. It is stored to stop a later pass re-adjudicating
    the pair, and a vector for "these two have no relation" is a vector that
    competes with real paths for a slot.
    """
    by_id = {str(node["id"]): node for node in nodes}
    rows: list[dict[str, Any]] = []
    for edge in edges:
        source = by_id.get(str(edge.get("source_id")))
        target = by_id.get(str(edge.get("target_id")))
        if source is None or target is None or edge.get("role") == "unrelated":
            continue
        rows.append(
            {
                # Composite, because an edge has no id of its own -- the table
                # is keyed by its endpoints, type and role -- and the vector
                # store needs something stable to key a record on.
                "id": (
                    f"{edge['source_id']}|{edge['type']}|{edge['role']}"
                    f"|{edge['target_id']}"
                ),
                "text": path_text(source, edge, target),
                "type": edge.get("type"),
                "role": edge.get("role"),
                "source_id": str(edge["source_id"]),
                "target_id": str(edge["target_id"]),
                # The union of both ends' turns. A path is evidenced by
                # everything either end rests on, which is what retrieval
                # follows to get from a matched path back to the transcript.
                "turn_indexes": sorted(
                    set(source.get("turn_indexes") or [])
                    | set(target.get("turn_indexes") or [])
                    | set(edge.get("turn_indexes") or [])
                ),
            }
        )
    return rows


#: One line saying what the rendering below it is.
#:
#: The reader's prompt introduces its context as "the context" and no more,
#: which was enough when the context was transcript turns: they read as a
#: conversation without being told. Arrow notation does not, and a model given
#: it unannounced has to infer the notation before it can use it. Cognee's own
#: graph template spends a sentence on exactly this.
#:
#: Said once, at the top, rather than per line.
_LEGEND = (
    "Facts drawn from the conversation, written as "
    "`subject --[relation]--> object`. Each side carries what it is, what it "
    "says, any measured value in square brackets and any date it happened on. "
    "A `supersedes` relation points from the statement that now holds to the "
    "one it replaced."
)


def _first_turn(node: dict[str, Any]) -> int:
    """The turn a statement was first made at, or 0 if it cites none.

    Shown on the timeline because the ordering prompt asks a reader to weigh
    turn numbers, and nothing else in this rendering has one: a path carries
    what two statements are to each other, not where either was said. Left off
    :func:`path_text`, which is what the vectors are built from -- a turn
    number matches no question, and changing that text would mean re-embedding
    the whole store to say nothing new.
    """
    turns = node.get("turn_indexes") or []
    return turns[0] if turns else 0


def _chronological(node: dict[str, Any]) -> tuple[str, int]:
    """Sort key putting a dated event where it belongs, ties broken by turn.

    Several events routinely share a day -- a deployment, a coverage figure and
    a sign-off can all be the tenth of April -- and a date is all ``precedes``
    has to go on, so it leaves those three in whatever order they arrived.
    Which was question similarity, and for a question that asks what happened
    first that is the one order guaranteed to be wrong.

    The turn is the tie-break because it is exact: the transcript states these
    things in the order they were said, and within a single day that is the
    best evidence of sequence there is. First turn rather than lowest, since a
    statement's turns are already sorted and the first is where it was raised.
    """
    return str(node.get("occurred_on")), _first_turn(node)


def graph_text(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> str:
    """Everything retrieved, as the relations it spells out, for a reader.

    The same rendering the vectors are built from, which is the point: the row
    that matched the question is the row the model is shown, rather than a
    pointer back to the transcript it was extracted from. A path already
    carries both endpoints in full -- name, type, kinds, description, any
    ``key=value`` and any date -- so there is no separate section listing the
    nodes again.

    Ordered by the rank of the source node, since *nodes* arrives sorted by
    similarity to the question: the relation whose subject was asked about
    comes before one that arrived through a neighbour.

    **``precedes`` is collapsed into one timeline.** Derivation emits it for
    every pair of dated events, so it grows with the square of them: measured
    on one transcript, 252 of 474 retrieved edges, and on a question about
    contradictions 52 of 58 -- 24,000 characters of date arithmetic answering
    nothing that was asked. A list sorted by date asserts all of it in a
    fraction of the room, and ordering is what the pairs were for. The pairs
    stay in the vector store, where the day count on an arbitrary two of them
    is the literal answer to an interval question; it is only the rendering
    that does not need them all. An edge missing a date on either end is left
    as a path, since the timeline has nowhere to put it.

    The timeline leads, for the reason the note leads the turns: it settles
    what happened when, and a model that reads the relations first has already
    decided.

    Nodes on no retrieved path are listed last, on their own. They matched the
    question directly and dropping them would lose the only thing some
    questions have -- a statement nothing was ever related to is still an
    answer. Named as unrelated so the model does not read the absence of a
    relation as an assertion about one.
    """
    rank = {str(node["id"]): position for position, node in enumerate(nodes)}
    last = len(rank)
    rows = sorted(
        path_rows(nodes, edges),
        key=lambda row: (
            rank.get(row["source_id"], last),
            rank.get(row["target_id"], last),
        ),
    )

    by_id = {str(node["id"]): node for node in nodes}
    dated: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    for row in rows:
        ends = [by_id.get(row["source_id"]), by_id.get(row["target_id"])]
        if row["role"] == "precedes" and all(
            end is not None and end.get("occurred_on") is not None for end in ends
        ):
            for end in ends:
                dated.setdefault(str(end["id"]), end)  # type: ignore[index]
            continue
        kept.append(row)

    timeline = [
        f"{node['occurred_on']} (turn {_first_turn(node)})  {node_text(node)}"
        for node in sorted(dated.values(), key=_chronological)
    ]

    seen = (
        {row["source_id"] for row in kept}
        | {row["target_id"] for row in kept}
        | dated.keys()
    )
    alone = [node_text(node) for node in nodes if str(node["id"]) not in seen]

    sections = [_LEGEND]
    if timeline:
        sections.append("In the order they happened:\n" + "\n".join(timeline))
    if kept:
        sections.append("\n".join(row["text"] for row in kept))
    if alone:
        sections.append("Matched, with no relation retrieved:\n" + "\n".join(alone))
    return "\n\n".join(sections)
