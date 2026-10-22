# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the graph knows about a set of nodes, as prose for a prompt.

The turns a retrieval returns say what was said. They do not say how the
sayings stand to each other, and that is the part a transcript hides: turn 12
and turn 48 both state a quota, both read as fact, and nothing in either turn
says the other exists. A model given both picks one, confidently.

This renders a retrieved subgraph into text that goes in front of the context.
It lives here rather than inside the retriever because every route wants the
same block -- the node route, the hybrid route and anything agentic built later
-- and the document pipeline learned this the expensive way, by having three
callers render it separately until one of them printed relation names raw and
told the model nothing.

**The two layers are rendered differently, on purpose.** A ``knowledge`` edge
reads better arrow-shaped: its role *is* the content, a transcript yields dozens
of them, and "the dashboard API is a component of the budget tracker" spends
three lines saying what one arrow says. A ``semantic`` edge gets a sentence
instead, because there the role is a schema term the reader does not know --
"statement A supersedes statement B" requires knowing what the word means here,
while "the user later revised this: X is current, Y is what they said before"
does not, and it is the sentence the answer needs to contain anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from auto_ontology.conversations.semantic.store import subjects_of

__all__ = ["SEMANTIC_HEADINGS", "note_text"]

#: How each semantic role reads as an instruction, and the order groups appear.
#:
#: Ordered by what goes wrong when the model does not know. A superseded value
#: produces a confidently wrong answer, which is the worst outcome and comes
#: first. An unresolved contradiction produces a wrong answer that at least
#: could have been hedged. Ordering and refinement change how an answer reads
#: rather than whether it is right, so they come last.
#:
#: Each entry is ``(role, heading, connector)``. The connector joins the two
#: sides and has to match the role: "rather than" reads as a correction, which
#: is right for a revision and actively misleading for an ordering -- rendered
#: under ``precedes`` it tells a model to prefer the earlier event over the
#: later one, when the point is that both happened.
#:
#: **The hedging wording is deliberate, and was measured.** One contradiction
#: question lost a nugget by reporting both readings where the rubric asked
#: which was correct, so these headings were rewritten to demand a verdict.
#: Against the same arm with the old wording that cost contradiction resolution
#: 0.938 -> 0.438: it was already this layer's strongest ability, at 0.938
#: against 0.625 without a graph, and pushing for a verdict broke the four
#: questions it was getting right to chase the one it was not.
SEMANTIC_HEADINGS = (
    (
        "supersedes",
        "The user revised these during the conversation. The first value is "
        "current; the second is what they said earlier and is no longer true:",
        "rather than",
    ),
    (
        "contradicts",
        "The user said both of these and never resolved the disagreement. Do "
        "not pick one silently -- say what was said on each occasion:",
        "but also",
    ),
    # Reaches only the orderings with no date on either end. A dated ordering
    # goes through :func:`_timeline` instead.
    (
        "precedes",
        "These events are ordered relative to each other, earlier first:",
        "and then",
    ),
    (
        "refines",
        "The first of each pair is a more specific version of the second. Both hold:",
        "which narrows",
    ),
)


#: What to do with everything above, said last so it is read last.
#:
#: One sentence, about preferences only. A preference question is a *request*:
#: "help me set up caching, keep it simple" is scored on whether the reply uses
#: localStorage, not on whether it can recite that the user dislikes heavy
#: dependencies, and a block labelling lines ``[preference]`` invites the
#: recital. Adding it took preference following from 0.500 to 1.000.
#:
#: It said more once. A clause asking for a direct answer in as few words as it
#: takes -- aimed at readers that replied "I need to find two dates:" instead of
#: "21 days" -- cost abstention 0.500 -> 0.250 and summarization 0.750 -> 0.646,
#: because one ability wants a refusal and another wants length. Ten abilities
#: do not share a closing instruction, which is the argument for Cognee's
#: per-ability prompts and against saying anything more here.
_CLOSING = (
    "Where a preference is listed above, satisfy it in what you produce rather "
    "than restating it."
)


def _turns(value: Any) -> str:
    indexes = value or []
    if not indexes:
        return ""
    if len(indexes) == 1:
        return f" (turn {indexes[0]})"
    return f" (turns {', '.join(str(i) for i in sorted(indexes))})"


def _sitting(node: dict[str, Any], sessions: Sessions) -> str:
    """Which sitting this node was said in, phrased the way the question is.

    A transcript is several sittings days apart, and a reader asked "what did I
    decide in our second conversation" has no way to answer from a turn number:
    nothing in the note says where one sitting ends. The turn index orders
    everything already, so this adds no ordering -- it only lets a question
    phrased in sittings find the turns it means.

    One-based, as "of 3" implies, and matching how
    :mod:`~auto_ontology.conversations.embedding_rows` renders the same fact on
    the turn rows -- a reader seeing both should not have to notice that one
    counts from zero.

    Nothing is rendered for a single-sitting transcript, where "session 1 of 1"
    is a line of context that rules nothing out, nor for a node spanning every
    sitting, where it says only that the subject ran throughout.
    """
    if not sessions.count or sessions.count < 2:
        return ""
    found = sorted(sessions.of(node))
    if not found or len(found) == sessions.count:
        return ""
    if len(found) == 1:
        return f", session {found[0] + 1} of {sessions.count}"
    listed = " and ".join(str(i + 1) for i in found)
    return f", sessions {listed} of {sessions.count}"


class Sessions:
    """Resolves a node to the sittings its turns fall in.

    A small object rather than a bare mapping because the renderer needs the
    total as well as the lookup, and because the common case -- a caller that
    has no mapping -- then reads as an empty one instead of a ``None`` check at
    every call site.
    """

    def __init__(self, by_turn: dict[tuple[str, int], int] | None = None) -> None:
        self._by_turn = by_turn or {}
        self.count = len(set(self._by_turn.values()))

    def of(self, node: dict[str, Any]) -> set[int]:
        conversation_id = str(node.get("conversation_id") or "")
        return {
            session
            for index in node.get("turn_indexes") or ()
            if (session := self._by_turn.get((conversation_id, int(index)))) is not None
        }


def _pair_line(edge: dict[str, Any], connector: str) -> str:
    """One semantic edge as a sentence a reader can act on.

    The *descriptions* carry the values, so they are what is rendered; a line
    naming only the nodes ("Daily Quota supersedes Daily Quota") tells a model
    two things have the same name and nothing about either figure.
    """
    source = (edge.get("source_description") or edge.get("source_name") or "").strip()
    target = (edge.get("target_description") or edge.get("target_name") or "").strip()
    line = (
        f"  - {source}{_turns(edge.get('source_turns'))}"
        f"\n    {connector}: {target}{_turns(edge.get('target_turns'))}"
    )
    reason = (edge.get("reason") or "").strip()
    return f"{line}\n    {reason}" if reason else line


def _timeline(
    edges: Sequence[dict[str, Any]], readings: frozenset[str]
) -> tuple[str, list[dict[str, Any]]]:
    """Dated orderings as one sorted list, and the undated ones left over.

    Orderings are stored all-pairs, so that an interval question naming two
    events that are not adjacent still finds a single edge carrying the gap.
    Rendered pairwise that is quadratic: eight matched events become
    twenty-eight three-line sentences, six kilobytes saying what eight dated
    lines say. The same eight collapse here to eight lines, and the interval
    between *any* two of them is the difference of two visible dates.

    Subtraction is not what fails. Finding two dates a hundred turns apart is
    what fails, and a sorted list has already done that.

    Undated orderings are returned rather than rendered. "The deployment
    precedes the uptime window that follows it" has no date to sort by, and
    putting it on a timeline would require inventing a position for it.

    *readings* are nodes carrying a measured value, and they are kept **off**
    the timeline entirely. A dated list reads as a revision chain: shown
    "65% on Mar 27, 78% on Mar 28, 85% on Apr 9" a reader answers 85%, which is
    right only if those measure the same thing, and here they did not. Currency
    is what ``supersedes`` is for, and it says so explicitly or not at all. A
    reading's date still reaches the reader through its statement line.
    """
    dated: dict[str, tuple[Any, str]] = {}
    undated: list[dict[str, Any]] = []
    for edge in edges:
        ends = (
            (edge.get("source_id"), edge.get("source_occurred_on")),
            (edge.get("target_id"), edge.get("target_occurred_on")),
        )
        if any(str(row_id) in readings for row_id, _ in ends):
            continue
        if not all(when for _, when in ends):
            undated.append(edge)
            continue
        for side, (row_id, when) in zip(("source", "target"), ends, strict=True):
            body = (
                edge.get(f"{side}_description") or edge.get(f"{side}_name") or ""
            ).strip()
            dated[str(row_id)] = (
                when,
                f"  - {when.isoformat()}  {body}{_turns(edge.get(f'{side}_turns'))}",
            )
    lines = [line for _, line in sorted(dated.values(), key=lambda row: row[0])]
    return ("\n".join(lines), undated)


def _node_line(
    node: dict[str, Any], subjects: dict[str, list[str]], sessions: Sessions
) -> str:
    """One statement, led by what it is about.

    The subject is not decoration. "API response time improved to 220ms" and
    "the autocomplete API response time is 280ms" are measurements of different
    things, and a reader asked which is faster will compare them as though they
    were not. The subject lives on a knowledge edge, which this block used to
    drop while being the longest and most prominent part of the note.
    """
    kinds = "/".join(node.get("kinds") or ())
    body = (node.get("description") or node.get("name") or "").strip()
    about = subjects.get(str(node.get("id") or ""), [])
    lead = f"{', '.join(about)}: " if about else ""
    where = f"{_turns(node.get('turn_indexes'))}{_sitting(node, sessions)}"
    return f"  - [{kinds}] {lead}{body}{where}"


def _thing_line(node: dict[str, Any], sessions: Sessions) -> str:
    name = (node.get("name") or "").strip()
    description = (node.get("description") or "").strip()
    tail = f" -- {description}" if description else ""
    where = f"{_turns(node.get('turn_indexes'))}{_sitting(node, sessions)}"
    return f"  - {name} ({node.get('type', '')}){tail}{where}"


def note_text(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    *,
    session_index_by_turn: dict[tuple[str, int], int] | None = None,
) -> str:
    """The prompt block for a retrieved subgraph.

    *nodes* are things and statements mixed, as retrieval returns them; *edges*
    are both layers mixed. Both are split here rather than by the caller, so
    there is one place that decides how a subgraph reads.

    Empty string when there is nothing to say. A heading over an empty list
    spends context telling a model that the graph found nothing, which is not a
    fact it can use.

    Ordered things, then how they relate, then what was said, then how the
    sayings stand to each other. Each layer refers to the one above it, so a
    reader meets a name before it is used. The statement list is also the part
    that survives when adjudication has not been run, and alone it is still a
    summary of what the conversation established.

    *session_index_by_turn* comes from
    :func:`~auto_ontology.conversations.semantic.store.session_index_by_turn`
    and is what lets a line say which sitting it was said in. Optional, because
    a caller rendering rows it already holds should not have to open a database
    to do it; omitted, the lines read as they did before.
    """
    sessions = Sessions(session_index_by_turn)
    things = [n for n in nodes if list(n.get("kinds") or ()) == ["object"]]
    statements = [n for n in nodes if list(n.get("kinds") or ()) != ["object"]]
    # Computed before ``about`` is dropped below: those edges are the clearest
    # statement of a subject even though they read poorly as arrows.
    subjects = subjects_of(edges)
    readings = frozenset(
        str(n["id"]) for n in statements if n.get("id") and n.get("value")
    )
    # ``about`` is dropped: it attaches a statement to the thing it concerns,
    # and the statement's own text already names that thing. "API daily quota
    # 1,200 calls/day --about--> OpenWeather API v2.5" sits above a line
    # reading "The new daily quota for the OpenWeather API is 1,200 calls per
    # day". It is 61% of the knowledge layer and the most repetitive part of it.
    knowledge = [
        e for e in edges if e.get("type") == "knowledge" and e.get("role") != "about"
    ]
    semantic = [e for e in edges if e.get("type") == "semantic"]

    blocks: list[str] = []
    if things:
        blocks.append(
            "Things this conversation is about:\n"
            + "\n".join(_thing_line(n, sessions) for n in things)
        )
    if knowledge:
        blocks.append(
            "How they relate:\n"
            + "\n".join(
                f"  - {e.get('source_name')} --{e.get('role')}--> "
                f"{e.get('target_name')}{_turns(e.get('turn_indexes'))}"
                for e in knowledge
            )
        )
    if statements:
        blocks.append(
            "What this conversation established, extracted from the full "
            "transcript:\n"
            + "\n".join(_node_line(n, subjects, sessions) for n in statements)
        )

    by_role: dict[str, list[dict[str, Any]]] = {}
    for edge in semantic:
        by_role.setdefault(str(edge.get("role")), []).append(edge)

    timeline, by_role["precedes"] = _timeline(by_role.get("precedes") or [], readings)

    for role, heading, connector in SEMANTIC_HEADINGS:
        rows = by_role.get(role)
        if rows:
            blocks.append(
                heading + "\n" + "\n".join(_pair_line(row, connector) for row in rows)
            )

    if timeline:
        blocks.append("When these happened, earliest first:\n" + timeline)

    if blocks:
        blocks.append(_CLOSING)
    return "\n\n".join(blocks)
