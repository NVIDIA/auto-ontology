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
of them, and "the reporting API is a component of the dispatch tool" spends
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
#: the lightweight option, not on whether it can recite that the user dislikes
#: heavy dependencies, and a block labelling lines ``[preference]`` invites the
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


#: What the unrealised kinds mean, glossed where they are used.
#:
#: These tags were rendered and never explained, and an unexplained tag reads
#: as decoration. Asked which settings were *enforced*, a reader shown a
#: ``[proposed]`` line holding a config the assistant had offered as an example
#: listed its settings as enforced; the right answer was that the transcript
#: never says any were adopted, and the tag was the only thing in the context
#: that said so. Measured at 0.0 on the same question in four rounds of four.
#:
#: Here rather than in ``_CLOSING`` because it is about how to read the lines
#: directly above it, and a closing instruction is read as applying to the
#: answer instead of to the evidence -- which is how the one closing clause
#: that over-reached cost abstention 0.500 -> 0.250.
#:
#: Only the kinds actually present are glossed. A sentence explaining a tag
#: that appears nowhere spends context ruling out something the reader could
#: not have concluded.
#:
#: The ``proposed`` gloss says what to *do* as well as what the tag means,
#: because stating only the meaning was not enough. Asked which ESLint rules
#: were enforced, a reader given the gloss answered with the three rules and
#: then wrote "this configuration is tagged [proposed] -- not something
#: confirmed as actually enforced". It had read the tag, understood it, and
#: still led with the list; the gold is that the transcript does not say, and
#: the answer scored 0.0. A reader that knows what a tag means still has to be
#: told that the honest answer is sometimes nothing.
_UNREALISED = {
    "proposed": (
        "[proposed] was suggested and never confirmed as done, so if the "
        "question asks what was enforced, adopted, chosen or actually done "
        "and the only lines bearing on it are tagged this way, say the "
        "transcript does not say rather than reciting them"
    ),
    "target": "[target] is a goal, not a measurement",
}


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

    The subject is not decoration. "API response time improved to 140ms" and
    "the address lookup API response time is 95ms" are measurements of different
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


#: How the half-edge block reads, per relation. Phrased as a possession rather
#: than an arrow, because these carry no turns and must not read like the
#: evidenced relations above them.
_NEIGHBOUR_HEADINGS = {
    "has_component": "is made up of",
    "built_with": "is built with",
    "covers": "covers",
    "refines": "refines",
}


def neighbourhood_text(
    nodes: Sequence[dict[str, Any]],
    half_edges: Sequence[dict[str, Any]],
) -> str:
    """What the retrieved things connect to, as names and nothing else.

    *half_edges* comes from
    :func:`~auto_ontology.conversations.semantic.store.fetch_half_edges`: one
    end retrieved, the other not. Rendered apart from the evidenced relations
    and without turn numbers, because that is exactly what distinguishes them
    -- the far end was never retrieved, so no turn here can be cited for it.
    A reader that wants one has to be told the name and nothing more.

    The point is to stop the note hiding what the graph knows. A question about
    a feature is usually a question about its parts, and a part is named for
    what it does, not for the feature it belongs to -- ``Request cancellation
    on a stale query`` shares no wording with "improving the address lookup"
    and cannot be reached by embedding at any budget. The edge is the only route,
    and it was being dropped for the far end it names.

    Empty string when there is nothing across the boundary.
    """
    inside = {str(n["id"]): n for n in nodes if n.get("id")}
    grouped: dict[tuple[str, str], list[str]] = {}
    for edge in half_edges:
        source, target = str(edge.get("source_id")), str(edge.get("target_id"))
        anchor, other = (
            (source, edge.get("target_name"))
            if source in inside
            else (target, edge.get("source_name"))
        )
        if anchor not in inside or not other:
            continue
        names = grouped.setdefault((anchor, str(edge.get("role"))), [])
        # By name. A containment relation extracted in both directions anchors
        # on the same node twice -- measured once in 778 edges, a feature and
        # one of its parts recorded as containing each other -- and the store
        # dedupes by (source, target, role), where a flipped pair is two keys.
        # Listing the far end twice reads as two components, so the display
        # collapses them; the contradictory edge itself is a graph defect and
        # ``has_component`` being antisymmetric is checkable without a model.
        if str(other) not in names:
            names.append(str(other))

    lines: list[str] = []
    for node in nodes:
        anchor = str(node.get("id"))
        for role, phrasing in _NEIGHBOUR_HEADINGS.items():
            names = grouped.get((anchor, role))
            if names:
                lines.append(f"- {node.get('name')} {phrasing}: {', '.join(names)}")
    if not lines:
        return ""
    return (
        "Also recorded about these, named but not quoted -- ask for a name to "
        "see what was said about it:\n" + "\n".join(lines)
    )


def note_text(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    *,
    half_edges: Sequence[dict[str, Any]] = (),
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
    # In the order they were said, which retrieval does not preserve and
    # nothing else in the note supplies. ``precedes`` orders only the 13 of 131
    # statements carrying both an event kind and a date, so a question asking
    # the order in which topics were *raised* -- as against when they happened
    # -- has the turn index and nothing else to go on. Measured at 0.550 over
    # four rounds on exactly that question, against 1.000 on its twin.
    statements = sorted(
        (n for n in nodes if list(n.get("kinds") or ()) != ["object"]),
        key=lambda n: (min(n.get("turn_indexes") or [10**6]), str(n.get("name") or "")),
    )
    asked = [n for n in statements if "request" in (n.get("kinds") or ())]
    statements = [n for n in statements if "request" not in (n.get("kinds") or ())]
    # Computed before ``about`` is dropped below: those edges are the clearest
    # statement of a subject even though they read poorly as arrows.
    subjects = subjects_of(edges)
    readings = frozenset(
        str(n["id"]) for n in statements if n.get("id") and n.get("value")
    )
    # ``about`` is dropped: it attaches a statement to the thing it concerns,
    # and the statement's own text already names that thing. "Daily label cap
    # 4,800 parcels/day --about--> Shipping Rates API v1.4" sits above a line
    # reading "The new daily label cap for the Shipping Rates API is 4,800
    # parcels per day". It is 61% of the knowledge layer and its most
    # repetitive part.
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
        marked = {
            kind
            for node in statements
            for kind in node.get("kinds") or ()
            if kind in _UNREALISED
        }
        # Sorting the lines is not enough on its own: a reader with no reason
        # to think a list is ordered falls back on the turn numbers, which is
        # what it was doing before the sort and why the sort alone moved
        # nothing. The order is stated here, next to the lines it describes,
        # for the same reason the kinds gloss is.
        heading = (
            "What this conversation established, extracted from the full "
            "transcript and listed in the order it was said, earliest first"
        )
        if marked:
            heading += (
                ". Read the tags: "
                + "; ".join(
                    text for kind, text in _UNREALISED.items() if kind in marked
                )
                + ". Nothing tagged this way is something that was done:"
            )
        else:
            heading += ":"
        blocks.append(
            heading
            + "\n"
            + "\n".join(_node_line(n, subjects, sessions) for n in statements)
        )

    # Apart from the statements above, and the separation is the point. Asked
    # how many concerns the user mentioned wanting handled, a reader given one
    # undifferentiated list of statements has every feature in the graph to
    # choose from and answered 6 against a gold of 4; cognee answered 23. The
    # set to count has to be a set before it can be counted.
    if asked:
        blocks.append(
            "What the user asked for help with, as against what the project "
            "already has. These are requests, not descriptions:\n"
            + "\n".join(_node_line(n, subjects, sessions) for n in asked)
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

    # Last of the graph blocks, and after the timeline: everything above is
    # evidenced and this is not, so it reads as an appendix rather than as more
    # of the same.
    if blocks:
        neighbours = neighbourhood_text(things, half_edges)
        if neighbours:
            blocks.append(neighbours)

    if blocks:
        blocks.append(_CLOSING)
    return "\n\n".join(blocks)
