# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide what each statement is *about*, once the whole transcript is extracted.

Extraction reads one window at a time, and a window is too small to answer this.
Asked inside turn 19 what "the prototype's fetch call latency" is a reading of,
the only candidate that turn names is ``Chrome v112.0.5615`` -- so the subject
comes back as the browser, or as another statement, or as the vague noun itself
promoted to a node. Measured on one BEAM transcript, 337 of 475 subject
attributions pointed at another *statement* rather than at a thing.

That is not a wording problem, and three cheaper fixes were measured and
rejected before this one: asking the extractor harder (``about -> statement``
still beat ``about -> thing`` 197 to 35), resolving the subject transitively
through the statement it pointed at (109 of 378 reached a thing directly, 104
never did, and the hops produced technology dumps), and keying revisions on the
statement-subject anyway (which is what split the latency readings in the first
place).

What makes the question answerable here and not there is the **vocabulary**. By
this point the transcript's things are all known, so the model can be handed the
closed list and told to choose from it. That is also why the answer is a label
and not a name: a free-text subject can be a thing that does not exist, and
there is no way to tell that from a thing named slightly differently.

Why a statement can only be about a thing. ``about`` answers "what carries this
property", and only a thing carries one: a latency reading belongs to the app,
not to another latency reading. Statements do relate to each other, but through
``supersedes``, ``contradicts`` and ``refines``, which
:mod:`~auto_ontology.conversations.semantic.derive` settles afterwards from the
subject this pass assigns. Letting a statement be a subject here is precisely
the defect being removed.

The turn text goes with the question because the node text alone often cannot
settle it. "The prototype's fetch call latency improved to 220ms" resolves by
elimination when only one thing in the list is an application, and not at all
when two are.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auto_ontology.conversations.semantic.extract import GraphDraft, NodeDraft

logger = logging.getLogger(__name__)

__all__ = ["SUBJECT_ROLES", "assign_subjects"]


#: Edge roles that answer "what is this statement about". Dropped from the
#: extraction draft and re-decided here, rather than corrected in place: a role
#: the extractor chose is a role it chose for a statement-shaped subject, and
#: keeping the ones that happen to point at a thing would leave the graph with
#: two sources of truth for the same question. The other knowledge roles --
#: ``built_with``, ``has_component``, the ``_at`` verbs -- are untouched; they
#: join two *things*, which one window is big enough to see.
SUBJECT_ROLES = ("about", "measured_by", "constrains")

_TRAILING_TYPE = re.compile(r"\([^()]*\)\s*$")

#: Turns of transcript per call. At ~4,700 chars a turn on a BEAM transcript
#: this is ~47K chars of text plus the things list and the statements drafted
#: from those turns, which leaves the answer short enough to stay reliable. One
#: turn per call would be 100 calls for a payload this size; the whole
#: transcript in one call would be 494K chars, past the window.
TURNS_PER_CALL = 10


_SYSTEM = """\
You decide what each statement in a conversation is ABOUT: which thing carries
the property, figure or fact it states.

You are given the complete list of things the conversation talks about. Choose
from that list and nothing else. You may not name a thing that is not listed,
and you may not answer with another statement.

How to choose:
  - A measurement belongs to what is being measured. "The prototype's fetch
    call latency improved to 220ms" is about the application, even though the
    sentence says "the prototype" and the application is named something else.
    The conversation renames its own subjects constantly; resolve the vague
    noun to the thing it refers to.
  - Pick the thing the figure is a property OF, not something merely present
    while it was measured. A latency measured in a browser is about the app,
    not about the browser; a quota enforced by an API is about the API.
  - Prefer the most specific thing that genuinely carries it. An autocomplete
    response time is about the autocomplete feature if the feature is listed,
    and about the application only if it is not.
  - A fact about a decision is about the thing decided on. "Switched to
    localStorage for caching" is about the application whose caching changed.
  - When two things could both carry it, read the surrounding turns: they say
    which one was under discussion.
  - When nothing in the list plausibly carries it, say so by leaving the
    statement out of your answer. An omitted statement loses one edge; a wrong
    one attaches a figure to the wrong thing and makes the graph state
    something false.
"""

_TEMPLATE = """\
## Things in this conversation

{things}

## Transcript, turns {first}-{last}

{text}

## Statements drafted from these turns

{statements}

For each statement, give the one thing it is about. Identify both by their `s`
and `n` labels, or by their exact names as written above -- either is read, but
nothing outside these two lists is. Omit any statement no listed thing carries.
"""


class SubjectChoice(BaseModel):
    """One statement and the thing it is about, both as labels."""

    statement: str = Field(
        default="", description="The statement's `s` label, or its exact name."
    )
    thing: str = Field(
        default="",
        description=(
            "The `n` label of the thing it is about, or that thing's exact name."
        ),
    )


class SubjectChoices(BaseModel):
    choices: list[SubjectChoice] = Field(default_factory=list)


def _is_thing(node: NodeDraft) -> bool:
    return list(node.kinds) == ["object"]


def _first_turn(node: NodeDraft) -> int:
    return min(node.turn_indexes) if node.turn_indexes else 10**9


def _render_things(things: Sequence[NodeDraft]) -> str:
    """The closed vocabulary, labelled.

    The type and description go with each one because the choice is often
    between two things of different kinds -- an application and a browser -- and
    the name alone does not say which is which.
    """
    lines = []
    for label, node in zip(_labels("n", things), things):
        detail = node.description.strip().split("\n")[0][:160]
        lines.append(
            f"- {label} | {node.name} | {node.type}"
            + (f" | {detail}" if detail else "")
        )
    return "\n".join(lines)


def _render_statements(statements: Sequence[NodeDraft]) -> str:
    lines = []
    for label, node in zip(_labels("s", statements), statements):
        detail = node.description.strip().split("\n")[0][:200]
        reading = f" | {node.key}={node.value}" if node.key and node.value else ""
        lines.append(f"- {label} | {node.name}{reading} | {detail}")
    return "\n".join(lines)


def _labels(prefix: str, items: Sequence[Any]) -> list[str]:
    return [f"{prefix}{i + 1}" for i in range(len(items))]


def _drop_drafted_subjects(graph: GraphDraft) -> int:
    """Remove the extractor's subject edges, in place. Returns how many went.

    Only the ones whose source is a statement: a thing that ``constrains``
    another thing is a relation this pass has no opinion about and does not
    replace.

    Matched on the normalised name, because that is what an ``EdgeDraft``
    endpoint is -- :func:`extract_graph` normalises both ends as it accumulates
    them, and the raw name would silently match nothing.
    """
    from auto_ontology.conversations.semantic.extract import normalise_name

    statements = {
        normalise_name(node.name) for node in graph.nodes if not _is_thing(node)
    }
    keep = [
        edge
        for edge in graph.edges
        if not (edge.role in SUBJECT_ROLES and edge.source in statements)
    ]
    dropped = len(graph.edges) - len(keep)
    graph.edges[:] = keep
    return dropped


def _batch(
    statements: Sequence[NodeDraft], turn_text: dict[int, str]
) -> list[tuple[list[int], list[NodeDraft]]]:
    """Statements grouped with the turns they came from, ``TURNS_PER_CALL`` at a time.

    Keyed on the *earliest* turn a statement cites, so each statement is asked
    about exactly once even when it was accumulated across several windows. The
    text of its later turns is then sometimes absent from the call -- which
    costs less than asking twice and having to reconcile two answers.
    """
    ordered = sorted(turn_text)
    batches = []
    for start in range(0, len(ordered), TURNS_PER_CALL):
        indexes = ordered[start : start + TURNS_PER_CALL]
        window = set(indexes)
        members = [node for node in statements if _first_turn(node) in window]
        if members:
            batches.append((indexes, members))
    return batches


def _look_up(offered: dict[str, NodeDraft], answer: str) -> NodeDraft | None:
    """One answer against the list this call was shown, however it was worded.

    The prompt lists each node as ``- s18 | name | detail``, and a run that
    answered in bare labels once will answer in whole lines the next time: a
    build dropped 42 of 42 subjects to ``s18 | User retrying after hitting rate
    limit concern``, and another offered ``name (Type)``, both of which name
    exactly one node and neither of which is a key. So each ``|`` segment is
    tried in turn, and a trailing parenthesised type is stripped.

    This widens the wording, not the vocabulary. Every candidate is still looked
    up in *offered*, so an invented subject resolves to nothing exactly as
    before -- which is the only thing the lookup was ever protecting.
    """
    for piece in [answer, *answer.split("|")]:
        candidate = piece.strip()
        if not candidate:
            continue
        node = offered.get(candidate) or offered.get(
            _TRAILING_TYPE.sub("", candidate).strip()
        )
        if node is not None:
            return node
    return None


def _accept(
    choices: Sequence[SubjectChoice],
    things: Sequence[NodeDraft],
    statements: Sequence[NodeDraft],
) -> dict[str, str]:
    """Resolve one batch's answer into statement name -> thing name.

    Either form is accepted, the label or the name. Labels are what the prompt
    asks for, and the first live run answered 412 of 412 with names instead --
    `s25 -> Weather app` rather than `s25 -> n7` -- which a label-only lookup
    discarded in full. The labels were never the safeguard: what makes an
    invented subject detectable is that the answer has to resolve inside the
    list this call was *shown*, and a name checked against that list resolves
    exactly as narrowly as a label does.

    Names are unique per transcript by the time this runs -- ``_disambiguate``
    guarantees it -- so one map over both sides cannot confuse a statement with
    a thing. The roles are still enforced separately: a subject must resolve to
    a thing, and a statement must be one this call asked about.

    Everything is checked against what this call was shown, and an unknown
    answer is dropped on its own rather than failing the batch: the alternative
    costs the nine good answers beside it, and there is no smaller question to
    retry. A statement answered twice keeps neither answer, because the second
    is evidence the first was a guess.
    """
    thing_by_label = dict(zip(_labels("n", things), things))
    thing_by_label.update({node.name: node for node in things})
    statement_by_label = dict(zip(_labels("s", statements), statements))
    statement_by_label.update({node.name: node for node in statements})

    resolved: dict[str, str] = {}
    twice: set[str] = set()
    for choice in choices:
        statement = _look_up(statement_by_label, choice.statement)
        thing = _look_up(thing_by_label, choice.thing)
        if statement is None or thing is None:
            logger.warning(
                "Ignoring subject %r -> %r: not offered in this call.",
                choice.statement,
                choice.thing,
            )
            continue
        if statement.name in resolved:
            twice.add(statement.name)
            continue
        resolved[statement.name] = thing.name

    for name in twice:
        logger.warning("Statement %r was given two subjects -- taking neither.", name)
        resolved.pop(name, None)
    return resolved


def assign_subjects(graph: GraphDraft, turns: Sequence[dict[str, Any]]) -> int:
    """Replace every statement's subject edge with one chosen from the things.

    In place on *graph*, returning how many ``about`` edges were written. Runs
    after consolidation and disambiguation, so the names it answers in are the
    names that will be stored, and before the write, so the edges land in the
    same transaction as everything else.

    A failed call leaves that batch's statements with no subject at all, which
    is deliberate. The drafted subjects are dropped first and not restored: they
    are the defect, and a graph missing ten subjects is recoverable by rerunning
    the relate pass, while a graph asserting that one latency reading is a
    property of another is not distinguishable later from a true one.
    """
    from auto_ontology.conversations.semantic.extract import EdgeDraft, normalise_name
    from auto_ontology.conversations.semantic.windows import turn_block

    things = sorted((node for node in graph.nodes if _is_thing(node)), key=_first_turn)
    statements = sorted(
        (node for node in graph.nodes if not _is_thing(node)), key=_first_turn
    )
    if not things or not statements:
        logger.info(
            "Conversation has %d thing(s) and %d statement(s) -- no subject to assign.",
            len(things),
            len(statements),
        )
        return 0

    dropped = _drop_drafted_subjects(graph)
    turn_text = {
        int(turn["turn_index"]): turn_block(
            str(turn.get("question") or "").strip(),
            str(turn.get("answer") or "").strip(),
        )
        for turn in turns
    }
    batches = _batch(statements, turn_text)
    logger.info(
        "Assigning subjects for %d statement(s) over %d call(s), from %d thing(s); "
        "dropped %d drafted subject edge(s).",
        len(statements),
        len(batches),
        len(things),
        dropped,
    )

    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    node_by_name = {node.name: node for node in graph.nodes}
    written = 0
    for indexes, members in batches:
        prompt = _TEMPLATE.format(
            things=_render_things(things),
            first=indexes[0],
            last=indexes[-1],
            text="\n\n".join(f"### turn {i}\n{turn_text[i]}" for i in indexes),
            statements=_render_statements(members),
        )
        try:
            answer = invoke_with_structured_output(
                get_llm_client(temperature=0.0, max_tokens=4096),
                [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
                SubjectChoices,
            )
        except Exception:
            logger.exception(
                "Could not assign subjects for turns %d-%d -- %d statement(s) "
                "keep no subject.",
                indexes[0],
                indexes[-1],
                len(members),
            )
            continue
        if answer is None:
            continue

        for statement_name, thing_name in _accept(
            answer.choices, things, members
        ).items():
            # Normalised, as every other ``EdgeDraft`` endpoint is: the write
            # resolves an edge's ends through a map keyed by normalised name,
            # and a raw name is dropped there without a word. Measured once:
            # 334 subject edges assigned, 334 reported written, 0 in the table.
            graph.edges.append(
                EdgeDraft(
                    source=normalise_name(statement_name),
                    target=normalise_name(thing_name),
                    role="about",
                    turn_indexes=set(node_by_name[statement_name].turn_indexes),
                    evidence="",
                )
            )
            written += 1

    logger.info(
        "Assigned a subject to %d of %d statement(s).", written, len(statements)
    )
    return written
