# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subject assignment that may read more of the transcript before answering.

Same question as :mod:`~auto_ontology.conversations.semantic.subjects` -- which
thing does each statement describe -- and the same closed vocabulary to answer
it in. What differs is that the batch is no longer all the evidence there is:
the model may ask for turns it was not shown and answer afterwards.

The case this exists for. A statement is asked about in the batch of the
earliest turn that cites it, and a thing may be introduced thirty turns later,
so the candidate arrives as a name, a type and a line of description while the
turns that would settle it sit outside the call. Measured on one BEAM
transcript, twelve ``about`` edges point at a thing first said *after* their
own statement -- the widest gaps being thirty-one and twenty-eight turns -- and
eleven of the twelve are right, resolved from the description alone. So this is
not a common failure; it is the one shape of the question where the pass is
guessing rather than reading, and the tools make reading possible.

What is deliberately *not* claimed for it. The defect that prompted this --
a meeting's constraint attached to a request node from the turn beside it --
is not of that shape and this pass will not fix it: turn 0 was in front of the
batch already, and the subject the constraint wanted had been filtered out of
the vocabulary before the question was asked. More reading does not add a
candidate that was never offered. Expect this to earn its keep on vague nouns
("the prototype", "it") with two plausible bearers, and nowhere else.

Shape of the loop. One batch at a time, as before, and within a batch an
alternation of *act* and *observe*: the model returns either a tool call or its
answer, a tool call is executed and its result appended, and the next turn of
the loop sees it. Capped twice over -- on tool calls and on iterations --
because a loop that cannot end is worse than a thin answer, and an exhausted
budget answers from what it has rather than failing.

Why the house's action-union shape rather than native tool calling: it is what
:mod:`~auto_ontology.retrieval.text_to_sql.agents.information_agent` does, one
structured-output call per step reads the same as the rest of this package, and
the tools here are two in number with one argument each.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auto_ontology.conversations.semantic.subjects import (
    SubjectChoice,
    _accept,
    _batch,
    _drop_drafted_subjects,
    _first_turn,
    _is_thing,
    _labels,
    _render_statements,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auto_ontology.conversations.semantic.extract import GraphDraft, NodeDraft

logger = logging.getLogger(__name__)

__all__ = ["assign_subjects_agentic"]


#: Tool calls one batch may make. Six is two reads for each of the three
#: hardest statements a ten-turn batch has held, which is the most that has
#: ever looked necessary; the cap is on the batch rather than the statement
#: because the model decides which of its statements are hard and that is the
#: judgement worth leaving to it.
MAX_TOOL_CALLS = 6

#: Steps of the act/observe loop, tool calls and the final answer together.
#: Strictly above ``MAX_TOOL_CALLS`` so that a batch which spends its whole
#: tool budget still has a step left to answer in.
MAX_ITERATIONS = 8

#: Characters of one tool result passed back. A turn averages ~4,700 and the
#: cap is per call rather than per turn so that reading five turns cannot
#: quietly outgrow the batch text the answer is supposed to rest on.
MAX_OBSERVATION_CHARS = 12_000

#: Turns one ``find_turns`` call may name. The tool answers where a phrase was
#: used, and a phrase used in forty turns is not a lead worth following.
MAX_SEARCH_HITS = 5


#: Keys offered to the regrouping phase at once. One call for a transcript's
#: whole crop of them -- five on the BEAM transcript this was measured on --
#: rather than one call each, because the question is the same question and the
#: candidates are found without a model.
MAX_REGROUP_KEYS = 12


class Action(StrEnum):
    """What the model wants to do next."""

    TOOL = "tool"
    ANSWER = "answer"


class ToolName(StrEnum):
    """The two things it may ask for."""

    READ_TURNS = "read_turns"
    FIND_TURNS = "find_turns"


class Regrouping(BaseModel):
    """One property key and the single thing that really carries it."""

    key: str = Field(default="", description="The property key, exactly as shown.")
    thing: str = Field(
        default="",
        description="The one thing the readings belong to, by `n` label or exact name.",
    )


class AgentStep(BaseModel):
    """One step: either a tool call or the batch's answer."""

    action: Action = Field(
        description=(
            "'tool' to read more of the transcript first, 'answer' to give the "
            "subjects now."
        )
    )
    tool: ToolName | None = Field(
        default=None, description="Which tool, when action is 'tool'."
    )
    turns: list[int] = Field(
        default_factory=list,
        description="Turn numbers to read, for 'read_turns'.",
    )
    query: str = Field(
        default="",
        description="A phrase to look for in the transcript, for 'find_turns'.",
    )
    reason: str = Field(
        default="",
        description="One short line: what is unresolved, or why these answers.",
    )
    choices: list[SubjectChoice] = Field(
        default_factory=list,
        description="The statement/thing pairs, when action is 'answer'.",
    )
    regroupings: list[Regrouping] = Field(
        default_factory=list,
        description="The key/subject pairs, when answering the regrouping question.",
    )


_SYSTEM = """\
You decide what each statement in a conversation is ABOUT: which thing carries
the property, figure or fact it states.

You are given the complete list of things the conversation talks about. Choose
from that list and nothing else. You may not name a thing that is not listed,
and you may not answer with another statement.

How to choose:
  - A measurement belongs to what is being measured. "The prototype's label
    render latency improved to 140ms" is about the application, even though the
    sentence says "the prototype" and the application is named something else.
    The conversation renames its own subjects constantly; resolve the vague
    noun to the thing it refers to.
  - Pick the thing the figure is a property OF, not something merely present
    while it was measured. A latency measured in a browser is about the app,
    not about the browser; a quota enforced by an API is about the API.
  - Prefer the most specific thing that genuinely carries it. An address lookup
    response time is about the address lookup feature if the feature is listed,
    and about the application only if it is not.
  - A fact about a decision is about the thing decided on. "Switched to on-disk
    storage for caching" is about the application whose caching changed.
  - When two things could both carry it, read the surrounding turns: they say
    which one was under discussion.
  - When nothing in the list plausibly carries it, say so by leaving the
    statement out of your answer. An omitted statement loses one edge; a wrong
    one attaches a figure to the wrong thing and makes the graph state
    something false.
  - A statement marked REQUEST is the exception to all of the above: it is
    about the PERSON who asked, not about the part they asked about. Choose the
    person from the list. Asking for the error messages to be friendlier is a
    fact about the asker, not about the messages, and filing it under the
    messages makes it indistinguishable from a description of how they already
    behave. If no person is listed, leave it out.

You are shown only some of the transcript, and each thing is labelled with the
turn it was first said in. When a thing you are considering was introduced
outside the turns you can see, or when a statement names its subject by a vague
noun you cannot resolve from the turns you have, read more before answering:

  - `read_turns` with turn numbers returns those turns in full. Use it on the
    turn a candidate was first said in.
  - `find_turns` with a phrase returns the turns that use it. Use it when you
    need to know where a word like "the prototype" was used concretely.

Reading is not required. Answer directly when the turns you have already settle
it, and spend the tool calls on the statements that are genuinely unresolved.
"""

_REGROUP_SYSTEM = """\
You are shown measurements from one conversation that were recorded under the
SAME property key but attached to DIFFERENT subjects. For each key, say whether
those readings are one quantity that changed over the conversation, or separate
quantities that happen to share a name.

This matters because a revision is recognised by key and subject together. Two
readings of one quantity filed under two subjects are never compared, and the
fact that the value changed is lost without a trace.

How to decide:
  - One quantity. "3,000 labels a day" on the API and "4,800 labels a day" on
    the rate limiter are the same cap, stated twice as it was raised. Answer
    with the subject that genuinely carries it -- the thing the cap is OF.
  - Separate quantities. A bundle size on one library and a bundle size on a
    different library are two sizes; an uptime of the application and an uptime
    of its monitoring are two readings of two things. Leave these alone.
  - When the turns do not settle it, read them. Say nothing rather than guess:
    a wrong merge asserts that a value changed when it did not.

Answer only for the keys that are one quantity, naming the one subject that
carries it, exactly as it is written in the list.
"""

_REGROUP_TEMPLATE = """\
## Keys recorded against more than one subject

{candidates}

## Transcript, turns {first}-{last}

{text}

For each key that is really ONE quantity, give the key and the single subject
that carries it. Omit any key whose readings are separate quantities. Read more
turns first if you need them.
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


def _render_things(things: Sequence[NodeDraft]) -> str:
    """The closed vocabulary, labelled, with the turn each was first said in.

    The turn is the one addition over the non-agentic pass. Without it the
    model cannot tell which candidates it is seeing the evidence for and which
    it is taking on description alone, and so cannot tell what is worth a read.
    """
    lines = []
    for label, node in zip(_labels("n", things), things):
        detail = node.description.strip().split("\n")[0][:160]
        first = _first_turn(node)
        where = f" | first said in turn {first}" if node.turn_indexes else ""
        lines.append(
            f"- {label} | {node.name} | {node.type}{where}"
            + (f" | {detail}" if detail else "")
        )
    return "\n".join(lines)


def _read_turns(wanted: Sequence[int], turn_text: dict[int, str]) -> str:
    known = [index for index in dict.fromkeys(wanted) if index in turn_text]
    if not known:
        return "No such turns."
    blocks = [f"### turn {index}\n{turn_text[index]}" for index in known]
    return _clip("\n\n".join(blocks))


def _find_turns(query: str, turn_text: dict[int, str]) -> str:
    needle = query.strip().casefold()
    if not needle:
        return "No phrase given."
    hits = [
        index for index, text in sorted(turn_text.items()) if needle in text.casefold()
    ]
    if not hits:
        return f"No turn uses {query!r}."
    shown = hits[:MAX_SEARCH_HITS]
    blocks = [f"### turn {index}\n{turn_text[index]}" for index in shown]
    more = (
        f"\n\n({len(hits)} turns use it; showing {len(shown)}: {shown})"
        if len(hits) > len(shown)
        else ""
    )
    return _clip("\n\n".join(blocks) + more)


def _clip(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return text[:MAX_OBSERVATION_CHARS] + "\n[truncated]"


def _subjects_of(
    graph: GraphDraft, statements: Sequence[NodeDraft]
) -> dict[str, list[str]]:
    """Statement name -> the names of the things it was just made about.

    Read back off the drafted edges rather than tracked alongside them: the
    acceptance step is what decides which answers became edges, and a second
    record of the same decision is a second thing to keep true.
    """
    from auto_ontology.conversations.semantic.extract import normalise_name

    by_normalised = {normalise_name(node.name): node.name for node in graph.nodes}
    out: dict[str, list[str]] = {}
    wanted = {node.name for node in statements}
    for edge in graph.edges:
        if edge.role != "about":
            continue
        source = by_normalised.get(edge.source)
        target = by_normalised.get(edge.target)
        if source in wanted and target:
            out.setdefault(source, []).append(target)
    return out


def _regroup_candidates(
    statements: Sequence[NodeDraft], subjects: dict[str, list[str]]
) -> dict[str, list[NodeDraft]]:
    """Keys whose readings were attached to more than one thing.

    Found without a model, which is what makes this cheap enough to always run.
    A key under two subjects is one of two things and nothing else: a quantity
    whose subject was decided inconsistently -- in which case the revision
    between its readings does not exist and never will -- or two quantities
    that share a name, which is common and wanted. Only a model separates them,
    but only arithmetic is needed to find the question.

    Measured on one BEAM transcript: five keys, among them a daily ceiling
    recorded against the API at 4,800 and against the rate limiter at 3,000,
    which is a revision the graph does not hold.
    """
    from auto_ontology.conversations.semantic.extract import UNREALISED_KINDS

    by_key: dict[str, list[NodeDraft]] = {}
    for node in statements:
        key = (node.key or "").strip()
        if not key or not (node.value or "").strip():
            continue
        if set(node.kinds) & set(UNREALISED_KINDS):
            continue
        by_key.setdefault(key, []).append(node)

    candidates = {}
    for key, nodes in sorted(by_key.items()):
        named = {name for node in nodes for name in subjects.get(node.name, ())}
        values = {(node.value or "").strip().casefold() for node in nodes}
        if len(named) > 1 and len(values) > 1:
            candidates[key] = nodes
    return dict(list(candidates.items())[:MAX_REGROUP_KEYS])


def _render_candidates(
    candidates: dict[str, list[NodeDraft]], subjects: dict[str, list[str]]
) -> str:
    lines = []
    for key, nodes in candidates.items():
        lines.append(f"- key `{key}`")
        for node in sorted(nodes, key=_first_turn):
            where = ", ".join(subjects.get(node.name, ())) or "(no subject)"
            lines.append(
                f"    - {node.value} | {node.name} | turn {_first_turn(node)} "
                f"| currently about: {where}"
            )
    return "\n".join(lines)


def _run_tool(step: AgentStep, turn_text: dict[int, str]) -> str:
    if step.tool == ToolName.READ_TURNS:
        return _read_turns(step.turns, turn_text)
    if step.tool == ToolName.FIND_TURNS:
        return _find_turns(step.query, turn_text)
    return "Unknown tool. Answer with what you have."


def _converse(
    messages: list[Any], turn_text: dict[int, str], label: str
) -> AgentStep | None:
    """Run the act/observe loop and return the step that answered, if one did.

    Shared by both phases because the loop is the same loop: the question
    differs, the budget and the tools do not.
    """
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    calls = 0
    for _ in range(MAX_ITERATIONS):
        try:
            step = invoke_with_structured_output(
                get_llm_client(temperature=0.0, max_tokens=4096),
                messages,
                AgentStep,
            )
        except Exception:
            logger.exception("Agent step failed for %s.", label)
            return None
        if step is None:
            return None
        if step.action == Action.ANSWER or calls >= MAX_TOOL_CALLS:
            return step
        observation = _run_tool(step, turn_text)
        calls += 1
        logger.debug(
            "%s: %s(%s) -- %s", label, step.tool, step.turns or step.query, step.reason
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Result of {step.tool} ({calls} of {MAX_TOOL_CALLS} "
                    f"tool calls used):\n\n{observation}"
                )
            )
        )
    return None


def _regroup(
    graph: GraphDraft,
    statements: Sequence[NodeDraft],
    things: Sequence[NodeDraft],
    turn_text: dict[int, str],
) -> int:
    """Re-point the subjects of readings that were split across two things.

    Returns how many ``about`` edges moved. The answer is a **subject**, not a
    revision edge, and that is the point: ``derive`` reads the subject and
    emits ``supersedes`` itself, so correcting the subject leaves one source of
    truth for what a reading belongs to. Writing the revision here instead
    would put a second one beside it, and the two would disagree the first time
    a later pass changed either.

    The absent revision is the defect this reaches and a check of the emitted
    edges cannot: a pair that was never grouped emits nothing, so there is
    nothing downstream to inspect. What makes it findable at all is that the
    readings still share a key.
    """
    from auto_ontology.conversations.semantic.extract import normalise_name

    subjects = _subjects_of(graph, statements)
    candidates = _regroup_candidates(statements, subjects)
    if not candidates:
        return 0

    indexes = sorted(turn_text)
    cited = sorted(
        {_first_turn(node) for nodes in candidates.values() for node in nodes}
    )
    prompt = _REGROUP_TEMPLATE.format(
        candidates=_render_candidates(candidates, subjects),
        first=indexes[0] if indexes else 0,
        last=indexes[-1] if indexes else 0,
        text="\n\n".join(
            f"### turn {i}\n{turn_text[i]}" for i in cited if i in turn_text
        ),
    )
    step = _converse(
        [SystemMessage(content=_REGROUP_SYSTEM), HumanMessage(content=prompt)],
        turn_text,
        f"regrouping {len(candidates)} key(s)",
    )
    if step is None:
        return 0

    thing_by_name = {node.name: node for node in things}
    thing_by_name.update(dict(zip(_labels("n", things), things)))
    moved = 0
    for answer in step.regroupings:
        nodes = candidates.get(answer.key.strip())
        target = thing_by_name.get(answer.thing.strip())
        if not nodes or target is None:
            logger.debug(
                "Regrouping ignored: key=%r thing=%r is outside what was asked.",
                answer.key,
                answer.thing,
            )
            continue
        wanted = normalise_name(target.name)
        sources = {normalise_name(node.name) for node in nodes}
        for edge in graph.edges:
            if (
                edge.role == "about"
                and edge.source in sources
                and edge.target != wanted
            ):
                edge.target = wanted
                moved += 1
    if moved:
        logger.info(
            "Regrouped %d subject edge(s) over %d key(s) recorded against more "
            "than one thing.",
            moved,
            len(candidates),
        )
    return moved


def assign_subjects_agentic(graph: GraphDraft, turns: Sequence[dict[str, Any]]) -> int:
    """Replace every statement's subject edge, reading more turns when needed.

    Signature and effect match
    :func:`~auto_ontology.conversations.semantic.subjects.assign_subjects` --
    in place on *graph*, returning how many ``about`` edges were written -- so
    the pipeline can choose between them at the call site and the two can be
    compared on one transcript without anything else differing.

    A batch that exhausts its budget without answering writes nothing for its
    statements, and a failed call likewise. Both leave the statements with no
    subject, which is the same bargain the non-agentic pass strikes: a missing
    edge is recoverable by rerunning, a wrong one is indistinguishable later
    from a true one.
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
        "Assigning subjects agentically for %d statement(s) over %d batch(es), "
        "from %d thing(s); dropped %d drafted subject edge(s).",
        len(statements),
        len(batches),
        len(things),
        dropped,
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
        step = _converse(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
            turn_text,
            f"turns {indexes[0]}-{indexes[-1]}",
        )
        choices: list[SubjectChoice] = list(step.choices) if step else []

        for statement_name, thing_name in _accept(choices, things, members).items():
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
    # Second phase, and the one the first cannot do for itself: a reading split
    # from its own revision across two subjects is a question about the whole
    # transcript, and every batch has answered by now.
    _regroup(graph, statements, things, turn_text)
    return written
