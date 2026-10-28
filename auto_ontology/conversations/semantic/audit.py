# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checking the edges no other pass looks at: the derived ones and the subjects.

:mod:`verify` reads *extracted* edges back against their turns, and runs before
consolidation, so it has seen nothing of what the later phases produce. The
subject pass chooses a subject and nothing checks it; derivation emits
``supersedes``, ``contradicts`` and ``precedes`` from field comparisons and
nothing checks those either. Two false edges of the twenty-four derived on one
BEAM transcript were found by reading them by hand:

    contradicts  Error rate reduced from 9% to 2%   ->  Error rate 9% before
                 null checks

Both were extracted from one turn, which says the rate *went from* 9% to 2% --
a before and an after in one sentence, recorded as the transcript disagreeing
with itself.

    supersedes   180ms spikes under rapid input     ->  Queue wait above 450ms

A peak-spike reading and an average-wait reading, chained because the extractor
filed both under one property key.

Two instruments here, and the cheap one first.

:func:`audit_structure` is arithmetic. A containment relation holding in both
directions, an edge from a node to itself, a ``precedes`` disagreeing with the
dates it was derived from: none of these need a model to be known wrong, and
one of each kind exists in the graph this was written against. Free, so it
always runs.

:func:`audit_revisions` puts the pairs that survive to a model, with the turns
they came from, and asks the one question arithmetic cannot: is this one
quantity that changed, or two quantities that share a key. Twenty-four pairs is
one call.

:func:`check_subjects` asks the same of the ``about`` edges, and is worth
having even though the subject pass just chose them, because it asks a
different question. Choosing a subject is a hundred-and-fifty-way decision made
from names and one-line descriptions; confirming one is a yes-or-no question
asked with the turn in front of it. A model that cannot find the right name in
a long ballot can still see that the name it picked is wrong, and that is the
failure this is for -- a reading filed against a plausible neighbour rather
than its own subject is invisible to everything downstream, and silently
decides which readings group.

Narrowed to the statements carrying a property key. There are two hundred and
forty-seven ``about`` edges and only forty-one such statements, and the rest
feed nothing: a subject that files a topic under a heading changes no other
edge, while a subject on a *keyed* reading decides which readings group and
therefore which revisions exist at all.

Everything here runs **before** the edges it judges are written, and that is
not a preference but the only arrangement that does anything: nothing in the
store deletes or retargets an edge, so a check placed after a write can print
and nothing more. :func:`resolve_mutual` and :func:`check_subjects` work on the
draft graph; :func:`check_derived` sits in the gap between ``derive_edges``
returning and ``write_edges`` committing, which is the one moment a derived
edge both exists and can still be changed.

Removal is confined to the verdicts that say there is nothing to keep -- a pair
that is not a revision, a containment holding both ways, an edge to itself.
Everything else is repaired in place: a pair facing the wrong way is turned
round, a reading filed under the wrong thing is moved. The precedent for
keeping removal narrow is :mod:`verify`, which dropped edges on its first outing
and cost ``information_extraction`` half its score -- a doubtful edge deleted is
one the reader has no way to ask for again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "RevisionReport",
    "StructureReport",
    "SubjectReport",
    "audit_revisions",
    "audit_structure",
    "check_derived",
    "check_subjects",
    "resolve_mutual",
    "review_merges",
    "unsound_pairs",
]


#: Roles whose meaning forbids the reverse edge. A component of a thing is not
#: a thing that contains it, and a reading does not supersede what supersedes
#: it. ``refines`` is not here: a pair of statements can each add to the other.
ANTISYMMETRIC = ("has_component", "builds", "supersedes", "precedes", "covers")

#: Revision pairs put to the model in one call. The transcript this was written
#: against has twenty-four in total, so the batch exists for the transcript
#: that has two hundred rather than for this one.
BATCH_PAIRS = 24

#: Where an edge sat in the list it was checked from, carried on the row so a
#: verdict can be traced back to the object it judged.
PLACE = "_place"

RevisionVerdict = Literal["changed", "reversed", "separate"]


@dataclass
class StructureReport:
    """What arithmetic alone found wrong."""

    #: Pairs where a role holds in both directions. Logically impossible.
    mutual: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    #: Edges from a node to itself.
    loops: list[dict[str, Any]] = field(default_factory=list)
    #: ``precedes`` edges whose endpoints' dates say the opposite.
    misdated: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.mutual) + len(self.loops) + len(self.misdated)


@dataclass
class PairVerdict:
    """One revision pair's verdict, with the model's reason."""

    edge: dict[str, Any]
    verdict: RevisionVerdict
    reason: str = ""


@dataclass
class RevisionReport:
    """What the model made of the derived revision edges."""

    changed: list[PairVerdict] = field(default_factory=list)
    reversed_: list[PairVerdict] = field(default_factory=list)
    separate: list[PairVerdict] = field(default_factory=list)
    #: Pairs never put to a model, because their turns were not available.
    unchecked: list[dict[str, Any]] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.changed) + len(self.reversed_) + len(self.separate)


class PairAnswer(BaseModel):
    """One pair, judged."""

    number: int = Field(default=0, description="The pair's number, as shown.")
    verdict: Literal["changed", "reversed", "separate"] = Field(
        default="changed",
        description=(
            "'changed' if one quantity moved, 'reversed' if it moved the other "
            "way, 'separate' if these are two different quantities."
        ),
    )
    reason: str = Field(default="", description="One short clause.")


class PairAnswers(BaseModel):
    answers: list[PairAnswer] = Field(default_factory=list)


_SYSTEM = """You check claims that a value in a conversation was revised.

You are shown turns of a transcript, then numbered PAIRS of readings that were
recorded under the same property for the same thing. Each pair was read as one
quantity whose value changed. Say whether that is what the turns describe.

  changed    The turns describe one quantity, and this value replaced that one.
  reversed   One quantity, but the pair has the direction wrong: the reading
             called current is the earlier one.
  separate   Two different quantities that happen to share a property name. A
             peak reading beside an average reading is two measurements, not
             one that moved. A limit imposed from outside beside a limit
             enforced internally may be two limits.

Read "from one value to another" carefully. A turn saying a figure went FROM
something TO something else states one before and one after in a single
sentence. Both readings are true at once, of two moments, and the pair is
'changed' -- not a disagreement, however far apart the two values are.

A target or a goal is never one of the readings: those are excluded before you
see them, so if a pair looks like a goal beside a measurement, answer
'separate'.

Judge only what the turns say. When they do not settle it, answer 'changed' and
say so in the reason: leaving a doubtful revision in place costs less than
removing a true one, which the reader cannot ask for again.

Give a verdict for every pair, by its number. Keep each reason to one clause.
"""

_TEMPLATE = """\
## Transcript

{text}

## Pairs

{pairs}
"""


def audit_structure(
    nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
) -> StructureReport:
    """Everything about the edges that is wrong without reading any turn.

    No model and no transcript: these are contradictions inside the graph's own
    claims. One of each kind was present on the transcript this was written
    against -- a feature recorded as containing one of its parts alongside the
    exact reverse, which also made that part read as two components in the note
    the reader sees.
    """
    by_id = {str(node["id"]): node for node in nodes}
    report = StructureReport()

    seen: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for edge in edges:
        source, target = str(edge["source_id"]), str(edge["target_id"])
        role = str(edge.get("role") or "")
        if source == target:
            report.loops.append(dict(edge))
            continue
        if role in ANTISYMMETRIC:
            if (mirror := seen.get((target, source, role))) is not None:
                report.mutual.append((dict(mirror), dict(edge)))
            seen[(source, target, role)] = edge
        if role == "precedes":
            earlier = by_id.get(source, {}).get("occurred_on")
            later = by_id.get(target, {}).get("occurred_on")
            if earlier and later and earlier > later:
                report.misdated.append(dict(edge))

    if report.total:
        logger.info(
            "Structure check: %d mutual pair(s), %d self-loop(s), %d misdated "
            "precedes.",
            len(report.mutual),
            len(report.loops),
            len(report.misdated),
        )
    return report


def _describe(
    edge: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> str | None:
    source = by_id.get(str(edge["source_id"]))
    target = by_id.get(str(edge["target_id"]))
    if source is None or target is None:
        return None
    return (
        f"current: {source.get('name')} = {source.get('value')} "
        f"(turn {max(source.get('turn_indexes') or [0])})\n"
        f"   earlier: {target.get('name')} = {target.get('value')} "
        f"(turn {max(target.get('turn_indexes') or [0])})\n"
        f"   property: {source.get('attribute_key')}"
    )


def _turns_of(
    edge: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> list[int]:
    out: set[int] = set()
    for end in (edge["source_id"], edge["target_id"]):
        node = by_id.get(str(end))
        if node:
            out.update(node.get("turn_indexes") or ())
    return sorted(out)


def audit_revisions(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    texts: Mapping[int, str],
) -> RevisionReport:
    """Read each derived revision back against the turns its readings came from.

    *edges* is filtered here rather than by the caller: the roles this can
    judge are exactly the two derivation emits from a value comparison, and a
    caller passing the whole graph should get the two checked rather than an
    error.

    A pair whose turns are not in *texts* is left unchecked rather than guessed
    at, which is the same bargain :mod:`verify` strikes -- an unasked question
    is not a verdict.
    """
    by_id = {str(node["id"]): node for node in nodes}
    pairs = [
        edge for edge in edges if str(edge.get("role")) in ("supersedes", "contradicts")
    ]
    report = RevisionReport()
    if not pairs:
        return report

    ready: list[tuple[dict[str, Any], str, list[int]]] = []
    for edge in pairs:
        described = _describe(edge, by_id)
        turns = [index for index in _turns_of(edge, by_id) if index in texts]
        if described is None or not turns:
            report.unchecked.append(dict(edge))
            continue
        ready.append((dict(edge), described, turns))
    if not ready:
        return report

    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    for start in range(0, len(ready), BATCH_PAIRS):
        batch = ready[start : start + BATCH_PAIRS]
        indexes = sorted({index for _, _, turns in batch for index in turns})
        prompt = _TEMPLATE.format(
            text="\n\n".join(f"### turn {i}\n{texts[i]}" for i in indexes),
            pairs="\n".join(
                f"{number}. {described}"
                for number, (_, described, _) in enumerate(batch, start=1)
            ),
        )
        try:
            answer = invoke_with_structured_output(
                get_llm_client(temperature=0.0, max_tokens=2048),
                [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
                PairAnswers,
            )
        except Exception:
            logger.exception(
                "Could not check %d revision pair(s) -- they stay as derived.",
                len(batch),
            )
            report.unchecked.extend(edge for edge, _, _ in batch)
            continue
        if answer is None:
            report.unchecked.extend(edge for edge, _, _ in batch)
            continue

        judged: dict[int, PairAnswer] = {}
        for item in answer.answers:
            if 1 <= item.number <= len(batch) and item.number not in judged:
                judged[item.number] = item
        for number, (edge, _, _) in enumerate(batch, start=1):
            item = judged.get(number)
            if item is None:
                report.unchecked.append(edge)
                continue
            verdict = PairVerdict(edge=edge, verdict=item.verdict, reason=item.reason)
            if item.verdict == "separate":
                report.separate.append(verdict)
            elif item.verdict == "reversed":
                report.reversed_.append(verdict)
            else:
                report.changed.append(verdict)

    logger.info(
        "Revision check: %d changed, %d reversed, %d separate, %d unchecked, "
        "of %d derived pair(s).",
        len(report.changed),
        len(report.reversed_),
        len(report.separate),
        len(report.unchecked),
        len(pairs),
    )
    return report


_SUBJECT_SYSTEM = """You check which thing a measurement was recorded against.

You are shown turns of a transcript, then numbered READINGS. Each reading is a
value taken from the conversation together with the thing it was filed under.
Say whether the turns support that filing.

  right    The turns say this value is a property of that thing.
  wrong    The turns say it is a property of something else, which you name.
  unsure   The turns do not settle it.

A thing named more precisely than the turn is still right. If a turn states a
value of the whole system and the reading is filed under a part of it that the
turn has been discussing, that is right, not wrong.

Watch for a reading filed against a neighbouring measurement instead of the
thing measured. A duration belongs to whatever takes that long, not to another
figure mentioned nearby.

Answer 'unsure' rather than guessing. Give a verdict for every reading.
"""


class SubjectAnswer(BaseModel):
    """One reading's subject, judged."""

    number: int = Field(default=0, description="The reading's number, as shown.")
    verdict: Literal["right", "wrong", "unsure"] = Field(default="unsure")
    belongs_to: str = Field(
        default="",
        description="When 'wrong', the thing the turns say it belongs to.",
    )
    reason: str = Field(default="", description="One short clause.")


class SubjectAnswers(BaseModel):
    answers: list[SubjectAnswer] = Field(default_factory=list)


@dataclass
class SubjectVerdict:
    """One ``about`` edge's verdict, and what became of it."""

    statement: str
    filed_under: str
    verdict: Literal["right", "wrong", "unsure"]
    belongs_to: str = ""
    reason: str = ""
    #: Set when ``belongs_to`` resolved to a thing and the edge was retargeted.
    moved: bool = False


@dataclass
class SubjectReport:
    """What the model made of the subjects on keyed readings."""

    right: list[SubjectVerdict] = field(default_factory=list)
    wrong: list[SubjectVerdict] = field(default_factory=list)
    unsure: list[SubjectVerdict] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.right) + len(self.wrong) + len(self.unsure)

    @property
    def moved(self) -> int:
        return sum(1 for item in self.wrong if item.moved)


def check_subjects(graph: Any, turns: Sequence[Mapping[str, Any]]) -> SubjectReport:
    """Confirm every keyed reading's subject, and retarget the ones that are wrong.

    Before the write, for the same reason :func:`resolve_mutual` is: nothing
    deletes or rewrites a stored edge, so a check that runs afterwards can only
    ever print. Against the draft graph the endpoints are names, which is also
    what the model answers in, so a better subject can be looked up and the
    ``about`` edge pointed at it.

    A named subject that resolves to no thing is kept as a verdict and nothing
    else. That is the one asymmetry worth keeping from the report-only version:
    the model answers in free text, and writing a subject the graph does not
    have is worse than keeping one it does.

    Narrowed to statements carrying a property key. The others file a topic
    under a heading and change no other edge, while a subject on a keyed
    reading decides which readings group and therefore which revisions exist.
    """
    from auto_ontology.conversations.semantic.extract import normalise_name
    from auto_ontology.conversations.semantic.subjects import _is_thing
    from auto_ontology.conversations.semantic.windows import turn_block

    report = SubjectReport()
    nodes = list(getattr(graph, "nodes", ()))
    thing_by_key = {
        normalise_name(node.name): node for node in nodes if _is_thing(node)
    }
    node_by_key = {normalise_name(node.name): node for node in nodes}
    texts = {
        int(turn["turn_index"]): turn_block(
            str(turn.get("question") or "").strip(),
            str(turn.get("answer") or "").strip(),
        )
        for turn in turns
    }

    ready: list[tuple[Any, Any, Any, str, list[int]]] = []
    for edge in getattr(graph, "edges", ()):
        if str(getattr(edge, "role", "")) != "about":
            continue
        source = node_by_key.get(str(edge.source))
        target = node_by_key.get(str(edge.target))
        if source is None or target is None:
            continue
        if not (getattr(source, "key", "") or "").strip():
            continue
        cited = sorted(index for index in source.turn_indexes if index in texts)
        if not cited:
            report.unchecked.append(source.name)
            continue
        ready.append(
            (
                edge,
                source,
                target,
                f"{source.name}: {source.key} = {source.value} "
                f"(turn {max(cited)})\n   filed under: {target.name}",
                cited,
            )
        )
    if not ready:
        return report

    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    logger.info(
        "Checking the subject of %d keyed reading(s) in %d call(s).",
        len(ready),
        -(-len(ready) // BATCH_PAIRS),
    )
    for start in range(0, len(ready), BATCH_PAIRS):
        batch = ready[start : start + BATCH_PAIRS]
        indexes = sorted({index for *_, cited in batch for index in cited})
        prompt = _TEMPLATE.format(
            text="\n\n".join(f"### turn {i}\n{texts[i]}" for i in indexes),
            pairs="\n".join(
                f"{number}. {described}"
                for number, (_, _, _, described, _) in enumerate(batch, start=1)
            ),
        ).replace("## Pairs", "## Readings")
        try:
            answer = invoke_with_structured_output(
                get_llm_client(temperature=0.0, max_tokens=2048),
                [
                    SystemMessage(content=_SUBJECT_SYSTEM),
                    HumanMessage(content=prompt),
                ],
                SubjectAnswers,
            )
        except Exception:
            logger.exception(
                "Could not check %d subject(s) -- they stay as assigned.", len(batch)
            )
            report.unchecked.extend(source.name for _, source, *_ in batch)
            continue
        if answer is None:
            report.unchecked.extend(source.name for _, source, *_ in batch)
            continue

        judged: dict[int, SubjectAnswer] = {}
        for item in answer.answers:
            if 1 <= item.number <= len(batch) and item.number not in judged:
                judged[item.number] = item
        for number, (edge, source, target, _, _) in enumerate(batch, start=1):
            item = judged.get(number)
            if item is None:
                report.unchecked.append(source.name)
                continue
            verdict = SubjectVerdict(
                statement=source.name,
                filed_under=target.name,
                verdict=item.verdict,
                belongs_to=item.belongs_to,
                reason=item.reason,
            )
            if item.verdict == "wrong":
                better = thing_by_key.get(normalise_name(item.belongs_to))
                if better is not None and better.name != target.name:
                    edge.target = normalise_name(better.name)
                    verdict.moved = True
                    logger.info(
                        "Moved %r from %r to %r -- %s",
                        source.name,
                        target.name,
                        better.name,
                        item.reason,
                    )
                else:
                    logger.info(
                        "%r is filed under %r and the turns say %r, which is no "
                        "thing in this graph -- left alone.",
                        source.name,
                        target.name,
                        item.belongs_to,
                    )
            getattr(report, item.verdict).append(verdict)

    logger.info(
        "Subject check: %d right, %d wrong (%d moved), %d unsure, %d unchecked.",
        len(report.right),
        len(report.wrong),
        report.moved,
        len(report.unsure),
        len(report.unchecked),
    )
    return report


_MUTUAL_SYSTEM = """You fix containment relations recorded in both directions.

Two statements were taken from a conversation about the same two things, each
claiming the first contains, builds or comes before the second. Both cannot
hold. You are shown each direction with the sentence it was taken from.

Choose the direction the sentences support.

  a        The first direction is right; drop the second.
  b        The second direction is right; drop the first.
  neither  The sentences do not settle it; both are kept.

Read each sentence as evidence for the direction it is printed under, and
notice when it argues against it instead. A sentence saying one thing is added
while waiting for another makes the first a part of the second, whichever
direction it was filed as.

Prefer the direction where the container is the broader thing: a part may be
mentioned in a sentence about the whole, but a whole is not a part of its own
component.

Answer 'neither' rather than guessing. Give a verdict for every numbered pair.
"""

_MUTUAL_TEMPLATE = """\
## Pairs

{pairs}
"""


class MutualAnswer(BaseModel):
    """Which direction of one mutual pair to keep."""

    number: int = Field(default=0, description="The pair's number, as shown.")
    keep: Literal["a", "b", "neither"] = Field(default="neither")
    reason: str = Field(default="", description="One short clause.")


class MutualAnswers(BaseModel):
    answers: list[MutualAnswer] = Field(default_factory=list)


def resolve_mutual(graph: Any) -> int:
    """Drop the wrong half of every containment recorded in both directions.

    Before the graph is written rather than after, which is the whole reason
    this exists separately from :func:`audit_structure`: there is no path that
    deletes a stored edge, and an edge never written needs none. The two halves
    arrive from different extraction windows and nothing between them ever
    compared the two, so this is the first and only place they meet.

    Decided from the ``evidence`` the extractor already kept on each edge, not
    from the transcript. The sentence is on the row, which makes this one call
    with no lookup, and in practice the sentence under the wrong direction is
    the thing that refutes it.

    Returns how many edges were dropped. ``neither`` keeps both, on the same
    footing as everywhere else here: a relation wrongly removed is one the
    reader cannot ask for again, and a mutual pair at least still says the two
    things are related.
    """
    edges = list(getattr(graph, "edges", ()))
    seen: dict[tuple[str, str, str], Any] = {}
    pairs: list[tuple[Any, Any]] = []
    for edge in edges:
        role = str(getattr(edge, "role", "") or "")
        if role not in ANTISYMMETRIC:
            continue
        source = str(getattr(edge, "source", "")).casefold()
        target = str(getattr(edge, "target", "")).casefold()
        if source == target:
            continue
        if (mirror := seen.get((target, source, role))) is not None:
            pairs.append((mirror, edge))
        else:
            seen[(source, target, role)] = edge
    if not pairs:
        return 0

    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    described = "\n".join(
        f"{number}. a: {first.source} {first.role} {first.target}\n"
        f'   sentence: "{(first.evidence or "").strip()}"\n'
        f"   b: {second.source} {second.role} {second.target}\n"
        f'   sentence: "{(second.evidence or "").strip()}"'
        for number, (first, second) in enumerate(pairs, start=1)
    )
    try:
        answer = invoke_with_structured_output(
            get_llm_client(temperature=0.0, max_tokens=1024),
            [
                SystemMessage(content=_MUTUAL_SYSTEM),
                HumanMessage(content=_MUTUAL_TEMPLATE.format(pairs=described)),
            ],
            MutualAnswers,
        )
    except Exception:
        logger.exception(
            "Could not resolve %d mutual pair(s) -- both halves are kept.", len(pairs)
        )
        return 0
    if answer is None:
        return 0

    keep: dict[int, str] = {}
    for item in answer.answers:
        if 1 <= item.number <= len(pairs) and item.number not in keep:
            keep[item.number] = item.keep

    doomed: list[Any] = []
    for number, (first, second) in enumerate(pairs, start=1):
        chosen = keep.get(number, "neither")
        if chosen == "a":
            doomed.append(second)
        elif chosen == "b":
            doomed.append(first)
        else:
            logger.info(
                "Kept both directions of %s %s %s -- undecided.",
                first.source,
                first.role,
                first.target,
            )
    for edge in doomed:
        logger.info(
            "Dropped %s %s %s -- recorded in both directions.",
            edge.source,
            edge.role,
            edge.target,
        )
        graph.edges.remove(edge)
    return len(doomed)


_MERGE_SYSTEM = """You check whether two names from a conversation are one thing.

Each numbered GROUP is a name that will be kept, followed by one or more names
that will be folded into it and deleted. They were grouped because the names
embed closely and share a type. Each name is shown with what it was taken to
mean and the sentence it was taken from. Say whether the grouping is right.

The sentences are the evidence. Two names drawn from one sentence are usually
one thing; two names drawn from sentences about different parts of a system are
usually two, however alike the names read.

  same       One thing under two names. A longer and a shorter form of the
             same name, a plural, an abbreviation, a spelling.
  different  Two things that happen to be named alike. Two measurements of
             different parts of one system are the clearest case: the names
             differ by one word and the difference between the two values is
             the thing a reader wants.

Lean towards 'different' when the names differ by a word that narrows what is
being measured or built, and towards 'same' when they differ only in form.

Per group, list the names that do NOT belong to it. Leave that list empty when
the whole group is one thing. A name you leave out stays as its own node, which
costs a duplicate; a name you wrongly fold in deletes a distinct thing and
cannot be recovered.
"""

_MERGE_TEMPLATE = """\
## Groups

{groups}
"""


class MergeAnswer(BaseModel):
    """Which names in one group do not belong to it."""

    number: int = Field(default=0, description="The group's number, as shown.")
    verdict: Literal["same", "different"] = Field(default="same")
    not_the_same: list[str] = Field(
        default_factory=list,
        description="Names to keep separate. Empty when the group is one thing.",
    )
    reason: str = Field(default="", description="One short clause.")


class MergeAnswers(BaseModel):
    answers: list[MergeAnswer] = Field(default_factory=list)


def review_merges(merges: list[Any], nodes: Sequence[Any]) -> list[Any]:
    """Return the planned merges worth applying, having asked about each group.

    Written to be passed as ``consolidate_graph(review=...)``, which hands it
    the plan rather than the outcome. That is deliberate and is the answer to
    "who unmerges": nobody can. Absorbing a duplicate unions its turns and
    evidence into the survivor and deletes the row, so the only moment a merge
    is reversible is before it happens.

    A group is not rejected whole unless the model says so. It names the
    members that do not belong, they are dropped from the group, and the rest
    still merge -- a cluster of four spellings with one impostor is three good
    merges and one mistake, and refusing the lot would keep three duplicates to
    avoid one error.

    The names are checked against the ones shown, as everywhere else here: a
    model answering with a name this call did not raise is ignored rather than
    acted on.
    """
    if not merges:
        return []

    detail = {}
    for node in nodes:
        from auto_ontology.conversations.semantic.extract import normalise_name

        detail[normalise_name(node.name)] = node

    def describe(key: str) -> str:
        """One name, with enough of the transcript to tell it from its neighbour.

        The ``evidence`` quote matters more here than the description does. A
        description is the extractor's paraphrase and two near-identical names
        tend to get near-identical paraphrases, which is exactly the case this
        call exists to split; the quote is the sentence the name came out of,
        and two measurements of different things are almost always two
        different sentences. It is already on the node, so this costs no
        lookup -- the same reason the reversed-containment check reads it.
        """
        node = detail.get(key)
        if node is None:
            return key
        lines = [f"{node.name} [{node.type}]"]
        if first := (node.description or "").strip().split("\n")[0][:120]:
            lines.append(f"      meaning: {first}")
        if quote := (getattr(node, "evidence", "") or "").strip()[:200]:
            lines.append(f'      said as: "{quote}"')
        return "\n".join(lines)

    groups = "\n".join(
        f"{number}. keep: {describe(merge.canonical)}\n"
        + "\n".join(f"   fold in: {describe(name)}" for name in merge.duplicates)
        for number, merge in enumerate(merges, start=1)
    )

    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    logger.info("Reviewing %d planned merge(s).", len(merges))
    try:
        answer = invoke_with_structured_output(
            get_llm_client(temperature=0.0, max_tokens=2048),
            [
                SystemMessage(content=_MERGE_SYSTEM),
                HumanMessage(content=_MERGE_TEMPLATE.format(groups=groups)),
            ],
            MergeAnswers,
        )
    except Exception:
        logger.exception("Could not review the merges -- applying the plan as is.")
        return merges
    if answer is None:
        return merges

    judged: dict[int, MergeAnswer] = {}
    for item in answer.answers:
        if 1 <= item.number <= len(merges) and item.number not in judged:
            judged[item.number] = item

    kept: list[Any] = []
    rejected = 0
    for number, merge in enumerate(merges, start=1):
        item = judged.get(number)
        if item is None:
            kept.append(merge)
            continue
        from auto_ontology.conversations.semantic.extract import normalise_name

        offered = set(merge.duplicates)
        spared = {
            key
            for name in item.not_the_same
            if (key := normalise_name(name)) in offered
        }
        if item.verdict == "different" and not spared:
            spared = offered
        if not spared:
            kept.append(merge)
            continue
        survivors = tuple(name for name in merge.duplicates if name not in spared)
        rejected += len(spared)
        for name in sorted(spared):
            logger.info(
                "Kept %r apart from %r -- %s", name, merge.canonical, item.reason
            )
        if survivors:
            kept.append(
                type(merge)(
                    canonical=merge.canonical,
                    duplicates=survivors,
                    similarity=merge.similarity,
                )
            )

    logger.info(
        "Merge review: %d name(s) kept apart, %d group(s) still merging.",
        rejected,
        len(kept),
    )
    return kept


def check_derived(
    derived: list[Any],
    nodes: Sequence[Mapping[str, Any]],
    texts: Mapping[int, str],
) -> tuple[int, int]:
    """Correct the derived edges in place, before any of them is written.

    *derived* is mutated: the pairs that are not a revision at all are removed
    from it, and the ones recorded the wrong way round have their ends swapped.
    Returns ``(turned round, dropped)``.

    Here because this is the only moment that works. The derived edges do not
    exist before :func:`derive_edges` returns, and after
    ``write_edges`` nothing in the store deletes or retargets one, so a check
    on either side of this line could do nothing but print.

    Dropping is confined to the one verdict that says there is no relation to
    keep. A ``reversed`` pair is a real revision facing the wrong way, which is
    why it is swapped rather than removed -- deleting it would lose the true
    fact that the two readings are of one quantity.
    """
    if not derived:
        return 0, 0

    # Numbered, because both checks copy the rows they are given and a verdict
    # has to find its way back to the edge object it was about.
    rows = [
        {
            "source_id": str(edge.source_id),
            "target_id": str(edge.target_id),
            "role": str(edge.role),
            PLACE: number,
        }
        for number, edge in enumerate(derived)
    ]

    structure = audit_structure(nodes, rows)
    doomed = {row[PLACE] for row in structure.loops}
    doomed |= {row[PLACE] for _, row in structure.mutual}

    by_id = {str(node["id"]): node for node in nodes}

    def named(edge: Mapping[str, Any]) -> str:
        ends = [by_id.get(str(edge.get(end))) for end in ("source_id", "target_id")]
        return " / ".join(
            f"{node['name']} ({node.get('value') or '-'})" if node else "?"
            for node in ends
        )

    report = audit_revisions(nodes, rows, texts)
    turned = 0
    for item in report.reversed_:
        place = item.edge[PLACE]
        if place in doomed:
            continue
        # ``SemanticEdge`` is frozen, deliberately: an edge is a decision, and
        # a decision is replaced rather than edited.
        edge = derived[place]
        derived[place] = replace(
            edge, source_id=edge.target_id, target_id=edge.source_id
        )
        turned += 1
        logger.info("Turned round %s %s: %s", edge.role, named(item.edge), item.reason)
    for item in report.separate:
        logger.info(
            "Dropping %s %s: %s",
            item.edge.get("role"),
            named(item.edge),
            item.reason,
        )
    doomed |= {item.edge[PLACE] for item in report.separate}

    keep = [edge for number, edge in enumerate(derived) if number not in doomed]
    dropped = len(derived) - len(keep)
    derived[:] = keep

    logger.info(
        "Derived edges: %d turned round, %d dropped, %d written.",
        turned,
        dropped,
        len(derived),
    )
    return turned, dropped


def unsound_pairs(report: RevisionReport) -> list[dict[str, Any]]:
    """The pairs that are not a revision at all.

    Only ``separate``. A ``reversed`` verdict says the relation is real and
    pointing the wrong way, which is a repair rather than a removal, and
    removing it would lose the fact that the two readings are of one quantity.
    """
    return [item.edge for item in report.separate]
