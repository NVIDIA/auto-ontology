# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checking knowledge edges against the turns they were taken from.

Every other pass over the graph compares *nodes*. Consolidation merges names,
disambiguation separates them, the subject pass chooses among them, and the
write refuses a node it cannot key -- so an edge, once extracted, is never
looked at again by anything. This module is the one pass that reads an edge
back against the text that produced it and asks whether the text says that.

It is cheap because the attribution is already stored. An edge carries the
turns it was extracted from, so the question is never "is this true of the
transcript" -- which would need the whole transcript -- but "is this true of
these two turns", which needs about six hundred characters. Batching edges by
the turns they cite brings a hundred-turn graph in at roughly ten calls.

Three verdicts, and the third is the useful one:

``supported``
    The turns say it. Nothing to do.
``unsupported``
    The turns are about this, and do not say this. The edge is wrong.
``absent``
    The turns do not mention one of the endpoints at all. Usually a
    *misattribution* rather than a false claim: the relation may well be stated
    somewhere, but not here, so this edge cannot be checked from what it cites.

Separating the last two matters. An edge whose claim is contradicted by its own
evidence is a different defect from one that cites the wrong turn, and only the
first is a reason to disbelieve the relation itself.

Reporting, not deleting, by default. A verifier that removes edges on its first
run is how a correct graph quietly loses half its relations; ``apply=True`` is
there once the counts have been read.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime only
    from auto_ontology.conversations.semantic.extract import EdgeDraft, NodeDraft

logger = logging.getLogger(__name__)

#: Characters of turn text per call. Small on purpose: the whole point is that
#: an edge's own evidence is short, and a batch large enough to need summarising
#: is a batch where the model stops reading the turns one at a time.
BATCH_CHARS = 8000

#: Edges per call, whatever the text budget allows. A long run of edges on few
#: turns is the common shape -- per-turn extraction puts a dozen on one turn --
#: and a verdict list longer than this starts losing its place.
BATCH_EDGES = 40

Verdict = Literal["supported", "unsupported", "absent"]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class EdgeVerdict:
    """One edge's verdict, with the model's reason for it."""

    edge: EdgeDraft
    verdict: Verdict
    reason: str = ""

    @property
    def statement(self) -> str:
        return f"{self.edge.source} --{self.edge.role}--> {self.edge.target}"


@dataclass
class VerificationReport:
    """What the check found, bucketed by verdict."""

    supported: list[EdgeVerdict] = field(default_factory=list)
    unsupported: list[EdgeVerdict] = field(default_factory=list)
    absent: list[EdgeVerdict] = field(default_factory=list)
    #: Edges never put to a model, because the turns they cite were not
    #: available to this run. Not a verdict: an unasked question.
    unchecked: list[EdgeDraft] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.supported) + len(self.unsupported) + len(self.absent)

    def wrong(self) -> list[EdgeDraft]:
        """The edges whose own evidence contradicts them."""
        return [item.edge for item in self.unsupported]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_SYSTEM = """You check claims about a conversation against the conversation.

You are shown some turns of a transcript, then a numbered list of RELATIONS
that were extracted from exactly those turns. For each relation, say whether
the turns you were shown support it.

  supported    The turns state this relation, or state it plainly enough that a
               careful reader would agree. Wording need not match: "I added
               lru-cache to the caching layer" supports
               `Caching layer --built_with--> lru-cache`.
  unsupported  The turns talk about both ends and say something INCOMPATIBLE
               with this. If the turns say the retry logic was removed, `Retry
               logic --deployed_to--> Production` is unsupported. If the user
               chose vanilla JS over React, `Weather app --built_with-->
               React 18.2` is unsupported.
  absent       The turns never mention one of the two ends. Say this rather
               than guessing. It usually means the relation came from a
               different part of the conversation, which you cannot see.

A NAME MORE PRECISE THAN THE SENTENCE IS STILL SUPPORTED. This is the mistake
to avoid above all others. A turn writing "HTML and CSS" supports
`--built_with--> HTML5` and `--built_with--> CSS3`; a turn writing "the
OpenWeather API" supports `--built_with--> OpenWeather API v2.5`; a turn
writing "jest" supports `--built_with--> jest v29`. The relation is the claim
being checked, not the wording of the end's name, and these were all called
unsupported once -- which deleted true relations and made an answer listing the
project's technologies lose half of them.

Judge only what is written. Do not use what you know about how such software is
normally built: `Weather app --built_with--> JavaScript` is 'absent' if these
turns never say so, however likely it is.

A relation may be stated by either speaker, in a question or in an answer. The
user asking "can we add debouncing to the autocomplete?" and the assistant
answering with the code supports the relation between them.

Be sparing with 'unsupported'. It is the verdict that removes information, and
the cost of wrongly calling a true relation false is higher than the cost of
leaving a doubtful one in place. When the turns neither state nor contradict
the relation, 'absent' is the honest answer.

Give a verdict for every relation, identified by its number. Keep each reason
to one short clause."""


class _VerdictItem(BaseModel):
    number: int = Field(description="The relation's number in the list shown.")
    verdict: Literal["supported", "unsupported", "absent"] = Field(
        description="Whether the turns shown support the relation."
    )
    reason: str = Field(default="", description="One short clause. Why.")


class _VerdictResult(BaseModel):
    verdicts: list[_VerdictItem] = Field(
        default_factory=list, description="One entry per relation shown."
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def _batch(
    edges: Sequence[EdgeDraft], turns: Mapping[int, str]
) -> tuple[list[list[EdgeDraft]], list[EdgeDraft]]:
    """Group *edges* into calls, and separate the ones that cannot be checked.

    Sorted by first cited turn, so that consecutive batches are consecutive
    conversation. Edges on neighbouring turns usually describe the same piece of
    work, and showing them together is what lets the model read one stretch of
    transcript once instead of the same turn in six different calls.
    """
    # Partitioned in one pass, by identity. Two edges can compare equal and
    # still be separate entries in the graph, and a membership test would let
    # one unverifiable edge take an identical sibling out with it.
    unchecked: list[EdgeDraft] = []
    checkable: list[EdgeDraft] = []
    for edge in edges:
        cited = {index for index in edge.turn_indexes if index in turns}
        (checkable if cited else unchecked).append(edge)

    ordered = sorted(
        checkable,
        key=lambda edge: (
            min(index for index in edge.turn_indexes if index in turns),
            edge.source,
            edge.role,
        ),
    )

    batches: list[list[EdgeDraft]] = []
    current: list[EdgeDraft] = []
    shown: set[int] = set()
    for edge in ordered:
        cited = {index for index in edge.turn_indexes if index in turns}
        added = sum(len(turns[index]) for index in cited - shown)
        over_text = current and len(_render_turns(shown, turns)) + added > BATCH_CHARS
        if over_text or len(current) >= BATCH_EDGES:
            batches.append(current)
            current, shown = [], set()
        current.append(edge)
        shown |= cited
    if current:
        batches.append(current)
    return batches, unchecked


def _render_turns(indexes: set[int], turns: Mapping[int, str]) -> str:
    return "\n\n".join(f"[turn {index}] {turns[index]}" for index in sorted(indexes))


def _render_edges(edges: Sequence[EdgeDraft]) -> str:
    lines = []
    for number, edge in enumerate(edges, start=1):
        cites = ", ".join(str(index) for index in sorted(edge.turn_indexes))
        lines.append(
            f"{number}. {edge.source} --{edge.role}--> {edge.target}"
            f"   (from turn {cites})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_batch(
    edges: Sequence[EdgeDraft], turns: Mapping[int, str]
) -> list[EdgeVerdict]:
    """Put one batch to the model, and read back only what resolves."""
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    shown: set[int] = set()
    for edge in edges:
        shown |= {index for index in edge.turn_indexes if index in turns}

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=4096),
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"TURNS:\n\n{_render_turns(shown, turns)}\n\n"
                    f"RELATIONS EXTRACTED FROM THOSE TURNS:\n{_render_edges(edges)}"
                )
            ),
        ],
        _VerdictResult,
    )
    if result is None:
        logger.warning("No verdicts returned for %d edge(s); leaving them.", len(edges))
        return []

    # Keyed by the number shown, and a number outside the list is dropped. The
    # model is choosing from a closed set here exactly as the subject pass does,
    # and the safety of both is that an answer which does not resolve inside the
    # set is discarded rather than guessed at.
    seen: set[int] = set()
    verdicts: list[EdgeVerdict] = []
    for item in result.verdicts:
        if not 1 <= item.number <= len(edges):
            logger.warning("Ignoring verdict for relation %d: not shown.", item.number)
            continue
        if item.number in seen:
            logger.warning("Relation %d judged twice; taking the first.", item.number)
            continue
        seen.add(item.number)
        verdicts.append(
            EdgeVerdict(
                edge=edges[item.number - 1],
                verdict=item.verdict,
                reason=item.reason.strip(),
            )
        )

    missing = len(edges) - len(verdicts)
    if missing:
        logger.warning("%d of %d relation(s) went unjudged.", missing, len(edges))
    return verdicts


def verify_edges(
    edges: Sequence[EdgeDraft], turns: Mapping[int, str]
) -> VerificationReport:
    """Check each edge in *edges* against the text of the turns it cites.

    *turns* maps turn index to that turn's rendered text -- the same text the
    extractor was shown, which is why this is affordable: no turn is read that
    did not already produce an edge.

    Returns the report. Nothing is removed; see :func:`drop_unsupported`.
    """
    report = VerificationReport()
    batches, report.unchecked = _batch(edges, turns)
    if report.unchecked:
        logger.warning(
            "%d edge(s) cite no turn in this transcript and were not checked.",
            len(report.unchecked),
        )
    if not batches:
        return report

    logger.info(
        "Checking %d edge(s) against their turns in %d call(s).",
        sum(len(batch) for batch in batches),
        len(batches),
    )
    for number, batch in enumerate(batches, start=1):
        logger.info(
            "Verifying batch %d/%d (%d edges).", number, len(batches), len(batch)
        )
        for item in _verify_batch(batch, turns):
            getattr(report, item.verdict).append(item)

    _report(report)
    return report


def _report(report: VerificationReport) -> None:
    """Log the buckets, and name the edges in the two that are not clean."""
    logger.info(
        "Edge check: %d supported, %d unsupported, %d absent, of %d checked.",
        len(report.supported),
        len(report.unsupported),
        len(report.absent),
        report.checked,
    )
    for bucket, label in (
        (report.unsupported, "UNSUPPORTED"),
        (report.absent, "ABSENT"),
    ):
        for item in bucket[:20]:
            logger.info("  %s %s -- %s", label, item.statement, item.reason)
        if len(bucket) > 20:
            logger.info("  ... and %d more %s.", len(bucket) - 20, label)


# ---------------------------------------------------------------------------
# Node citations
# ---------------------------------------------------------------------------


#: Statements citing at least this many turns are checked. One turn cannot be
#: wrong about itself -- the statement was extracted from it -- so the claim
#: worth testing only appears once a second turn is attached.
MIN_CITED_TURNS = 2

CitationVerdict = Literal["supports", "denies", "unrelated"]


@dataclass
class NodeCitation:
    """One ``(statement, turn)`` pair's verdict."""

    node: NodeDraft
    turn_index: int
    verdict: CitationVerdict
    reason: str = ""


@dataclass
class CitationReport:
    """What the citation check found."""

    supports: list[NodeCitation] = field(default_factory=list)
    denies: list[NodeCitation] = field(default_factory=list)
    unrelated: list[NodeCitation] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.supports) + len(self.denies) + len(self.unrelated)

    def wrong(self) -> list[NodeCitation]:
        """Citations to detach: the turn does not support what it is cited for."""
        return self.denies + self.unrelated


_CITATION_SYSTEM = """You check which turns support a claim about a conversation.

You are shown some turns of a transcript, then numbered CLAIMS, each listing the
turns it was extracted from. For every claim-and-turn pair, say what that turn
does to that claim.

  supports    The turn states the claim, or states it plainly enough that a
              careful reader would agree.
  denies      The turn says the OPPOSITE. The user saying "I've never actually
              obtained an API key" denies the claim "OpenWeather API key
              obtained", however much the two sentences look alike.
  unrelated   The turn neither states nor contradicts the claim. It may be
              about the same topic and still say nothing about this.

'denies' is the verdict this check exists for, so do not soften it. A claim that
cites the turn retracting it is recorded backwards -- the graph ends up saying
that turn confirms what it refuses -- and that is worse than a missing citation,
because nothing downstream can tell it was ever in doubt.

Read the turn, not what is plausible. A claim may be true of the conversation
and still be unsupported by a particular turn.

Give a verdict for every pair, identified by the claim's number and the turn's
number. Keep each reason to one short clause."""


class _CitationItem(BaseModel):
    claim: int = Field(description="The claim's number in the list shown.")
    turn: int = Field(description="The turn number, as labelled in the text.")
    verdict: Literal["supports", "denies", "unrelated"] = Field(
        description="What that turn does to that claim."
    )
    reason: str = Field(default="", description="One short clause. Why.")


class _CitationResult(BaseModel):
    verdicts: list[_CitationItem] = Field(
        default_factory=list, description="One entry per claim-and-turn pair."
    )


def _render_claims(nodes: Sequence[NodeDraft]) -> str:
    lines = []
    for number, node in enumerate(nodes, start=1):
        cites = ", ".join(str(index) for index in sorted(node.turn_indexes))
        reading = f" [{node.key} = {node.value}]" if node.key else ""
        lines.append(
            f"{number}. {node.name}{reading}\n"
            f"   {node.description}\n"
            f"   cited from turns: {cites}"
        )
    return "\n".join(lines)


def _check_citations(
    nodes: Sequence[NodeDraft], turns: Mapping[int, str]
) -> list[NodeCitation]:
    """Put one batch of claims to the model and read back what resolves."""
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    shown: set[int] = set()
    for node in nodes:
        shown |= {index for index in node.turn_indexes if index in turns}

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=4096),
        [
            SystemMessage(content=_CITATION_SYSTEM),
            HumanMessage(
                content=(
                    f"TURNS:\n\n{_render_turns(shown, turns)}\n\n"
                    f"CLAIMS:\n{_render_claims(nodes)}"
                )
            ),
        ],
        _CitationResult,
    )
    if result is None:
        logger.warning(
            "No verdicts returned for %d claim(s); leaving them.", len(nodes)
        )
        return []

    seen: set[tuple[int, int]] = set()
    verdicts: list[NodeCitation] = []
    for item in result.verdicts:
        if not 1 <= item.claim <= len(nodes):
            logger.warning("Ignoring verdict for claim %d: not shown.", item.claim)
            continue
        node = nodes[item.claim - 1]
        # The turn has to be one this claim actually cites. A verdict on a turn
        # it never cited is about a pair that was never put to the model, and
        # acting on it would detach a citation nobody questioned.
        if item.turn not in node.turn_indexes:
            logger.warning(
                "Ignoring verdict on turn %d for %r: not cited by it.",
                item.turn,
                node.name,
            )
            continue
        if (item.claim, item.turn) in seen:
            continue
        seen.add((item.claim, item.turn))
        verdicts.append(
            NodeCitation(
                node=node,
                turn_index=item.turn,
                verdict=item.verdict,
                reason=item.reason.strip(),
            )
        )
    return verdicts


def verify_node_citations(
    nodes: Sequence[NodeDraft], turns: Mapping[int, str]
) -> CitationReport:
    """Check each multi-turn statement against every turn it cites.

    Only statements, and only those citing :data:`MIN_CITED_TURNS` or more. A
    *thing* legitimately spans every turn that mentions it -- ``Autocomplete``
    appearing in twenty turns is the graph working -- so there is no claim there
    to be wrong about. A statement citing several turns is asserting that each
    one says it, and that assertion is what this tests.
    """
    claims = [
        node
        for node in nodes
        if node.is_statement
        and len({index for index in node.turn_indexes if index in turns})
        >= MIN_CITED_TURNS
    ]
    report = CitationReport()
    if not claims:
        return report

    batches: list[list[NodeDraft]] = []
    current: list[NodeDraft] = []
    shown: set[int] = set()
    for node in sorted(claims, key=lambda n: (min(n.turn_indexes), n.name)):
        cited = {index for index in node.turn_indexes if index in turns}
        added = sum(len(turns[index]) for index in cited - shown)
        over_text = current and len(_render_turns(shown, turns)) + added > BATCH_CHARS
        if over_text or len(current) >= BATCH_EDGES:
            batches.append(current)
            current, shown = [], set()
        current.append(node)
        shown |= cited
    if current:
        batches.append(current)

    logger.info(
        "Checking %d multi-turn statement(s) against their turns in %d call(s).",
        len(claims),
        len(batches),
    )
    for number, batch in enumerate(batches, start=1):
        logger.info("Verifying claims %d/%d (%d).", number, len(batches), len(batch))
        for item in _check_citations(batch, turns):
            getattr(report, item.verdict).append(item)

    logger.info(
        "Citation check: %d supports, %d denies, %d unrelated, of %d checked.",
        len(report.supports),
        len(report.denies),
        len(report.unrelated),
        report.checked,
    )
    for item in report.denies:
        logger.info(
            "  DENIES turn %d contradicts %r -- %s",
            item.turn_index,
            item.node.name,
            item.reason,
        )
    for item in report.unrelated[:20]:
        logger.info(
            "  UNRELATED turn %d does not state %r -- %s",
            item.turn_index,
            item.node.name,
            item.reason,
        )
    return report


def detach_bad_citations(report: CitationReport) -> int:
    """Remove every unsupported turn from the node that cited it, in place.

    The node itself is kept, and so is its last turn: a statement citing nothing
    cannot be traced back to the transcript at all, which is a worse graph than
    one citing a turn loosely. A denial found here also means a node is
    *missing* -- the retraction belongs in the graph as its own statement -- and
    this cannot add it. Removing the false citation stops the graph claiming
    that turn confirms the claim; the missing node is extraction's job.
    """
    detached = 0
    for item in report.wrong():
        remaining = item.node.turn_indexes - {item.turn_index}
        if not remaining:
            logger.warning(
                "Keeping turn %d on %r: it is the only citation left.",
                item.turn_index,
                item.node.name,
            )
            continue
        item.node.turn_indexes = remaining
        item.node.verified_turn_indexes -= {item.turn_index}
        detached += 1
    logger.info("Detached %d citation(s).", detached)
    return detached


# ---------------------------------------------------------------------------
# Repairing a denial
# ---------------------------------------------------------------------------


_DENIAL_SYSTEM = """You record a retraction as its own statement.

You are shown ONE turn of a transcript and ONE claim that the turn contradicts.
The claim is in a knowledge graph; the retraction is not, which is the hole you
are filling. Write the turn's denial as a statement in its own right.

First decide WHICH OF TWO THINGS this is, because only one of them is a missing
statement:

  retraction   The SPEAKER takes back something they said. "I've never actually
               obtained an API key" against 'OpenWeather API key obtained'. The
               transcript holds both claims, the speaker is in conflict with
               themselves, and the retraction belongs in the graph beside what
               it denies.
  inaccuracy   The claim is simply recorded WRONGLY. The graph says the cache
               uses a Map and the code uses an LRU cache; nobody retracted
               anything, one node is just wrong. There is no new statement to
               add here -- the repair is to the existing node, which is not
               yours to make -- so say 'inaccuracy' and write nothing else of
               substance.

Then, for a retraction:

  name         Short, 1-8 words, naming the DENIAL -- 'Never obtained an API
               key'. Never the claim's own name: the graph already holds that,
               and a second node under it is the bug this is repairing.
  description  One sentence, in the transcript's terms, saying what the speaker
               actually said. Name what the denial is about, as the claim does.
  evidence     A VERBATIM span from the turn, copied exactly, in which the
               speaker denies it. Not a paraphrase. This is checked against the
               turn and the statement is discarded if it does not appear.
  value        If the claim carries a key, the opposite value for that same key
               -- 'no' against 'yes'. Leave empty if the claim has no key.

State only what the turn states. You are not deciding which side is true, and
you are not summarising the dispute: the claim stays in the graph, and a later
step draws the relation between the two. If the turn does not actually deny the
claim, return nothing."""


class _DenialItem(BaseModel):
    kind: Literal["retraction", "inaccuracy"] = Field(
        description="'retraction' if the speaker takes back something they said; "
        "'inaccuracy' if the claim is merely recorded wrongly."
    )
    name: str = Field(description="Short name for the denial, 1-8 words.")
    description: str = Field(description="One sentence, the transcript's terms.")
    evidence: str = Field(description="Verbatim span from the turn.")
    value: str = Field(
        default="", description="Opposite value, if the claim has a key."
    )


def _normalise(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text).strip().casefold()


def build_denials(
    report: CitationReport,
    turns: Mapping[int, str],
    already: Sequence[NodeDraft] = (),
) -> list[NodeDraft]:
    """A statement node for every retraction the citation check found.

    Only a *retraction* -- the speaker taking back something they said. A turn
    that merely shows a claim was recorded wrongly ('the cache uses a Map' where
    the code uses an LRU cache) is an inaccuracy in an existing node, not a
    missing statement, and the repair there is to the node rather than beside
    it. Measured on the first live run: four of six denials were one such
    inaccuracy reported from four turns, which would have added four
    near-identical statements for a single mistake.

    *already* is the graph's existing nodes, used only to stay out of the
    extractor's way. A turn that produced a statement of its own has been
    handled; repairing it again writes the retraction twice.

    The hole this fills is the one defect the rest of the pipeline cannot touch.
    A wrong edge can be dropped and a wrong citation detached, but a claim the
    speaker took back leaves the graph asserting the claim and holding no trace
    of the retraction -- and a pass that only removes things can never put the
    missing statement in.

    Restating, not reasoning. The model is given one turn and the claim that
    turn contradicts, and must return a verbatim span in which the speaker
    denies it; a statement whose quote is not in the turn is discarded. It
    therefore cannot assert anything the turn does not say, which is the only
    basis on which adding to a verified graph is defensible at all.

    The node is returned, not inserted. Added to the draft before consolidation
    it is indistinguishable from an extracted one: it is merged, named, given a
    subject, stored, embedded, and paired with the claim it denies by the
    adjudication pass, which already emits ``contradicts``. No new machinery
    downstream.
    """
    from auto_ontology.conversations.semantic.extract import NodeDraft
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    # Turns extraction already gave a statement of their own. If the prompt
    # produced 'Never obtained an API key' from turn 35, repairing turn 35 again
    # writes the same retraction twice; the extractor got there first, and one
    # of the two nodes is then a duplicate the graph cannot pair cleanly.
    spoken_for = {
        next(iter(node.turn_indexes))
        for node in already
        if node.is_statement and len(node.turn_indexes) == 1
    }

    built: list[NodeDraft] = []
    answered: set[str] = set()
    for item in report.denies:
        body = turns.get(item.turn_index)
        if body is None:
            continue
        if item.turn_index in spoken_for:
            logger.info(
                "Skipping turn %d: extraction already states it on its own.",
                item.turn_index,
            )
            continue

        claim = item.node
        # One retraction per claim, not per citation. Four turns denied one
        # 'caching with Map' node on the first live run, which would have put
        # four near-identical statements in the graph for a single conflict.
        if claim.name in answered:
            logger.info("Already recorded a denial of %r.", claim.name)
            continue
        reading = (
            f"\nThe claim records {claim.key} = {claim.value}." if claim.key else ""
        )
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0, max_tokens=1024),
            [
                SystemMessage(content=_DENIAL_SYSTEM),
                HumanMessage(
                    content=(
                        f"TURN {item.turn_index}:\n\n{body}\n\n"
                        f"CLAIM IT CONTRADICTS:\n{claim.name}\n"
                        f"{claim.description}{reading}"
                    )
                ),
            ],
            _DenialItem,
        )
        if result is None:
            logger.warning("No denial returned for %r.", claim.name)
            continue
        if result.kind != "retraction":
            logger.info(
                "Turn %d does not retract %r, it contradicts how it was "
                "recorded -- citation detached, nothing added.",
                item.turn_index,
                claim.name,
            )
            continue

        quote = _normalise(result.evidence)
        if not quote or quote not in _normalise(body):
            logger.warning(
                "Discarding denial of %r: quote %r is not in turn %d.",
                claim.name,
                result.evidence[:60],
                item.turn_index,
            )
            continue
        if _normalise(result.name) == _normalise(claim.name):
            logger.warning(
                "Discarding denial of %r: it reuses the claim's own name.", claim.name
            )
            continue

        built.append(
            NodeDraft(
                name=result.name.strip(),
                type=claim.type,
                # A retraction is a fact about what the speaker says is so, not
                # a reading and not an event: it has no value to track over time
                # and nothing happened on a date.
                kinds=["fact"],
                description=result.description.strip(),
                key=claim.key,
                value=result.value.strip() if claim.key else "",
                turn_indexes={item.turn_index},
                verified_turn_indexes={item.turn_index},
                evidence=result.evidence.strip(),
            )
        )
        answered.add(claim.name)
        logger.info(
            "Recorded %r, denying %r from turn %d.",
            built[-1].name,
            claim.name,
            item.turn_index,
        )

    logger.info("Built %d denial statement(s) of %d.", len(built), len(report.denies))
    return built


def drop_unsupported(
    edges: list[EdgeDraft], report: VerificationReport
) -> list[EdgeDraft]:
    """*edges* without the ones their own evidence contradicts.

    Only the ``unsupported`` bucket. An ``absent`` verdict says the edge cites
    the wrong turn, which is a defect in its attribution and not evidence
    against the relation, and dropping on it would delete every edge the
    extractor attributed loosely.
    """
    wrong = {id(edge) for edge in report.wrong()}
    kept = [edge for edge in edges if id(edge) not in wrong]
    logger.info(
        "Dropped %d unsupported edge(s) of %d.", len(edges) - len(kept), len(edges)
    )
    return kept
