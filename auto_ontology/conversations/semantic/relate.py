# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a transcript's attributes stand to one another.

Extraction gives each attribute its own evidence. This asks what holds
*between* attributes, which is what makes the set an ontology rather than a
list, and it is what :mod:`note` reads at query time.

Three steps:

1. **Propose.** Attributes whose vectors sit close together are candidate
   pairs. Cheap, and wrong on its own: "the quota is 1,000" and "the quota is
   1,200" embed almost identically, and so do "coverage is 78%" and "latency is
   250ms" relative to the rest of a transcript.
2. **Adjudicate.** A model says what the relation actually is and why, so the
   edge carries provenance. The ``unrelated`` verdicts are stored too -- they
   are most of the answer, and keeping them is what stops a later pass paying
   to reach them again.
3. **Nothing.** There is no grouping step. The document pipeline builds
   families because a corpus states one indicator in several publications and
   all the readings are simultaneously true; a transcript is one speaker over
   time, so two readings of one quantity are ordered, not rival, and the order
   is already the edge.

**Direction is the point.** Every relation here is asymmetric, and a reader
that ignores the direction inverts the answer:

``supersedes``
    Source replaces target. The transcript moved on; the source is current and
    the target is history. This is what a "what is the X now" question needs.
``contradicts``
    The two disagree and nothing in the transcript resolves which holds. Not
    the same as ``supersedes`` and kept apart because the right answer differs:
    a superseded value has one answer, a contradiction's correct answer is to
    say both were said.
``precedes``
    Source happened before target. Only between ``event`` attributes, and the
    only edge that can answer an interval question.
``refines``
    Source is a more specific statement of target -- a constraint narrowing a
    preference, a measurement of a state. Neither replaces the other.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ROLES",
    "SemanticEdge",
    "adjudicate",
    "candidate_pairs",
    "relate_nodes",
]

#: What a pair can be, and how each reads to the model deciding. Matches the
#: semantic half of ``role`` on ``conversation_edge``.
ROLES = {
    "supersedes": (
        "A and B state the same thing with different values, and A is the "
        "later one -- the speaker revised it, so A is current and B is history"
    ),
    "contradicts": (
        "A and B disagree and the transcript never settles which holds; both "
        "were said and neither was retracted"
    ),
    "precedes": "both are events, and A happened before B",
    "refines": (
        "A is a more specific statement of B -- a constraint narrowing a "
        "preference, or a measurement of a state -- and neither replaces the "
        "other"
    ),
    "unrelated": "no useful relation between them",
}

#: Relations whose direction must be checked against the transcript rather than
#: taken from the model's A/B ordering. The adjudicator is shown the turn
#: numbers, so it can get this right; these are the ones where getting it wrong
#: inverts the answer rather than merely mislabelling it.
DIRECTED = frozenset({"supersedes", "precedes", "refines"})

#: How close two attribute vectors must be to be worth asking about. Cosine
#: similarity, not distance.
CANDIDATE_SIMILARITY = 0.70

#: Pairs to adjudicate at most, per transcript. A transcript with 60
#: attributes has 1,770 pairs; proximity cuts that to a few dozen, and this
#: caps the tail so one unusually repetitive conversation cannot spend an hour.
MAX_PAIRS = 300

RelationName = Literal["supersedes", "contradicts", "precedes", "refines", "unrelated"]


class PairVerdict(BaseModel):
    """What one call says about one candidate pair."""

    role: RelationName
    reason: str | None = Field(
        None, description="One short sentence naming the distinction."
    )
    #: Which side the relation points *from*. Asked explicitly rather than
    #: assumed to be A, because the pair is presented in extraction order and
    #: the later statement is as often B as A.
    source: Literal["A", "B"] | None = Field(
        None, description="Which statement the relation runs from. Null if unrelated."
    )


@dataclass(frozen=True)
class SemanticEdge:
    """One semantic relation, in storage order.

    ``derived`` says which instrument produced it: a field comparison in
    :mod:`derive`, or a model call here. The two fail differently -- derivation
    is wrong only when the extractor keyed two properties alike, adjudication is
    wrong when the model misread -- and a reader auditing a surprising edge
    needs to know which one to doubt.
    """

    source_id: str
    target_id: str
    role: str
    reason: str | None
    distance: str | None
    derived: bool = False
    turn_indexes: list[int] = field(default_factory=list)


_SYSTEM = """You decide how two statements from one person's chat transcript \
relate to each other.

The statements were extracted from the same conversation, held over several \
sessions. Turn numbers tell you what was said first: a higher turn number is \
later in the conversation.

Each statement says what it is ABOUT. Two statements about different subjects \
are not a revision of each other, however alike they read: the app's average \
response time and the autocomplete's average response time are two quantities \
that share a word, not one quantity measured twice. The property key says the \
same thing more narrowly -- different keys are different properties unless \
the extractor keyed one property two ways.

Choose exactly one relation, and say which side it runs FROM.
"""

_TEMPLATE = """Statement A
  about: {a_about}
  name: {a_name}
  kind: {a_kind}
  property key: {a_key}
  says: {a_description}
  stated at turn(s): {a_turns}{a_date}

Statement B
  about: {b_about}
  name: {b_name}
  kind: {b_kind}
  property key: {b_key}
  says: {b_description}
  stated at turn(s): {b_turns}{b_date}

Relations:
{options}

If you choose supersedes, source must be the LATER statement (higher turn
number -- the user changed their mind, and the newer statement wins).

If you choose precedes, order by when the events HAPPENED, which is the date
written in the description. Do not order by "said on" or by turn number: a
transcript often mentions two events in one sitting, and mentions them in the
opposite order to the one they occurred in.
"""


def _options_block() -> str:
    return "\n".join(f"  {name} -- {gloss}" for name, gloss in ROLES.items())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    left = sum(x * x for x in a) ** 0.5
    right = sum(y * y for y in b) ** 0.5
    return dot / (left * right) if left and right else 0.0


def candidate_pairs(
    attributes: Sequence[dict],
    *,
    similarity: float = CANDIDATE_SIMILARITY,
    max_pairs: int = MAX_PAIRS,
) -> list[tuple[int, int, float]]:
    """Index pairs worth adjudicating, closest first.

    *attributes* carry a ``vector`` each. Pairs below *similarity* are not
    proposed; above it they are returned in descending order so that a
    transcript hitting *max_pairs* spends its budget on the closest pairs
    rather than the first ones alphabetically.

    ``event``/``event`` pairs are proposed at any similarity. Two events in one
    project rarely resemble each other -- "obtained an API key" and "completed
    the UI wireframes" share no vocabulary -- yet ``precedes`` between them is
    exactly what an interval question needs, and a similarity gate is the wrong
    instrument for finding them. Pairs whose dates both parsed are settled by
    :mod:`derive` and arrive here only in *skip*; this clause is what still
    reaches the undated ones, where reading is the only way to order them.
    """
    scored: list[tuple[int, int, float]] = []
    for i, j in itertools.combinations(range(len(attributes)), 2):
        left, right = attributes[i], attributes[j]
        both_events = "event" in (left.get("kinds") or ()) and "event" in (
            right.get("kinds") or ()
        )
        vectors = left.get("vector"), right.get("vector")
        score = _cosine(*vectors) if all(vectors) else 0.0
        if both_events or score >= similarity:
            scored.append((i, j, score))

    scored.sort(key=lambda pair: pair[2], reverse=True)
    if len(scored) > max_pairs:
        logger.info("Capping %d candidate pair(s) at %d.", len(scored), max_pairs)
    return scored[:max_pairs]


def _turns(attribute: dict) -> str:
    indexes = attribute.get("turn_indexes") or []
    return ", ".join(str(i) for i in sorted(indexes)) or "unknown"


def _date(attribute: dict) -> str:
    """The turn's date, labelled as what it is.

    This is when the statement was *made*, which for an ``event`` is often not
    when the event happened: "I got the key on March 10" can be said on March
    15, and the transcript carries only the latter as an anchor. Labelling it
    ``said on`` rather than ``dated`` keeps the adjudicator from ordering two
    events by when they were mentioned -- the description holds the real date,
    and the instruction below tells it to prefer that.
    """
    anchor = attribute.get("time_anchor")
    return f"\n  said on: {anchor}" if anchor else ""


def _about(attribute: dict) -> str:
    return ", ".join(attribute.get("about") or ()) or "not recorded"


def adjudicate(left: dict, right: dict) -> PairVerdict | None:
    """What relation holds between two attributes, if any."""
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    prompt = _TEMPLATE.format(
        a_about=_about(left),
        a_name=left.get("name", ""),
        a_kind=left.get("kind", ""),
        a_key=left.get("attribute_key") or "none",
        a_description=left.get("description", ""),
        a_turns=_turns(left),
        a_date=_date(left),
        b_about=_about(right),
        b_name=right.get("name", ""),
        b_kind=right.get("kind", ""),
        b_key=right.get("attribute_key") or "none",
        b_description=right.get("description", ""),
        b_turns=_turns(right),
        b_date=_date(right),
        options=_options_block(),
    )
    return invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=2048),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        PairVerdict,
    )


def _orient(verdict: PairVerdict, left: dict, right: dict) -> tuple[dict, dict]:
    """Put the pair in the order the relation runs.

    The model's ``source`` is taken when it gave one. When it did not, and the
    relation is one whose direction matters, the turn numbers decide: the later
    statement supersedes, the earlier one precedes. Falling back to the
    presentation order would make the direction depend on which attribute the
    extractor happened to find first.
    """
    if verdict.source == "B":
        return right, left
    if verdict.source == "A" or verdict.role not in DIRECTED:
        return left, right

    left_last = max(left.get("turn_indexes") or [-1])
    right_last = max(right.get("turn_indexes") or [-1])
    if verdict.role == "supersedes":
        return (left, right) if left_last >= right_last else (right, left)
    return (left, right) if left_last <= right_last else (right, left)


def _not_a_revision(left: dict, right: dict) -> str | None:
    """Why this pair cannot be ``supersedes``, or ``None`` if it can.

    ``supersedes`` is the one verdict the note renders as a claim about the
    present -- it prints the target under "no longer true" -- so a wrong one
    does not merely add noise, it tells the reader a fact has been retracted
    when it has not. These two cases are decidable from the rows, and the
    adjudicator got both wrong on the first transcript measured, so they are
    checked here rather than left to the prompt above:

    **Different subjects.** The weather app's API response time, improved to
    220ms in session three, was ruled a revision of the autocomplete's 280ms
    from session two. Two quantities, one word in common. The note then said
    the autocomplete now answers in 220ms, and the reader -- comparing against
    a 250ms fetch latency -- named the wrong one as faster. An unrecorded
    subject is not a mismatch: it is read off the knowledge edges, and a
    statement the extractor left unattached would otherwise be exempt from
    every revision it genuinely belongs to.

    **The same value twice.** One 220ms reading was ruled a revision of
    another 220ms reading, the extractor having keyed one measurement as both
    ``api_response_time`` and ``api_response_time_ms``. Nothing was revised, so
    there is nothing for a reader to prefer; it is a duplicate, and the note
    spends three lines implying a change that did not happen.
    """
    left_about = {name.casefold() for name in left.get("about") or ()}
    right_about = {name.casefold() for name in right.get("about") or ()}
    if left_about and right_about and not left_about & right_about:
        return f"different subjects ({_about(left)} / {_about(right)})"

    left_value = (left.get("value") or "").strip().casefold()
    right_value = (right.get("value") or "").strip().casefold()
    if left_value and left_value == right_value:
        return f"the same value on both sides ({left.get('value')})"
    return None


#: Adjudication calls in flight at once. The pairs are independent -- each is
#: one question about two statements and shares nothing with the others -- so
#: this is bounded by the API rather than by the work.
ADJUDICATION_WORKERS = 8


def _adjudicate_all(
    pairs: Sequence[tuple[int, int, float]], attributes: Sequence[dict]
) -> list[PairVerdict | None]:
    """Every pair's verdict, concurrently, **in the order of** *pairs*.

    Parallel because this is the slow stage and nothing in it is shared: 296
    pairs at three seconds each ran for fifteen minutes in series. The earlier
    choice to keep it sequential assumed the concurrency would come from running
    transcripts side by side, which is true of a twenty-conversation run and
    false of the one that matters while a single graph is being tuned -- there
    the fifteen minutes are the whole critical path.

    Only the calls are parallel. Every decision that follows -- rejecting a
    revision, orienting the edge, counting the roles -- stays sequential and in
    pair order, so the edges written do not depend on which call returned first.
    """
    started = time.monotonic()
    done = 0
    lock = threading.Lock()

    def judge(pair: tuple[int, int, float]) -> PairVerdict | None:
        nonlocal done
        left, right = attributes[pair[0]], attributes[pair[1]]
        try:
            return adjudicate(left, right)
        finally:
            with lock:
                done += 1
                if done % 20 == 1 or done == len(pairs):
                    logger.info(
                        "Adjudicated %d/%d pair(s) (%.0fs elapsed).",
                        done,
                        len(pairs),
                        time.monotonic() - started,
                    )

    # At least one worker: a pool sized zero raises, and an empty pair list
    # reaches here whenever a caller holds this helper on its own.
    workers = max(1, min(ADJUDICATION_WORKERS, len(pairs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(judge, pairs))


def relate_nodes(
    attributes: Sequence[dict],
    *,
    skip: set[frozenset[str]] | None = None,
) -> list[SemanticEdge]:
    """Adjudicate every candidate pair and return the edges, ``unrelated``
    included.

    *skip* holds pairs :mod:`derive` has already settled from the rows
    themselves. They are not re-asked: a revised value and an ordering of dated
    events are decided by comparing fields, and a second verdict on the same
    pair would leave two edges with nothing to say which is current. On a
    transcript whose properties are keyed consistently this is most of the
    candidates, which is where the saving is.

    The adjudication calls run concurrently; see :func:`_adjudicate_all`.
    Everything decided from their answers stays in pair order, so the graph does
    not depend on which call came back first.
    """
    pairs = candidate_pairs(attributes)
    if skip:
        kept = [
            (i, j, score)
            for i, j, score in pairs
            if frozenset((str(attributes[i]["id"]), str(attributes[j]["id"])))
            not in skip
        ]
        if len(kept) < len(pairs):
            logger.info(
                "Skipping %d pair(s) already decided from the rows.",
                len(pairs) - len(kept),
            )
        pairs = kept
    if not pairs:
        logger.info("No candidate pairs among %d attribute(s).", len(attributes))
        return []

    edges: list[SemanticEdge] = []
    counts: dict[str, int] = {}
    verdicts = _adjudicate_all(pairs, attributes)
    for (i, j, score), verdict in zip(pairs, verdicts):
        left, right = attributes[i], attributes[j]
        if verdict is None:
            logger.debug(
                "Adjudication failed for %r / %r.", left.get("name"), right.get("name")
            )
            continue
        if verdict.role == "supersedes" and (rejected := _not_a_revision(left, right)):
            logger.info(
                "Dropping supersedes %r / %r: %s.",
                left.get("name"),
                right.get("name"),
                rejected,
            )
            continue
        source, target = _orient(verdict, left, right)
        edges.append(
            SemanticEdge(
                source_id=str(source["id"]),
                target_id=str(target["id"]),
                role=verdict.role,
                reason=(verdict.reason or "").strip() or None,
                distance=f"{score:.4f}",
                derived=False,
                turn_indexes=sorted(
                    set(source.get("turn_indexes") or [])
                    | set(target.get("turn_indexes") or [])
                ),
            )
        )
        counts[verdict.role] = counts.get(verdict.role, 0) + 1

    logger.info(
        "Adjudicated %d pair(s): %s.",
        len(edges),
        ", ".join(f"{n} {r}" for r, n in sorted(counts.items())) or "none",
    )
    return edges
