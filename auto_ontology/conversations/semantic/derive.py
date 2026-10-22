# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic edges that follow from the rows and need no model to decide.

:mod:`relate` asks a model what holds between two statements. That is the right
instrument for ``refines`` and for disagreements spanning different properties,
and the wrong one for the two cases here, where the answer is already in the
nodes:

**A revised value.** Two statement nodes with one ``attribute_key``, attached to
one subject, with different ``value``s are two readings of one quantity -- not
by inference but by construction, because the extractor was told to reuse the
key. Which is current is then a question about turn order. Asking a model costs
a call and can come back wrong; comparing two integers cannot.

**An ordering of events.** Two ``event`` nodes with parsed ``occurred_on`` dates
order by subtraction. The model-based version of this was the worst-performing
part of the statement-only design: it was told to read dates out of prose and
ignore the turn anchor, and still ordered by the telling often enough to score
0.193 on event ordering against a published 0.788.

What is left for :mod:`relate` is what genuinely needs reading: whether one
statement narrows another, and whether two statements about *different*
properties disagree. Neither can be settled by comparing fields.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from typing import Any

from auto_ontology.conversations.semantic.extract import UNREALISED_KINDS
from auto_ontology.conversations.semantic.relate import SemanticEdge

logger = logging.getLogger(__name__)

__all__ = ["MAX_EVENT_PAIRS", "derive_edges"]

#: Event pairs to emit at most. Every pair rather than a chain of consecutive
#: events, because an interval question names two events that need not be
#: adjacent and the prompt only renders edges whose *both* ends were retrieved
#: -- a chain would leave the common case, two events with a third between them,
#: with no edge to render. Quadratic, hence the cap: 25 events is 300 pairs,
#: which is free, and the fallback past that is the chain.
MAX_EVENT_PAIRS = 600


def _last_turn(node: dict[str, Any]) -> int:
    return max(node.get("turn_indexes") or [-1])


def _occurred(node: dict[str, Any]) -> date | None:
    value = node.get("occurred_on")
    return value if isinstance(value, date) else None


def _kinds(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(node.get("kinds") or ())


def _revisions(nodes: Sequence[dict[str, Any]]) -> list[SemanticEdge]:
    """``supersedes`` between successive readings of one property.

    Chained, not every pair: with three readings of a quota the useful claims
    are that the third replaced the second and the second replaced the first.
    Emitting "third supersedes first" as well would let a reader conclude the
    first was replaced twice, and says nothing the chain does not.

    Two readings stated in the *same* turn are a ``contradicts`` instead. Turn
    order is the only thing that makes one of them later, so when it is absent
    there is no later one -- which is exactly the distinction the two roles are
    kept apart for.

    A ``target`` or ``proposed`` reading is not in the chain at all. Both are
    statements about a value that was never the value: a 290ms goal beside a
    measured 520ms is not the figure having changed, and derivation comparing
    fields has no way to tell the difference once they are in one group.

    Grouped by ``attribute_key`` *and* subject, within one transcript. One key
    can be carried by two things -- a test coverage figure belongs to a module
    and to the app containing it -- and chaining across them would read two
    measurements as one that moved.

    A statement with no subject recorded is grouped by its key alone. The
    subject is a ``knowledge`` edge, and one the extractor omitted would
    otherwise stop a statement being comparable with its own revision; measured
    on a BEAM transcript that was 21 of 79 readings, too many to drop. They
    group among themselves, never with a subjected reading, so the looser rule
    cannot widen the stricter one.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        key = (node.get("attribute_key") or "").strip()
        if not key or set(_kinds(node)) & set(UNREALISED_KINDS):
            continue
        # Under each of them: which subject makes two readings comparable is
        # not knowable here, and a pair sharing any one of them is a pair.
        for subject in sorted(node.get("about") or ()) or [""]:
            grouped[(key, subject)].append(node)

    edges: list[SemanticEdge] = []
    #: One statement can be about several things, so the same pair is reachable
    #: through each shared subject. The claim is the same claim each time.
    emitted: set[tuple[str, str]] = set()
    for (key, _subject), group in grouped.items():
        # Distinct values only. Two rows with the same value are the same
        # reading stated twice, which is not a revision of anything.
        by_value: dict[str, dict[str, Any]] = {}
        for node in group:
            value = (node.get("value") or "").strip().casefold()
            if value and value not in by_value:
                by_value[value] = node
        if len(by_value) < 2:
            continue

        ordered = sorted(by_value.values(), key=_last_turn)
        for earlier, later in itertools.pairwise(ordered):
            same_turn = _last_turn(earlier) == _last_turn(later)
            source, target = (earlier, later) if same_turn else (later, earlier)
            pair = (str(source["id"]), str(target["id"]))
            if pair in emitted:
                continue
            emitted.add(pair)
            edges.append(
                SemanticEdge(
                    source_id=str(source["id"]),
                    target_id=str(target["id"]),
                    role="contradicts" if same_turn else "supersedes",
                    reason=(
                        f"both state {key!r} in one turn, with different values"
                        if same_turn
                        else f"both state {key!r}: "
                        f"{earlier.get('value')!r} then {later.get('value')!r}"
                    ),
                    distance=None,
                    derived=True,
                    turn_indexes=sorted(
                        set(earlier.get("turn_indexes") or [])
                        | set(later.get("turn_indexes") or [])
                    ),
                )
            )
    return edges


def _orderings(nodes: Sequence[dict[str, Any]]) -> list[SemanticEdge]:
    """``precedes`` between dated events, by date.

    Events sharing a date get no edge. "A precedes B" is false when they
    happened on the same day, and there is no role for "simultaneous" because
    no question asks for one.
    """
    dated = [
        (node, when)
        for node in nodes
        if "event" in _kinds(node) and (when := _occurred(node))
    ]
    if len(dated) < 2:
        return []

    dated.sort(key=lambda pair: pair[1])
    pairs = [
        (earlier, early, later, late)
        for (earlier, early), (later, late) in itertools.combinations(dated, 2)
        if early < late
    ]
    if len(pairs) > MAX_EVENT_PAIRS:
        logger.info(
            "%d dated event(s) make %d pair(s) -- falling back to a chain.",
            len(dated),
            len(pairs),
        )
        pairs = [
            (earlier, early, later, late)
            for (earlier, early), (later, late) in itertools.pairwise(dated)
            if early < late
        ]

    return [
        SemanticEdge(
            source_id=str(earlier["id"]),
            target_id=str(later["id"]),
            role="precedes",
            # The interval, computed once here rather than left for a reader to
            # work out. "How many days between X and Y" is the question this
            # edge exists for, and a model that has to subtract two dates it
            # read out of prose is the step that was getting it wrong.
            reason=(
                f"{early.isoformat()} is {(late - early).days} day(s) "
                f"before {late.isoformat()}"
            ),
            distance=None,
            derived=True,
            turn_indexes=sorted(
                set(earlier.get("turn_indexes") or [])
                | set(later.get("turn_indexes") or [])
            ),
        )
        for earlier, early, later, late in pairs
    ]


def derive_edges(
    nodes: Sequence[dict[str, Any]],
) -> tuple[list[SemanticEdge], set[frozenset[str]]]:
    """Semantic edges that follow from the rows, and the pairs they account for.

    The second return value is what :mod:`relate` must not re-adjudicate.
    Unordered pairs, because a derived ``supersedes`` running later-to-earlier
    and a proposed candidate in extraction order are the same pair, and
    adjudicating it again would produce a second verdict with nothing to say
    which is current.
    """
    edges = _revisions(nodes) + _orderings(nodes)
    decided = {frozenset((edge.source_id, edge.target_id)) for edge in edges}

    counts: dict[str, int] = {}
    for edge in edges:
        counts[edge.role] = counts.get(edge.role, 0) + 1
    logger.info(
        "Derived %d semantic edge(s) without a model call: %s.",
        len(edges),
        ", ".join(f"{n} {r}" for r, n in sorted(counts.items())) or "none",
    )
    return edges, decided
