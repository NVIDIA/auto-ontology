# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collapse things the transcript named more than one way.

Identity across windows is exact string equality after
:func:`~auto_ontology.conversations.semantic.extract.normalise_name`, so "API
integration module" and "API module" stay two nodes forever. Each then carries
half the edges and half the turns, and a question that matches one of them gets
a subgraph missing whatever attached to the other. Cognee has the same
limitation in its default pipeline and ships the same remedy as an opt-in pass.

**Only things, never statements.** A statement's content *is* its value, and two
readings of one property are exactly what a revision is made of: merging
``coverage is 78%`` into ``coverage is 85%`` would delete the revision rather
than record it. ``supersedes`` is where that belongs, and it needs both rows.

**Only within one type.** A ``Feature`` and the ``Person`` who asked for it can
have similar names and are not the same node. Cognee defaults to the same
restriction and so does this, with no option to lift it: the failure it
prevents is silent, and the recall it costs is a node pair that stays split --
which is the status quo, not a regression.

The pass is conservative by construction. It runs on the drafts, before
anything is written, so a merge is a rewrite in memory rather than a surgery on
stored rows: there are no edges to re-point in Postgres, no vectors to purge
and nothing to undo. What reaches the store is the already-merged graph.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from auto_ontology.conversations.semantic.extract import (
    EdgeDraft,
    GraphDraft,
    NodeDraft,
    normalise_name,
)

logger = logging.getLogger(__name__)

__all__ = ["Merge", "consolidate_graph", "plan_merges"]

#: Cosine similarity at which two names of one type are taken to be one thing.
#:
#: 0.90 rather than Cognee's 0.85. Their pass is run deliberately by a user who
#: can inspect a dry run first; this one runs unattended inside every build, so
#: it is tuned for precision. At 0.85 "fetch call latency" and "autocomplete
#: input latency" are near enough to merge, and they are two measurements whose
#: difference is the answer to a question in this benchmark.
DEFAULT_THRESHOLD = 0.90

#: Types never merged, however close the names.
#:
#: A ``Timestamp`` is the clearest case: "March 10 2024" and "March 12 2024"
#: embed almost identically and are the two ends of an interval question.
PROTECTED_TYPES = frozenset({"Timestamp", "Subject"})


@dataclass(frozen=True)
class Merge:
    """One cluster, collapsed onto *canonical*."""

    canonical: str
    duplicates: tuple[str, ...]
    similarity: float


def _degree(edges: Sequence[EdgeDraft]) -> dict[str, int]:
    """How many edges touch each node, by normalised name."""
    counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        counts[edge.source] += 1
        counts[edge.target] += 1
    return counts


def _shares_a_name(left: str, right: str) -> bool:
    """Whether one normalised name's words are contained in the other's.

    Required *in addition to* similarity, because similarity alone does not
    work here. Measured on one transcript's 63 things: at 0.90 cosine the pass
    wanted to merge ``express`` with ``slack`` and ``javascript`` with ``axios``
    and ``node fetch``; at 0.85 it collapsed nineteen technologies onto one
    node; at 0.95 it merged nothing at all. Short technology names of one type
    embed close together because the embedding encodes the *register* -- a
    JavaScript tooling name -- far more strongly than the identity.

    Containment is what actually separates an alias from a sibling. "API
    module" is a subset of "API integration module"; ``express`` and ``slack``
    share no word. This is the signal Cognee's own example turns on too -- NYC
    against New York City -- it just happens to survive similarity there
    because proper nouns are not drawn from one small register.

    Single-word names are excluded. Containment is trivially satisfied by a
    one-word name inside any phrase that mentions it, which would merge
    ``cache`` into ``cache invalidation strategy``.
    """
    left_words = set(left.split())
    right_words = set(right.split())
    if len(left_words) < 2 or len(right_words) < 2:
        return False
    return left_words <= right_words or right_words <= left_words


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def plan_merges(
    nodes: Sequence[NodeDraft],
    edges: Sequence[EdgeDraft],
    vectors: Mapping[str, Sequence[float]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Merge]:
    """Which things to collapse, and onto what.

    *vectors* are keyed by normalised name and embed the **name alone**, not
    the name and description together the way the search vectors do. Two
    distinct features of one project routinely share most of a description --
    "part of the weather app, written this sprint" -- and comparing those
    merges things that merely sit near each other. The name is the claim about
    identity; the description is a claim about the thing.

    Single-link clustering: a node joins a cluster when it is close enough to
    *any* member, not to all of them. Aliases chain -- "API module", "API
    integration module", "the API integration module" -- and requiring every
    pair to clear the threshold would leave the ends of such a chain apart.
    Single-link is the failure-prone choice at a low threshold, which is the
    other reason :data:`DEFAULT_THRESHOLD` sits where it does.

    The canonical is the **most connected** member, because it is the one most
    of the graph already points at, so a merge rewrites the fewest edges and
    the name most of the transcript used survives. Ties go to the longest name
    -- "API integration module" over "API module", the more specific of the two
    being the safer thing to show a reader -- and then to the name itself, so
    the plan does not depend on dictionary order.
    """
    things = [
        node
        for node in nodes
        if list(node.kinds) == ["object"]
        and node.type not in PROTECTED_TYPES
        and vectors.get(normalise_name(node.name))
    ]
    if len(things) < 2:
        return []

    degree = _degree(edges)
    by_type: dict[str, list[NodeDraft]] = defaultdict(list)
    for node in things:
        by_type[node.type].append(node)

    merges: list[Merge] = []
    for group in by_type.values():
        if len(group) < 2:
            continue
        # Union-find over the pairs that clear the threshold.
        parent = {
            normalise_name(node.name): normalise_name(node.name) for node in group
        }
        closeness: dict[str, float] = {}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_key = normalise_name(left.name)
                right_key = normalise_name(right.name)
                if not _shares_a_name(left_key, right_key):
                    continue
                score = _cosine(vectors[left_key], vectors[right_key])
                if score < threshold:
                    continue
                left_root, right_root = (
                    find(normalise_name(left.name)),
                    find(normalise_name(right.name)),
                )
                if left_root != right_root:
                    parent[right_root] = left_root
                root = find(normalise_name(left.name))
                closeness[root] = max(closeness.get(root, 0.0), score)

        clustered: dict[str, list[NodeDraft]] = defaultdict(list)
        for node in group:
            clustered[find(normalise_name(node.name))].append(node)

        for root, members in clustered.items():
            if len(members) < 2:
                continue
            ranked = sorted(
                members,
                key=lambda n: (
                    -degree.get(normalise_name(n.name), 0),
                    -len(n.name),
                    normalise_name(n.name),
                ),
            )
            canonical, *duplicates = ranked
            merges.append(
                Merge(
                    canonical=normalise_name(canonical.name),
                    duplicates=tuple(normalise_name(n.name) for n in duplicates),
                    similarity=closeness.get(root, 0.0),
                )
            )
    return merges


def _absorb(canonical: NodeDraft, duplicate: NodeDraft) -> None:
    """Fold *duplicate* into *canonical*, losing nothing that was grounded."""
    canonical.turn_indexes |= duplicate.turn_indexes
    canonical.verified_turn_indexes |= duplicate.verified_turn_indexes
    if duplicate.description and duplicate.description not in canonical.description:
        canonical.description = (
            f"{canonical.description} {duplicate.description}".strip()
        )
    if not canonical.evidence:
        canonical.evidence = duplicate.evidence
    if canonical.time_anchor is None:
        canonical.time_anchor = duplicate.time_anchor
    if canonical.occurred_on is None:
        canonical.occurred_on = duplicate.occurred_on


def consolidate_graph(
    graph: GraphDraft,
    vectors: Mapping[str, Sequence[float]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Merge]:
    """Apply :func:`plan_merges` to *graph* in place, returning what was merged.

    Edges are re-pointed rather than dropped, and an edge that becomes a
    self-loop is removed: "the API module is a component of the API integration
    module" says nothing once those are one node. Edges that become duplicates
    of each other are folded together, keeping the union of their turns, so a
    relation stated about both spellings is evidenced by both.
    """
    merges = plan_merges(graph.nodes, graph.edges, vectors, threshold=threshold)
    if not merges:
        return []

    canonical_of: dict[str, str] = {}
    for merge in merges:
        for duplicate in merge.duplicates:
            canonical_of[duplicate] = merge.canonical

    by_name = {normalise_name(node.name): node for node in graph.nodes}
    for duplicate, canonical in canonical_of.items():
        if duplicate in by_name and canonical in by_name:
            _absorb(by_name[canonical], by_name[duplicate])

    graph.nodes = [
        node for node in graph.nodes if normalise_name(node.name) not in canonical_of
    ]

    folded: dict[tuple[str, str, str], EdgeDraft] = {}
    for edge in graph.edges:
        source = canonical_of.get(edge.source, edge.source)
        target = canonical_of.get(edge.target, edge.target)
        if source == target:
            continue
        key = (source, edge.role, target)
        existing = folded.get(key)
        if existing is None:
            edge.source, edge.target = source, target
            folded[key] = edge
            continue
        existing.turn_indexes |= edge.turn_indexes
        if not existing.evidence:
            existing.evidence = edge.evidence
    graph.edges = list(folded.values())

    logger.info(
        "Consolidated %d name(s) into %d node(s): %s",
        sum(len(m.duplicates) for m in merges),
        len(merges),
        "; ".join(
            f"{m.canonical} <- {', '.join(m.duplicates)} ({m.similarity:.2f})"
            for m in merges
        ),
    )
    return merges
