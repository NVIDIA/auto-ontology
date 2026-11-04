# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How to read a search hit's score, whichever kind of search produced it.

A **semantic** hit's ``score`` is its cosine distance (``1 - cos``, lower is
better), and every threshold in the clarification flow was tuned on that
scale: ``CLARIFY_MAX_DISTANCE``, the ambiguity gap, the collision margins, the
KB-vs-column cutoff.

A **hybrid** hit (``VDB_SEARCH_MODE=hybrid``) was ranked by reciprocal rank
fusion of a BM25 and a vector search. Its ``score`` is ``1 - rrf`` -- still
lower-is-better, so sorting by it is right -- but it is a function of *ranks*,
not of distance, and two hits' scores say nothing about how far apart they
are. A hybrid hit also carries the fields in :data:`HYBRID_SCORE_FIELDS`.

Each use of a score therefore has its own rule:

* **Is this hit a match at all?** :func:`passes_relevance_gate`: its cosine
  distance is within ``max_distance``, in both modes. For a hybrid hit that is
  the vector branch's distance; a hit only BM25 found has none and does not
  pass.
* **Which hit is best?** ``score``, lower first, in both modes.
* **Are two hits close?** -- the gaps, margins and cutoffs tuned in cosine
  units: :func:`cosine_distance`. On a hybrid hit that is its vector
  branch's cosine distance; a hit only BM25 found has none and counts as
  infinitely far, which makes it lose every closeness test. Those tests all
  fall back to the cautious path (defer to the LLM, don't treat as a match).
* **Is the top pair a tie?** :func:`ambiguous_pair`. Cosine: their gap is
  under ``max_gap``. Hybrid: the two branches do not agree on their order.

Why the hybrid gate is not on ``rrf``: a fused score is built from ranks, and
every search ranks *something* first, so no RRF threshold can turn away an
entity the catalog has no answer for. Measured by ``dev_tools.calibrate_vdb``
on the fixture, an RRF gate of 0.5 (ranked first by at least one branch)
accepted all 15 such entities, and even 1.0 (first in both) accepted 4. The
vector distance, the one absolute relevance signal a hybrid hit has, accepted
none of them at 0.70. RRF still decides the order.
"""

from __future__ import annotations

import math
from typing import Any

#: Fields a hybrid hit carries besides ``score``, as search rows name them.
HYBRID_SCORE_FIELDS = ("rrf", "lexical_score", "semantic_score", "semantic_distance")


def is_hybrid_hit(hit: dict[str, Any]) -> bool:
    return hit.get("rrf") is not None


def _float(value: Any, missing: float) -> float:
    try:
        return float(value) if value is not None else missing
    except (TypeError, ValueError):
        return missing


def passes_relevance_gate(hit: dict[str, Any], max_distance: float) -> bool:
    """Whether *hit* is a match at all: its cosine distance is within *max_distance*."""
    return cosine_distance(hit) <= max_distance


def cosine_distance(hit: dict[str, Any], missing: float = math.inf) -> float:
    """*hit*'s cosine distance, for comparisons tuned in cosine units.

    *missing* when there is none: no score, or a hybrid hit only BM25 found.
    """
    if is_hybrid_hit(hit):
        return _float(hit.get("semantic_distance"), missing)
    return _float(hit.get("score"), missing)


def _branches_agree(best: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether the BM25 and vector searches both put *best* ahead of *second*.

    A branch that found neither has no opinion; one that found only one of
    them prefers that one. Agreement needs at least one branch for *best* and
    none for *second*.
    """
    prefers_best = prefers_second = False
    for key in ("lexical_score", "semantic_score"):
        a, b = best.get(key), second.get(key)
        if a is None and b is None:
            continue
        a_score, b_score = _float(a, -math.inf), _float(b, -math.inf)
        if a_score > b_score:
            prefers_best = True
        elif b_score > a_score:
            prefers_second = True
    return prefers_best and not prefers_second


def ambiguous_pair(
    best: dict[str, Any], second: dict[str, Any], max_gap: float
) -> bool:
    """Whether *best* and *second* (both already past the gate) are a tie.

    Cosine: their distances are within *max_gap*. Hybrid: the two searches do
    not agree that *best* comes first -- one ranks *second* higher, or neither
    can tell them apart. *max_gap* is not used for hybrid hits.
    """
    if is_hybrid_hit(best) or is_hybrid_hit(second):
        return not _branches_agree(best, second)
    return cosine_distance(second) - cosine_distance(best) < max_gap
