# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Applying a rule: replaying its search, and labelling what the search finds.

A rule is a saved global search plus the tags to apply to everything it matches,
and this is the half that does the applying. :func:`find_targets` replays the
search through the *same* service the ``/search/global-search`` route calls, so
a rule labels exactly what the dialog that created it had on screen -- including
the list cap, since a rule that labelled more than the person could see would be
a rule they did not agree to. :func:`label_targets` writes the labels it found.

Two functions because only the second one belongs inside the transaction that
saves the rule, and the split is what keeps the search out of it. See each for
why.

Called once, when the rule is created. The tags it applies carry the rule's id
rather than a user id (see ``gsf.dal.tags.attach_tags_by_rule``), which is what
lets a tag's page say a rule did this and name it, and what makes deleting the
rule take those labels back.

Re-applying as the catalog grows is not wired up yet: nothing calls this on
ingest. It is written to be safe to call again -- labels it already applied are
skipped rather than rewritten -- so that hook is a call site, not a rewrite.
"""

from __future__ import annotations

import logging
from typing import Any

from gsf.dal import tags as tags_dal
from gsf.dal.tags import (
    TARGET_COLUMN,
    TARGET_COLUMN_ATTRIBUTE,
    TARGET_SQL_ATTRIBUTE,
    TARGET_TABLE,
    TARGET_TERM,
)
from gsf.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
)
from gsf.catalog.constants import Labels
from gsf.server.search import service as search_service

logger = logging.getLogger(__name__)

#: Which search hits can carry a tag, and as which kind of target.
#:
#: Global search answers over ten kinds and only five of them are taggable:
#: Databases and Schemas are containers a tag is deliberately not applied to
#: (see ``tag_target`` in ``gsf/dal/schema.py``), and the two analysis kinds have
#: no tag column at all. Hits of those kinds are dropped rather than refused --
#: a rule saved from the *All* tab legitimately matches them, and it still means
#: "label what you can".
#:
#: ``View`` is absent on purpose: a view is a ``catalog_table`` row whose
#: ``table_type`` says so, so a view hit arrives as ``Table`` and is labelled as
#: one. See ``SEARCH_TYPE_VIEW`` in ``gsf/dal/search.py``.
TAGGABLE_SEARCH_TYPES: dict[str, str] = {
    LABEL_TERM: TARGET_TERM,
    Labels.TABLE: TARGET_TABLE,
    Labels.COLUMN: TARGET_COLUMN,
    LABEL_COLUMN_ATTRIBUTE: TARGET_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE: TARGET_SQL_ATTRIBUTE,
}


def _taggable_targets(hits: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The ``(kind, item_id)`` pairs among *hits* that a tag can be applied to.

    Hits of an untaggable kind are dropped, as are hits with no id -- the search
    normalises a missing one to ``None`` rather than omitting the row, and a
    label needs something to point at.
    """
    targets = []
    for hit in hits:
        kind = TAGGABLE_SEARCH_TYPES.get(hit.get("type") or "")
        item_id = hit.get("id")
        if kind is not None and item_id is not None:
            targets.append((kind, item_id))
    return targets


def find_targets(
    *,
    search_term: str,
    text_match_option: str,
    filters: dict[str, Any],
) -> list[tuple[str, str]]:
    """Replay the rule's search and return what of it can carry a tag.

    Read-only, and called *before* the transaction that saves the rule rather
    than inside it. It needs nothing that transaction produces -- a rule is not
    a searchable object, so the search reads the same catalog either way -- and
    it is the slow half by an order of magnitude: measured against half a
    million columns it takes 70-400 ms, where the insert and the labels together
    take about 20 ms. A substring match has no index to use (the btree on
    ``name`` cannot serve ``ilike '%x%'``), so that cost grows with the catalog.
    Run inside the transaction, all of it was spent holding a pooled *write*
    connection and the locks on the freshly inserted rule row.

    Ordering the search first is not a weaker guarantee. The two writes still
    commit together, which is the whole of what atomicity buys here, and a
    ``SELECT`` takes no locks on what it finds: an object could already
    disappear between being matched and being labelled, and the transaction
    could only ever roll the labels back, not prevent it.

    *filters* is the stored JSON, read with the same defaults
    ``GlobalSearchFilters`` declares: it is written with ``exclude_none``, so a
    filter the dialog never sent is absent from the row rather than null, and
    reading it with ``.get`` is what keeps those two the same rule.

    A search that matches nothing is not an error, and returns no targets. A
    rule is a standing instruction -- "label what this finds" -- and finding
    nothing today says only that the catalog does not have it yet.
    """
    found = search_service.global_search(
        search_term=search_term,
        text_match_option=text_match_option,
        objects=filters.get("objects"),
        include_description=filters.get("description", False),
        include_synonyms=filters.get("synonyms", True),
    )
    hits = found["data"]
    targets = _taggable_targets(hits)
    logger.info(
        "Rule search %r matched %d objects, %d taggable",
        search_term,
        len(hits),
        len(targets),
    )
    return targets


def label_targets(
    *, rule_id: str, tag_ids: list[str], targets: list[tuple[str, str]]
) -> int:
    """Apply the rule's tags to *targets*, and say how many labels stuck.

    The transactional half: called inside the block that inserts the rule, so
    the rule and its labels are one unit of work and a rule that could not
    label the catalog is not left saved and inert.

    The count is of labels *applied*, not objects matched: an object already
    carrying the tag is not counted, because nothing was done to it.
    """
    applied = tags_dal.attach_tags_by_rule(
        rule_id=rule_id, tag_ids=tag_ids, targets=targets
    )
    logger.info(
        "Rule %s applied %d labels over %d targets", rule_id, applied, len(targets)
    )
    return applied
