# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a *stored* conversation graph's edges against the turns they cite.

The same judgement :mod:`~auto_ontology.conversations.semantic.verify` makes
during a build, applied to a graph already in the database instead of to a
fresh extraction draft. That difference is the whole reason this exists:
finding the wrong edges costs about five minutes, while rebuilding the graph to
act on what it found costs half an hour of extraction and produces a
*different* graph -- not the one the verdicts were read for.

Two moments, two powers. During a build the check can shape what is written, so
a contradicted edge is dropped before it lands. Afterwards it can only remove,
and the vectors and paths built from those edges have to be rebuilt behind it.

Reports by default. ``apply=True`` deletes the edges whose own turns contradict
them and re-embeds the path index, because the paths are one row per edge and
would otherwise go on offering the deleted ones to retrieval.

    python -m auto_ontology.conversations.semantic.repair --database-name beam_100K_2 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auto_ontology.conversations.semantic.extract import EdgeDraft

logger = logging.getLogger(__name__)

__all__ = ["RepairReport", "repair_stored_graph"]

#: A stored node or edge, as ``fetch_nodes`` and ``fetch_edges`` return it.
Row = dict[str, Any]


@dataclass(frozen=True)
class RepairReport:
    """What a pass found, and what it changed if it was allowed to."""

    #: Extracted knowledge edges judged. Derived ones are excluded.
    checked: int
    #: Of those, edges their own turns contradict.
    unsupported: int
    #: Rows actually removed. Zero unless ``apply``, and compared against
    #: ``unsupported`` -- a delete that matched nothing would otherwise look
    #: like a success.
    deleted: int
    #: Whether the graph was changed.
    applied: bool


def repair_stored_graph(
    *,
    database_name: str,
    conversation_id: str,
    apply: bool = False,
) -> RepairReport:
    """Judge *conversation_id*'s stored edges, and delete the wrong ones if asked.

    Raises :class:`ValueError` when the conversation has no graph, rather than
    reporting a clean zero: nothing to check and nothing built are different
    answers, and only one of them means the graph is fine.
    """
    from auto_ontology.conversations.semantic.extract import EdgeDraft
    from auto_ontology.conversations.semantic.store import fetch_edges, fetch_nodes
    from auto_ontology.conversations.semantic.verify import verify_edges
    from auto_ontology.conversations.store import fetch_turns
    from auto_ontology.conversations.semantic.windows import build_windows

    nodes = fetch_nodes(database_name, conversation_id=conversation_id)
    if not nodes:
        raise ValueError(
            f"conversation {conversation_id!r} has no nodes in {database_name!r}: "
            "build its graph first"
        )
    stored = fetch_edges([node["id"] for node in nodes], type_="knowledge")

    # Only the extracted edges are checkable. A derived edge asserts nothing
    # about the text -- ``precedes`` is two dates compared, ``supersedes`` is two
    # values of one key -- so asking a model whether a turn states it is asking
    # the wrong question, and a 'no' would delete arithmetic that is correct.
    claimed = [edge for edge in stored if not edge.get("derived")]
    logger.info(
        "%d knowledge edge(s), %d of them extracted from the text.",
        len(stored),
        len(claimed),
    )

    # Rendered through ``build_windows`` rather than read off the rows, so the
    # verifier is shown character-for-character what the extractor was shown. A
    # turn labelled one way here and another way there would let the check turn
    # on formatting instead of on whether the edge is right.
    turns = fetch_turns(database_name, spans=[(conversation_id, 0, 10**9)])
    texts: dict[int, str] = {}
    for window in build_windows(turns, max_chars=1):
        texts.update(zip(window.turn_indexes, window.texts))

    # Keyed by identity, so the verdict can be traced back to the row to delete.
    # Two edges can carry the same names and roles after a merge, and matching
    # on those would delete a sibling the model never judged.
    rows: dict[int, Row] = {}
    drafts: list[EdgeDraft] = []
    for row in claimed:
        draft = EdgeDraft(
            source=row["source_name"],
            target=row["target_name"],
            role=row["role"],
            turn_indexes=set(row["turn_indexes"] or ()),
            evidence=row.get("evidence") or "",
        )
        rows[id(draft)] = row
        drafts.append(draft)

    _check_citations(nodes, texts)

    wrong = verify_edges(drafts, texts).wrong()
    if not wrong:
        logger.info("Nothing to delete.")
        return RepairReport(len(claimed), 0, 0, applied=apply)

    if not apply:
        logger.info(
            "%d edge(s) are unsupported by their own turns. Re-run with apply "
            "to delete them.",
            len(wrong),
        )
        return RepairReport(len(claimed), len(wrong), 0, applied=False)

    deleted = _delete(rows, wrong)
    _reembed_paths(database_name=database_name, conversation_id=conversation_id)
    return RepairReport(len(claimed), len(wrong), deleted, applied=True)


def _check_citations(nodes: list[Row], texts: dict[int, str]) -> None:
    """Report, for every multi-turn statement, which of its turns support it.

    Read-only even under ``apply``. Detaching a citation here would rewrite
    ``turn_indexes`` on a stored row, and the vectors, the note and the paths
    were all built from the turns as they stand; a graph whose citations and
    whose embeddings disagree is harder to reason about than one with a known
    bad citation in it. A 'denies' verdict also means a node is *missing*, and
    the only place that can be fixed is extraction -- so the useful output here
    is the list, which says whether the extraction rule worked.
    """
    from auto_ontology.conversations.semantic.extract import NodeDraft
    from auto_ontology.conversations.semantic.verify import verify_node_citations

    drafts = [
        NodeDraft(
            name=row["name"],
            type=row["type"],
            kinds=list(row["kinds"] or ()),
            description=row.get("description") or "",
            key=row.get("attribute_key") or "",
            value=row.get("value") or "",
            turn_indexes=set(row["turn_indexes"] or ()),
        )
        for row in nodes
    ]
    verify_node_citations(drafts, texts)


def _delete(rows: dict[int, Row], wrong: list[EdgeDraft]) -> int:
    """Delete one row per unsupported edge, by its whole primary key.

    ``type`` included: a semantic-layer edge can share the endpoints and role of
    a knowledge edge, and is a different row.
    """
    from sqlalchemy import and_, delete, or_

    from auto_ontology.dal import schema as s
    from auto_ontology.dal.session import store

    conditions = []
    for draft in wrong:
        row = rows[id(draft)]
        conditions.append(
            and_(
                s.conversation_edge.c.source_id == row["source_id"],
                s.conversation_edge.c.target_id == row["target_id"],
                s.conversation_edge.c.type == row["type"],
                s.conversation_edge.c.role == row["role"],
            )
        )

    # RETURNING, because ``query_write`` hands back rows rather than a rowcount
    # and a delete that matched nothing would otherwise log as a success.
    deleted = store().query_write(
        delete(s.conversation_edge)
        .where(or_(*conditions))
        .returning(s.conversation_edge.c.role)
    )
    logger.info("Deleted %d edge row(s) of %d asked for.", len(deleted), len(wrong))
    return len(deleted)


def _reembed_paths(*, database_name: str, conversation_id: str) -> None:
    """Rebuild the path index, which still carries the deleted edges.

    With ``reset`` on. The paths are one row per edge, so an incremental write
    would leave every deleted edge in the index and retrieval would go on
    offering relations the graph no longer holds.
    """
    from auto_ontology.conversations.semantic.pipeline import _embed_paths

    count = _embed_paths(
        database_name=database_name,
        conversation_id=conversation_id,
        reset=True,
    )
    logger.info("Re-embedded %d path(s).", count)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversation", help="Conversation id, e.g. 2")
    parser.add_argument(
        "--database-name",
        required=True,
        help="The scope its graph was built under.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the unsupported edges and re-embed the paths. Without it "
        "nothing is written and the run is a read-only report.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    repair_stored_graph(
        database_name=args.database_name,
        conversation_id=args.conversation,
        apply=args.apply,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
