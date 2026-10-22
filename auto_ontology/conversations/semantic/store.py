# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes over ``conversation_node`` and ``conversation_edge``."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_, delete, insert, select

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_edges",
    "fetch_nodes",
    "replace_graph",
    "session_index_by_turn",
    "subjects_of",
    "write_edges",
]

_COLUMNS = (
    s.conversation_node.c.id,
    s.conversation_node.c.conversation_id,
    s.conversation_node.c.name,
    s.conversation_node.c.type,
    s.conversation_node.c.kinds,
    s.conversation_node.c.description,
    s.conversation_node.c.attribute_key,
    s.conversation_node.c.value,
    s.conversation_node.c.turn_indexes,
    s.conversation_node.c.evidence,
    s.conversation_node.c.time_anchor,
    s.conversation_node.c.occurred_on,
)


def _database_id(database_name: str) -> str | None:
    rows = store().query_read(
        select(s.catalog_database.c.id).where(
            s.catalog_database.c.name == database_name
        )
    )
    return str(rows[0]["id"]) if rows else None


def fetch_nodes(
    database_name: str,
    *,
    conversation_id: str | None = None,
    ids: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Nodes of one memory.

    *ids* restricts to specific rows, which is what retrieval uses after a
    vector search; *conversation_id* restricts to one transcript, which is what
    the relate pass uses. Both omitted reads every node in the memory.

    *kinds* keeps only nodes carrying at least one of them. This is the route an
    aggregate question needs -- "how many bugs did I fix" is a count over
    ``event``, not a similarity search -- and the GIN index on ``kinds`` is what
    makes it an index scan rather than a table sweep.
    """
    query = (
        select(*_COLUMNS)
        .select_from(
            s.conversation_node.join(
                s.catalog_database,
                s.catalog_database.c.id == s.conversation_node.c.database_id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
    )
    if conversation_id is not None:
        query = query.where(s.conversation_node.c.conversation_id == conversation_id)
    if ids is not None:
        if not ids:
            return []
        query = query.where(s.conversation_node.c.id.in_(list(ids)))
    if kinds:
        query = query.where(s.conversation_node.c.kinds.overlap(list(kinds)))
    return store().query_read(query.order_by(s.conversation_node.c.name))


def session_index_by_turn(database_name: str) -> dict[tuple[str, int], int]:
    """Which sitting each turn belongs to, as ``(conversation_id, turn_index)``.

    A node records the turns it was drawn from and nothing about sittings, and
    it does not need to: ``conversation.session_index`` already says which
    sitting a turn is in, so a node's sittings are a lookup rather than a column
    to populate and keep true. Denormalising it onto the node would add a
    migration and a second place for the same fact to be wrong.

    Keyed on the conversation as well as the turn because turn numbering is
    per-transcript and twenty of them share a ``database_name`` -- turn 5 exists
    in each, in a different sitting each time.

    The whole memory in one query, because the caller renders a subgraph that
    may span transcripts and the table is small: one row per turn.
    """
    rows = store().query_read(
        select(
            s.conversation.c.conversation_id,
            s.conversation.c.turn_index,
            s.conversation.c.session_index,
        )
        .select_from(
            s.conversation.join(
                s.catalog_database,
                s.catalog_database.c.id == s.conversation.c.database_id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
    )
    return {
        (str(row["conversation_id"]), int(row["turn_index"])): int(row["session_index"])
        for row in rows
        if row["turn_index"] is not None and row["session_index"] is not None
    }


def subjects_of(edges: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """What each statement is about, read off the knowledge edges already here.

    No extra query: the subject of a statement *is* a knowledge edge, and the
    caller was given those. Several are possible -- a latency is measured on a
    feature and in a browser -- and all are kept, because which one
    disambiguates depends on the question.

    Both readers of this need the same answer for the same reason. The note
    prefixes a rendered statement with its subject so "average response time
    280ms" is not read as the app's; adjudication compares subjects so that
    reading is not written into the graph as a revision in the first place.
    """
    found: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("type") != "knowledge":
            continue
        # The subject is the ``about`` edge, not every knowledge edge leaving
        # the statement. Without this filter a statement's `built_with` and
        # `obtained_at` targets are read as subjects too, and the two callers
        # fail in opposite directions: the note prefixes a reading with a
        # library's name, and adjudication compares the wrong things. Measured:
        # 'OpenWeather API key obtained' carries only `obtained_at` ->
        # '2024-03-10', so its subject read as that date, and the retraction
        # 'Never obtained an API key' was rejected as a revision of it for
        # naming a different one -- when the honest answer is that the first
        # statement has no recorded subject at all, which this guard allows.
        if edge.get("role") != "about":
            continue
        name = (edge.get("target_name") or "").strip()
        source = str(edge.get("source_id") or "")
        if name and source and name not in found.setdefault(source, []):
            found[source].append(name)
    return found


def fetch_edges(
    ids: Sequence[str],
    *,
    type_: str | None = None,
) -> list[dict[str, Any]]:
    """Every edge with **both** ends inside *ids*, ``unrelated`` excluded.

    Both ends, which is what keeps this bounded and useful: a central node
    carries dozens of edges across a transcript, but only the ones joining two
    nodes already in front of the model can change how it reads them. A
    one-ended expansion would pull in the whole graph from any node.

    *type_* restricts to one layer. Omitted returns both, which is the normal
    case -- the knowledge edges say what the things are to each other and the
    semantic edges say which statements still hold, and answering a question
    usually needs both.

    ``unrelated`` is stored and never read back here. It exists to stop a later
    pass re-adjudicating a pair, and rendering "these two are unrelated" into a
    prompt would spend context saying nothing.
    """
    if not ids:
        return []
    edge = s.conversation_edge
    source = s.conversation_node.alias("source")
    target = s.conversation_node.alias("target")
    query = (
        select(
            edge.c.source_id,
            edge.c.target_id,
            edge.c.type,
            edge.c.role,
            edge.c.statement,
            edge.c.reason,
            edge.c.derived,
            edge.c.turn_indexes,
            edge.c.evidence,
            source.c.name.label("source_name"),
            source.c.type.label("source_type"),
            source.c.description.label("source_description"),
            source.c.turn_indexes.label("source_turns"),
            source.c.occurred_on.label("source_occurred_on"),
            target.c.name.label("target_name"),
            target.c.type.label("target_type"),
            target.c.description.label("target_description"),
            target.c.turn_indexes.label("target_turns"),
            target.c.occurred_on.label("target_occurred_on"),
        )
        .select_from(
            edge.join(source, source.c.id == edge.c.source_id).join(
                target, target.c.id == edge.c.target_id
            )
        )
        .where(
            edge.c.source_id.in_(list(ids)),
            edge.c.target_id.in_(list(ids)),
            edge.c.role != "unrelated",
        )
    )
    if type_ is not None:
        query = query.where(edge.c.type == type_)
    return store().query_read(query)


def _require_distinct_names(nodes: Sequence[dict[str, Any]]) -> None:
    """Refuse a node list that would put two rows under one name.

    Raised, not repaired. The obvious repair -- rename the second -- cannot be
    done here: this function's caller returns ids keyed by name, so a caller
    that still knows the node by the name it passed in would look up the
    renamed row and be handed the other one's id. Two nodes then share an id,
    and the first visible symptom is a derived edge from a node to itself,
    several steps from the naming problem that caused it.

    So the invariant is checked at the boundary and enforced upstream, where
    the drafts can still be renamed before anything is keyed by them. See
    ``pipeline._disambiguate``.
    """
    seen: dict[str, dict[str, Any]] = {}
    for node in nodes:
        normalized = node["normalized_name"]
        if normalized in seen:
            raise ValueError(
                f"two nodes share the name {node['name']!r} (properties "
                f"{seen[normalized].get('attribute_key')!r} and "
                f"{node.get('attribute_key')!r}); names must be unique within "
                "a transcript. Disambiguate the drafts before writing them."
            )
        seen[normalized] = node


def replace_graph(
    *,
    database_name: str,
    conversation_id: str,
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """Write one transcript's nodes and knowledge edges, replacing what was there.

    Returns the node ids keyed by normalised name, so the caller can write
    semantic edges and vectors against the ids that landed.

    Delete-then-insert rather than upsert. A node has no stable key -- it is
    whatever the extractor named it this time -- so a second run that renames
    "Fetch latency" to "API response latency" would upsert a new row and leave
    the old one, and retrieval would return both as though the transcript stated
    it twice. The whole transcript's graph is one unit.

    One transaction, so a failed re-extraction leaves the previous run intact
    rather than a transcript with nodes and no edges.
    """
    with write_transaction():
        handle = store()
        database_id = _database_id(database_name)
        if database_id is None:
            raise ValueError(
                f"no database named {database_name!r}: ingest the conversation first"
            )

        handle.query_write(
            delete(s.conversation_node).where(
                and_(
                    s.conversation_node.c.database_id == database_id,
                    s.conversation_node.c.conversation_id == conversation_id,
                )
            )
        )

        _require_distinct_names(nodes)
        by_name: dict[str, str] = {}
        for node in nodes:
            rows = handle.query_write(
                insert(s.conversation_node)
                .values(
                    database_id=database_id,
                    conversation_id=conversation_id,
                    name=node["name"],
                    normalized_name=node["normalized_name"],
                    type=node["type"],
                    kinds=list(node["kinds"]),
                    description=node.get("description") or None,
                    attribute_key=node.get("attribute_key") or None,
                    value=node.get("value") or None,
                    turn_indexes=sorted(node.get("turn_indexes") or []),
                    evidence=node.get("evidence") or None,
                    time_anchor=node.get("time_anchor"),
                    occurred_on=node.get("occurred_on"),
                )
                .returning(s.conversation_node.c.id)
            )
            by_name[node["normalized_name"]] = str(rows[0]["id"])

        written = 0
        unresolved: list[str] = []
        for edge in edges:
            source, target = by_name.get(edge["source"]), by_name.get(edge["target"])
            if not source or not target:
                unresolved.append(
                    f"{edge['source']!r} --{edge['role']}--> {edge['target']!r}"
                )
                continue
            written += 1
            handle.query_write(
                insert(s.conversation_edge).values(
                    source_id=source,
                    target_id=target,
                    type="knowledge",
                    role=edge["role"],
                    statement=edge.get("statement"),
                    turn_indexes=sorted(edge.get("turn_indexes") or []),
                    evidence=edge.get("evidence") or None,
                    derived=False,
                )
            )

    # What landed, not what was handed over. An edge naming a node that is not
    # in the write is skipped above, and reporting the input count made a pass
    # that produced 334 subject edges and wrote none of them look like it had
    # worked -- the names were right and in the wrong form, which is exactly the
    # failure a count of the input cannot show.
    if unresolved:
        listed = "\n  ".join(unresolved[:10])
        more = (
            f"\n  ... and {len(unresolved) - 10} more" if len(unresolved) > 10 else ""
        )
        logger.warning(
            "Dropped %d of %d edge(s) naming a node that was not written -- an "
            "endpoint has to be a normalised node name:\n  %s%s",
            len(unresolved),
            len(edges),
            listed,
            more,
        )
    logger.info(
        "Wrote %d node(s) and %d knowledge edge(s) for conversation %s.",
        len(by_name),
        written,
        conversation_id,
    )
    return by_name


def write_edges(edges: Sequence[Any]) -> int:
    """Insert semantic edges. Returns how many landed.

    Runs after :func:`replace_graph`, which has already cleared the transcript's
    previous nodes -- and this table cascades from them, so the old edges are
    gone with their endpoints and there is nothing to delete here.

    Duplicates are dropped rather than rejected. The primary key is
    ``(source_id, target_id, type, role)`` and two derivations landing on the
    same one are saying the same thing twice -- there is no second fact to
    lose, unlike two nodes under one name. Logged, because the usual cause is
    upstream: two drafts that resolved to a single id, which makes every edge
    between them an edge from a node to itself.
    """
    if not edges:
        return 0

    seen: set[tuple[str, str, str]] = set()
    unique = []
    for edge in edges:
        key = (str(edge.source_id), str(edge.target_id), edge.role)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    if len(unique) != len(edges):
        logger.warning(
            "Dropped %d duplicate semantic edge(s) of %d.",
            len(edges) - len(unique),
            len(edges),
        )
    edges = unique

    with write_transaction():
        handle = store()
        for edge in edges:
            handle.query_write(
                insert(s.conversation_edge).values(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    type="semantic",
                    role=edge.role,
                    reason=edge.reason,
                    distance=edge.distance,
                    derived=edge.derived,
                    turn_indexes=list(edge.turn_indexes),
                )
            )
    return len(edges)
