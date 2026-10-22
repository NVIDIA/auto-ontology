# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Populate one transcript's graph, end to end.

Runs after ingest, against the turns already in the store, and builds the two
layers in order:

1. read the transcript and window it,
2. extract nodes and ``knowledge`` edges,
3. collapse things the transcript named more than one way,
4. decide what each statement is about, choosing from the things now that they
   are all known, and write both, replacing whatever the last run left,
5. settle the property keys, so one measurement named two ways is one property,
   then derive the ``semantic`` edges that follow from the rows -- revised
   values, dated events,
6. adjudicate the semantic edges that need reading -- ``refines``, and
   disagreements spanning different properties,
7. embed the nodes into ``vdb.conversation_semantic``.

The order is not arbitrary. The semantic layer sits on top of the knowledge
layer by construction: both derivation and adjudication need the row ids, so
neither can run before the write. The embed goes last because it is the only
step recoverable on its own -- a missing vector can be backfilled from the
stored rows, while a missing node means re-reading the transcript through the
model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from auto_ontology.conversations.semantic import store as semantic_store
from auto_ontology.conversations.semantic.consolidate import consolidate_graph
from auto_ontology.conversations.semantic.derive import derive_edges
from auto_ontology.conversations.semantic.extract import (
    extract_graph,
    node_text,
    normalise_name,
)
from auto_ontology.conversations.semantic.keys import reconcile_keys
from auto_ontology.conversations.semantic.relate import relate_nodes
from auto_ontology.conversations.semantic.subjects import assign_subjects
from auto_ontology.dal.schema import CONVERSATION_NODE_KINDS
from auto_ontology.conversations.semantic.windows import WINDOW_CHARS, build_windows
from auto_ontology.conversations.store import fetch_turns

logger = logging.getLogger(__name__)

__all__ = [
    "GraphResult",
    "populate_conversation_graph",
    "relate_conversation",
]


@dataclass(frozen=True)
class GraphResult:
    """What one transcript's pass produced."""

    conversation_id: str
    nodes: int
    things: int
    statements: int
    knowledge_edges: int
    derived_edges: int
    semantic_edges: int
    vectors: int
    paths: int
    seconds: float


def _edge_statement(source: str, role: str, target: str) -> str:
    """A knowledge edge as a sentence, rendered once at write time.

    So that every reader of the row gets the same wording. The document
    pipeline learned this the expensive way, by having three callers render
    relations separately until one of them printed the role raw and told the
    model nothing.
    """
    return f"{source} {role.replace('_', ' ')} {target}"


def populate_conversation_graph(
    *,
    database_name: str,
    conversation_id: str,
    relate: bool = True,
    reset_vectors: bool = False,
    window_chars: int = WINDOW_CHARS,
    draft_cache: Path | None = None,
    verify: bool = False,
) -> GraphResult:
    """Extract, relate, store and embed one transcript's graph.

    *relate* turns off the **adjudication** pass only. The derived semantic
    edges are free and always written: skipping them would leave a transcript
    whose revised values look like unrelated readings. Useful for getting the
    graph in place and measuring the node route before paying for ``refines``.

    *reset_vectors* clears the semantic collection before writing. Off by
    default because several transcripts normally share a ``database_name`` and
    each would otherwise wipe the last; the first conversation of a fresh run
    passes it once.

    *window_chars* is how much turn text one extraction call is shown. Pass 1
    for one turn per call, which is cognee's unit: a chunk there is a single
    user-assistant turn. The default packs several turns together, which on a
    BEAM transcript came out as one to four of them -- the turns are long.

    *draft_cache* is a file to read the extraction draft from, and to write it
    to when it is not there or no longer matches. For retrying the stages after
    extraction, which is where the tuning happens and where an hour of model
    calls has twice been lost to a bug in a stage that costs seconds. Off by
    default, and it should stay off in a measured run: a cached draft is a claim
    about what the current code does, and a measurement wants it tested.

    *verify* reads the extracted graph back against the turns it came from.
    Edges whose own turns contradict them are **dropped**; edges merely
    uncited, and statements citing a turn that denies them, are reported and
    kept. The asymmetry is deliberate: a wrong edge asserts something the
    transcript does not say, while a wrong citation means a node is *missing*,
    and removing things cannot add the one that is absent. Measured on one
    transcript: 15 edges of 179 dropped, 8 citations of 123 flagged. Checks the
    part of the graph no other pass examines -- everything else compares nodes.
    """
    started = time.monotonic()
    turns = fetch_turns(database_name, spans=[(conversation_id, 0, 10**9)])
    if not turns:
        raise ValueError(
            f"conversation {conversation_id!r} has no turns in {database_name!r}: "
            "ingest it first"
        )

    windows = build_windows(turns, max_chars=window_chars)
    logger.info(
        "Conversation %s: %d turn(s) in %d window(s).",
        conversation_id,
        len(turns),
        len(windows),
    )

    graph = None
    digest = ""
    if draft_cache is not None:
        from auto_ontology.conversations.semantic.draft_cache import (
            fingerprint,
            load_draft,
        )

        digest = fingerprint(windows)
        graph = load_draft(draft_cache, digest)

    if graph is None:
        graph = extract_graph(windows)
        if draft_cache is not None:
            from auto_ontology.conversations.semantic.draft_cache import save_draft

            # Before consolidation, which mutates the draft: the saved point has
            # to be the output of extraction alone, or the fingerprint stops
            # describing what produced the file.
            save_draft(draft_cache, digest, graph)

    if verify:
        # Before consolidation renames anything: the edge is judged in the
        # extractor's own words, which are the words the turns are written in.
        # The subject edges are not here to be checked, and do not need to be --
        # they are chosen from a closed list of nodes rather than asserted.
        from auto_ontology.conversations.semantic.verify import (
            build_denials,
            detach_bad_citations,
            verify_edges,
            verify_node_citations,
        )

        texts = _turn_texts(windows)
        # Reported, not dropped. Measured once with dropping on: 21 edges of 213
        # removed, and `information_extraction` fell from 0.917 to 0.417,
        # because two of the deletions were ``built_with html5`` and
        # ``built_with css3`` -- rejected for naming HTML5 where the turn wrote
        # HTML -- and the answer to "what technologies did I use" lost two of
        # its four items. The verdict conflates a contradicted relation with one
        # whose node is named more precisely than the sentence, and only the
        # first is a reason to delete anything.
        verify_edges(graph.edges, texts)
        # Before consolidation for the same reason, and before the subject pass
        # because a statement citing a turn that denies it is a statement whose
        # subject was chosen from the wrong evidence.
        citations = verify_node_citations(graph.nodes, texts)
        # Built before the citations are detached, while the claims still name
        # the turns that contradict them -- that pairing is the whole input.
        graph.nodes.extend(build_denials(citations, texts, graph.nodes))
        detach_bad_citations(citations)

    consolidate_graph(graph, _embed_names(graph.nodes))
    _disambiguate(graph.nodes)
    # After the names are settled and before the write: the subjects are
    # answered in the names that will be stored, and land in the same
    # transaction as the edges they replace.
    assign_subjects(graph, turns)
    _check_writable(graph.nodes)
    name_of = {normalise_name(node.name): node for node in graph.nodes}

    node_ids = semantic_store.replace_graph(
        database_name=database_name,
        conversation_id=conversation_id,
        nodes=[
            {
                "name": node.name,
                "normalized_name": normalise_name(node.name),
                "type": node.type,
                "kinds": node.kinds,
                "description": node.description,
                "attribute_key": node.key,
                "value": node.value,
                "turn_indexes": sorted(node.turn_indexes),
                "evidence": node.evidence,
                "time_anchor": node.time_anchor,
                "occurred_on": node.occurred_on,
            }
            for node in graph.nodes
        ],
        edges=[
            {
                "source": edge.source,
                "target": edge.target,
                "role": edge.role,
                "statement": _edge_statement(
                    name_of[edge.source].name
                    if edge.source in name_of
                    else edge.source,
                    edge.role,
                    name_of[edge.target].name
                    if edge.target in name_of
                    else edge.target,
                ),
                "turn_indexes": sorted(edge.turn_indexes),
                "evidence": edge.evidence,
            }
            for edge in graph.edges
        ],
    )

    # The ids that landed, paired back with the fields derivation reads.
    stored = [
        {
            "id": node_id,
            "name": node.name,
            "kinds": node.kinds,
            "description": node.description,
            "attribute_key": node.key,
            "value": node.value,
            "turn_indexes": sorted(node.turn_indexes),
            "time_anchor": node.time_anchor,
            "occurred_on": node.occurred_on,
            "vector": node.vector,
        }
        for node in graph.nodes
        if (node_id := node_ids.get(normalise_name(node.name)))
    ]
    statements = [row for row in stored if row["kinds"] != ["object"]]
    _attach_subjects(stored, statements)
    reconcile_keys(statements)

    derived, decided = derive_edges(statements)
    semantic = semantic_store.write_edges(derived)
    if relate and len(statements) > 1:
        _embed_for_relate(statements)
        semantic += semantic_store.write_edges(relate_nodes(statements, skip=decided))

    vectors = _embed(
        database_name=database_name,
        conversation_id=conversation_id,
        stored=stored,
        reset=reset_vectors,
    )
    paths = _embed_paths(
        database_name=database_name,
        conversation_id=conversation_id,
        reset=reset_vectors,
    )

    elapsed = time.monotonic() - started
    things = len(stored) - len(statements)
    logger.info(
        "Conversation %s done: %d node(s) (%d thing, %d statement), "
        "%d knowledge edge(s), %d semantic edge(s), %d vector(s), %d path(s) "
        "in %.1fs.",
        conversation_id,
        len(stored),
        things,
        len(statements),
        len(graph.edges),
        semantic,
        vectors,
        paths,
        elapsed,
    )
    return GraphResult(
        conversation_id=conversation_id,
        nodes=len(stored),
        things=things,
        statements=len(statements),
        knowledge_edges=len(graph.edges),
        derived_edges=len(derived),
        semantic_edges=semantic,
        vectors=vectors,
        paths=paths,
        seconds=elapsed,
    )


def _turn_texts(windows: list) -> dict[int, str]:
    """Turn index to turn text, taken from the windows themselves.

    From the windows rather than from ``fetch_turns`` so that the verifier reads
    exactly the characters the extractor read. A turn rendered one way for
    extraction and another way for checking would let a quote fail to match for
    reasons that have nothing to do with whether the edge is right.
    """
    texts: dict[int, str] = {}
    for window in windows:
        texts.update(zip(window.turn_indexes, window.texts))
    return texts


def _attach_subjects(stored: list[dict], statements: list[dict]) -> None:
    """Put each statement's subject on its row, for adjudication to compare.

    Read back from the store rather than from the extraction drafts, so that
    both callers of :func:`~auto_ontology.conversations.semantic.relate.\
relate_nodes` get the subject by the same route and the shape the note sees at
    query time is the shape the adjudicator saw at build time.

    *stored* is passed whole, not just the statements: a subject is a thing,
    and :func:`~auto_ontology.conversations.semantic.store.fetch_edges` wants
    both ends of an edge in the id set.
    """
    edges = semantic_store.fetch_edges([row["id"] for row in stored])
    subjects = semantic_store.subjects_of(edges)
    for row in statements:
        row["about"] = subjects.get(str(row["id"]), [])


def _forget_paths(*, database_name: str, nodes: list[dict]) -> None:
    """Drop the path vectors of the semantic edges about to be replaced.

    Re-adjudication deletes rows that already have vectors, and the edges it
    is most likely to drop are the wrong ones -- which is the point of running
    it. Left behind, they would keep matching a path whose edge no longer
    exists, so the retune would change the graph and not the answers.

    By id rather than by resetting the collection: several transcripts share a
    ``database_name``, and a reset to retune one of them would cost the other
    nineteen their paths. Enumerated *before* the delete, because a path's id
    is composed from its endpoints and there is no way back to it afterwards.
    """
    from auto_ontology.conversations.semantic.paths import path_rows
    from auto_ontology.vdb import get_one_hop_path_vdb

    ids = [str(row["id"]) for row in nodes]
    stale = [
        row["id"]
        for row in path_rows(nodes, semantic_store.fetch_edges(ids, type_="semantic"))
    ]
    if not stale:
        return
    try:
        vdb = get_one_hop_path_vdb(database_name=database_name)
        for path_id in stale:
            vdb.delete_by_id(path_id)
    except Exception:
        logger.exception(
            "Could not drop %d stale path vector(s) -- retrieval may still "
            "match an edge this pass replaced.",
            len(stale),
        )
        return
    logger.info("Dropped %d path vector(s) of the edges being replaced.", len(stale))


def _check_writable(nodes: list) -> None:
    """Check the drafted nodes against the table's constraints, all at once.

    The write is one statement per node inside one transaction, so Postgres
    reports the *first* row it rejects and rolls the rest back. With an hour of
    extraction behind it, that turns a graph with three unrelated schema
    problems into three hours and three runs. This finds all of them in
    milliseconds and names every offending node.

    It is not a substitute for the constraints, which still run and still have
    the last word. It only moves the common failures earlier, and only the ones
    checkable from the draft alone -- a foreign key to a database row, for
    instance, is not.
    """
    problems: list[str] = []
    valid = set(CONVERSATION_NODE_KINDS)

    seen: dict[str, str] = {}
    for node in nodes:
        normalized = normalise_name(node.name)
        if normalized in seen:
            problems.append(
                f"name {node.name!r} used twice (as {seen[normalized]!r} "
                f"and {node.key!r})"
            )
        seen[normalized] = node.key or node.type

        if not node.kinds:
            problems.append(f"{node.name!r} has no kinds")
        elif unknown := sorted(set(node.kinds) - valid):
            problems.append(f"{node.name!r} has unknown kind(s) {unknown}")

        if not normalized:
            problems.append(f"{node.name!r} normalises to an empty name")

    if problems:
        listed = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(
            f"{len(problems)} drafted node(s) cannot be written:\n  {listed}{more}"
        )


def _disambiguate(nodes: list) -> None:
    """Rename any node whose name another node already has, in place.

    One name per transcript is required twice over: by
    ``uq_conversation_node_name``, and by this module, which pairs stored ids
    back to drafts by name. Two drafts under one name resolve to one id, and
    then a derived edge between two nodes becomes an edge from a node to
    itself -- which is how this was found, as a duplicate edge key three steps
    downstream of the naming problem that caused it.

    Done here rather than in ``store.replace_graph`` for exactly that reason.
    Renaming at the write is too late: the caller still knows the node by the
    name it passed in, so the lookup that follows hands it the other node's id.
    By the time anything is keyed by name, the names have to be settled.

    A statement's identity is its property and value, deliberately, so two
    statements that differ in property and happen to share a name are two
    nodes with one name. The extractor is asked for names of 1-8 words that
    include what distinguishes them; when it overruns that and writes one long
    name twice, this is the backstop.

    Renaming rather than merging: merging two readings is the error that
    identity exists to prevent, and nothing here can tell "one fact said twice"
    from "a value that moved".
    """
    seen: set[str] = set()
    for node in nodes:
        normalized = normalise_name(node.name)
        if normalized not in seen:
            seen.add(normalized)
            continue

        original = node.name
        suffix = node.value or node.key or node.type
        renamed = f"{original} ({suffix})"
        attempt = 2
        while normalise_name(renamed) in seen:
            renamed = f"{original} ({suffix} {attempt})"
            attempt += 1

        logger.warning(
            "Two nodes named %r; renaming the later one to %r.", original, renamed
        )
        node.name = renamed
        # Embedded from the old name during consolidation, so it no longer
        # describes this node. Cleared rather than corrected, which lets the
        # vector write re-embed it with the rest of the missing ones.
        node.vector = None
        seen.add(normalise_name(renamed))


def _embed_names(nodes: list) -> dict[str, list[float]]:
    """Embed each thing's name, for the consolidation pass only.

    Names alone, and only for things: see
    :mod:`~auto_ontology.conversations.semantic.consolidate` for why a
    statement is never a merge candidate and why the description is left out.

    Failure returns nothing, which turns consolidation off for this transcript
    rather than failing the build. A graph with two names for one thing is the
    behaviour every run before this one had.
    """
    names = [
        normalise_name(node.name) for node in nodes if list(node.kinds) == ["object"]
    ]
    if len(names) < 2:
        return {}

    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import embed_texts

    try:
        vectors = embed_texts(names, get_embed_params(), "passage")
    except Exception:
        logger.exception(
            "Could not embed %d name(s) -- skipping consolidation, so a thing "
            "named two ways stays two nodes.",
            len(names),
        )
        return {}
    return {name: vector for name, vector in zip(names, vectors) if vector}


def _embed_for_relate(stored: list[dict]) -> None:
    """Fill in the vectors the candidate proposal needs.

    Extraction no longer embeds as it goes -- identity is the property key and
    value now, not vector proximity -- so the vectors arrive null and the one
    caller that needs them asks for them here. Failure leaves them null, which
    costs :func:`candidate_pairs` its similarity gate and still proposes every
    ``event``/``event`` pair.
    """
    missing = [row for row in stored if not row.get("vector")]
    if not missing:
        return

    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import embed_texts

    try:
        vectors = embed_texts(
            [node_text(row["name"], row["description"]) for row in missing],
            get_embed_params(),
            "passage",
        )
    except Exception:
        logger.exception(
            "Could not embed %d node(s) for candidate proposal -- adjudication "
            "falls back to event pairs only.",
            len(missing),
        )
        return
    for row, vector in zip(missing, vectors):
        row["vector"] = vector


def relate_conversation(*, database_name: str, conversation_id: str) -> int:
    """Re-derive and re-adjudicate semantic edges for an extracted transcript.

    The second half of :func:`populate_conversation_graph`, runnable on its own.
    Extraction costs one model call per window and an hour on a long transcript;
    the role vocabulary and the adjudicator's prompt are the parts that get
    tuned, and re-reading the transcript to retune them is the wrong price to
    pay.

    Only the semantic layer is replaced. The knowledge edges came out of the
    same read as the nodes and are as stable as they are.
    """
    from sqlalchemy import and_, delete

    from auto_ontology.dal import schema as sch
    from auto_ontology.dal.session import store, write_transaction

    nodes = [
        dict(row)
        for row in semantic_store.fetch_nodes(
            database_name, conversation_id=conversation_id
        )
    ]
    statements = [row for row in nodes if list(row.get("kinds") or []) != ["object"]]
    if len(statements) < 2:
        logger.info(
            "Conversation %s has %d statement(s) -- nothing to relate.",
            conversation_id,
            len(statements),
        )
        return 0

    _attach_subjects(nodes, statements)
    reconcile_keys(statements)
    ids = [str(row["id"]) for row in statements]
    _forget_paths(database_name=database_name, nodes=nodes)
    with write_transaction():
        store().query_write(
            delete(sch.conversation_edge).where(
                and_(
                    sch.conversation_edge.c.type == "semantic",
                    sch.conversation_edge.c.source_id.in_(ids),
                    sch.conversation_edge.c.target_id.in_(ids),
                )
            )
        )

    derived, decided = derive_edges(statements)
    written = semantic_store.write_edges(derived)
    _embed_for_relate(statements)
    written += semantic_store.write_edges(relate_nodes(statements, skip=decided))
    _embed_paths(
        database_name=database_name, conversation_id=conversation_id, reset=False
    )
    return written


def _embed(
    *,
    database_name: str,
    conversation_id: str,
    stored: list[dict],
    reset: bool,
) -> int:
    """Write the nodes' vectors to ``vdb.conversation_semantic``.

    Things and statements share one collection. A question names a *thing* as
    often as it names something said about one -- "what did I pick for caching"
    matches the thing, "the cache is localStorage" matches the statement -- and
    a search that could only reach one of them would miss whichever the question
    happened not to phrase.

    Failure is logged rather than raised. The graph, its turn links and its
    edges are the expensive part and are already committed; a missing vector
    costs this transcript its concept route until a backfill, which is
    recoverable from the stored rows.
    """
    if not stored:
        return 0

    from auto_ontology.catalog.constants import Labels
    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import embed_texts
    from auto_ontology.utils.embedding_rows import build_embed_row
    from auto_ontology.vdb import get_conversation_semantic_vdb
    from auto_ontology.vdb.records import to_vdb_records

    texts = [node_text(row["name"], row["description"]) for row in stored]
    try:
        missing = [i for i, row in enumerate(stored) if not row.get("vector")]
        if missing:
            fresh = embed_texts(
                [texts[i] for i in missing], get_embed_params(), "passage"
            )
            for position, vector in zip(missing, fresh):
                stored[position]["vector"] = vector

        rows = []
        for text, row in zip(texts, stored):
            if not row.get("vector"):
                continue
            embed_row = build_embed_row(
                text=text,
                node_id=str(row["id"]),
                label=Labels.CONVERSATION,
                name=row["name"],
                schema_name="",
                database_name=database_name,
                extra={
                    "conversation_id": conversation_id,
                    # Comma-joined rather than a list: the vector store's
                    # metadata is flat, and retrieval only ever asks whether a
                    # kind is present.
                    "kinds": ",".join(row["kinds"]),
                    # The turns this node points at, so a retrieval that has
                    # the vector hit already knows what to read without going
                    # back to Postgres for the row.
                    "turn_indexes": row["turn_indexes"],
                },
            )
            embed_row["metadata"]["embedding"] = row["vector"]
            rows.append(embed_row)

        records = to_vdb_records(rows)
        if not records:
            logger.warning(
                "Conversation %s: no node vector survived embedding.",
                conversation_id,
            )
            return 0
        vdb = get_conversation_semantic_vdb(database_name=database_name, reset=reset)
        return int(vdb.run(records))
    except Exception:
        logger.exception(
            "Could not index %d node(s) of conversation %s for search -- the "
            "graph itself is unaffected.",
            len(stored),
            conversation_id,
        )
        return 0


def _embed_paths(*, database_name: str, conversation_id: str, reset: bool) -> int:
    """Embed every edge as a path into ``vdb.1hop_pathes``.

    Read back from the store rather than assembled from the drafts, because by
    now the edges come from three places -- extraction, derivation and
    adjudication -- and the rows are the one view that has all of them with the
    ids they landed under.

    Failure is logged rather than raised, for the same reason the node embed's
    is: the graph is committed, and a missing path vector costs this transcript
    its path route until a backfill.
    """
    from auto_ontology.catalog.constants import Labels
    from auto_ontology.conversations.semantic.paths import path_rows
    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import embed_texts
    from auto_ontology.utils.embedding_rows import build_embed_row
    from auto_ontology.vdb import get_one_hop_path_vdb
    from auto_ontology.vdb.records import to_vdb_records

    try:
        nodes = semantic_store.fetch_nodes(
            database_name, conversation_id=conversation_id
        )
        edges = semantic_store.fetch_edges([str(row["id"]) for row in nodes])
        rows = path_rows(nodes, edges)
        if not rows:
            logger.info("Conversation %s has no paths to embed.", conversation_id)
            return 0

        vectors = embed_texts(
            [row["text"] for row in rows], get_embed_params(), "passage"
        )
        records = []
        for row, vector in zip(rows, vectors):
            if not vector:
                continue
            embed_row = build_embed_row(
                text=row["text"],
                node_id=row["id"],
                label=Labels.CONVERSATION,
                name=f"{row['type']}:{row['role']}",
                schema_name="",
                database_name=database_name,
                extra={
                    "conversation_id": conversation_id,
                    "type": row["type"],
                    "role": row["role"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "turn_indexes": row["turn_indexes"],
                },
            )
            embed_row["metadata"]["embedding"] = vector
            records.append(embed_row)

        prepared = to_vdb_records(records)
        if not prepared:
            logger.warning(
                "Conversation %s: no path vector survived embedding.",
                conversation_id,
            )
            return 0
        vdb = get_one_hop_path_vdb(database_name=database_name, reset=reset)
        return int(vdb.run(prepared))
    except Exception:
        logger.exception(
            "Could not index the paths of conversation %s -- the graph itself "
            "is unaffected.",
            conversation_id,
        )
        return 0
