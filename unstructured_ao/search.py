# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finding passages to answer from, through the ontology.

Two routes, in opposite directions:

- :func:`attribute_search` — ``query → attribute vectors →
  unstructured_attribute__chunk → chunks``. A question matches a *concept* and
  inherits that concept's evidence, rather than matching passage text. Each
  attribute is already grounded in the chunks that support it: the ones the
  extractor cited plus the ones linking found.
- :func:`chunk_ontology` — the reverse, for a caller that has already found its
  passages some other way: ``chunks → unstructured_attribute__chunk →
  attributes → families, relations``. It annotates rather than retrieves.

The first is capped by construction, which is worth knowing before choosing it:
only chunks linked to the attributes a question happens to match can ever be
delivered, measured at 72.5% of what plain chunk search delivers, and ordering
the candidates by question similarity already extracts all of it. The rest is
unreachable rather than mis-ranked.

This module holds the traversal, the ordering and the labelling. What it
deliberately does not hold is anything about scoring a retrieval against a gold
set — page keys, rank cut-offs, qrels — which belongs to whoever is measuring.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, select

from auto_ontology.dal.schema import (
    unstructured_attribute,
    unstructured_attribute__chunk,
    unstructured_chunk,
    unstructured_document,
)
from unstructured_ao import ontology

#: An LLM-cited chunk is evidence the model demonstrably read, so it outranks a
#: chunk that merely embedded close to the attribute.
SOURCE_RANK = {"llm": 0, "vector": 1}


def joinable(text: str) -> str:
    """A chunk's text reduced to what every store agrees on.

    Postgres keeps the chunk text as it was written, and a store need not.
    An external store may hand the same text back with its whitespace
    normalised. Comparing raw strings would then report disagreement between
    stores that otherwise contain the same content.
    """
    return " ".join((text or "").split())


@dataclass(frozen=True)
class Passage:
    """A chunk, and how the ontology reached it."""

    text: str
    attribute_id: str
    #: The attribute's name, with whatever distinguishes this reading from its
    #: rivals appended — what to put above the passage in a prompt.
    label: str
    source: str
    score: float | None
    document: str | None = None
    page: int | None = None
    document_id: str | None = None
    chunk_id: str | None = None


@dataclass
class AttributeSearch:
    """What :func:`attribute_search` found.

    *passages* is what to answer from, in the order to present it. *ranked* is
    the same evidence ordered by strength instead, for a caller ranking pages
    or documents — a different question from what to read first, and the two
    orders disagree on purpose. *note* is the ontology's warning about the set.
    """

    passages: list[Passage] = field(default_factory=list)
    ranked: list[Passage] = field(default_factory=list)
    note: str = ""

    @property
    def chunks(self) -> list[str]:
        return [p.text for p in self.passages]

    @property
    def labels(self) -> list[str]:
        return [p.label for p in self.passages]

    @property
    def ranked_pages(self) -> list[tuple[str | None, int]]:
        """``(document, page)``, best first, each page once.

        A page ranks by its strongest piece of evidence, the way a chunk search
        ranks a page by its closest hit — so a page reached through several
        attributes keeps its best position rather than its last.
        """
        return list(
            dict.fromkeys(
                (p.document, p.page) for p in self.ranked if p.page is not None
            )
        )


@dataclass
class ChunkOntology:
    """What the ontology knows about passages someone else retrieved.

    *labels* is per passage, in the order they were given. *links* is every
    attribute/chunk link behind those labels, with the position of the passage
    it belongs to, for a caller that wants to record what was available rather
    than only what was rendered.
    """

    labels: list[list[str]] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""


def _matched_attribute_ids(query: str, k: int) -> list[str]:
    """Attribute ids matching *query*, best first, across every document."""
    from unstructured_ao.vectorstore import get_attribute_store

    hits = get_attribute_store().similarity_search_with_score(query=query, k=k)
    ids = []
    for doc, _score in hits:
        attribute_id = getattr(doc, "id", None) or doc.metadata.get("id")
        if attribute_id:
            ids.append(str(attribute_id))
    return ids


def _linked_chunk_rows(
    conn: Connection, attribute_ids: list[str]
) -> list[dict[str, Any]]:
    """Every chunk linked to *attribute_ids*, with its page and document."""
    link = unstructured_attribute__chunk
    statement = (
        select(
            link.c.attribute_id,
            link.c.source,
            link.c.score,
            unstructured_attribute.c.name.label("attribute_name"),
            unstructured_chunk.c.id.label("chunk_id"),
            unstructured_chunk.c.document_id,
            unstructured_chunk.c.content,
            unstructured_chunk.c.page_number,
            unstructured_document.c.name.label("document_name"),
        )
        .select_from(
            link.join(
                unstructured_attribute,
                unstructured_attribute.c.id == link.c.attribute_id,
            )
            .join(unstructured_chunk, unstructured_chunk.c.id == link.c.chunk_id)
            .join(
                unstructured_document,
                unstructured_document.c.id == unstructured_chunk.c.document_id,
            )
        )
        .where(link.c.attribute_id.in_(attribute_ids))
    )
    return [dict(row) for row in conn.execute(statement).mappings()]


def _with_family_members(matched: list[str], family_index: dict) -> list[str]:
    """*matched*, each followed immediately by its family's other members.

    Deferring the members until after every match looks fairer and measured
    worse — ndcg@10 0.4433 to 0.3828, recall@10 0.5041 to 0.4163, with @1
    unmoved, so the loss was entirely in positions 2 onward. A top attribute's
    family members are the same measure for another place, period or source:
    they belong at rank 2, not rank 11.
    """
    ordered: list[str] = []
    for attribute_id in matched:
        if attribute_id not in ordered:
            ordered.append(attribute_id)
        family_id = family_index["by_attribute"].get(attribute_id)
        if not family_id:
            continue
        for member in family_index["families"][family_id]["members"]:
            if member["attribute_id"] not in ordered:
                ordered.append(member["attribute_id"])
    return ordered


def _interleave(
    rows: list[dict[str, Any]], attribute_ids: list[str]
) -> list[dict[str, Any]]:
    """One chunk per attribute in turn, best attribute first.

    Not attribute-major. Attributes carry a median of five chunk links and a
    generator sees ten of them, so draining them in rank order spent the entire
    budget on the single best attribute: across the questions this route
    refused while plain chunk search answered, all five chunks came from one
    attribute every time, and the chunk actually holding the answer was usually
    reachable through a lower-ranked attribute and cut. One chunk per attribute
    in turn spends the same budget across that many concepts.

    Within an attribute, cited chunks come before linked ones, then closer
    before further.
    """
    per_attribute: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_attribute.setdefault(str(row["attribute_id"]), []).append(row)
    for group in per_attribute.values():
        group.sort(
            key=lambda r: (
                SOURCE_RANK.get(r["source"], 2),
                float(r["score"]) if r["score"] is not None else 0.0,
            )
        )

    queues = [per_attribute.get(attribute_id, []) for attribute_id in attribute_ids]
    ordered: list[dict[str, Any]] = []
    for depth in range(max((len(queue) for queue in queues), default=0)):
        for queue in queues:
            if depth < len(queue):
                ordered.append(queue[depth])
    return ordered


def _label(row: dict[str, Any], family_index: dict) -> str:
    """The attribute's name, and which reading it is when it has rivals."""
    attribute_id = str(row["attribute_id"])
    family_id = family_index["by_attribute"].get(attribute_id)
    members = family_index["families"].get(family_id, {}).get("members", [])
    member = next((m for m in members if m["attribute_id"] == attribute_id), None)
    label = row["attribute_name"]
    qualifier = (member or {}).get("qualifier")
    # Extraction often names the country in the attribute itself, and
    # "Austria ... — Austria" reads as two different facts.
    if qualifier and qualifier.lower() not in label.lower():
        return f"{label} — {qualifier}"
    return label


def attribute_search(
    query: str,
    *,
    attribute_k: int = 10,
    context_k: int = 10,
    chunk_cap: int | None = None,
    order: dict[str, int] | None = None,
) -> AttributeSearch:
    """Concepts matching *query*, resolved to the passages that support them.

    *attribute_k* concepts are matched, their families joined on behind them,
    and their chunks interleaved; the first *context_k* distinct ones are
    returned to answer from. *chunk_cap* bounds how deep the evidence behind
    them is collected for ``ranked``, and defaults to *context_k*.

    *order* re-sorts the candidates before the cap, as ``{joinable text:
    rank}`` — so it changes which passages are delivered, not only their order.
    Interleaving is indicator-major: a passage carrying the answer lands fourth
    whenever its indicator ranked fourth, however good the passage is. Passing
    a question-to-passage ranking here tests whether indicators *select* worse
    evidence or merely *present* it worse, which is a diagnostic rather than a
    design — a route that needs another retriever's ranking to order its own
    output has conceded the ranking.
    """
    from auto_ontology.dal.session import get_engine

    chunk_cap = context_k if chunk_cap is None else chunk_cap
    matched = _matched_attribute_ids(query, attribute_k)
    if not matched:
        return AttributeSearch()

    engine = get_engine()
    with engine.connect() as conn:
        family_index = ontology.index_families(ontology.family_rows(conn, matched))
        attribute_ids = _with_family_members(matched, family_index)
        rows = _linked_chunk_rows(conn, attribute_ids)

    ordered = _interleave(rows, attribute_ids)
    if order is not None:
        unranked = len(order)
        ordered.sort(key=lambda r: order.get(joinable(r["content"]), unranked))

    accepted: list[Passage] = []
    seen: set[str] = set()
    used_families: list[str] = []
    for row in ordered:
        content = row["content"] or ""
        if not content or content in seen:
            continue
        if len(seen) >= chunk_cap:
            break
        seen.add(content)
        attribute_id = str(row["attribute_id"])
        accepted.append(
            Passage(
                text=content,
                attribute_id=attribute_id,
                label=_label(row, family_index),
                source=row["source"],
                score=float(row["score"]) if row["score"] is not None else None,
                document=row["document_name"],
                page=row["page_number"],
                document_id=str(row["document_id"]),
                chunk_id=str(row["chunk_id"]),
            )
        )
        # Only families a delivered chunk actually came through are worth
        # warning about; the rest would be a note about absent evidence.
        family_id = family_index["by_attribute"].get(attribute_id)
        if family_id and family_id not in used_families:
            used_families.append(family_id)

    passages = accepted[:context_k]

    # Ranked by evidence strength, not by the order the interleave produced.
    # Interleaving exists so the passages come from many concepts rather than
    # one; carrying that into a ranking puts a weak chunk from the tenth concept
    # above a strong one from the second.
    ranked = sorted(
        (p for p in accepted if p.page is not None),
        key=lambda p: (SOURCE_RANK.get(p.source, 2), p.score or 0.0),
    )

    # Relations between the attributes actually delivered, families across
    # everything collected: a family warns about a reading that could be
    # mistaken for one in front of the model, which the deeper evidence can
    # still be, while a relation is a statement about two sources both present.
    delivered = list(dict.fromkeys(p.attribute_id for p in passages))
    with engine.connect() as conn:
        relations = ontology.relation_rows(conn, delivered)
    note = ontology.note_text(family_index["families"], used_families, relations)
    return AttributeSearch(passages=passages, ranked=ranked, note=note)


def chunk_search(query: str, *, k: int = 10) -> list[Passage]:
    """The passages nearest *query*, best first.

    Plain vector search over the chunks, with no ontology in the loop. It is
    here because it is the strongest retrieval measured — the route through
    concepts is capped below it, and annotating these results is what
    :func:`chunk_ontology` is for — and because the product cannot borrow the
    benchmark's copy of the same index.

    The passages carry no label: nothing has said what they are readings of
    yet. :func:`ask` fills that in.
    """
    from auto_ontology.dal.session import get_engine
    from unstructured_ao.vectorstore import get_chunk_store

    hits = get_chunk_store().similarity_search_with_score(query=query, k=k)
    scores: dict[str, float] = {}
    for doc, score in hits:
        chunk_id = getattr(doc, "id", None) or doc.metadata.get("id")
        if chunk_id and str(chunk_id) not in scores:
            scores[str(chunk_id)] = float(score)
    if not scores:
        return []

    # One round trip for the page and document: the vector table stores the
    # embedded text and the document id, not the page a caller has to cite.
    statement = (
        select(
            unstructured_chunk.c.id,
            unstructured_chunk.c.document_id,
            unstructured_chunk.c.content,
            unstructured_chunk.c.page_number,
            unstructured_document.c.name.label("document_name"),
        )
        .select_from(
            unstructured_chunk.join(
                unstructured_document,
                unstructured_document.c.id == unstructured_chunk.c.document_id,
            )
        )
        .where(unstructured_chunk.c.id.in_(list(scores)))
    )
    with get_engine().connect() as conn:
        rows = {str(r["id"]): dict(r) for r in conn.execute(statement).mappings()}

    # Back into the order the store returned, which the id round trip lost.
    return [
        Passage(
            text=rows[chunk_id]["content"],
            attribute_id="",
            label="",
            source="vector",
            score=score,
            document=rows[chunk_id]["document_name"],
            page=rows[chunk_id]["page_number"],
            document_id=str(rows[chunk_id]["document_id"]),
            chunk_id=chunk_id,
        )
        for chunk_id, score in scores.items()
        if chunk_id in rows
    ]


_chunk_ids: dict[str, str] | None = None
_chunk_ids_lock = threading.Lock()


def chunk_id_by_text(conn: Connection | None = None) -> dict[str, str]:
    """``{joinable chunk text: chunk id}`` for the whole corpus, read once.

    Text is the only key shared with a store that was populated by an outside
    pipeline, and it has to be normalised before it joins. Held in memory
    rather than queried per request: the corpus is a couple of thousand rows,
    and no index would serve a match on normalised text anyway.
    """
    global _chunk_ids
    if _chunk_ids is not None:
        return _chunk_ids

    # Outside the lock: borrowing a connection re-enters this function, and a
    # plain Lock cannot be taken twice by the same thread.
    if conn is None:
        from auto_ontology.dal.session import get_engine

        with get_engine().connect() as owned:
            return chunk_id_by_text(owned)

    with _chunk_ids_lock:
        # Two threads that got this far both read the corpus and build the same
        # map; the lock is here so neither sees the other's half-built one.
        if _chunk_ids is None:
            rows = conn.execute(
                select(unstructured_chunk.c.id, unstructured_chunk.c.content)
            ).mappings()
            _chunk_ids = {joinable(r["content"]): str(r["id"]) for r in rows}
    return _chunk_ids


def _chunk_attribute_rows(
    conn: Connection, chunk_ids: list[str]
) -> list[dict[str, Any]]:
    """Every attribute reached from *chunk_ids*, with how it was reached."""
    link = unstructured_attribute__chunk
    statement = (
        select(
            link.c.chunk_id,
            link.c.attribute_id,
            link.c.source,
            link.c.score,
            unstructured_attribute.c.name.label("attribute_name"),
        )
        .select_from(
            link.join(
                unstructured_attribute,
                unstructured_attribute.c.id == link.c.attribute_id,
            )
        )
        .where(link.c.chunk_id.in_(chunk_ids))
    )
    return [dict(row) for row in conn.execute(statement).mappings()]


def chunk_ontology(chunks: list[str]) -> ChunkOntology:
    """What the ontology knows about passages retrieved some other way.

    The inverse of :func:`attribute_search`: instead of reaching passages
    through concepts, this reaches the concepts a passage is a reading of, and
    from them the families and relations those concepts carry. A passage this
    ontology has never seen simply contributes nothing.
    """
    if not chunks:
        return ChunkOntology()

    from auto_ontology.dal.session import get_engine

    by_text = chunk_id_by_text()
    # Position of each passage, so a link can say which one it belongs to.
    position: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        chunk_id = by_text.get(joinable(chunk))
        if chunk_id is not None:
            position.setdefault(chunk_id, index)
    if not position:
        return ChunkOntology(labels=[[] for _ in chunks])

    chunk_ids = list(position)
    engine = get_engine()
    with engine.connect() as conn:
        rows = _chunk_attribute_rows(conn, chunk_ids)

    links = [
        {
            "attribute_id": str(row["attribute_id"]),
            "name": row["attribute_name"],
            "source": row["source"],
            "score": float(row["score"]) if row["score"] is not None else None,
            "chunk": position[str(row["chunk_id"])],
        }
        for row in rows
    ]

    labels: list[list[str]] = [[] for _ in chunks]
    for link in links:
        names = labels[link["chunk"]]
        if link["name"] not in names:
            names.append(link["name"])

    # In delivery order, so the note reads in the order the passages do.
    ordered_ids = list(
        dict.fromkeys(
            link["attribute_id"] for link in sorted(links, key=lambda x: x["chunk"])
        )
    )
    with engine.connect() as conn:
        note = ontology.ontology_note(ordered_ids, conn)
    return ChunkOntology(labels=labels, links=links, note=note)
