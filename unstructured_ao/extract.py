# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM extraction of terms and attributes from unstructured document text.

Design mirrors illumex's ``embed_service.extract_attributes_from_text()`` and
``TermsExtractor`` but simplified for one term per document (the document's
core concept / topic) and per-term attribute extraction.

The document is read window by window rather than as a single truncated
prefix, so an attribute introduced on page 40 is still found.  Each window
carries the list of attributes found so far and is asked to reuse an existing
name when it describes one of them, which deduplicates while the model still
has the context to judge; :func:`_merge_near_duplicates` cleans up the
paraphrases that slip through.

Attribute types
---------------
* ``sql``  — reusable numeric calculation (e.g. "Employment rate", "Headcount
  of posted workers").  These parallel illumex's ``SqlAttributeItem``.
* ``ref``  — enumerated literal facts / classifications (e.g. list of EU
  member states, occupational categories).  These parallel illumex's
  ``RefAttributeItem``.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sqlalchemy import Connection

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import unstructured_ao.config as config
from auto_ontology.utils.llm_invoke import safe_invoke_structured_nr
from unstructured_ao.windows import Window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------


class TermExtraction(BaseModel):
    """The single business term that represents this document."""

    name: str = Field(
        description=(
            "1–4 word noun phrase naming the core concept of the document. "
            "E.g. 'EU Labour Mobility', 'Posted Workers Directive', 'Employment Rate'."
        )
    )
    description: str = Field(
        description="One sentence describing what this document is about."
    )


class AttributeItem(BaseModel):
    """One attribute extracted from a window of the document."""

    name: str = Field(description="Short, human-readable attribute name (≤10 words).")
    description: str = Field(
        description="One sentence describing what this attribute captures."
    )
    attribute_type: Literal["sql", "ref"] = Field(
        description=(
            "'sql' for a reusable numeric calculation or metric; "
            "'ref' for a list of named categories, entities, or literal facts."
        )
    )
    source_chunks: list[int] = Field(
        default_factory=list,
        description=(
            "Labels of the numbered blocks in this window that state or support "
            "the attribute, e.g. [2, 5]. Use only labels shown above."
        ),
    )
    evidence: str = Field(
        default="",
        description=(
            "A short quote (≤200 characters) copied EXACTLY, character for "
            "character, from one of the labelled blocks."
        ),
    )


class AttributeExtractionResult(BaseModel):
    """All attributes extracted from one window."""

    attributes: list[AttributeItem] = Field(
        description="Attributes supported by this window, typically 0–6."
    )


@dataclass
class _Draft:
    """An attribute accumulated across the windows that mention it."""

    name: str
    description: str
    attribute_type: str
    chunk_ids: set[str] = field(default_factory=set)
    #: Subset of ``chunk_ids`` whose link was confirmed by a verbatim quote.
    verified_chunk_ids: set[str] = field(default_factory=set)
    #: Embedding of ``"<name>: <description>"``, set by the merge pass.  Kept
    #: so the attribute vector store can reuse it instead of paying to embed
    #: the same text a second time.
    vector: list[float] | None = None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_TERM_SYSTEM = """You are a data ontology expert.

Given the text of a document, extract the single business term that best
represents the document's core concept.

Rules:
- Return exactly ONE term.
- The name must be a 1–4 word noun phrase (no verbs, no articles).
- Do not use generic words like "Report", "Document", "Analysis", "Study".
- The description must be one complete sentence that would help someone
  understand what domain this document covers.
"""

_ATTRIBUTE_SYSTEM = """You are a data ontology expert specialising in HR and
labour-market policy documents.

You are shown ONE WINDOW of a longer document, split into numbered blocks, and
the attributes already extracted from earlier windows.  Extract the structured
attributes this window supports — measurable facts or reusable reference lists
that could enrich a data catalogue entry for the document's business term.

Attribute types:
  sql  — a numeric metric or calculation that a data analyst would compute
         (e.g. "Employment rate", "Number of posted workers", "Wage growth %").
  ref  — a named list of categories, entities, or literal facts enumerated in
         the document (e.g. "EU member states covered", "Occupational groups",
         "Legal instruments cited").

Rules:
- If this window describes an attribute that is ALREADY IN THE LIST, return it
  again using its EXACT existing name, with the block labels and quote from
  THIS window. Do not rename it and do not invent a variant of it.
- Only add a new attribute for a genuinely new concept.
- Return an empty list when the window supports nothing — most windows of a
  long document add little, and padding the list corrupts the catalogue.
- Every name must be short (≤10 words) and specific to this document.
- Every description must be one complete sentence.
- source_chunks must contain only labels shown in this window, and only blocks
  that actually state the attribute — not blocks that are merely nearby.
- evidence must be copied verbatim from one of those blocks. Do not paraphrase,
  summarise, or reformat it; it is checked against the block text.
- Do not invent attributes not supported by the text.
"""


def _format_known(drafts: list[_Draft]) -> str:
    if not drafts:
        return "(none yet — this is the first window)"
    return "\n".join(
        f"- {d.name} [{d.attribute_type}]: {d.description}" for d in drafts
    )


# ---------------------------------------------------------------------------
# Attribute identity
# ---------------------------------------------------------------------------


def _normalise_name(name: str) -> str:
    """Key under which two spellings of one attribute name collapse."""
    stripped = re.sub(r"[^a-z0-9]+", " ", name.casefold())
    return re.sub(r"\s+", " ", stripped).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _merge_near_duplicates(
    drafts: list[_Draft],
    *,
    threshold: float = config.ATTRIBUTE_MERGE_THRESHOLD,
) -> list[_Draft]:
    """Fold same-typed attributes whose text embeds nearly identically.

    The per-window reuse instruction catches most repeats, but a model that
    writes "Headcount of posted workers" in one window and "Number of posted
    workers" in another produces two rows for one concept; their links belong
    on a single attribute.
    """
    if len(drafts) < 2:
        return drafts

    from unstructured_ao.vectorstore import build_embeddings

    try:
        vectors = build_embeddings().embed_documents(
            [f"{d.name}: {d.description}" for d in drafts]
        )
    except Exception:
        logger.exception(
            "Embedding-based attribute merge failed — keeping %d name-matched "
            "attribute(s) as extracted.",
            len(drafts),
        )
        return drafts

    kept: list[tuple[_Draft, list[float]]] = []
    for draft, vector in zip(drafts, vectors):
        draft.vector = vector
        for other, other_vector in kept:
            if other.attribute_type != draft.attribute_type:
                continue
            if _cosine(vector, other_vector) < threshold:
                continue
            other.chunk_ids |= draft.chunk_ids
            other.verified_chunk_ids |= draft.verified_chunk_ids
            logger.debug("Merged attribute %r into %r.", draft.name, other.name)
            break
        else:
            kept.append((draft, vector))

    if len(kept) < len(drafts):
        logger.info("Merged %d near-duplicate attribute(s).", len(drafts) - len(kept))
    return [d for d, _ in kept]


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_term(text: str) -> TermExtraction | None:
    """Extract the single core business term from *text*.

    Tries the non-reasoning LLM (fast/cheap) and falls back to the reasoning
    LLM if it exhausts its retries.  Returns ``None`` only when both fail, so
    the caller can decide whether to skip or raise.
    """
    messages = [
        SystemMessage(content=_TERM_SYSTEM),
        HumanMessage(content=f"DOCUMENT TEXT:\n\n{text}"),
    ]
    result = safe_invoke_structured_nr(messages, TermExtraction)
    if result is None:
        logger.warning("Term extraction returned None — document will have no term.")
    return result


def extract_window_attributes(
    window: Window,
    term_name: str,
    known: list[_Draft],
) -> list[AttributeItem]:
    """Extract the attributes *window* supports, reusing names from *known*.

    Returns an empty list when both the non-reasoning and the reasoning LLM
    fail, which costs the window's attributes but not the document's.
    """
    messages = [
        SystemMessage(content=_ATTRIBUTE_SYSTEM),
        HumanMessage(
            content=(
                f"BUSINESS TERM: {term_name}\n\n"
                f"ATTRIBUTES ALREADY FOUND:\n{_format_known(known)}\n\n"
                f"DOCUMENT WINDOW:\n\n{window.labelled_text}"
            )
        ),
    ]
    result = safe_invoke_structured_nr(messages, AttributeExtractionResult)
    if result is None:
        logger.warning("Attribute extraction returned None for term %r.", term_name)
        return []
    return result.attributes


def _attach_links(draft: _Draft, window: Window, item: AttributeItem) -> None:
    """Record the chunks of *window* that support *item* on *draft*.

    A verbatim quote identifies its chunk on its own; the labels the model
    returned are taken only when they fall inside the window.  When neither
    resolves the attribute simply gets no llm link here — the vector step
    searches the whole document afterwards, so linking every chunk in the
    window "just in case" would add false links for no recall.
    """
    quoted = window.find_quote(item.evidence)
    if quoted:
        draft.chunk_ids.add(quoted)
        draft.verified_chunk_ids.add(quoted)

    resolved = 0
    for label in item.source_chunks:
        chunk_id = window.resolve(label)
        if chunk_id is None:
            logger.debug(
                "Attribute %r cited label [%s] outside this window — ignored.",
                item.name,
                label,
            )
            continue
        draft.chunk_ids.add(chunk_id)
        resolved += 1

    if not quoted and not resolved:
        logger.debug("Attribute %r cited nothing resolvable in this window.", item.name)


def attribute_text(name: str, description: str | None) -> str:
    """The text an attribute is embedded as.

    One definition, used by the merge pass and by the vector store, so the
    vectors a search compares are the vectors the merge compared.
    """
    return f"{name}: {description or ''}"


def index_attributes(
    document_id: str,
    term_id: str,
    stored: list[tuple[str, "_Draft"]],
) -> None:
    """Write attribute embeddings to the attribute vector store.

    Reuses the vectors the merge pass already computed; only attributes that
    skipped it — a document with a single attribute, or one whose merge
    embedding call failed — are embedded here.

    Failure is logged rather than raised: the attributes and their grounded
    links are the expensive part and are already in the transaction, and a
    missing vector can be backfilled from the stored rows later.
    """
    if not stored:
        return

    from unstructured_ao.vectorstore import build_embeddings, get_attribute_store

    texts = [attribute_text(d.name, d.description) for _, d in stored]
    try:
        missing = [i for i, (_, d) in enumerate(stored) if d.vector is None]
        if missing:
            fresh = build_embeddings().embed_documents([texts[i] for i in missing])
            for i, vector in zip(missing, fresh):
                stored[i][1].vector = vector

        vectors = [d.vector for _, d in stored]
        get_attribute_store(vector_size=len(vectors[0])).add_embeddings(
            texts=texts,
            embeddings=vectors,
            metadatas=[
                {
                    "document_id": document_id,
                    "term_id": term_id,
                    "attribute_type": d.attribute_type,
                }
                for _, d in stored
            ],
            ids=[attribute_id for attribute_id, _ in stored],
        )
    except Exception:
        logger.exception(
            "Could not index %d attribute(s) of document %s for search — the "
            "attributes themselves are unaffected.",
            len(stored),
            document_id,
        )
        return

    logger.info(
        "Indexed %d attribute(s) of document %s for search.", len(stored), document_id
    )


def run_extraction(
    document_id: str,
    windows: list[Window],
    conn: "Connection",  # type: ignore[name-defined]
) -> None:
    """Extract and persist the term, attributes and grounded links.

    Writes to ``unstructured_term``, ``unstructured_attribute`` and the
    ``source='llm'`` rows of ``unstructured_attribute__chunk``.  The retrieve
    step (``retrieve.link_document_attributes``) runs afterwards and adds the
    ``source='vector'`` links for supporting chunks no window cited.
    """
    from unstructured_ao.dal import (
        insert_attribute,
        insert_attribute_chunk_link,
        insert_term,
    )

    if not windows:
        logger.warning("Document %s has no text to extract from.", document_id)
        return

    # The title and abstract live at the front, so the first window identifies
    # the document; re-asking per window would just invite disagreement.
    term_extraction = extract_term(windows[0].text)
    if term_extraction is None:
        logger.error(
            "Skipping attribute extraction for document %s — term extraction failed.",
            document_id,
        )
        return

    term_id = insert_term(
        conn,
        document_id=document_id,
        name=term_extraction.name,
        description=term_extraction.description,
    )
    logger.info(
        "Term %r (id=%s) extracted for document %s.",
        term_extraction.name,
        term_id,
        document_id,
    )

    drafts: dict[str, _Draft] = {}
    for i, window in enumerate(windows, 1):
        items = extract_window_attributes(
            window, term_extraction.name, list(drafts.values())
        )
        for item in items:
            key = _normalise_name(item.name)
            draft = drafts.get(key)
            if draft is None:
                draft = _Draft(
                    name=item.name,
                    description=item.description,
                    attribute_type=item.attribute_type,
                )
                drafts[key] = draft
            _attach_links(draft, window, item)
        logger.info(
            "Window %d/%d: %d attribute mention(s), %d distinct attribute(s) so far.",
            i,
            len(windows),
            len(items),
            len(drafts),
        )

    merged = _merge_near_duplicates(list(drafts.values()))

    linked = 0
    verified = 0
    stored: list[tuple[str, _Draft]] = []
    for draft in merged:
        attribute_id = insert_attribute(
            conn,
            term_id=term_id,
            name=draft.name,
            description=draft.description,
            attribute_type=draft.attribute_type,
        )
        for chunk_id in sorted(draft.chunk_ids):
            insert_attribute_chunk_link(
                conn,
                attribute_id=attribute_id,
                chunk_id=chunk_id,
                source="llm",
            )
        linked += len(draft.chunk_ids)
        verified += len(draft.verified_chunk_ids)
        stored.append((attribute_id, draft))

    index_attributes(document_id, term_id, stored)

    logger.info(
        "Extracted %d attribute(s) for term %r from %d window(s); "
        "%d grounded link(s), %d quote-verified (document %s).",
        len(merged),
        term_extraction.name,
        len(windows),
        linked,
        verified,
        document_id,
    )
