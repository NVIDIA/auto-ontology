# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM extraction of terms and attributes from unstructured document text.

Design mirrors illumex's ``embed_service.extract_attributes_from_text()`` and
``TermsExtractor`` but simplified for one term per document (the document's
core concept / topic) and per-term attribute extraction.

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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sqlalchemy import Connection

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auto_ontology.utils.llm_invoke import (
    get_non_reasoning_llm_client,
    invoke_with_structured_output,
)

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
    """One attribute extracted from the document."""

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


class AttributeExtractionResult(BaseModel):
    """All attributes extracted for a given document term."""

    attributes: list[AttributeItem] = Field(
        description="Extracted attributes, typically 3–10 per document."
    )


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

Given the text of a document and its primary business term, extract structured
attributes — measurable facts or reusable reference lists that could enrich a
data catalogue entry for this term.

Attribute types:
  sql  — a numeric metric or calculation that a data analyst would compute
         (e.g. "Employment rate", "Number of posted workers", "Wage growth %").
  ref  — a named list of categories, entities, or literal facts enumerated in
         the document (e.g. "EU member states covered", "Occupational groups",
         "Legal instruments cited").

Rules:
- Extract 3–10 attributes; quality over quantity.
- Every name must be short (≤10 words) and specific to this document.
- Every description must be one complete sentence.
- Do not invent attributes not supported by the text.
- Do not repeat the same concept under different names.
"""


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_term(text: str, *, max_chars: int = 8_000) -> TermExtraction | None:
    """Extract the single core business term from *text*.

    Uses the non-reasoning LLM (fast/cheap).  Falls back to ``None`` on
    persistent failure so the caller can decide whether to skip or raise.
    """
    excerpt = text[:max_chars]
    llm = get_non_reasoning_llm_client()
    messages = [
        SystemMessage(content=_TERM_SYSTEM),
        HumanMessage(content=f"DOCUMENT TEXT:\n\n{excerpt}"),
    ]
    result = invoke_with_structured_output(llm, messages, TermExtraction)
    if result is None:
        logger.warning("Term extraction returned None — document will have no term.")
    return result


def extract_attributes(
    text: str,
    term_name: str,
    *,
    max_chars: int = 12_000,
) -> list[AttributeItem]:
    """Extract attributes for *term_name* from *text*.

    Returns an empty list on persistent LLM failure.
    """
    excerpt = text[:max_chars]
    llm = get_non_reasoning_llm_client()
    messages = [
        SystemMessage(content=_ATTRIBUTE_SYSTEM),
        HumanMessage(
            content=(f"BUSINESS TERM: {term_name}\n\nDOCUMENT TEXT:\n\n{excerpt}")
        ),
    ]
    result = invoke_with_structured_output(llm, messages, AttributeExtractionResult)
    if result is None:
        logger.warning("Attribute extraction returned None for term %r.", term_name)
        return []
    return result.attributes


def run_extraction(
    document_id: str,
    full_text: str,
    conn: "Connection",  # type: ignore[name-defined]
) -> None:
    """Extract and persist the term and attributes for *document_id*.

    Writes to ``unstructured_term`` and ``unstructured_attribute``.  The
    retrieve step (``retrieve.link_attributes_to_chunks``) runs afterwards to
    build the attribute → chunk pointer table.
    """
    from unstructured.dal import insert_attribute, insert_term

    term_extraction = extract_term(full_text)
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

    attributes = extract_attributes(full_text, term_extraction.name)
    for attr in attributes:
        insert_attribute(
            conn,
            term_id=term_id,
            name=attr.name,
            description=attr.description,
            attribute_type=attr.attribute_type,
        )
    logger.info(
        "Extracted %d attribute(s) for term %r (document %s).",
        len(attributes),
        term_extraction.name,
        document_id,
    )
