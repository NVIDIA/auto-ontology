# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for the unstructured ingest pipeline."""

from __future__ import annotations

import os

#: Which document extractor ``ingest`` runs: ``"auto"``, ``"hosted"``,
#: or ``"docling"``.
#:
#: - ``auto`` uses hosted extraction for PDF, Docling for DOCX/PPTX/HTML, and
#:   the built-in plain-text reader for TXT.
#: - ``hosted`` runs no model locally.  Layout comes from NVIDIA's page-elements
#:   service and figures from the gateway's vision model, so the only local work
#:   is parsing and rasterising the PDF.  1.70 s/page.  Needs the
#:   ``UNSTRUCTURED_PAGE_ELEMENTS_*`` triplet, which defaults to the hosted
#:   inference gateway and its ``sk-`` key.
#: - ``docling`` runs page layout on the local CPU, which costs 3.0GB of torch
#:   and CUDA wheels on linux-x86_64 and crashes on Apple's MPS backend, so it
#:   is pinned to CPU.  2.05 s/page.  Install the ``unstructured-docling`` extra.
#:
#: ``hosted`` takes text from the PDF's own text layer, so an image-only page
#: yields nothing unless a figure is detected on it.  A scanned corpus needs OCR
#: wired in; this one did not.
#:
#: They write the same chunk shape but not the same chunks, so switching this
#: against an already-populated table mixes two corpora.  Point ``CHUNK_TABLE``
#: somewhere else when comparing them.
EXTRACTOR = os.environ.get("UNSTRUCTURED_EXTRACTOR", "auto")

#: Where files uploaded through ``/api/documents`` are kept, one subdirectory
#: per upload so two files with the same name do not collide.  The stored path
#: is the document's unique key, so it must survive restarts: mount a volume
#: here in a container.
UPLOAD_DIR = os.environ.get(
    "UNSTRUCTURED_UPLOAD_DIR",
    os.path.join(os.path.expanduser("~"), ".auto_ontology", "documents"),
)

#: Most words an uploaded document may hold.  Counted over the extracted text,
#: so it is enforced after extraction: a file over it ends up ``failed``.
MAX_DOCUMENT_WORDS = 500_000

#: Whether an upload's background task goes on from ingest to term and
#: attribute extraction.  That pass makes several LLM calls per document, so it
#: is the slow and paid part; turn it off to ingest chunks only.
EXTRACT_ON_UPLOAD = os.environ.get(
    "UNSTRUCTURED_EXTRACT_ON_UPLOAD", "true"
).lower() in (
    "1",
    "true",
    "yes",
)

#: Table holding unstructured chunk embeddings, written by the v2
#: ``PGVectorStore`` API.  Must be stable across ingest and retrieve runs — a
#: mismatch means retrieval queries the wrong table (or an empty one).
CHUNK_SCHEMA = "vdb"
CHUNK_TABLE = os.environ.get("UNSTRUCTURED_CHUNK_TABLE", "unstructured_chunks")

#: Table holding attribute embeddings — ``"<name>: <description>"`` per row,
#: keyed by ``unstructured_attribute.id``.  Separate from the chunk table
#: because searching it answers a different question: chunks find the passage
#: that mentions something, attributes find the concept itself.
ATTRIBUTE_SCHEMA = "vdb"
ATTRIBUTE_TABLE = "unstructured_attributes"

#: Cosine-similarity top-k used when linking attributes to chunks.
RETRIEVE_TOP_K = 5

#: Characters of chunk text per extraction window.  Extraction reads the whole
#: document one window at a time, so this trades LLM calls against how much
#: context the model has when deciding whether an attribute is new.
EXTRACT_WINDOW_CHARS = 10_000

#: Cosine similarity above which two same-typed attributes are treated as one.
#: Catches the drift the per-window reuse instruction misses — "Headcount of
#: posted workers" against "Number of posted workers".
ATTRIBUTE_MERGE_THRESHOLD = 0.9
