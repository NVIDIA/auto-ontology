# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for the unstructured ingest pipeline.

Import this module before nemo_retriever so that the Ray resource override is
in place before Ray initialises.  Apple Silicon advertises as a GPU; the
override forces Ray to plan for CPU-only NIM endpoints instead.  Only the
``nemo`` extractor touches Ray, but the override is set unconditionally because
it has to land before any import that might start a cluster.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

#: Which document extractor ``ingest`` runs: ``"hosted"``, ``"docling"``, or
#: ``"nemo"``.
#:
#: All three scored the same over the 318 ViDoRe v3 HR queries — nDCG@5 0.2237,
#: 0.2219 and 0.2317, with every paired bootstrap interval crossing zero — so
#: the default is chosen on what each one costs to run, not on quality:
#:
#: - ``hosted`` runs no model locally.  Layout comes from NVIDIA's page-elements
#:   service and figures from the gateway's vision model, so the only local work
#:   is parsing and rasterising the PDF.  1.70 s/page.  Needs the
#:   ``UNSTRUCTURED_PAGE_ELEMENTS_*`` triplet, which defaults to the hosted
#:   inference gateway and its ``sk-`` key.
#: - ``docling`` runs page layout on the local CPU, which costs 3.0GB of torch
#:   and CUDA wheels on linux-x86_64 and crashes on Apple's MPS backend, so it
#:   is pinned to CPU.  2.05 s/page.  Install the ``unstructured-docling`` extra.
#: - ``nemo`` is NeMo Retriever's extract + embed graph.  Fastest at 0.76 s/page,
#:   but needs a Ray cluster and ``RAY_AUTH_MODE=disabled`` to start.  Install
#:   the ``unstructured-nemo`` extra.
#:
#: ``hosted`` takes text from the PDF's own text layer, so an image-only page
#: yields nothing unless a figure is detected on it.  A scanned corpus needs OCR
#: wired in; this one did not.
#:
#: They write the same chunk shape but not the same chunks, so switching this
#: against an already-populated table mixes two corpora.  Point ``CHUNK_TABLE``
#: somewhere else when comparing them.
EXTRACTOR = os.environ.get("UNSTRUCTURED_EXTRACTOR", "hosted")

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
