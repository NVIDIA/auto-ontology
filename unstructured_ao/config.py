# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for the unstructured ingest pipeline.

Import this module before nemo_retriever so that the Ray resource override is
in place before Ray initialises.  Apple Silicon advertises as a GPU; the
override forces Ray to plan for CPU-only NIM endpoints instead.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

#: Table holding unstructured chunk embeddings, written by the v2
#: ``PGVectorStore`` API.  Must be stable across ingest and retrieve runs — a
#: mismatch means retrieval queries the wrong table (or an empty one).
CHUNK_SCHEMA = "vdb"
CHUNK_TABLE = "unstructured_chunks"

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
