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
from pathlib import Path

os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data" / "vidore_v3_hr"

#: langchain_postgres collection name for unstructured chunk embeddings.
#: Must be stable across ingest and retrieve runs — a mismatch means retrieval
#: queries the wrong table (or an empty one).
CHUNK_COLLECTION = "unstructured_chunks"

#: Cosine-similarity top-k used when linking attributes to chunks.
RETRIEVE_TOP_K = 5
