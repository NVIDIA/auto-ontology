# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which vector store this process uses, and what that means for embedding.

Kept free of heavy imports on purpose: :mod:`auto_ontology.utils.embedding` and
:mod:`auto_ontology.vdb.records` both ask :func:`embeds_server_side`, and they
sit on the import path of everything that writes to a store.
"""

from __future__ import annotations

import os
from typing import Literal

VdbBackend = Literal["postgres", "elasticsearch"]

_BACKENDS: tuple[VdbBackend, ...] = ("postgres", "elasticsearch")


def vdb_backend() -> VdbBackend:
    """The configured backend: ``VDB_BACKEND``, defaulting to ``postgres``.

    Read on every call rather than once at import, so a test can switch it
    with ``monkeypatch.setenv``.
    """
    value = os.environ.get("VDB_BACKEND", "").strip().lower() or "postgres"
    if value not in _BACKENDS:
        raise ValueError(f"VDB_BACKEND={value!r} is not one of {', '.join(_BACKENDS)}.")
    return value  # type: ignore[return-value]


def embeds_server_side() -> bool:
    """Whether the store embeds text itself, so the app must not.

    True for Elasticsearch, whose ``semantic_text`` field calls the NVIDIA
    inference endpoint on write (as a ``passage``) and on search (as a
    ``query``). Rows then reach the store as text alone, and the app's own
    embedding step is skipped rather than run and thrown away.
    """
    return vdb_backend() == "elasticsearch"


SearchMode = Literal["semantic", "hybrid"]

_SEARCH_MODES: tuple[SearchMode, ...] = ("semantic", "hybrid")


def vdb_search_mode() -> SearchMode:
    """The configured search mode: ``VDB_SEARCH_MODE``, defaulting to ``semantic``.

    ``hybrid`` fuses a BM25 search over row names with the vector search, by
    a weighted sum of their normalised scores. It needs Elasticsearch, and it changes what a
    hit's score means -- see
    :mod:`auto_ontology.retrieval.data_access.hit_scores` -- so asking for it
    on Postgres is an error rather than something to quietly ignore.
    """
    value = os.environ.get("VDB_SEARCH_MODE", "").strip().lower() or "semantic"
    if value not in _SEARCH_MODES:
        raise ValueError(
            f"VDB_SEARCH_MODE={value!r} is not one of {', '.join(_SEARCH_MODES)}."
        )
    if value == "hybrid" and vdb_backend() != "elasticsearch":
        raise ValueError("VDB_SEARCH_MODE=hybrid requires VDB_BACKEND=elasticsearch.")
    return value  # type: ignore[return-value]
