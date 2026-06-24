# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Singleton NeMo Retriever wired to the app pgvector store."""

from __future__ import annotations

from nemo_retriever.graph.retriever import Retriever

from gsf.utils.embedding import get_embed_kwargs
from gsf.vdb import get_semantic_vdb, get_vdb

_retriever: Retriever | None = None
_taxonomies_retriever: Retriever | None = None


def get_tabular_retriever() -> Retriever:
    """Singleton retriever for the tabular / data-layer collection."""
    global _retriever
    if _retriever is None:
        vdb = get_vdb()
        _retriever = Retriever(
            vdb_kwargs={"vdb": vdb},
            embed_kwargs=get_embed_kwargs(),
        )
    return _retriever


def get_taxonomies_retriever() -> Retriever:
    """Singleton retriever for the semantic-layer (taxonomies) collection."""
    global _taxonomies_retriever
    if _taxonomies_retriever is None:
        _taxonomies_retriever = Retriever(
            vdb_kwargs={"vdb": get_semantic_vdb()},
            embed_kwargs=get_embed_kwargs(),
        )
    return _taxonomies_retriever
