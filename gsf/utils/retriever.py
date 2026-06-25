# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Singleton NeMo Retriever wired to the app pgvector store."""

from __future__ import annotations

from nemo_retriever.graph.retriever import Retriever

from gsf.utils.embedding import get_embed_kwargs
from gsf.vdb import get_data_vdb, get_semantic_vdb

_data_retriever: Retriever | None = None
_semantic_retriever: Retriever | None = None


def get_data_objects_retriever() -> Retriever:
    """Singleton retriever for the tabular / data-layer collection."""
    global _data_retriever
    if _data_retriever is None:
        vdb = get_data_vdb()
        _data_retriever = Retriever(
            vdb_kwargs={"vdb": vdb},
            embed_kwargs=get_embed_kwargs(),
        )
    return _data_retriever


def get_semantic_objects_retriever() -> Retriever:
    """Singleton retriever for the semantic-layer (taxonomies) collection."""
    global _semantic_retriever
    if _semantic_retriever is None:
        _semantic_retriever = Retriever(
            vdb_kwargs={"vdb": get_semantic_vdb()},
            embed_kwargs=get_embed_kwargs(),
        )
    return _semantic_retriever
