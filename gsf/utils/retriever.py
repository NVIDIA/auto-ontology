# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Singleton NeMo Retriever wired to the app pgvector store."""

from __future__ import annotations

import logging

from nemo_retriever.graph.retriever import Retriever

from gsf.utils.embedding import get_embed_kwargs
from gsf.vdb import get_data_vdb, get_semantic_vdb
from gsf.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)

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


def _close_retriever_vdb(retriever: Retriever | None) -> None:
    if retriever is None:
        return
    vdb = (retriever.vdb_kwargs or {}).get("vdb")
    if isinstance(vdb, PostgresVDB):
        try:
            vdb.close()
        except Exception:
            logger.exception("Failed to close PostgresVDB on retriever cleanup")


def close_retrievers() -> None:
    """Dispose singleton retriever VDB pools so CLI/debug processes can exit.

    PGEngine keeps a background asyncio loop and ThreadPoolExecutor workers;
    without an explicit close, interpreter shutdown joins those workers and
    appears to hang after the coverage result is already printed.
    """
    global _data_retriever, _semantic_retriever
    _close_retriever_vdb(_data_retriever)
    _close_retriever_vdb(_semantic_retriever)
    _data_retriever = None
    _semantic_retriever = None
