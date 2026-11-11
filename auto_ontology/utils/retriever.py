# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector search over a pgvector collection, and the app's two singletons."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from openai import APIConnectionError, APIStatusError

from auto_ontology.utils.embedding import EmbedParams, embed_texts, get_embed_params
from auto_ontology.vdb import get_data_vdb, get_semantic_vdb
from auto_ontology.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)

# A search embeds one short string, so it should answer in seconds; the
# ingest-sized default would let one stalled request hold a question for ten
# minutes before the retry in ``Retriever.query`` ever got to see the failure.
# That loop is the only retry on this path -- the client's own are off -- so a
# dead endpoint costs three 60s attempts, not three times six.
_QUERY_TIMEOUT_S = 60.0
_QUERY_ATTEMPTS = 3
_TRANSIENT_TOKENS = (
    "502",
    "Bad Gateway",
    "503",
    "Service Unavailable",
    "504",
    "Gateway Timeout",
    "Timeout",
    "Connection",
)


def _is_transient(exc: Exception) -> bool:
    """Whether *exc* is an endpoint blip rather than a problem with the query.

    The client's errors are classified by type: ``APIConnectionError`` covers
    timeouts too, whose message ("Request timed out.") names no status.
    """
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (408, 429) or exc.status_code >= 500
    return any(token in str(exc) for token in _TRANSIENT_TOKENS)


class Retriever:
    """Embed a query and run a k-NN search against one :class:`PostgresVDB`.

    ``vdb_kwargs["vdb"]`` is part of the interface, not an implementation
    detail: :func:`~auto_ontology.retrieval.data_access.semantic_search.search_semantic_index`
    reads the store's ``metadata_filter_format`` off it to decide how to build
    the ``where`` filter.
    """

    def __init__(self, vdb: PostgresVDB, embed_params: EmbedParams) -> None:
        self.vdb_kwargs: dict[str, Any] = {"vdb": vdb}
        self.embed_params = embed_params

    @property
    def vdb(self) -> PostgresVDB:
        return self.vdb_kwargs["vdb"]

    def query(
        self,
        text: str,
        top_k: int = 10,
        vdb_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """The *top_k* nearest rows to *text*, best first.

        Each hit is ``{"text", "metadata", "_distance"}``, with ``metadata``
        the row's full stored metadata. ``vdb_kwargs["where"]``, when given,
        is the store's metadata filter. A timeout, a dropped connection or a
        5xx/408/429 from the embedding endpoint is retried with backoff; when
        the query still cannot be embedded this raises, with the endpoint's
        HTTP status in the message, so callers can tell a failure from an
        empty result.
        """
        vector = self._embed_query(text)
        if vector is None:
            raise ValueError("Cannot search for a blank query.")
        where = (vdb_kwargs or {}).get("where")
        hits = self.vdb.retrieval([vector], top_k=top_k, where=where)[0]
        return [
            {
                "text": (hit.get("text") or "").strip(),
                "metadata": hit.get("metadata") or {},
                "_distance": float(hit.get("_distance", 0.0)),
            }
            for hit in hits[:top_k]
        ]

    def _embed_query(self, text: str) -> list[float] | None:
        for attempt in range(1, _QUERY_ATTEMPTS + 1):
            try:
                return embed_texts(
                    [text],
                    self.embed_params,
                    "query",
                    timeout=_QUERY_TIMEOUT_S,
                    retries=0,
                )[0]
            except Exception as exc:
                if attempt == _QUERY_ATTEMPTS or not _is_transient(exc):
                    raise
                wait = 2**attempt + random.uniform(0, 1)
                logger.warning(
                    "Embedding the query failed on attempt %d/%d, retrying in %.1fs: %s",
                    attempt,
                    _QUERY_ATTEMPTS,
                    wait,
                    str(exc)[:200],
                )
                time.sleep(wait)
        raise AssertionError("unreachable")


_data_retriever: Retriever | None = None
_semantic_retriever: Retriever | None = None


def get_data_objects_retriever() -> Retriever:
    """Singleton retriever for the tabular / data-layer collection."""
    global _data_retriever
    if _data_retriever is None:
        _data_retriever = Retriever(get_data_vdb(), get_embed_params())
    return _data_retriever


def get_semantic_objects_retriever() -> Retriever:
    """Singleton retriever for the semantic-layer (taxonomies) collection."""
    global _semantic_retriever
    if _semantic_retriever is None:
        _semantic_retriever = Retriever(get_semantic_vdb(), get_embed_params())
    return _semantic_retriever


def _close_retriever_vdb(retriever: Retriever | None) -> None:
    if retriever is None:
        return
    try:
        retriever.vdb.close()
    except Exception:
        logger.exception("Failed to close PostgresVDB on retriever cleanup")


def close_retrievers() -> None:
    """Dispose singleton retriever VDB pools so CLI/debug processes can exit.

    ``PGEngine`` keeps a background asyncio loop and ThreadPoolExecutor workers,
    so a process that never disposes them exits later than it should. Calling
    this explicitly is the tidy path; ``PostgresVDB.close`` bounds the wait and
    skips it outright once the loop thread is gone, so neither route can wedge
    at shutdown the way it once did.
    """
    global _data_retriever, _semantic_retriever
    _close_retriever_vdb(_data_retriever)
    _close_retriever_vdb(_semantic_retriever)
    _data_retriever = None
    _semantic_retriever = None
