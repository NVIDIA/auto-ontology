# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector search over a pgvector collection, and the app's two singletons."""

from __future__ import annotations

import logging
import os
import threading
from array import array
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any

from auto_ontology.utils.embedding import EmbedParams, embed_texts, get_embed_params
from auto_ontology.vdb import get_data_vdb, get_semantic_vdb
from auto_ontology.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)

# A search embeds one short string, so it should answer in seconds; the
# ingest-sized default would let one stalled request hold a question for ten
# minutes before ``_query_with_retry`` ever got to see the failure. That loop
# is the only retry on this path -- the client's own are off -- so a dead
# endpoint costs three 60s attempts, not three times six.
_QUERY_TIMEOUT_S = 60.0

# Query vectors kept per process. One 2048-dimension vector is 16 KB as packed
# doubles, so the default holds about 16 MB. Zero turns the cache off.
_QUERY_CACHE_SIZE = int(os.environ.get("AUTO_ONTOLOGY_QUERY_EMBED_CACHE_SIZE", "1024"))


class _QueryEmbeddingCache:
    """Embed each distinct query string once.

    One question embeds the same few strings again and again: each entity is
    searched for attributes during retrieval, for tables during preparation,
    again to rank hub siblings, and once more by every reconstruction that
    goes looking for a missing table; a full-pipeline retry repeats all of it.
    Each of those was its own request to the embedding endpoint, on the
    question's critical path whenever nothing else was running, and each one a
    chance to be rate-limited into a multi-second backoff.

    The cached value is the endpoint's own answer for that exact request --
    same model, same input type, same text -- held as packed doubles, so a hit
    returns bit-for-bit the vector a new request would have. Nothing is
    approximated and nothing is batched: a batch is a different request and is
    not guaranteed the same floats.

    Concurrent requests for a string not yet cached share one call. Retrieval
    searches the full question under two labels at the same moment, and without
    this both would miss and both would embed it. A failure is never stored:
    every waiter gets the exception, and the next request tries again.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._vectors: OrderedDict[tuple, array] = OrderedDict()
        self._pending: dict[tuple, Future] = {}

    def get(self, key: tuple, compute: Any) -> list[float] | None:
        """The vector for *key*, calling *compute* only if nobody has it yet.

        ``None`` from *compute* (a blank query) is returned and not kept.
        """
        if self._max_entries <= 0:
            return compute()

        with self._lock:
            packed = self._vectors.get(key)
            if packed is not None:
                self._vectors.move_to_end(key)
                return packed.tolist()
            pending = self._pending.get(key)
            if pending is None:
                pending = self._pending[key] = Future()
                owner = True
            else:
                owner = False

        if not owner:
            vector = pending.result()
            return None if vector is None else list(vector)

        try:
            vector = compute()
        except BaseException as exc:
            with self._lock:
                self._pending.pop(key, None)
            pending.set_exception(exc)
            raise

        # Packing can only fail for something that is not a flat list of
        # numbers. That is not this cache's to judge: the vector goes on to the
        # store exactly as it would have, and is simply not kept. Whatever
        # happens here, the waiters are released.
        packed: array | None = None
        if vector is not None:
            try:
                packed = array("d", vector)
            except (TypeError, ValueError, OverflowError):
                packed = None

        with self._lock:
            self._pending.pop(key, None)
            if packed is not None:
                self._vectors[key] = packed
                while len(self._vectors) > self._max_entries:
                    self._vectors.popitem(last=False)
        pending.set_result(vector)
        return vector

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()


_query_embeddings = _QueryEmbeddingCache(_QUERY_CACHE_SIZE)


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
        is the store's metadata filter. Raises when the query cannot be
        embedded, with the endpoint's HTTP status in the message, so callers
        can tell a transient failure from an empty result.
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
        """The query vector for *text*, embedded at most once per process.

        The key is everything that decides the request ``embed_texts`` sends:
        where it goes, which model answers, the modality, and the text as it
        is sent (stripped). See :class:`_QueryEmbeddingCache`.

        The embedding function itself is part of the key as well. In the
        running service it is always the same function; a test that swaps in
        its own embedder gets its own entries instead of vectors another
        embedder produced for the same string.
        """
        params = self.embed_params
        embed = embed_texts
        key = (
            embed,
            params.embed_invoke_url,
            params.model_name,
            params.embed_modality,
            text.strip() if isinstance(text, str) else text,
        )
        return _query_embeddings.get(
            key,
            lambda: embed(
                [text],
                params,
                "query",
                timeout=_QUERY_TIMEOUT_S,
                retries=0,
            )[0],
        )


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
