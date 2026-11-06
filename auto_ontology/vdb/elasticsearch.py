# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elasticsearch vector store, embedding through a ``semantic_text`` field.

Unlike :class:`~auto_ontology.vdb.postgres.PostgresVDB`, this store is handed
text, not vectors. The index maps a plain ``text`` field and copies it into
``content``, a ``semantic_text`` field bound to an ``nvidia`` inference
endpoint (see :mod:`auto_ontology.infra.elasticsearch`). Elasticsearch embeds
each row on write as a ``passage`` and each search string as a ``query``, so
the asymmetric ``input_type`` handling the app otherwise does itself is the
endpoint's job here.

``text`` stays a lexical field for hybrid search (``VDB_SEARCH_MODE=hybrid``):
an ES|QL ``FORK`` runs BM25 over ``text`` and a vector search over
``content``, and ``FUSE`` merges them by reciprocal rank fusion on the
server. See :meth:`ElasticsearchVDB._hybrid_search` for what a hybrid hit
carries and :mod:`auto_ontology.retrieval.data_access.hit_scores` for how the
app reads it.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from typing import Any, Callable, Iterable

from elasticsearch import BadRequestError, Elasticsearch, NotFoundError, helpers

from auto_ontology.infra.elasticsearch import EmbedEndpoint, ensure_embed_endpoint
from auto_ontology.vdb.backend import SearchMode

logger = logging.getLogger(__name__)

#: Metadata keys indexed as ``keyword`` so they can be filtered on. Everything
#: else in a row's metadata is kept in ``_source`` and returned with hits, but
#: not indexed -- ``content_metadata`` alone would otherwise grow the mapping
#: with every new key a row builder adds.
_FILTERABLE_METADATA = (
    "id",
    "label",
    "database_name",
    "schema_name",
    "name",
    "source",
    "document_type",
)

# Retry policy for rows the endpoint refused, by way of Elasticsearch. Same
# shape and reasoning as auto_ontology.utils.embedding's chunk retry, which
# this replaces on this backend: the failure is a rate limit, and the quota
# needs wall-clock time to refill. Elasticsearch has already retried each
# request a few times before reporting the row as failed.
_RETRY_ATTEMPTS = 4
#: Ids per delete_by_query in delete_by_database, under the default 65,536
#: terms limit with room to spare.
_DELETE_BATCH = 10_000
_RETRY_BASE_SECONDS = 15.0
_RETRY_MAX_SECONDS = 120.0

#: A bulk request waits for every row in it to be embedded; at ingest sizes
#: that is minutes, not the client's default ten seconds.
_REQUEST_TIMEOUT_S = 600.0

# Indices and endpoints this process has already checked, so a store built
# per request does not cost two round trips each time.
_ensured: set[tuple[str, str]] = set()
_ensured_lock = threading.Lock()


def _flatten(records: Iterable) -> Iterable[dict]:
    """Yield record dicts from a possibly-nested list of records."""
    for item in records:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _flatten(item)


def _retryable(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


# How the inference plugin quotes the model endpoint's HTTP status inside its
# own error: "... from inference entity id [x] status [500]. Error message ...".
_UPSTREAM_STATUS = re.compile(r"status \[(\d{3})\]")


def upstream_status(error: object) -> int | None:
    """The model endpoint's HTTP status quoted in an inference error, if any.

    Elasticsearch does not pass every upstream status through as its own: a
    429 arrives as 429, but a 5xx from the model endpoint arrives as a 400,
    with the real status only in the message. Read as-is, a transient outage
    at NVIDIA would look like a malformed request and never be retried.
    *error* is a bulk item's ``error`` dict, an exception, or a message.
    """
    match = _UPSTREAM_STATUS.search(str(error))
    return int(match.group(1)) if match else None


def is_retryable_failure(status: int, error: object) -> bool:
    """Whether a request that failed with *status* and *error* is worth retrying."""
    if _retryable(status):
        return True
    upstream = upstream_status(error)
    return upstream is not None and _retryable(upstream)


def _filter_clauses(where: dict[str, Any] | None) -> list[dict]:
    """``{"label": "Column", "database_name": [...]}`` as ``term``/``terms`` clauses."""
    clauses: list[dict] = []
    for key, value in (where or {}).items():
        if value is None:
            continue
        field_name = f"metadata.{key}"
        if isinstance(value, (list, tuple, set)):
            clauses.append({"terms": {field_name: [str(v) for v in value]}})
        else:
            clauses.append({"term": {field_name: str(value)}})
    return clauses


# Index names are interpolated into ES|QL (FROM cannot take a parameter); they
# come from config, but are checked so that a bad one fails here, not as a
# confusing parse error.
_INDEX_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")

#: Rows each branch of a hybrid search contributes to the fusion. Larger than
#: any per-label ``top_k`` the app asks for: a row's fused rank depends on its
#: rank in each branch, and a branch cut at ``top_k`` would drop the rows that
#: the *other* branch ranks highly.
DEFAULT_RANK_WINDOW = 50

#: Elasticsearch's own RRF default.
DEFAULT_RANK_CONSTANT = 60

_HYBRID_QUERY = """
FROM {index} METADATA _id, _index, _score
| FORK
    ( WHERE MATCH(text, ?query){filters}
      | SORT _score DESC | LIMIT ?window | EVAL lexical_score = _score )
    ( WHERE MATCH(content, ?query){filters}
      | SORT _score DESC | LIMIT ?window | EVAL semantic_score = _score )
| FUSE RRF WITH {{"rank_constant": {rank_constant}}}
| SORT _score DESC
| LIMIT ?top_k
| KEEP _id, _score, lexical_score, semantic_score
"""


def _esql_filters(where: dict[str, Any] | None) -> tuple[str, list[dict]]:
    """``where`` as ES|QL conditions (``AND ...``) and their named parameters.

    Values are always parameters. Keys become field names in the query text,
    so only the indexed metadata keys are accepted.
    """
    conditions: list[str] = []
    params: list[dict] = []
    for key, value in (where or {}).items():
        if value is None:
            continue
        if key not in _FILTERABLE_METADATA:
            raise ValueError(f"Cannot filter on metadata key {key!r}.")
        values = list(value) if isinstance(value, (list, tuple, set)) else [value]
        names = []
        for item in values:
            name = f"f{len(params)}"
            params.append({name: str(item)})
            names.append(f"?{name}")
        if len(names) == 1:
            conditions.append(f"metadata.{key} == {names[0]}")
        else:
            conditions.append(f"metadata.{key} IN ({', '.join(names)})")
    return "".join(f" AND {c}" for c in conditions), params


def score_to_distance(score: float) -> float:
    """Cosine distance from an Elasticsearch kNN score.

    Under ``cosine`` -- and ``dot_product`` on the unit-length vectors NVIDIA
    returns; see ``infra.elasticsearch._COSINE_SCORED`` -- Elasticsearch scores
    a hit ``(1 + cos) / 2``, so
    ``1 - cos`` -- pgvector's ``<=>``, and what every distance threshold in the
    app was tuned against -- is ``2 * (1 - score)``.
    """
    return 2.0 * (1.0 - float(score))


class ElasticsearchVDB:
    """Vector store backed by an Elasticsearch index with a ``semantic_text`` field."""

    metadata_filter_format = "dict"
    embeds_server_side = True

    def __init__(
        self,
        *,
        client: Elasticsearch,
        index_name: str,
        endpoint: EmbedEndpoint,
        database_name: str | None = None,
        reset: bool = False,
        search_mode: SearchMode = "semantic",
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        rank_window: int = DEFAULT_RANK_WINDOW,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not _INDEX_NAME.fullmatch(index_name):
            raise ValueError(f"Not a valid Elasticsearch index name: {index_name!r}")
        self._client = client
        self.index_name = index_name
        self.collection_name = index_name
        self._endpoint = endpoint
        self.database_name = database_name
        self.search_mode = search_mode
        self._rank_constant = int(rank_constant)
        self._rank_window = int(rank_window)
        self._sleep = sleep
        # Every row this store writes carries the generation, so a reset can
        # remove exactly the rows that predate it.
        self._generation = uuid.uuid4().hex
        # A reset is deferred to the first successful write rather than done
        # here, as PostgresVDB does. On that backend the app has embedded the
        # rows before the store is even built, so a dead endpoint never gets as
        # far as the reset. Here the write *is* the embed, and resetting first
        # would leave a failed ingest with no rows at all instead of the
        # previous run's.
        self._reset_pending = bool(database_name and reset)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _mappings(self) -> dict:
        return {
            "dynamic": False,
            "properties": {
                "text": {"type": "text", "copy_to": "content"},
                "content": {
                    "type": "semantic_text",
                    "inference_id": self._endpoint.inference_id,
                },
                "generation": {"type": "keyword"},
                "metadata": {
                    "type": "object",
                    "dynamic": False,
                    "properties": {
                        key: {"type": "keyword"} for key in _FILTERABLE_METADATA
                    },
                },
            },
        }

    def create_index(self, **kwargs: Any) -> str:
        """Ensure the inference endpoint and the index exist."""
        key = (self.index_name, self._endpoint.inference_id)
        with _ensured_lock:
            if key in _ensured:
                return self.index_name
        ensure_embed_endpoint(self._client, self._endpoint)
        if not self._client.indices.exists(index=self.index_name):
            try:
                self._client.indices.create(
                    index=self.index_name, mappings=self._mappings()
                )
                logger.info("Created Elasticsearch index %s", self.index_name)
            except BadRequestError as exc:
                if "resource_already_exists_exception" not in str(exc):
                    raise
        with _ensured_lock:
            _ensured.add(key)
        return self.index_name

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _documents(self, records: list, generation: str | None = None) -> list[dict]:
        docs: list[dict] = []
        for record in _flatten(records):
            metadata = record.get("metadata") or {}
            text = (
                record.get("text") or record.get("content") or metadata.get("content")
            )
            if not isinstance(text, str) or not text.strip():
                continue
            docs.append(
                {
                    "text": text,
                    "generation": generation or self._generation,
                    "metadata": {
                        "document_type": record.get("document_type"),
                        # A vector computed by an older code path is not
                        # ours to store; the endpoint embeds the text.
                        **{k: v for k, v in metadata.items() if k != "embedding"},
                    },
                }
            )
        return docs

    def _bulk(self, docs: list[dict], refresh: str | bool) -> int:
        """Index *docs*, retrying the rows that failed on a retryable status.

        Returns the number indexed. Rows that failed for good are logged with
        the first reason Elasticsearch gave; a bad key or an unknown model
        fails every row the same way, and one line says so.
        """
        pending = docs
        written = 0
        delay = _RETRY_BASE_SECONDS
        last_error: object = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            operations: list[dict] = []
            for doc in pending:
                operations.append({"index": {"_index": self.index_name}})
                operations.append(doc)
            response = self._client.options(request_timeout=_REQUEST_TIMEOUT_S).bulk(
                operations=operations, refresh=refresh
            )
            retry: list[dict] = []
            fatal: list[dict] = []
            for doc, item in zip(pending, response["items"]):
                result = item.get("index") or {}
                status = int(result.get("status", 500))
                if status < 300:
                    written += 1
                elif is_retryable_failure(status, result.get("error")):
                    retry.append(doc)
                    last_error = result.get("error")
                else:
                    fatal.append(result)
            if fatal:
                logger.error(
                    "ElasticsearchVDB: %d row(s) rejected by %s -- %s",
                    len(fatal),
                    self.index_name,
                    fatal[0].get("error"),
                )
            if not retry:
                break
            if attempt == _RETRY_ATTEMPTS:
                logger.error(
                    "ElasticsearchVDB: giving up on %d row(s) after %d attempts -- %s",
                    len(retry),
                    attempt,
                    last_error,
                )
                break
            logger.warning(
                "ElasticsearchVDB: %d row(s) failed on attempt %d/%d (%s); "
                "retrying in %.0fs",
                len(retry),
                attempt,
                _RETRY_ATTEMPTS,
                last_error,
                delay,
            )
            self._sleep(delay)
            delay = min(delay * 2, _RETRY_MAX_SECONDS)
            pending = retry
        return written

    def write_to_index(
        self,
        records: list,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> int:
        """Bulk-index records, returning the number of rows written.

        Raises ``RuntimeError`` when there were rows to write and none were
        written. On this backend the write is the embed, so this is where a
        dead endpoint surfaces -- the same point at which
        :func:`~auto_ontology.utils.embedding.embed_rows` raises on the other.
        """
        docs = self._documents(records, kwargs.get("generation"))
        if not docs:
            logger.info("ElasticsearchVDB.write_to_index: no rows to insert")
            return 0
        self.create_index()

        written = 0
        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
        for index, batch in enumerate(batches):
            # Wait for visibility once, on the last batch: an interactive save
            # has to be searchable by the next request, but waiting per batch
            # would add a refresh interval (5s on Serverless) to every one.
            last = index == len(batches) - 1
            written += self._bulk(batch, refresh="wait_for" if last else False)
            if written and self._reset_pending:
                self._finish_reset()

        if not written:
            raise RuntimeError(
                f"Elasticsearch wrote 0/{len(docs)} rows to {self.index_name}; "
                f"see the errors above (inference endpoint "
                f"{self._endpoint.inference_id})."
            )
        logger.info(
            "ElasticsearchVDB.write_to_index: inserted %d/%d rows into %s",
            written,
            len(docs),
            self.index_name,
        )
        return written

    def _finish_reset(self) -> None:
        """Delete this database's rows from earlier generations."""
        self._reset_pending = False
        deleted = self._delete(
            {
                "bool": {
                    "filter": [
                        {"term": {"metadata.database_name": self.database_name}}
                    ],
                    "must_not": [{"term": {"generation": self._generation}}],
                }
            }
        )
        logger.info(
            "ElasticsearchVDB: reset removed %d earlier row(s) for database %s",
            deleted,
            self.database_name,
        )

    def run(self, records: list) -> int:
        """Create the index if needed, then write records to it."""
        self.create_index()
        return self.write_to_index(records)

    def replace_rows(self, node_ids: Iterable[str], records: list) -> int:
        """Write *records*, then drop the older rows of the ids that got a new one.

        Write first, the order a reset uses: the write is the embed here, so
        deleting first would leave an edit the endpoint rejects with no row at
        all. The new rows carry their own generation, which is what tells them
        apart from the old ones sharing their id. An id whose new row was
        rejected keeps its old one.
        """
        ids = sorted({str(node_id) for node_id in node_ids})
        if not ids:
            return 0
        generation = uuid.uuid4().hex
        self.create_index()
        # Raises when nothing was written, before anything is deleted.
        written = self.write_to_index(records, generation=generation)
        rewritten = self._ids_in_generation(ids, generation)
        if rewritten:
            self._delete(
                {
                    "bool": {
                        "filter": [{"terms": {"metadata.id": rewritten}}],
                        "must_not": [{"term": {"generation": generation}}],
                    }
                }
            )
        return written

    def _ids_in_generation(self, ids: list[str], generation: str) -> list[str]:
        """Which of *ids* have a row in *generation*."""
        response = self._client.search(
            index=self.index_name,
            size=0,
            query={
                "bool": {
                    "filter": [
                        {"terms": {"metadata.id": ids}},
                        {"term": {"generation": generation}},
                    ]
                }
            },
            aggs={"ids": {"terms": {"field": "metadata.id", "size": len(ids)}}},
        )
        return [bucket["key"] for bucket in response["aggregations"]["ids"]["buckets"]]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete(self, query: dict) -> int:
        try:
            response = self._client.delete_by_query(
                index=self.index_name,
                query=query,
                refresh=True,
                conflicts="proceed",
            )
        except NotFoundError:
            return 0
        return int(response.get("deleted", 0))

    def delete_by_database(self, database_name: str) -> list[str]:
        """Delete every row for *database_name*; returns the deleted rows' ids.

        Ids rather than a count, the contract PostgresVDB has and dal.reset
        relies on. They are collected first and the rows then deleted by id,
        so the list is exactly what was removed.
        """
        try:
            ids = [
                hit["_id"]
                for hit in helpers.scan(
                    self._client,
                    index=self.index_name,
                    query={
                        "query": {"term": {"metadata.database_name": database_name}}
                    },
                    _source=False,
                )
            ]
        except NotFoundError:
            return []
        for start in range(0, len(ids), _DELETE_BATCH):
            self._delete({"ids": {"values": ids[start : start + _DELETE_BATCH]}})
        return ids

    def delete_all(self) -> int:
        """Delete every row, keeping the index and its mapping."""
        deleted = self._delete({"match_all": {}})
        logger.info(
            "ElasticsearchVDB.delete_all: deleted %d rows from %s",
            deleted,
            self.index_name,
        )
        return deleted

    def delete_by_id(self, node_id: str) -> int:
        """Delete every row whose ``metadata["id"]`` is *node_id*."""
        return self._delete({"term": {"metadata.id": str(node_id)}})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def retrieval(
        self,
        queries: list,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        """The best rows for each query *text*, best first.

        In ``semantic`` mode, the nearest rows by cosine; in ``hybrid`` mode,
        BM25 and vector results fused by reciprocal rank (see
        :meth:`_hybrid_search`). Either way ``_distance`` orders the hits,
        lower first.

        Raises on any error but a missing index (nothing ingested yet, which is
        an empty result). That differs from :class:`PostgresVDB`, which logs
        and returns nothing: here a search error is usually the inference
        endpoint failing to embed the query, and
        ``semantic_search._query_with_retry`` needs to see it to retry.
        """
        search = (
            self._hybrid_search
            if self.search_mode == "hybrid"
            else self._semantic_search
        )
        started = time.perf_counter()
        results = [search(text, top_k, kwargs.get("where")) for text in queries]
        logger.debug(
            "vdb.retrieval mode=%s total=%.3fs queries=%d",
            self.search_mode,
            time.perf_counter() - started,
            len(queries),
        )
        return results

    def _semantic_search(
        self, text: str, top_k: int, where: dict[str, Any] | None
    ) -> list[dict]:
        """Nearest rows by cosine, with ``_distance`` = ``1 - cos``.

        The filter goes inside the ``knn`` clause, so it is applied while the
        graph is searched: ``top_k`` matching rows come back even when most of
        the index does not match, which a filter around the query would not
        guarantee.
        """
        knn: dict[str, Any] = {
            "field": "content",
            "k": top_k,
            "num_candidates": max(100, top_k * 10),
            "query_vector_builder": {"text_embedding": {"model_text": text}},
        }
        filters = _filter_clauses(where)
        if filters:
            knn["filter"] = filters
        try:
            response = self._client.search(
                index=self.index_name,
                size=top_k,
                query={"knn": knn},
                source_excludes=["generation"],
            )
        except NotFoundError:
            return []
        return [
            {
                "text": hit["_source"].get("text", ""),
                "metadata": hit["_source"].get("metadata") or {},
                "_distance": score_to_distance(hit["_score"]),
            }
            for hit in response["hits"]["hits"]
        ]

    def _hybrid_search(
        self, text: str, top_k: int, where: dict[str, Any] | None
    ) -> list[dict]:
        """BM25 and vector results fused by reciprocal rank, on the server.

        Each hit carries, beside ``text`` and ``metadata``:

        * ``_rrf`` -- the fused score normalised to (0, 1]: the raw RRF score
          over its maximum, ``2 / (k + 1)``, which a row ranked first by both
          branches reaches.
        * ``_distance`` -- ``1 - _rrf``, so the app's lower-is-better ordering
          still sorts hybrid hits correctly. It is *not* a cosine distance and
          must not be compared with the cosine thresholds; ``hit_scores`` reads
          the fields below for that.
        * ``_lexical_score`` / ``_semantic_score`` -- each branch's own score,
          ``None`` when that branch did not return the row. Comparing two
          rows' scores within a branch is comparing their ranks in it.
        * ``_semantic_distance`` -- the cosine distance, from
          ``_semantic_score``, or ``None``.

        ``FUSE`` keeps only mapped columns, so the full metadata comes from a
        follow-up ``mget`` of the ``top_k`` ids.
        """
        filters, filter_params = _esql_filters(where)
        query = _HYBRID_QUERY.format(
            index=self.index_name,
            filters=filters,
            rank_constant=self._rank_constant,
        )
        params = [
            {"query": text},
            {"window": max(self._rank_window, top_k)},
            {"top_k": top_k},
            *filter_params,
        ]
        try:
            response = self._client.esql.query(query=query, params=params)
        except BadRequestError as exc:
            if "Unknown index" in str(exc):
                return []
            raise
        columns = [column["name"] for column in response["columns"]]
        rows = [dict(zip(columns, values)) for values in response["values"]]
        if not rows:
            return []

        sources = {
            doc["_id"]: doc["_source"]
            for doc in self._client.mget(
                index=self.index_name,
                ids=[row["_id"] for row in rows],
                source_excludes=["generation"],
            )["docs"]
            if doc.get("found")
        }
        max_rrf = 2.0 / (self._rank_constant + 1)
        hits: list[dict] = []
        for row in rows:
            source = sources.get(row["_id"])
            if source is None:
                # Deleted between the search and the mget.
                continue
            rrf = min(float(row["_score"]) / max_rrf, 1.0)
            semantic = row.get("semantic_score")
            hits.append(
                {
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata") or {},
                    "_distance": 1.0 - rrf,
                    "_rrf": rrf,
                    "_lexical_score": row.get("lexical_score"),
                    "_semantic_score": semantic,
                    "_semantic_distance": (
                        score_to_distance(semantic) if semantic is not None else None
                    ),
                }
            )
        return hits

    def close(self) -> None:
        """No-op: the client is shared per process (see ``get_elasticsearch_client``)."""
