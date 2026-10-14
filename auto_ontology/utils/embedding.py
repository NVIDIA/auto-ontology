# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text embedding against the configured NIM / inference-api endpoint.

Both endpoints speak the OpenAI ``/embeddings`` API, plus three NVIDIA fields
sent through ``extra_body``. ``input_type`` is the one that matters: NVIDIA's
embedding models are asymmetric, so a stored row must be embedded as a
``passage`` and a search string as a ``query``. Mixing them up raises no error
-- it just makes every search worse.

Ingest and query MUST use the same model; a mismatch produces garbage results
or a dimension error from pgvector.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

import httpx
from openai import APIStatusError, Omit, OpenAI

from auto_ontology.utils.model_config import resolve
from auto_ontology.vdb.records import write_embedded_rows

if TYPE_CHECKING:
    import pandas as pd

    from auto_ontology.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)

InputType = Literal["passage", "query"]

# Each field falls back to DEFAULT_MODELS_<field>.
_EMBED_ENDPOINT = resolve("EMBED", "ENDPOINT")
_EMBED_MODEL = resolve("EMBED", "MODEL")
_EMBED_API_KEY = resolve("EMBED", "API_KEY")

# Per-request retries inside the client, for 408/409/429/5xx and dropped
# connections. They back off for seconds; the chunk-level retry below is the
# slow one, for a rate limit that needs minutes to clear.
_REQUEST_RETRIES = 5

# Interactive writes (an edited column, a saved analysis) run inside a
# request handler, so a dead endpoint has to surface in about a minute: one
# 60s attempt and one retry for a transient blip, instead of the ingest-sized
# 600s and five retries.
_INTERACTIVE_TIMEOUT_S = 60.0
_INTERACTIVE_RETRIES = 1

# Retry policy for a chunk the endpoint refused. Delays are long on purpose:
# the failure being retried is a rate limit, and the quota needs wall-clock time
# to refill — an immediate retry just burns another attempt against a closed
# window. Measured recovery after saturating the endpoint was under 3 minutes,
# which 15s doubling to 120s covers across four attempts.
_RETRY_ATTEMPTS = 4
_RETRY_BASE_SECONDS = 15.0
_RETRY_MAX_SECONDS = 120.0

# Why a chunk came back unembedded, kept on the frame rather than on every row.
_ERROR_ATTR = "embed_error"
# Set when retrying cannot help: the request is wrong (bad key, unknown model,
# wrong path), not the endpoint busy.
_FATAL_ATTR = "embed_error_fatal"


@dataclass(frozen=True)
class EmbedParams:
    """Where and how to embed. The key is kept out of ``repr`` and so out of logs."""

    model_name: str
    embed_invoke_url: str
    api_key: str = field(default="", repr=False)
    embed_modality: str = "text"
    inference_batch_size: int = 32
    nim_http_max_concurrent: int = 32
    request_timeout_s: float = 600.0


def get_embed_params() -> EmbedParams:
    return EmbedParams(
        embed_invoke_url=_EMBED_ENDPOINT,
        model_name=_EMBED_MODEL,
        api_key=_EMBED_API_KEY,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# A keyless self-hosted NIM expects no Authorization header, and the client
# only sends none when the header is omitted per request.
_NO_AUTH = {"Authorization": Omit()}

_clients: dict[tuple[str, str], OpenAI] = {}
_clients_lock = threading.Lock()


def _base_url(endpoint: str) -> str:
    """*endpoint* without a trailing ``/embeddings``, which the client appends.

    Configs carry both forms -- the built-in defaults end at ``/v1``, older
    deployments spell out ``/v1/embeddings`` -- and both have to keep working.
    """
    url = endpoint.rstrip("/")
    return url.removesuffix("/embeddings")


def _new_client(params: EmbedParams, http_client: httpx.Client | None = None) -> OpenAI:
    return OpenAI(
        base_url=_base_url(params.embed_invoke_url),
        # The client refuses to start without a key; a keyless endpoint gets a
        # placeholder here and no header at all, see _NO_AUTH.
        api_key=params.api_key or "unused",
        max_retries=_REQUEST_RETRIES,
        http_client=http_client,
    )


def _client(params: EmbedParams) -> OpenAI:
    """One client per endpoint and key, so its connection pool is shared."""
    key = (_base_url(params.embed_invoke_url), params.api_key)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = _new_client(params)
        return client


def embed_texts(
    texts: list[str],
    params: EmbedParams,
    input_type: InputType,
    *,
    timeout: float | None = None,
    retries: int | None = None,
) -> list[list[float] | None]:
    """Embed *texts*, returning one vector per text in the same order.

    Blank texts are not sent and come back as ``None``. Raises the client's
    error when a request fails after its retries -- an
    :class:`openai.APIStatusError` carries the HTTP status in its message.
    *timeout* (seconds per request) and *retries* (per request) default to the
    ingest-sized ``params.request_timeout_s`` and ``_REQUEST_RETRIES``; a caller
    with its own retry loop passes ``retries=0`` so the two do not multiply.
    """
    stripped = [text.strip() if isinstance(text, str) else "" for text in texts]
    to_send = [i for i, text in enumerate(stripped) if text]
    vectors: list[list[float] | None] = [None] * len(texts)
    if not to_send:
        return vectors

    client = _client(params)
    if retries is not None:
        client = client.with_options(max_retries=retries)
    batches = list(itertools.batched(to_send, max(1, params.inference_batch_size)))

    def _embed_batch(indices: tuple[int, ...]) -> None:
        response = client.embeddings.create(
            model=params.model_name,
            input=[stripped[i] for i in indices],
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": "END",
                "modality": params.embed_modality,
            },
            timeout=timeout or params.request_timeout_s,
            extra_headers=None if params.api_key else _NO_AUTH,
        )
        returned = {item.index for item in response.data}
        if returned != set(range(len(indices))):
            # Placing what came back and leaving the rest None would let the
            # caller persist a partial batch without noticing.
            raise RuntimeError(
                f"Embedding endpoint returned {len(response.data)} vector(s) "
                f"for {len(indices)} input(s)."
            )
        for item in response.data:
            vectors[indices[item.index]] = list(item.embedding)

    if len(batches) == 1:
        _embed_batch(batches[0])
    else:
        workers = min(len(batches), max(1, params.nim_http_max_concurrent))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() so the first failed batch re-raises here.
            list(pool.map(_embed_batch, batches))
    return vectors


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _is_fatal(exc: Exception) -> bool:
    """Whether *exc* says the request itself is wrong, so waiting cannot help.

    Any 4xx except 408 (timeout), 409 and 429 (rate limit): a bad key, an
    unknown model id, a wrong path. The client has already retried the rest.
    """
    return isinstance(exc, APIStatusError) and (
        400 <= exc.status_code < 500 and exc.status_code not in (408, 409, 429)
    )


def _embed_frame(frame: "pd.DataFrame", params: EmbedParams) -> "pd.DataFrame":
    """Embed *frame*'s ``text`` column as passages into ``metadata["embedding"]``.

    Never raises. A failure leaves every row without an embedding and the
    reason in ``frame.attrs``, which is what lets the caller retry the chunk,
    log why it failed, and still write the chunks that worked.
    """
    out = frame.copy()
    # A fresh dict per row: the caller's row dicts are not ours to mutate.
    out["metadata"] = [
        dict(m) if isinstance(m, dict) else {}
        for m in out.get("metadata", [None] * len(out))
    ]
    try:
        vectors = embed_texts(list(out.get("text", [""] * len(out))), params, "passage")
    except Exception as exc:
        out.attrs[_ERROR_ATTR] = f"{type(exc).__name__}: {exc}"
        out.attrs[_FATAL_ATTR] = _is_fatal(exc)
        return out
    for metadata, vector in zip(out["metadata"], vectors):
        if vector is not None:
            metadata["embedding"] = vector
    return out


def embed_chunk_size(params: EmbedParams) -> int:
    """Rows per embed call.

    One chunk is one full wave through the endpoint — ``batch_size`` texts per
    request times ``max_concurrent`` requests in flight — so the HTTP pool stays
    saturated while each chunk still finishes in seconds rather than minutes.
    That cadence is what the per-chunk log line, the streaming write and the
    retry all hang off: smaller chunks mean more frequent progress and less work
    lost to — or redone by — a mid-run failure, larger ones mean less pgvector
    round-tripping.
    """
    return max(1, params.inference_batch_size * params.nim_http_max_concurrent)


def _embedded_row_count(frame: "pd.DataFrame") -> int:
    """Rows in *frame* that came back with a non-empty embedding."""
    return sum(
        1
        for metadata in frame.get("metadata", [])
        if isinstance(metadata, dict) and metadata.get("embedding")
    )


def _failure_reason(frame: "pd.DataFrame | None") -> str:
    """The endpoint's error for a chunk that came back unembedded."""
    if frame is None or frame.empty:
        return "embed returned no rows"
    return (
        frame.attrs.get(_ERROR_ATTR) or "no embeddings returned and no error reported"
    )


def _embed_with_retry(
    embed: "Callable[[pd.DataFrame], pd.DataFrame | None]",
    part: "pd.DataFrame",
    *,
    label: str,
    attempts: int = _RETRY_ATTEMPTS,
    sleep: "Callable[[float], None]" = time.sleep,
) -> "pd.DataFrame | None":
    """Embed *part*, retrying while the endpoint returns nothing usable.

    Retries the whole chunk on top of the client's per-request retries because
    a sustained run exhausts the endpoint's quota for minutes, longer than those
    back off for, and without this the chunk's rows are silently dropped and the
    catalog ends up missing entries no one notices. A request the endpoint
    rejects as wrong -- a bad key, an unknown model -- is not retried: waiting
    does not fix it.

    Returns the embedded frame, or the last empty result once attempts run out;
    the caller decides what to do with a chunk that never succeeded.
    """
    delay = _RETRY_BASE_SECONDS
    embedded = None
    for attempt in range(1, attempts + 1):
        embedded = embed(part)
        if (
            embedded is not None
            and not embedded.empty
            and _embedded_row_count(embedded)
        ):
            if attempt > 1:
                logger.info("%s: recovered on attempt %d/%d", label, attempt, attempts)
            return embedded
        reason = _failure_reason(embedded)
        fatal = embedded is not None and embedded.attrs.get(_FATAL_ATTR, False)
        if attempt == attempts or fatal:
            logger.error(
                "%s: giving up after %d attempt(s) — %s", label, attempt, reason
            )
            return embedded
        logger.warning(
            "%s: attempt %d/%d failed (%s); retrying in %.0fs",
            label,
            attempt,
            attempts,
            reason,
            delay,
        )
        sleep(delay)
        delay = min(delay * 2, _RETRY_MAX_SECONDS)
    return embedded


def batch_embed_chunks(
    rows: "list[dict] | pd.DataFrame",
    params: EmbedParams,
    *,
    label: str = "",
    chunk_size: int | None = None,
) -> "Iterator[pd.DataFrame]":
    """Embed *rows* chunk by chunk, yielding each chunk as it comes back.

    Generating instead of returning is what lets the caller write each chunk to
    the VDB while the next one is still embedding — see
    :func:`auto_ontology.ingestion_service.ingest.run_ingest`. Yields nothing when *rows*
    is empty; the caller decides whether that is an error.
    """
    import pandas as pd

    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    total = len(frame)
    if total == 0:
        return

    size = chunk_size or embed_chunk_size(params)
    chunks = math.ceil(total / size)
    prefix = f"Embed{f' {label}' if label else ''}"
    logger.info(
        "%s: %d row(s) in %d chunk(s) of %d via %s (model %s, batch %s, "
        "concurrency %s)",
        prefix,
        total,
        chunks,
        size,
        params.embed_invoke_url,
        params.model_name,
        params.inference_batch_size,
        params.nim_http_max_concurrent,
    )

    started = time.monotonic()
    done = 0
    embedded_total = 0

    for index in range(chunks):
        part = frame.iloc[index * size : (index + 1) * size]
        chunk_label = f"{prefix} chunk {index + 1}/{chunks}"
        chunk_started = time.monotonic()
        embedded = _embed_with_retry(
            lambda p: _embed_frame(p, params), part, label=chunk_label
        )
        chunk_elapsed = time.monotonic() - chunk_started

        if embedded is None or embedded.empty:
            logger.warning(
                "%s: embed returned nothing for %d row(s) in %.1fs",
                chunk_label,
                len(part),
                chunk_elapsed,
            )
            continue

        with_embeddings = _embedded_row_count(embedded)
        done += len(part)
        embedded_total += with_embeddings
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        log = logger.info if with_embeddings else logger.warning
        log(
            "%s chunk %d/%d: %d/%d row(s) embedded in %.1fs "
            "(%d/%d done, %.0f rows/s, ~%.0fs left)",
            prefix,
            index + 1,
            chunks,
            with_embeddings,
            len(part),
            chunk_elapsed,
            done,
            total,
            rate,
            (total - done) / rate if rate > 0 else 0.0,
        )
        yield embedded

    logger.info(
        "%s: %d/%d row(s) embedded in %.1fs",
        prefix,
        embedded_total,
        total,
        time.monotonic() - started,
    )


def batch_embed(
    rows: "list[dict] | pd.DataFrame",
    params: EmbedParams,
) -> "pd.DataFrame":
    """Embed *rows* and return one DataFrame with the embeddings.

    A concatenating wrapper over :func:`batch_embed_chunks`, for the callers
    that want the whole result in hand. Prefer the generator when the rows are
    going straight into a VDB — it bounds peak memory and starts persisting
    before the last chunk is embedded.

    Returns an empty DataFrame when *rows* is empty or the embed step yields
    nothing; the caller decides whether that is an error.
    """
    import pandas as pd

    frames = list(batch_embed_chunks(rows, params))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def embed_rows(
    rows: list[dict[str, Any]],
    params: EmbedParams,
    *,
    what: str,
) -> list[dict[str, Any]]:
    """Embed already-built rows as passages, all or nothing.

    For the small, interactive writes -- one analysis, one attribute, one
    edited column -- where a failure should reach the caller instead of being
    retried for minutes the way an ingest chunk is: one 60s attempt and one
    retry, so a dead endpoint fails the request in about two minutes rather
    than an hour. Returns copies of the rows
    that came back with an embedding, ready for
    :func:`~auto_ontology.vdb.records.write_embedded_rows`. Raises
    ``RuntimeError`` when the endpoint fails or returns no embeddings; *what*
    names the rows in that message.
    """
    if not rows:
        return []
    try:
        vectors = embed_texts(
            [row.get("text") or "" for row in rows],
            params,
            "passage",
            timeout=_INTERACTIVE_TIMEOUT_S,
            retries=_INTERACTIVE_RETRIES,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding {len(rows)} {what} row(s) via {params.embed_invoke_url} "
            f"failed: {exc}"
        ) from exc

    embedded = [
        {**row, "metadata": {**(row.get("metadata") or {}), "embedding": vector}}
        for row, vector in zip(rows, vectors)
        if vector
    ]
    if not embedded:
        raise RuntimeError(
            f"Embedding step produced 0/{len(rows)} {what} rows with embeddings."
        )
    return embedded


def embed_rows_into_vdb(
    rows: list[dict[str, Any]],
    params: EmbedParams,
    vdb: "PostgresVDB",
    *,
    what: str,
) -> int:
    """:func:`embed_rows`, then append the result to *vdb*.

    Returns the number of rows written.
    """
    if not rows:
        return 0
    before = time.time()
    embedded = embed_rows(rows, params, what=what)
    written = write_embedded_rows(vdb, embedded)
    logger.info(
        "Embedded and appended %d/%d %s row(s) via %s in %.2fs.",
        len(embedded),
        len(rows),
        what,
        type(vdb).__name__,
        time.time() - before,
    )
    return written


def embed_docs_into_vdb(
    docs: list[dict],
    embed_params: EmbedParams,
    vdb: "PostgresVDB",
    database_name: str | None = None,
) -> int:
    """Embed *docs* and upsert them into *vdb*.

    Each doc must have at least ``id``, ``name``, ``label``, and ``text`` keys
    (the shape returned by ``fetch_sql_attribute_docs`` and
    ``fetch_suggested_sql_attribute_docs``). An optional ``source`` key is
    carried through to the metadata unchanged (empty string when absent) —
    used by SqlAttribute docs so retrieval hits know their provenance
    (``"sql"``, ``"bridgeTable"``, …) without an extra lookup.

    Returns the number of rows successfully embedded and ingested.
    Raises ``RuntimeError`` when the embedding call produces zero embedded rows
    so the caller can decide how to handle the failure.
    """
    if not docs:
        return 0

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = (
            f"auto_ontology:{node_id}"
            if node_id is not None
            else "auto_ontology:unknown"
        )
        tabular_fields = {
            "id": node_id,
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "source_path": path,
            "database_name": database_name,
            "source": item.get("source", ""),
        }
        rows.append(
            {
                "text": (item.get("text") or "").strip(),
                "path": path,
                "page_number": -1,
                "metadata": {
                    **tabular_fields,
                    "content_metadata": dict(tabular_fields),
                },
            }
        )
    label = str(docs[0].get("label") or "document")
    return embed_rows_into_vdb(rows, embed_params, vdb, what=label)
