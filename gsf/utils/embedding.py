# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM text-embedding config shared by ingest pipelines and the retriever."""

from __future__ import annotations

import logging
import math

import time
from typing import TYPE_CHECKING, Callable, Iterator

from nemo_retriever.common.params.models import EmbedParams

from gsf.utils.model_config import resolve

if TYPE_CHECKING:
    import pandas as pd

    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

# Remote NIM embedding endpoint — no local GPU required.
# MUST match the model used at ingest time; a mismatch produces garbage results
# or a dimension error from pgvector. Each field falls back to DEFAULT_MODELS_<field>.
_EMBED_ENDPOINT = resolve("EMBED", "ENDPOINT")
_EMBED_MODEL = resolve("EMBED", "MODEL")
_EMBED_API_KEY = resolve("EMBED", "API_KEY")

# Retry policy for a chunk the endpoint refused. Delays are long on purpose:
# the failure being retried is a rate limit, and the quota needs wall-clock time
# to refill — an immediate retry just burns another attempt against a closed
# window. Measured recovery after saturating the endpoint was under 3 minutes,
# which 15s doubling to 120s covers across four attempts.
_RETRY_ATTEMPTS = 4
_RETRY_BASE_SECONDS = 15.0
_RETRY_MAX_SECONDS = 120.0


def get_embed_kwargs() -> dict[str, str]:
    """Keyword args for ``Retriever`` embed configuration."""
    return {
        "model_name": _EMBED_MODEL,
        "embed_invoke_url": _EMBED_ENDPOINT,
        "api_key": _EMBED_API_KEY,
    }


def get_embed_params() -> EmbedParams:
    return EmbedParams(
        embed_invoke_url=_EMBED_ENDPOINT,
        model_name=_EMBED_MODEL,
        api_key=_EMBED_API_KEY,
        embed_modality="text",
    )


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
    batch = getattr(params, "inference_batch_size", None) or 32
    concurrent = getattr(params, "nim_http_max_concurrent", None) or 32
    return max(1, int(batch) * int(concurrent))


def _embedded_row_count(frame: "pd.DataFrame") -> int:
    """Rows in *frame* that came back with a non-empty embedding.

    The embed step never raises: ``embed_text_main_text_embed`` catches its own
    exceptions and returns every row with ``{"embedding": [], "error": ...}``.
    So "did this work?" is only answerable by counting, which is why every
    caller here does.
    """
    return sum(
        1
        for metadata in frame.get("metadata", [])
        if isinstance(metadata, dict) and metadata.get("embedding")
    )


def _failure_reason(frame: "pd.DataFrame | None") -> str:
    """The endpoint's error for a chunk that came back unembedded.

    ``embed_text_main_text_embed`` swallows the exception and stores its string
    on every row instead of raising, so this is the only place the underlying
    cause — a 429, a dropped connection — is still readable. Without it the logs
    say a chunk failed but never why.
    """
    if frame is None or frame.empty:
        return "embed returned no rows"
    for column in ("text_embeddings_1b_v2", "text_embeddings"):
        if column not in frame:
            continue
        for payload in frame[column]:
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])
    return "no embeddings returned and no error reported"


def _embed_with_retry(
    embed: "Callable[[pd.DataFrame], pd.DataFrame | None]",
    part: "pd.DataFrame",
    *,
    label: str,
    attempts: int = _RETRY_ATTEMPTS,
    sleep: "Callable[[float], None]" = time.sleep,
) -> "pd.DataFrame | None":
    """Embed *part*, retrying while the endpoint returns nothing usable.

    Retries the whole chunk rather than individual requests because that is the
    only granularity available: the library catches per-frame, so one refused
    request zeroes every row it was given. Retrying is what makes a rate limit
    survivable — the endpoint returns 429 once a sustained run exhausts its
    quota, and without this those rows are silently dropped and the catalog
    ends up missing entries no one notices.

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
        if attempt == attempts:
            logger.error(
                "%s: giving up after %d attempts — %s", label, attempts, reason
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

    **This is the only place GSF touches ``_BatchEmbedActor``.** It is a private
    library symbol, so the dependency is deliberately confined to the one import
    below rather than repeated at every call site.

    ``_BatchEmbedActor`` is an *archetype* operator: it resolves to a CPU or GPU
    variant on first use and caches that variant on the instance. Constructing
    it once and calling it per chunk therefore pays the variant's endpoint probe
    (a 5s timeout against cloud endpoints, which have no ``/v1/health/ready``)
    once per ingest. Running each chunk through its own
    :class:`~nemo_retriever.graph.Graph` would not: ``Graph.execute`` re-resolves
    the archetype every call, so the probe would run once per chunk.

    Generating instead of returning is what lets the caller write each chunk to
    the VDB while the next one is still embedding — see
    :func:`gsf.ingestion_service.ingest.run_ingest`. Yields nothing when *rows*
    is empty; the caller decides whether that is an error.
    """
    import pandas as pd

    from nemo_retriever.operators.embed.operators import _BatchEmbedActor

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
        getattr(params, "embed_invoke_url", None)
        or getattr(params, "embedding_endpoint", None),
        getattr(params, "model_name", None),
        getattr(params, "inference_batch_size", None),
        getattr(params, "nim_http_max_concurrent", None),
    )

    actor = _BatchEmbedActor(params=params)
    started = time.monotonic()
    done = 0
    embedded_total = 0

    for index in range(chunks):
        part = frame.iloc[index * size : (index + 1) * size]
        chunk_label = f"{prefix} chunk {index + 1}/{chunks}"
        chunk_started = time.monotonic()
        embedded = _embed_with_retry(actor, part, label=chunk_label)
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


def embed_docs_into_vdb(
    docs: list[dict],
    embed_params: "EmbedParams",
    vdb: "VDB",
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
    import time

    import pandas as pd

    from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
    from nemo_retriever.operators.vdb import IngestVdbOperator

    if not docs:
        return 0

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = f"gsf:{node_id}" if node_id is not None else "gsf:unknown"
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
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {
                    **tabular_fields,
                    "content_metadata": dict(tabular_fields),
                },
            }
        )

    before = time.time()
    embedded = embed_text_main_text_embed(
        pd.DataFrame(rows),
        model_name=embed_params.model_name,
        embed_invoke_url=embed_params.embed_invoke_url,
        api_key=embed_params.api_key,
        embed_modality=embed_params.embed_modality,
    )

    with_embeddings = [
        r
        for r in embedded.to_dict(orient="records")
        if (r.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        raise RuntimeError(
            f"Embedding step produced 0/{len(embedded)} rows with embeddings; "
            f"check upstream embed errors (often a transient "
            f"{embed_params.embed_invoke_url} 5xx)."
        )

    IngestVdbOperator(vdb=vdb)(with_embeddings)
    logger.info(
        "Embedded %d/%d row(s) via %s in %.2fs.",
        len(with_embeddings),
        len(embedded),
        type(vdb).__name__,
        time.time() - before,
    )
    return len(with_embeddings)
