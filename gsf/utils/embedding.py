# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM text-embedding config shared by ingest pipelines and the retriever."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


def batch_embed(
    rows: "list[dict] | pd.DataFrame",
    params: EmbedParams,
) -> "pd.DataFrame":
    """Embed *rows* in one batch and return the DataFrame with embeddings.

    **This is the only place GSF touches ``_BatchEmbedActor``.** It is a private
    library symbol, so the dependency is deliberately confined to the two lines
    below rather than repeated at every call site — see
    ``docs/refactor/drop-neo4j/PLAN.md`` § Phase 1.

    ``_BatchEmbedActor`` is an *archetype* operator: it resolves to a CPU or GPU
    variant only when a :class:`~nemo_retriever.graph.Graph` executes it, so it
    cannot be called directly the way ``TabularFetchEmbeddingsOp`` and
    ``IngestVdbOperator`` can. Hence the one-node graph.

    Returns an empty DataFrame when *rows* is empty or the embed step yields
    nothing; the caller decides whether that is an error.
    """
    import pandas as pd

    from nemo_retriever.graph import Graph
    from nemo_retriever.operators.embed.operators import _BatchEmbedActor

    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()

    results = (Graph() >> _BatchEmbedActor(params=params)).execute(frame)
    embedded = results[0] if results else None
    if embedded is None:
        return pd.DataFrame()
    return embedded


def embed_docs_into_vdb(
    docs: list[dict],
    embed_params: "EmbedParams",
    vdb: "VDB",
    database_name: str | None = None,
) -> int:
    """Embed *docs* and upsert them into *vdb*.

    Each doc must have at least ``id``, ``name``, ``label``, and ``text`` keys
    (the shape returned by ``fetch_sql_attribute_docs`` and
    ``fetch_suggested_sql_attribute_docs``).

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
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
        tabular_fields = {
            "id": node_id,
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "source_path": path,
            "database_name": database_name,
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
