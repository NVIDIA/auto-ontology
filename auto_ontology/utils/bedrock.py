# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon Bedrock, shared by the chat, embedding, and rerank clients.

A model configured as ``bedrock/<id>`` is called natively on Bedrock and signed
with the standard AWS credential chain (``AWS_PROFILE``, ``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY``, SSO, an instance or pod role), so its triplet needs no
API key and no endpoint. The id is a Bedrock model or inference-profile id, or a
model ARN. Requests go to ``AWS_REGION``, else ``AWS_DEFAULT_REGION``, else the
profile's region. An ARN names its own region instead, for a model Bedrock does
not offer in that one: reranking is offered in fewer regions than chat and
embedding models.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import threading
from typing import Any, Literal

PREFIX = "bedrock/"

# Cohere takes up to 96 texts per request; Titan embeds one text per request.
_COHERE_MAX_TEXTS = 96
# The widest vector each model family offers.
_COHERE_V4_DIMENSIONS = 1536
_TITAN_V2_DIMENSIONS = 1024

# Bedrock rejects a rerank request with more sources than this, and an empty or
# longer document.
_RERANK_MAX_SOURCES = 1000
_RERANK_MAX_CHARS = 32000


class UnsupportedModelError(ValueError):
    """The configured Bedrock model cannot serve this request; retrying cannot help."""


def is_bedrock_model(model: str) -> bool:
    """Whether *model* is served by Amazon Bedrock."""
    return model.startswith(PREFIX)


def model_id(model: str) -> str:
    """The Bedrock model id, inference-profile id, or ARN in *model*."""
    return model.removeprefix(PREFIX)


def arn_region(model: str) -> str | None:
    """The region an ARN names; ``None`` for an id, which uses the configured one."""
    parts = model_id(model).split(":")
    if parts[0] != "arn" or len(parts) < 4:
        return None
    return parts[3] or None


def http_status(exc: BaseException) -> int | None:
    """The HTTP status of a failed Bedrock call; ``None`` for any other error."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status if isinstance(status, int) else None


_clients: dict[tuple[str, str | None, float, int], Any] = {}
_clients_lock = threading.Lock()


def _region(region: str | None) -> str | None:
    """*region*, else ``AWS_REGION``, else ``AWS_DEFAULT_REGION``.

    ``None`` leaves the choice to boto3, which reads the profile's region.
    botocore reads ``AWS_DEFAULT_REGION`` and the profile but never
    ``AWS_REGION``, which langchain-aws (the chat client) reads first: resolving
    it here keeps the embedding and rerank calls in the chat model's region.
    """
    return (
        region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or None
    )


def _client(service: str, region: str | None, *, timeout: float, attempts: int) -> Any:
    """One boto3 client per service, region, and retry policy, shared by threads."""
    region = _region(region)
    key = (service, region, timeout, attempts)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            import boto3
            from botocore.config import Config

            client = _clients[key] = boto3.session.Session().client(
                service,
                region_name=region,
                config=Config(
                    read_timeout=timeout,
                    retries={"mode": "standard", "total_max_attempts": attempts},
                ),
            )
        return client


def _invoke(client: Any, model: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.invoke_model(
        modelId=model,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())


def _cohere(
    client: Any, model: str, texts: list[str], input_type: Literal["passage", "query"]
) -> list[list[float]]:
    body: dict[str, Any] = {
        "texts": texts,
        # Cohere embeds a query and the passages it should find differently.
        "input_type": "search_query" if input_type == "query" else "search_document",
        "embedding_types": ["float"],
    }
    if "embed-v4" in model:
        body["output_dimension"] = _COHERE_V4_DIMENSIONS
    vectors = _invoke(client, model, body)["embeddings"]
    # Keyed by embedding type when embedding_types is sent; a bare list otherwise.
    vectors = vectors["float"] if isinstance(vectors, dict) else vectors
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Bedrock returned {len(vectors)} vector(s) for {len(texts)} input(s)."
        )
    return vectors


def _titan(client: Any, model: str, text: str) -> list[float]:
    body: dict[str, Any] = {"inputText": text}
    if "embed-text-v2" in model:
        body |= {"dimensions": _TITAN_V2_DIMENSIONS, "normalize": True}
    return _invoke(client, model, body)["embedding"]


def embed(
    texts: list[str],
    model: str,
    input_type: Literal["passage", "query"],
    *,
    timeout: float,
    retries: int,
) -> list[list[float]]:
    """One vector per text, in order, from a Cohere or Titan embedding model.

    *retries* is per request, on throttling, 5xx, and dropped connections.
    """
    resolved = model_id(model)
    is_cohere = "cohere.embed" in resolved
    if not is_cohere and "titan-embed-text" not in resolved:
        raise UnsupportedModelError(
            f"Bedrock embedding model {resolved!r} is not supported; use a Cohere "
            "(cohere.embed-*) or Titan text (amazon.titan-embed-text-*) model."
        )
    client = _client(
        "bedrock-runtime", arn_region(model), timeout=timeout, attempts=retries + 1
    )
    if is_cohere:
        return [
            vector
            for batch in itertools.batched(texts, _COHERE_MAX_TEXTS)
            for vector in _cohere(client, resolved, list(batch), input_type)
        ]
    return [_titan(client, resolved, text) for text in texts]


def rerank_scores(
    query: str, documents: list[str], model: str, *, timeout: float
) -> list[float]:
    """One relevance score per document, higher is better; ``-inf`` for a blank one.

    Calls Bedrock's Rerank API once, without retrying: the caller picks the
    fallback when it fails.
    """
    resolved = model_id(model)
    client = _client(
        "bedrock-agent-runtime", arn_region(model), timeout=timeout, attempts=1
    )
    arn = (
        resolved
        if resolved.startswith("arn:")
        else f"arn:aws:bedrock:{client.meta.region_name}::foundation-model/{resolved}"
    )
    scores = [-math.inf] * len(documents)
    sendable = [i for i, document in enumerate(documents) if document.strip()]
    for chunk in itertools.batched(sendable, _RERANK_MAX_SOURCES):
        response = client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": documents[i][:_RERANK_MAX_CHARS]},
                    },
                }
                for i in chunk
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": len(chunk),
                    "modelConfiguration": {"modelArn": arn},
                },
            },
        )
        for result in response["results"]:
            scores[chunk[result["index"]]] = float(result["relevanceScore"])
    return scores
