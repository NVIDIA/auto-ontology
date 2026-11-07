# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-derived Elasticsearch connection config and the embedding endpoint.

The client is shared per process: it pools its own connections, and the
services ask for a store on every request.

The inference endpoint is Elasticsearch's ``nvidia`` service, built from the
same ``EMBED_*`` triplet the app uses when it embeds for itself, so switching
``VDB_BACKEND`` does not mean configuring the model twice.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

from elasticsearch import BadRequestError, Elasticsearch, NotFoundError

logger = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ElasticsearchConfig:
    """Where the cluster is and how to authenticate. Secrets stay out of ``repr``."""

    url: str
    api_key: str = field(default="", repr=False)
    username: str = ""
    password: str = field(default="", repr=False)
    verify_certs: bool = True
    ca_certs: str | None = None
    #: Prepended to each collection name to form the index name.
    index_prefix: str = "auto-ontology"
    inference_id: str = "auto-ontology-embed"


def get_elasticsearch_config() -> ElasticsearchConfig:
    """Build the config from ``ES_*`` env vars; ``ES_URL`` is required."""
    url = os.environ.get("ES_URL", "").strip()
    if not url:
        raise ValueError("VDB_BACKEND=elasticsearch requires ES_URL.")
    return ElasticsearchConfig(
        url=url,
        api_key=os.environ.get("ES_API_KEY", ""),
        username=os.environ.get("ES_USERNAME", ""),
        password=os.environ.get("ES_PASSWORD", ""),
        verify_certs=os.environ.get("ES_VERIFY_CERTS", "true").strip().lower() in _TRUE,
        ca_certs=os.environ.get("ES_CA_CERTS") or None,
        index_prefix=os.environ.get("ES_INDEX_PREFIX", "") or "auto-ontology",
        inference_id=os.environ.get("ES_INFERENCE_ID", "") or "auto-ontology-embed",
    )


_clients: dict[ElasticsearchConfig, Elasticsearch] = {}
_clients_lock = threading.Lock()


def get_elasticsearch_client(config: ElasticsearchConfig) -> Elasticsearch:
    """One client per config, so its connection pool is shared."""
    with _clients_lock:
        client = _clients.get(config)
        if client is None:
            auth: dict = {}
            if config.api_key:
                auth["api_key"] = config.api_key
            elif config.username:
                auth["basic_auth"] = (config.username, config.password)
            client = _clients[config] = Elasticsearch(
                config.url,
                verify_certs=config.verify_certs,
                ca_certs=config.ca_certs,
                # A self-signed local cluster is the only reason to turn
                # verification off, and it is deliberate there.
                ssl_show_warn=config.verify_certs,
                retry_on_timeout=True,
                **auth,
            )
        return client


@dataclass(frozen=True)
class EmbedEndpoint:
    """The ``nvidia`` inference endpoint Elasticsearch embeds through."""

    inference_id: str
    model_id: str
    api_key: str = field(repr=False)
    #: Full ``.../embeddings`` URL. The service's own default is
    #: build.nvidia.com's, which is what an ``nvapi-`` key expects.
    url: str | None = None
    requests_per_minute: int | None = None


def _embeddings_url(endpoint: str) -> str:
    """*endpoint* ending in ``/embeddings``, which the ``nvidia`` service needs.

    The app's ``EMBED_ENDPOINT`` is a base URL (``.../v1``) the OpenAI client
    appends to; Elasticsearch takes the full path.
    """
    return endpoint.rstrip("/").removesuffix("/embeddings") + "/embeddings"


def get_embed_endpoint(inference_id: str) -> EmbedEndpoint:
    """The endpoint for this process's ``EMBED_*`` triplet.

    ``ES_EMBED_URL`` overrides the URL derived from ``EMBED_ENDPOINT``: the two
    differ when Elasticsearch reaches the model by a different route than the
    app does (a NIM on a private network, or the fake embedder in tests).
    """
    # Imported here: auto_ontology.utils imports the retriever, which imports
    # auto_ontology.vdb, which imports this module.
    from auto_ontology.utils.embedding import _TEXT_PREFIXED_MODELS
    from auto_ontology.utils.model_config import resolve

    model_id = resolve("EMBED", "MODEL")
    if any(model in model_id for model in _TEXT_PREFIXED_MODELS):
        # Such a deployment ignores input_type and needs a different text
        # prefix per side; semantic_text sends the text unchanged, so every
        # vector would share one direction and ranking would be near random.
        raise ValueError(
            f"EMBED_MODEL={model_id!r} needs a query/passage text prefix, which "
            "Elasticsearch cannot add. Use a model that honours input_type "
            "(e.g. nvidia/nemotron-3.5-embed-1b on build.nvidia.com) or "
            "VDB_BACKEND=postgres."
        )
    rpm = os.environ.get("ES_EMBED_RATE_LIMIT_RPM", "").strip()
    url = os.environ.get("ES_EMBED_URL", "").strip() or _embeddings_url(
        resolve("EMBED", "ENDPOINT")
    )
    return EmbedEndpoint(
        inference_id=inference_id,
        model_id=model_id,
        api_key=resolve("EMBED", "API_KEY"),
        url=url,
        requests_per_minute=int(rpm) if rpm else None,
    )


#: Similarities under which ``_score = (1 + cos) / 2``, the relation
#: ``ElasticsearchVDB`` inverts to report cosine distance. ``dot_product`` is
#: one because NVIDIA's embedding models return unit-length vectors, for which
#: the dot product *is* the cosine. ``l2_norm`` scores ``1 / (1 + d^2)`` and is
#: not.
_COSINE_SCORED = ("cosine", "dot_product")


def _check_similarity(endpoint: EmbedEndpoint, settings: dict) -> None:
    similarity = settings.get("similarity")
    if similarity not in _COSINE_SCORED:
        raise RuntimeError(
            f"Inference endpoint {endpoint.inference_id!r} scores by "
            f"{similarity!r}; distances are only meaningful under "
            f"{' or '.join(_COSINE_SCORED)}."
        )


def ensure_embed_endpoint(client: Elasticsearch, endpoint: EmbedEndpoint) -> None:
    """Create *endpoint* unless it exists; refuse one that points elsewhere.

    An existing endpoint with a different model is an error rather than
    something to overwrite: every row already indexed through it was embedded
    by the old model, and mixing models produces garbage rankings without
    raising anything. The fix is to delete the endpoint and its indices and
    re-ingest.
    """
    try:
        existing = client.inference.get(
            task_type="text_embedding", inference_id=endpoint.inference_id
        )
    except NotFoundError:
        existing = None

    if existing is not None:
        settings = (existing.get("endpoints") or [{}])[0].get("service_settings", {})
        if settings.get("model_id") != endpoint.model_id:
            raise RuntimeError(
                f"Inference endpoint {endpoint.inference_id!r} embeds with "
                f"{settings.get('model_id')!r}, not EMBED_MODEL="
                f"{endpoint.model_id!r}. Delete the endpoint and the "
                "auto-ontology indices, then re-ingest, or set ES_INFERENCE_ID "
                "to a new id."
            )
        _check_similarity(endpoint, settings)
        return

    # No ``similarity``: the service picks one (``cosine`` on Serverless,
    # ``dot_product`` on 9.5), and on 9.5 setting it fails with a server-side
    # NullPointerException. Whichever it picks is checked below.
    service_settings: dict = {
        "model_id": endpoint.model_id,
        "api_key": endpoint.api_key,
    }
    if endpoint.url:
        service_settings["url"] = endpoint.url
    if endpoint.requests_per_minute:
        service_settings["rate_limit"] = {
            "requests_per_minute": endpoint.requests_per_minute
        }
    try:
        created = client.inference.put(
            task_type="text_embedding",
            inference_id=endpoint.inference_id,
            body={
                "service": "nvidia",
                "service_settings": service_settings,
                # The app sent truncate=END when it embedded for itself.
                "task_settings": {"truncate": "END"},
                # One vector per row, as the Postgres store keeps. The rows are
                # short descriptions; chunking them would only change scores.
                "chunking_settings": {"strategy": "none"},
            },
        )
    except BadRequestError as exc:
        # Another process created it between the GET and the PUT, and checked
        # it.
        if "already exists" not in str(exc):
            raise
        return
    _check_similarity(endpoint, created.get("service_settings") or {})
    logger.info(
        "Created Elasticsearch inference endpoint %s (nvidia, %s)",
        endpoint.inference_id,
        endpoint.model_id,
    )
