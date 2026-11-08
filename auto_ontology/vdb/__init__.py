# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDB initialization — data layer and semantic layer collections.

``VDB_BACKEND`` picks the store: ``postgres`` (pgvector, the default) or
``elasticsearch`` (``semantic_text`` over an NVIDIA inference endpoint). Both
implement :class:`~auto_ontology.vdb.base.VectorStore`. ``VDB_SEARCH_MODE=hybrid``
(Elasticsearch only) fuses BM25 with the vector search for identifier-shaped
queries; ``HYBRID_RRF_RANK_CONSTANT`` and ``HYBRID_RRF_RANK_WINDOW`` tune the
fusion.
"""

import os

from auto_ontology.infra.elasticsearch import (
    get_elasticsearch_client,
    get_elasticsearch_config,
    get_embed_endpoint,
)
from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.backend import vdb_backend, vdb_search_mode
from auto_ontology.vdb.base import VectorStore
from auto_ontology.vdb.elasticsearch import (
    DEFAULT_RANK_CONSTANT,
    DEFAULT_RANK_WINDOW,
    ElasticsearchVDB,
)
from auto_ontology.vdb.postgres import PostgresVDB

DATA_VDB_COLLECTION: str = "data_objects_layer"
SEMANTIC_VDB_COLLECTION: str = "semantic_layer"
VDB_SCHEMA: str = "vdb"


def get_vdb(
    *,
    database_name: str | None = None,
    collection_name: str = DATA_VDB_COLLECTION,
    reset: bool = False,
) -> VectorStore:
    """Build the configured store for *collection_name*.

    When database_name is provided, the VDB will use it to reset old embeddings for the given database.
    """
    # Checked on every backend, so that hybrid on Postgres fails here.
    search_mode = vdb_search_mode()
    if vdb_backend() == "elasticsearch":
        config = get_elasticsearch_config()
        return ElasticsearchVDB(
            client=get_elasticsearch_client(config),
            index_name=f"{config.index_prefix}-{collection_name}",
            endpoint=get_embed_endpoint(config.inference_id),
            database_name=database_name,
            reset=reset,
            search_mode=search_mode,
            rank_constant=int(
                os.environ.get("HYBRID_RRF_RANK_CONSTANT") or DEFAULT_RANK_CONSTANT
            ),
            rank_window=int(
                os.environ.get("HYBRID_RRF_RANK_WINDOW") or DEFAULT_RANK_WINDOW
            ),
        )

    connection_string = get_postgres_connection_string()
    kwargs: dict = {
        "connection_string": connection_string,
        "collection_name": collection_name,
        "schema_name": VDB_SCHEMA,
    }
    if database_name:
        kwargs["database_name"] = database_name
    return PostgresVDB(reset=reset, **kwargs)


def get_data_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> VectorStore:
    """Backward-compatible alias for :func:`get_vdb` (tabular / data layer)."""
    return get_vdb(database_name=database_name, reset=reset)


def get_semantic_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> VectorStore:
    """Build the configured store for the semantic layer."""
    return get_vdb(
        database_name=database_name,
        collection_name=SEMANTIC_VDB_COLLECTION,
        reset=reset,
    )
