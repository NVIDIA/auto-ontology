# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDB initialization — data layer, semantic layer and conversation collections."""

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB

DATA_VDB_COLLECTION: str = "data_objects_layer"
SEMANTIC_VDB_COLLECTION: str = "semantic_layer"

#: Chat transcripts, one vector per question/answer pair.
#:
#: Its own collection rather than a ``label`` inside the data layer. The two
#: hold different kinds of thing and are searched for different reasons --
#: "which table holds invoices" against a catalog, "what did I say about the
#: deadline" against a transcript -- and sharing an index means every tabular
#: query competes with however many turns a conversation happens to have. A
#: hundred-turn chat would outnumber a small catalog's nodes on its own, and
#: the losing side would be the catalog.
CONVERSATION_VDB_COLLECTION: str = "conversation"

#: What a transcript is about, one vector per graph node.
#:
#: Separate from the turn collection because the two are searched against
#: different things. A turn vector carries a transcript's wording, so it matches
#: a question that reuses it; a node vector carries a *concept*, so it matches a
#: question that asks for the thing by another name. Pooling them would also let
#: a hundred turns outvote the dozen nodes that summarise them, which is the
#: retrieval the node route exists to avoid.
CONVERSATION_SEMANTIC_VDB_COLLECTION: str = "conversation_semantic"

#: One vector per edge, embedded as the whole path it spells out:
#: ``source (type) [info] --role--> target (type) [info]``.
#:
#: A third collection rather than more rows in the node one, because a path is
#: not a node and matching it answers a different question. "What is the project
#: built with" names neither endpoint's description and no statement at all --
#: it names the *relation*, and only a vector carrying both ends and the verb
#: between them can match it. Searching nodes for that question returns the
#: project, the framework, or both, and the reader still has to guess what joins
#: them.
#:
#: One hop, hence the name. Two-hop paths are the obvious extension and are
#: quadratic in the node count, so the collection is named for its depth rather
#: than for the idea, leaving room for a sibling rather than a silent change of
#: meaning.
ONE_HOP_PATH_VDB_COLLECTION: str = "1hop_pathes"

VDB_SCHEMA: str = "vdb"


def get_vdb(
    *,
    database_name: str | None = None,
    collection_name: str = DATA_VDB_COLLECTION,
    reset: bool = False,
) -> PostgresVDB:
    """Build a PostgresVDB pointed at the local pgvector-enabled Postgres.

    When database_name is provided, the VDB will use it to reset old embeddings for the given database.
    """
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
) -> PostgresVDB:
    """Backward-compatible alias for :func:`get_vdb` (tabular / data layer)."""
    return get_vdb(database_name=database_name, reset=reset)


def get_semantic_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> PostgresVDB:
    """Build a PostgresVDB for the semantic layer."""
    return get_vdb(
        database_name=database_name,
        collection_name=SEMANTIC_VDB_COLLECTION,
        reset=reset,
    )


def get_conversation_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> PostgresVDB:
    """Build a PostgresVDB for chat transcripts."""
    return get_vdb(
        database_name=database_name,
        collection_name=CONVERSATION_VDB_COLLECTION,
        reset=reset,
    )


def get_conversation_semantic_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> PostgresVDB:
    """Build a PostgresVDB for a transcript's graph nodes."""
    return get_vdb(
        database_name=database_name,
        collection_name=CONVERSATION_SEMANTIC_VDB_COLLECTION,
        reset=reset,
    )


def get_one_hop_path_vdb(
    *, database_name: str | None = None, reset: bool = False
) -> PostgresVDB:
    """Build a PostgresVDB for a transcript's edges, embedded as paths."""
    return get_vdb(
        database_name=database_name,
        collection_name=ONE_HOP_PATH_VDB_COLLECTION,
        reset=reset,
    )
