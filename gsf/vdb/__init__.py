# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VDB initialization and configuration.
"""

from gsf.vdb.config import get_postgres_connection_string
from gsf.vdb.postgres import PostgresVDB

VDB_COLLECTION: str = "nv_ingest_tabular"
VDB_SCHEMA: str = "vdb"


def get_vdb(*, database_name: str | None = None) -> PostgresVDB:
    """Build a PostgresVDB pointed at the local pgvector-enabled Postgres.

    When database_name is provided, the VDB will use it to reset old embeddings for the given database.
    """
    kwargs: dict = {
        "connection_string": get_postgres_connection_string(),
        "collection_name": VDB_COLLECTION,
        "schema_name": VDB_SCHEMA,
    }
    if database_name:
        kwargs["database_name"] = database_name
    return PostgresVDB(**kwargs)
