# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j-backed schema / node lookups.

Builds the :class:`Schema` objects consumed by the SQL parser plus generic
``get_item_by_id`` helpers.

All direct Neo4j calls live in gsf/neo4j/datasources.py.
This module only keeps the pure-Python ``get_schemas_by_ids`` that assembles
pandas DataFrames into :class:`Schema` objects.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.ingestion.model.schema import Schema

from gsf.neo4j.datasources import (
    get_all_schemas_ids,
    get_item_by_id,
    get_schemas_from_graph_by_ids,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_all_schemas_ids",
    "get_item_by_id",
    "get_schemas_by_ids",
]


def get_schemas_by_ids(relevant_schemas_ids: list | None = None) -> dict:
    """Assemble :class:`Schema` objects from Neo4j catalog data.

    Fetches raw column/table rows via :func:`gsf.neo4j.datasources.get_schemas_from_graph_by_ids`
    then builds the in-memory :class:`Schema` map consumed by the SQL parser.
    """
    before_get_all = time.time()
    data_array = get_schemas_from_graph_by_ids(relevant_schemas_ids)
    logger.info(f"time took to get all data from graph: {time.time() - before_get_all}")
    data_df = pd.DataFrame(data_array)
    dbs = list(data_df["database_name"].unique())

    schemas = data_df[["database_name", "table_schema"]]
    schemas = schemas.drop_duplicates().to_dict(orient="records")

    all_schemas = {}
    schema_dfs = {}
    dbs_nodes = {}
    for database_name in dbs:
        database_node = Neo4jNode(
            name=database_name, label=Labels.DB, props={"name": database_name}
        )
        dbs_nodes[database_name] = database_node

    tables_df = data_df[
        ["database_name", "table_schema", "table_name", "table_id"]
    ].drop_duplicates(subset=["database_name", "table_schema", "table_name"])
    tables_df = tables_df.rename(columns={"table_id": "id"})

    unique_schemas = data_df.table_schema.unique()
    for table_schema in unique_schemas:
        schema_tables_df = tables_df.loc[tables_df["table_schema"] == table_schema]
        schema_dfs[table_schema] = {
            "tables": schema_tables_df.to_dict(orient="records")
        }

    for table_schema in unique_schemas:
        columns_df = data_df.loc[data_df["table_schema"] == table_schema].rename(
            columns={"column_id": "id"}
        )
        schema_dfs[table_schema]["columns"] = columns_df.to_dict(orient="records")

    before_modify_all = time.time()
    for schema in schemas:
        table_schema: str = schema.get("table_schema")
        if not table_schema:
            continue

        schema_database_name: str = schema["database_name"]
        schema_database_node = dbs_nodes[schema_database_name]
        tables_df = pd.DataFrame(schema_dfs[table_schema]["tables"])
        columns_df = pd.DataFrame(schema_dfs[table_schema]["columns"])

        all_schemas[table_schema.lower()] = Schema(
            schema_database_node,
            tables_df,
            columns_df,
            table_schema,
            is_creation_mode=False,
        )
    logger.info(
        f"total time it took to create all schemas nodes: {time.time() - before_modify_all}"
    )
    logger.info(f"total time for get_schemas_by_ids(): {time.time() - before_get_all}")
    return all_schemas
