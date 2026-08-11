# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn tables/columns DataFrames into Schema aggregates.

Forked verbatim in Phase 1 of the drop-Neo4j refactor
(``docs/refactor/drop-neo4j/PLAN.md``) from::

    nemo_retriever.tabular_data.ingestion.parsers.schemas_parser

(NeMo-Retriever, Apache-2.0). Behaviour is unchanged; only imports and the
``Neo4jNode`` -> ``CatalogNode`` rename differ.
"""

from datetime import datetime, timezone

from gsf.catalog.constants import Labels
from gsf.catalog.model.node import CatalogNode
from gsf.catalog.model.schema import Schema
from gsf.catalog.store.schemas import get_table_ids, get_column_ids
import logging

logger = logging.getLogger(__name__)


def parse_df(tables_df, columns_df, database_name: str, database_node=None):
    """
    Every schema manager assumes a single database in the input file
    :param tables_df: DataFrame with columns: table_schema, table_name
    :param columns_df: DataFrame with columns: table_schema, table_name, column_name, ...
    :param database_name: name of the database, taken from the connector
    :param database_node: optional existing Neo4j database node
    Assumption: the file contains schemas of a single database
    :return:
    """
    if not database_node:
        database_node = CatalogNode(
            name=database_name,
            label=Labels.DB,
            props={"name": database_name, "pulled": datetime.now(timezone.utc)},
            match_props={"name": database_name},
        )

    tables_df = get_table_ids(tables_df, database_name)
    columns_df = get_column_ids(columns_df, database_name)

    unique_schema_names = tables_df["table_schema"].unique()
    schemas = {}

    for schema_name in unique_schema_names:
        schema_tables_df = tables_df.loc[tables_df["table_schema"] == schema_name]
        schema_columns_df = columns_df.loc[columns_df["table_schema"] == schema_name]
        logger.info(f"Started parsing schema {schema_name}.")
        schema = Schema(database_node, schema_tables_df, schema_columns_df)
        schema.create_schema_node(schema_name)
        schemas.update({schema.get_schema_name().lower(): schema})
        logger.info(f"Finished parsing schema {schema.get_schema_name()}.")

    return schemas, database_node
