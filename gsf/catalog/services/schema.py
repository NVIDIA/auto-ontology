# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add one schema's nodes and edges to the store."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import repeat
from gsf.catalog.normalize import chunks
from gsf.catalog.store.schemas import (
    PendingRows,
    add_schemas_edge,
    merge_schema_edges,
    merge_schema_nodes,
)
from gsf.catalog.constants import Labels
from gsf.catalog.model.schema import Schema

logger = logging.getLogger(__name__)


def add_table(table_edges, pending: PendingRows):
    merge_schema_edges(table_edges, Labels.TABLE, Labels.COLUMN, pending)


def add_schema(
    schema: Schema,
    latest_timestamp: datetime,
    num_workers: int,
):
    """
    Add all the nodes and edges of the given schema.
    If the schema exists then:
    new nodes - will be added with the given latest_timestamp.
    remaining nodes - the latest_timestamp will be updated.
    missing nodes - the property delete=True will be added to these nodes.
    """
    # Scoped to this call: the column rows staged below are claimed by the edge
    # passes further down, and anything left over dies with this frame.
    pending = PendingRows()

    try:
        db_schema_edge = schema.get_db_schema_edge()
        add_schemas_edge(db_schema_edge, latest_timestamp)

        table_column_nodes_chunks = list(
            chunks(
                [
                    {
                        "label": [Labels.TABLE],
                        "match_props": x["match_props"],
                        "props": x["props"],
                    }
                    for x in schema.get_table_nodes()
                ],
                500,
            )
        ) + list(
            chunks(
                [
                    {
                        "label": [Labels.COLUMN],
                        "match_props": x["match_props"],
                        "props": x["props"],
                    }
                    for x in schema.get_column_nodes()
                ],
                500,
            )
        )
        for table_column_nodes in table_column_nodes_chunks:
            merge_schema_nodes(table_column_nodes, latest_timestamp, pending)

        edges_chunks = list(chunks(schema.get_schema_to_tables_edges(), 500))
        for edges in edges_chunks:
            merge_schema_edges(edges, Labels.SCHEMA, Labels.TABLE, pending)

        edges_per_table = schema.get_edges_per_table()
        with ThreadPoolExecutor(num_workers) as executor:
            # `list(...)` is load-bearing: an exception in a worker is stored in
            # its future and re-raised only when consumed. Left unconsumed,
            # every failure to write a table's columns is discarded in silence
            # and this function reports success having written nothing.
            list(executor.map(add_table, edges_per_table, repeat(pending)))

    except Exception as err:
        logger.error(f"Failed adding schema: {schema.get_schema_name()}")
        logger.exception(err)
        raise
