# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What changed in a schema since the last ingest.

Storage-agnostic: pandas over two frames — one read back from the store, one
freshly parsed from the source — plus calls back into the store to apply the
result. Both backends share it, because two copies of the subtlest code in the
write path would have to be fixed twice or drift apart. It was already silently
broken for every re-ingest once; once is enough.

The store calls go through ``gsf.catalog.store.*``, the selector, so this
resolves to whichever backend is active. They are imported inside the functions
rather than at module scope: the selector imports the implementations, which
import this module, and a top-level import would close that circle.
"""

from __future__ import annotations

import logging

import pandas as pd

from gsf.catalog.constants import Labels
from gsf.catalog.normalize import chunks

logger = logging.getLogger(__name__)


# Keys shared by both sides of the column diff.
#
# Upstream merged on ["database", "schema", "table_name", "column_name"].
# Neither frame has a "schema" column -- both call it "table_schema" -- and
# "database" exists only on the graph side, so every merge raised KeyError and
# the column diff never ran.
#
# "database" is redundant anyway: a Schema belongs to exactly one database.
_COLUMN_MERGE_KEYS = ["table_schema", "table_name", "column_name"]


def accumulate_added_column_props(added_column, edges_to_add, new_schema):
    new_table_node_props = new_schema.get_table_node_props(added_column.table_name)
    new_table_node_match_props = new_schema.get_table_node_match_props(
        added_column.table_name
    )
    edges_to_add.append(
        {
            "from_label": Labels.TABLE,
            "from_identProps": new_table_node_match_props,
            "v_props": new_table_node_props,
            "to_label": Labels.COLUMN,
            "to_identProps": added_column.match_props_files,
            "u_props": added_column.props_files,
            "optional_edge_props": {},
        }
    )


def accumulate_updated_column(
    column_in_intersection, items_to_update_in_graph, new_schema
):
    # verify that the node with the correct id is in hand
    # new_schema.replace_id(column_in_intersection.props_y["id"], column_in_intersection.props_x["id"])
    new_column_node_props = column_in_intersection.props_files
    new_column_node_props["id"] = column_in_intersection.props_graph["id"]
    items_to_update_in_graph.append(
        {
            "id": column_in_intersection.props_graph["id"],
            "label": Labels.COLUMN,
            "props": new_column_node_props,
        }
    )


def update_diff_from_existing_schema(new_schema, latest_timestamp):
    from gsf.catalog.store.db import (
        add_schemas_edge_batch,
        delete_columns_batch,
        delete_table,
        update_properties_in_graph_batch,
    )
    from gsf.catalog.store.schemas import add_schemas_edge, load_schema_from_graph

    try:
        # load existing schema
        schema_name = new_schema.get_schema_name()
        database_name = new_schema.get_database_name()

        existing_schema = load_schema_from_graph(database_name, schema_name)
        if existing_schema is None:
            return

        existing_schema_node = existing_schema.get_schema_node()

        existing_table_names = existing_schema.tables_df.table_name.unique()
        new_table_names = new_schema.tables_df.table_name.unique()
        tables_names_to_add = set(new_table_names) - set(existing_table_names)
        logger.info(
            f"Tables to add in schema {schema_name}: {len(tables_names_to_add)}"
        )

        for table_name in tables_names_to_add:
            edge_params = {}
            add_schemas_edge(
                [
                    existing_schema_node,
                    new_schema.get_table_node(table_name),
                    edge_params,
                ],
                latest_timestamp,
            )

            column_names = new_schema.get_table_columns_by_table_name(table_name)
            for column_name in column_names:
                add_schemas_edge(
                    [
                        new_schema.get_table_node(table_name),
                        new_schema.get_column_node(column_name, table_name),
                        edge_params,
                    ],
                    latest_timestamp,
                )

        tables_names_to_delete = set(existing_table_names) - set(new_table_names)
        logger.info(
            f"Tables to delete in schema {schema_name}: {len(tables_names_to_delete)}"
        )
        for deleted_table_name in tables_names_to_delete:
            deleted_table_node_props = existing_schema.get_table_node_props(
                deleted_table_name
            )
            delete_table(deleted_table_node_props["id"])

        # If a table appears in both schemas, identify columns to add and columns to delete.
        columns_merge = pd.merge(
            existing_schema.columns_df,
            new_schema.columns_df,
            on=_COLUMN_MERGE_KEYS,
            how="left",
            suffixes=("_graph", "_files"),
        )
        deleted_columns = columns_merge.loc[
            columns_merge["column_name_lower_files"].isnull()
        ]
        logger.info(
            f"Columns to delete in schema {schema_name}: {len(deleted_columns)}"
        )
        ids_to_delete = deleted_columns["props_graph"].apply(lambda p: p["id"]).tolist()
        delete_columns_batch(ids_to_delete)

        columns_merge = pd.merge(
            existing_schema.columns_df,
            new_schema.columns_df,
            on=_COLUMN_MERGE_KEYS,
            how="right",
            suffixes=("_graph", "_files"),
        )
        added_columns = columns_merge.loc[
            columns_merge["column_name_lower_graph"].isnull()
        ]
        logger.info(f"Columns to add in schema {schema_name}: {len(added_columns)}")
        edges_to_merge = []
        added_columns.apply(
            lambda x: accumulate_added_column_props(x, edges_to_merge, new_schema),
            axis=1,
        )
        add_schemas_edge_batch(edges_to_merge, created=latest_timestamp)

        columns_merge = pd.merge(
            existing_schema.columns_df,
            new_schema.columns_df,
            on=_COLUMN_MERGE_KEYS,
            how="inner",
            suffixes=("_graph", "_files"),
        )

        list_of_props = ["data_type", "is_nullable"]
        if len(list_of_props) > 0:
            column_diffs = []
            for prop in list_of_props:
                column_diff = columns_merge[
                    columns_merge[f"{prop}_graph"].astype(str)
                    != columns_merge[f"{prop}_files"].astype(str)
                ]
                column_diffs.append(column_diff)
            columns_to_update = pd.concat(column_diffs, ignore_index=True, axis=0)
            columns_to_update.drop_duplicates(
                set(columns_to_update.columns)
                - set(
                    [
                        "props_graph",
                        "props_files",
                        "match_props_graph",
                        "match_props_files",
                    ]
                ),
                inplace=True,
            )

            items_to_update_in_graph = []
            columns_to_update.apply(
                lambda x: accumulate_updated_column(
                    x, items_to_update_in_graph, new_schema
                ),
                axis=1,
            )
            items_to_update_in_graph_chunks = list(
                chunks(items_to_update_in_graph, 1000)
            )
            len_chunks = len(items_to_update_in_graph_chunks)
            for i, chunk in enumerate(items_to_update_in_graph_chunks):
                logger.info(f"Updating columns chunk {i + 1}/{len_chunks}")
                update_properties_in_graph_batch(chunk)

    except Exception as err:
        raise Exception(f'Error in "update_diff_from_existing_schema": {err}')
