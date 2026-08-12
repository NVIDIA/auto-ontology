# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.datasources``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import list is explicit rather than ``import *``: it *is* the frozen public
surface both implementations must provide, it doubles as a porting checklist,
and a star-import would silently paper over a function the Postgres side has not
implemented yet.

**Functions only.** ``TABLE_COUNTS_SUBQUERY`` is Cypher, so it is not re-exported
here — ``gsf/dal/exploration.py`` imports it from the Neo4j implementation
directly until Phase 9 ports that module and the constant disappears with it.

Phase 11 deletes this file and promotes ``pg/datasources.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.datasources import (  # noqa: F401
        fetch_databases,
        fetch_schemas_for_database,
        fetch_all_schema_ids,
        fetch_schema_ids_for_database,
        fetch_schemas_by_ids,
        fetch_tables_for_schema,
        fetch_sorted_tables,
        fetch_table_by_id,
        fetch_table_by_name,
        fetch_tables_by_ids,
        fetch_all_tables_without_term,
        fetch_join_neighbors,
        fetch_join_edges,
        fetch_columns_for_table,
        count_columns_for_table,
        fetch_parent_table_id_for_column,
        fetch_table_context,
        fetch_col_table_contexts,
        store_column_sample_values,
        store_column_uniqueness,
        fetch_tables_and_columns_by_node_ids,
        apply_metadata_batch,
        patch_catalog_node,
        fetch_node_properties_by_id,
        fetch_item_by_id,
        fetch_bridge_table_candidates,
    )
else:
    from gsf.dal.neo4j.datasources import (  # noqa: F401
        fetch_databases,
        fetch_schemas_for_database,
        fetch_all_schema_ids,
        fetch_schema_ids_for_database,
        fetch_schemas_by_ids,
        fetch_tables_for_schema,
        fetch_sorted_tables,
        fetch_table_by_id,
        fetch_table_by_name,
        fetch_tables_by_ids,
        fetch_all_tables_without_term,
        fetch_join_neighbors,
        fetch_join_edges,
        fetch_columns_for_table,
        count_columns_for_table,
        fetch_parent_table_id_for_column,
        fetch_table_context,
        fetch_col_table_contexts,
        store_column_sample_values,
        store_column_uniqueness,
        fetch_tables_and_columns_by_node_ids,
        apply_metadata_batch,
        patch_catalog_node,
        fetch_node_properties_by_id,
        fetch_item_by_id,
        fetch_bridge_table_candidates,
    )

__all__ = [
    "fetch_databases",
    "fetch_schemas_for_database",
    "fetch_all_schema_ids",
    "fetch_schema_ids_for_database",
    "fetch_schemas_by_ids",
    "fetch_tables_for_schema",
    "fetch_sorted_tables",
    "fetch_table_by_id",
    "fetch_table_by_name",
    "fetch_tables_by_ids",
    "fetch_all_tables_without_term",
    "fetch_join_neighbors",
    "fetch_join_edges",
    "fetch_columns_for_table",
    "count_columns_for_table",
    "fetch_parent_table_id_for_column",
    "fetch_table_context",
    "fetch_col_table_contexts",
    "store_column_sample_values",
    "store_column_uniqueness",
    "fetch_tables_and_columns_by_node_ids",
    "apply_metadata_batch",
    "patch_catalog_node",
    "fetch_node_properties_by_id",
    "fetch_item_by_id",
    "fetch_bridge_table_candidates",
]
