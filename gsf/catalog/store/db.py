# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.catalog.store.db``.

Resolves to the Neo4j or Postgres implementation once, at import, based on
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import lists are explicit rather than ``import *`` for the same reason the
DAL's are: they *are* the frozen public surface both implementations must
provide, they double as a porting checklist, and a star-import would silently
paper over a function the Postgres side has not implemented yet.

Phase 11 deletes this file and promotes ``pg/db.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.catalog.store.pg.db import (  # noqa: F401
        db_exists,
        update_node_property,
        delete_schema,
        add_schemas_edge_batch,
        accumulate_added_column_props,
        accumulate_updated_column,
        update_diff_from_existing_schema,
        delete_table,
        update_properties_in_graph_batch,
        delete_columns_batch,
        delete_column,
    )
else:
    from gsf.catalog.store.neo4j.db import (  # noqa: F401
        db_exists,
        update_node_property,
        delete_schema,
        add_schemas_edge_batch,
        accumulate_added_column_props,
        accumulate_updated_column,
        update_diff_from_existing_schema,
        delete_table,
        update_properties_in_graph_batch,
        delete_columns_batch,
        delete_column,
    )

__all__ = [
    "db_exists",
    "update_node_property",
    "delete_schema",
    "add_schemas_edge_batch",
    "accumulate_added_column_props",
    "accumulate_updated_column",
    "update_diff_from_existing_schema",
    "delete_table",
    "update_properties_in_graph_batch",
    "delete_columns_batch",
    "delete_column",
]
