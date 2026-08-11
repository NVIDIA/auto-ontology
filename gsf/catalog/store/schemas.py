# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.catalog.store.schemas``.

Resolves to the Neo4j or Postgres implementation once, at import, based on
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import lists are explicit rather than ``import *`` for the same reason the
DAL's are: they *are* the frozen public surface both implementations must
provide, they double as a porting checklist, and a star-import would silently
paper over a function the Postgres side has not implemented yet.

Phase 11 deletes this file and promotes ``pg/schemas.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.catalog.store.pg.schemas import (  # noqa: F401
        load_schema_from_graph,
        get_schemas_ids_and_names,
        get_schema_columns,
        get_schema_tables,
        get_table_ids,
        get_column_ids,
        add_schemas_edge,
        delete_old_fks,
        add_fks,
        reset_pks,
        add_pks,
        merge_schema_nodes,
        merge_schema_edges,
    )
else:
    from gsf.catalog.store.neo4j.schemas import (  # noqa: F401
        load_schema_from_graph,
        get_schemas_ids_and_names,
        get_schema_columns,
        get_schema_tables,
        get_table_ids,
        get_column_ids,
        add_schemas_edge,
        delete_old_fks,
        add_fks,
        reset_pks,
        add_pks,
        merge_schema_nodes,
        merge_schema_edges,
    )

__all__ = [
    "load_schema_from_graph",
    "get_schemas_ids_and_names",
    "get_schema_columns",
    "get_schema_tables",
    "get_table_ids",
    "get_column_ids",
    "add_schemas_edge",
    "delete_old_fks",
    "add_fks",
    "reset_pks",
    "add_pks",
    "merge_schema_nodes",
    "merge_schema_edges",
]
