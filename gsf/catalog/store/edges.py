# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.catalog.store.edges``.

Resolves to the Neo4j or Postgres implementation once, at import, based on
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import lists are explicit rather than ``import *`` for the same reason the
DAL's are: they *are* the frozen public surface both implementations must
provide, they double as a porting checklist, and a star-import would silently
paper over a function the Postgres side has not implemented yet.

Phase 11 deletes this file and promotes ``pg/edges.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.catalog.store.pg.edges import (  # noqa: F401
        is_flat_dict,
        check_properties_compatibility_with_neo4j,
        prepare_edge,
        prepare_node,
        add_edges,
        get_node_properties_by_id,
        delete_bulk_of_nodes,
        detach_bulk_of_nodes,
        get_node_id_by_name_and_label,
    )
else:
    from gsf.catalog.store.neo4j.edges import (  # noqa: F401
        is_flat_dict,
        check_properties_compatibility_with_neo4j,
        prepare_edge,
        prepare_node,
        add_edges,
        get_node_properties_by_id,
        delete_bulk_of_nodes,
        detach_bulk_of_nodes,
        get_node_id_by_name_and_label,
    )

__all__ = [
    "is_flat_dict",
    "check_properties_compatibility_with_neo4j",
    "prepare_edge",
    "prepare_node",
    "add_edges",
    "get_node_properties_by_id",
    "delete_bulk_of_nodes",
    "detach_bulk_of_nodes",
    "get_node_id_by_name_and_label",
]
