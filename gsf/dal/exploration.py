# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.exploration``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

``MAX_EXPLORATION_GRAPH_NODES`` is re-exported too: the router reads it to
document its own cap, and a per-backend value would let the two disagree about
how large a payload can get.

Phase 11 deletes this file and promotes ``pg/exploration.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.exploration import (  # noqa: F401
        MAX_EXPLORATION_GRAPH_NODES,
        fetch_data_exploration_edges,
        fetch_table_exploration_details,
        fetch_exploration_related_nodes,
        fetch_table_zones_map,
        fetch_data_exploration_graph,
        fetch_semantic_exploration_graph,
    )
else:
    from gsf.dal.neo4j.exploration import (  # noqa: F401
        MAX_EXPLORATION_GRAPH_NODES,
        fetch_data_exploration_edges,
        fetch_table_exploration_details,
        fetch_exploration_related_nodes,
        fetch_table_zones_map,
        fetch_data_exploration_graph,
        fetch_semantic_exploration_graph,
    )

__all__ = [
    "MAX_EXPLORATION_GRAPH_NODES",
    "fetch_data_exploration_edges",
    "fetch_table_exploration_details",
    "fetch_exploration_related_nodes",
    "fetch_table_zones_map",
    "fetch_data_exploration_graph",
    "fetch_semantic_exploration_graph",
]
