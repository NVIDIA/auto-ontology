# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exploration service layer.

Keeps the HTTP router independent from the DAL and provides the
domain boundary for future Exploration-specific orchestration.
"""

from __future__ import annotations

from auto_ontology.dal.exploration import (
    MAX_EXPLORATION_GRAPH_NODES,
    fetch_column_attribute_exploration_details,
    fetch_column_exploration_details,
    fetch_data_exploration_edges,
    fetch_data_exploration_graph,
    fetch_exploration_related_nodes,
    fetch_semantic_exploration_graph,
    fetch_semantic_link_path,
    fetch_sql_attribute_exploration_details,
    fetch_sql_exploration_details,
    fetch_table_exploration_details,
    fetch_table_zones_map,
    fetch_term_exploration_details,
)

__all__ = [
    "MAX_EXPLORATION_GRAPH_NODES",
    "fetch_column_attribute_exploration_details",
    "fetch_column_exploration_details",
    "fetch_data_exploration_edges",
    "fetch_data_exploration_graph",
    "fetch_exploration_related_nodes",
    "fetch_semantic_exploration_graph",
    "fetch_semantic_link_path",
    "fetch_sql_attribute_exploration_details",
    "fetch_sql_exploration_details",
    "fetch_table_exploration_details",
    "fetch_table_zones_map",
    "fetch_term_exploration_details",
]
