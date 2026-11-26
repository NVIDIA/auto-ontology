# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Apache AGE projection of the catalog and semantic layer.

Postgres tables stay the source of truth. When ``AGE_GRAPH_ENABLED`` is set,
:func:`auto_ontology.graph.projection.sync_graph` rebuilds an openCypher graph
of the same rows inside the same database, and
:mod:`auto_ontology.graph.queries` reads it. Nothing else in Auto Ontology
depends on this package.
"""
