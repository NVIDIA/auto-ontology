# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.custom_analyses`` — CustomAnalysis reads and writes.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

Classes are selected alongside the functions and must be the *same objects* on
both sides: callers ``except`` these exception types and ``isinstance`` these
dataclasses, and a per-backend copy would stop matching silently.

Phase 11 deletes this file and promotes ``pg/custom_analyses.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.custom_analyses import (  # noqa: F401
        CustomAnalysisNameConflict,
        CustomAnalysisSqlConflict,
        CustomAnalysisSqlError,
        list_custom_analyses,
        find_analysis_by_name,
        find_analysis_by_sql,
        detach_existing_sql_edges,
        fetch_custom_analyses,
        embed_custom_analyses,
        fetch_custom_analyses_with_sql,
        get_custom_analysis_by_id,
        delete_custom_analysis_node,
        fetch_tables_from_custom_analyses,
    )
else:
    from gsf.dal.neo4j.custom_analyses import (  # noqa: F401
        CustomAnalysisNameConflict,
        CustomAnalysisSqlConflict,
        CustomAnalysisSqlError,
        list_custom_analyses,
        find_analysis_by_name,
        find_analysis_by_sql,
        detach_existing_sql_edges,
        fetch_custom_analyses,
        embed_custom_analyses,
        fetch_custom_analyses_with_sql,
        get_custom_analysis_by_id,
        delete_custom_analysis_node,
        fetch_tables_from_custom_analyses,
    )

__all__ = [
    "CustomAnalysisNameConflict",
    "CustomAnalysisSqlConflict",
    "CustomAnalysisSqlError",
    "list_custom_analyses",
    "find_analysis_by_name",
    "find_analysis_by_sql",
    "detach_existing_sql_edges",
    "fetch_custom_analyses",
    "embed_custom_analyses",
    "fetch_custom_analyses_with_sql",
    "get_custom_analysis_by_id",
    "delete_custom_analysis_node",
    "fetch_tables_from_custom_analyses",
]
