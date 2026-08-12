# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.pql_analyses`` — PqlAnalysis reads and writes.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

Classes are selected alongside the functions and must be the *same objects* on
both sides: callers ``except`` these exception types and ``isinstance`` these
dataclasses, and a per-backend copy would stop matching silently.

Phase 11 deletes this file and promotes ``pg/pql_analyses.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.pql_analyses import (  # noqa: F401
        PqlAnalysisNameConflict,
        PqlAnalysisPqlConflict,
        list_pql_analyses,
        find_pql_analysis_by_name,
        find_pql_analysis_by_pql,
        get_pql_analysis_by_id,
        fetch_pql_analyses_by_ids,
        upsert_pql_analysis_node,
        delete_pql_analysis_node,
        embed_pql_analyses,
    )
else:
    from gsf.dal.neo4j.pql_analyses import (  # noqa: F401
        PqlAnalysisNameConflict,
        PqlAnalysisPqlConflict,
        list_pql_analyses,
        find_pql_analysis_by_name,
        find_pql_analysis_by_pql,
        get_pql_analysis_by_id,
        fetch_pql_analyses_by_ids,
        upsert_pql_analysis_node,
        delete_pql_analysis_node,
        embed_pql_analyses,
    )

__all__ = [
    "PqlAnalysisNameConflict",
    "PqlAnalysisPqlConflict",
    "list_pql_analyses",
    "find_pql_analysis_by_name",
    "find_pql_analysis_by_pql",
    "get_pql_analysis_by_id",
    "fetch_pql_analyses_by_ids",
    "upsert_pql_analysis_node",
    "delete_pql_analysis_node",
    "embed_pql_analyses",
]
