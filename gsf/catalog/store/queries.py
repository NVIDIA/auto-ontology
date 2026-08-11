# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.catalog.store.queries``.

Resolves to the Neo4j or Postgres implementation once, at import, based on
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import lists are explicit rather than ``import *`` for the same reason the
DAL's are: they *are* the frozen public surface both implementations must
provide, they double as a porting checklist, and a star-import would silently
paper over a function the Postgres side has not implemented yet.

Phase 11 deletes this file and promotes ``pg/queries.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.catalog.store.pg.queries import (  # noqa: F401
        add_query,
        get_sql_by_full_query,
        get_sql_counters,
        update_counters_and_timestamps_for_query_and_affected_data,
        load_sqls_to_tables,
        get_candidate_sql_ids,
    )
else:
    from gsf.catalog.store.neo4j.queries import (  # noqa: F401
        add_query,
        get_sql_by_full_query,
        get_sql_counters,
        update_counters_and_timestamps_for_query_and_affected_data,
        load_sqls_to_tables,
        get_candidate_sql_ids,
    )

__all__ = [
    "add_query",
    "get_sql_by_full_query",
    "get_sql_counters",
    "update_counters_and_timestamps_for_query_and_affected_data",
    "load_sqls_to_tables",
    "get_candidate_sql_ids",
]
