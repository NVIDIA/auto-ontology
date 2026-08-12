# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.reset`` — per-database reset.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

Classes are selected alongside the functions and must be the *same objects* on
both sides: callers ``except`` these exception types and ``isinstance`` these
dataclasses, and a per-backend copy would stop matching silently.

Phase 11 deletes this file and promotes ``pg/reset.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.reset import (  # noqa: F401
        ResetResult,
        delete_semantic_layer,
        delete_data_layer,
        delete_all_data,
    )
else:
    from gsf.dal.neo4j.reset import (  # noqa: F401
        ResetResult,
        delete_semantic_layer,
        delete_data_layer,
        delete_all_data,
    )

__all__ = [
    "ResetResult",
    "delete_semantic_layer",
    "delete_data_layer",
    "delete_all_data",
]
