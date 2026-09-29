# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two readiness checks, shared by both services.

The API server and the ingestion service depend on the same Postgres and the
same migrated schema, so they should answer the same way about it. Kept here
rather than duplicated per router because two copies of a health check drift
into disagreeing about whether the system is up, and the one that is wrong is
whichever you are not looking at.

Neither raises. A readiness endpoint has to answer *because* a dependency is
down; turning that into a 500 loses the distinction between "not ready" and
"broken", which is the whole signal.
"""

from __future__ import annotations

from auto_ontology.dal.connections import verify_connectivity
from auto_ontology.dal.schema_version import schema_state


def check_store() -> dict[str, str]:
    """Probe the pooled connection the application actually uses.

    Through the DAL rather than a fresh ``psycopg.connect``: a raw connection
    proves the server is reachable, which is not the same as proving this
    process can get a usable connection out of its pool — and the pool is what
    every request depends on.
    """
    try:
        verify_connectivity()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": (str(exc) or type(exc).__name__)[:200]}


def check_schema() -> dict[str, str]:
    """Report whether the database carries the revision this build expects.

    Both services refuse to *start* on a stale schema, so in practice this
    catches the case that gate cannot: a schema that changes under an already
    running process — a migration applied mid-flight, a table dropped by hand.
    """
    state = schema_state()
    if state.current:
        return {"status": "ok", "revision": state.applied or ""}
    return {"status": "error", "detail": state.detail[:200]}
