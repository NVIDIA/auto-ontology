# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which data store backs the catalog and semantic layer.

One switch, read once at import, selecting between the Neo4j implementation and
the Postgres one for the duration of the process. See
``docs/refactor/drop-neo4j/PLAN.md`` § "Keeping main green" for why the choice
is whole-store rather than per-domain: cross-domain reads are pervasive —
``fetch_table_exploration_details`` spans Table, Term *and* ColumnAttribute in
one statement — so any boundary you could draw is crossed by an existing query.

It lives in :mod:`gsf.infra` rather than under ``gsf.dal`` because both
``gsf.dal`` (reads) and ``gsf.catalog.store`` (writes) select on it, and
``gsf.catalog`` must not depend on ``gsf.dal``.

**Read once, at import.** A value that could change mid-process would let one
request read Neo4j and the next read Postgres, which is worse than either.
Tests that need the other backend must patch the selector module, not this
value.

The three processes — API, ingestion service, and the *spawned* chat worker —
each read it independently, so it has to be set in the environment of all
three, not just the one you started by hand.
"""

from __future__ import annotations

import os

NEO4J = "neo4j"
POSTGRES = "postgres"

_VALID = frozenset({NEO4J, POSTGRES})

#: Default stays ``neo4j`` until Phase 11 flips it. Every phase up to then lands
#: a Postgres implementation alongside the Neo4j one without changing what runs.
STORE = os.environ.get("GSF_STORE", NEO4J).strip().lower()

if STORE not in _VALID:
    raise RuntimeError(
        f"GSF_STORE must be one of {sorted(_VALID)}, got {STORE!r}. "
        f"Unset it to use the default ({NEO4J})."
    )

#: Convenience flag; prefer this over comparing strings at call sites.
USE_PG = STORE == POSTGRES
