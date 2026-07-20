# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live, read-only database probing for the text-to-SQL pipeline.

During SQL construction the agent otherwise works "blind" — writing SQL purely
from static schema / semantic context. This package lets the pipeline run cheap,
bounded, read-only queries against the live database so the model grounds its
SQL in observed values (real enum spellings, value ranges, row counts) instead
of guessing.

This package:
- :class:`ProbeExecutor` — the single choke point that guarantees probes are
  ``SELECT``-only, ``LIMIT``-bounded, and capped by a call budget.
- :func:`find_literal_mismatches` (:mod:`literal_check`) — compare filter
  literals in generated SQL against real distinct DB values so wrong
  case/spelling can be repaired.
"""

from gsf.retrieval.text_to_sql.db_probe.config import (
    is_db_probe_proactive,
    DB_PROBE_MAX_CALLS,
    DB_PROBE_MAX_ROWS,
    DB_PROBE_TIMEOUT_S,
)
from gsf.retrieval.text_to_sql.db_probe.executor import (
    ProbeExecutor,
    is_read_only_select,
)
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_literal_mismatches,
)

__all__ = [
    "is_db_probe_proactive",
    "DB_PROBE_MAX_CALLS",
    "DB_PROBE_MAX_ROWS",
    "DB_PROBE_TIMEOUT_S",
    "ProbeExecutor",
    "is_read_only_select",
    "build_value_repair_error",
    "find_literal_mismatches",
]
